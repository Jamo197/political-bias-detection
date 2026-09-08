"""LLM-as-a-Judge calibration script.

Runs an OpenRouter judge over the human-annotated chunks saved by
``run_streamlit.py``, computes agreement metrics (QWK for ordinal
R_top/R_ideo, standard Cohen's κ for nominal N_info/A_caus), and
produces a calibration report.

Usage (from the project root):

    python src/run_llm_judge.py --limit 5
    python src/run_llm_judge.py --model openai/gpt-5-mini --concurrency 4

Each run writes a per-run snapshot (annotations JSONL) and the JSON agreement
report into a timestamped subfolder under ``RAG Analysis/<run_id>/`` (similar to
how ``evaluate_metrics.py`` organizes results). The append-only log still
accumulates across runs.

For calibration iterations, edit the JUDGE_SYSTEM_PROMPT constant at the top
of this file or pass ``--prompt-file``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm

from run_streamlit import load_existing_annotations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env.local")

DEFAULT_MODEL = "deepseek/deepseek-v4-flash-0731"
DEFAULT_ANNOTATIONS_PATH = "RAG Analysis/qualitative_annotations.jsonl"
DEFAULT_OUTPUT_ROOT = "RAG Analysis"
DEFAULT_OUTPUT_LOG = "RAG Analysis/llm_judge_annotations.jsonl"

# TODO: Update prompt to get them better understand the task and the dimensions.
# also maybe add more informations, like the justification of the RAG and the poltical bias and its error
JUDGE_SYSTEM_PROMPT = """You are an exacting, skeptical political science judge evaluating Information Retrieval (RAG) performance in political stance and bias classification.

You evaluate whether a retrieved text passage provides valid, actionable, and causally utilized evidence for classifying an input text's political bias.

--- EVALUATION TARGETS ---
1. INPUT TEXT: The text to be classified (social media post, speech snippet, quote).
2. RETRIEVED CHUNK: The contextual passage retrieved from a parliamentary speech.
3. GENERATOR JUSTIFICATION: The classifier's stated rationale for its assigned bias score.

--- SCORING RUBRICS & STRICT BOUNDARIES ---

1. Topical Relevance (R_top) — [Integer 1 to 5]
Measure topic alignment strictly by specific policy target and legislative mechanism:
• [1] IRRELEVANT: Completely different topic, domain, or societal sphere.
• [2] BROAD THEMATIC DOMAIN ONLY: Shares high-level domain (e.g., public finance, social welfare, environment), but targets different policies, different debates, or different eras. 
  -> NOTE: If Input is a generic complaint about taxes/costs and the Chunk is a debate on a specific federal budget or Schuldenbremse, this is STRICTLY Level 2.
• [3] RELATED SUB-ISSUE: Same sub-policy domain, but discusses different specific mechanisms, opposing bills, or non-overlapping details.
• [4] DIRECT POLICY OVERLAP: Both texts debate the EXACT same policy measure, bill, or specific institutional mechanism (e.g., both specifically debate the Schuldenbremse or Bürgergeld rates), differing only in rhetorical framing.
• [5] IDENTICAL TARGET: Exact identical legislative bill, motion, entity, or specific quote referenced in both texts.

2. Ideological Specificity (R_ideo) — [Integer 1 to 5]
Measure the ideological clarity of the RETRIEVED CHUNK on its own merits:
• [1] PROCEDURAL / NEUTRAL: Bureaucratic announcements, committee schedules, uncontroversial administrative statements.
• [2] DESCRIPTIVE REPORTING: Mentions political conflict objectively without taking an ideological stance.
• [3] VALUE-LADEN RHETORIC: Uses polarized adjectives, general critique, or emotional appeals without clear programmatic ideology.
• [4] DISTINCT IDEOLOGICAL POSITION: Clear, partisan policy stance (e.g., fiscal hawkishness, deregulation, welfare expansion) expressed in parliamentary debate or public speech.
• [5] BINDING MANIFESTO / FORMAL PROGRAMME: Explicit reference to official party programmes, election manifestos, formal coalition agreements, or binding roll-call votes.
  -> NOTE: A speaker shouting "Not with us as CDU/CSU!" on the parliament floor is Level 4 (parliamentary debate rhetoric), NOT Level 5.

3. Information Delta (N_info) — [Integer 1 to 3]
Does the chunk provide necessary, actionable knowledge to classify the INPUT TEXT's ideological stance?
*HARD CONSTRAINT:* If R_top <= 2, N_info MUST be 1. Irrelevant or broadly thematic chunks cannot provide actionable context for the input text.
• [1] ZERO DELTA / DISTRACTING: Provides no new actionable clues for the input text, repeats what the input already made obvious, or introduces off-topic facts that risk topic drift.
• [2] SUPPLEMENTARY BACKGROUND: Clarifies contextual details (dates, acronyms, institutional settings) that marginally help interpret the input text.
• [3] ESSENTIAL GROUNDING: Provides missing contextual facts directly required to identify the input text's stance (e.g., identifies an obscure policy term or decodes dog whistles).

4. Attribution / Faithfulness (A_caus) — [Binary 0 or 1]
Did the generator justification demonstrably rely on THIS specific chunk?
• [0] NO / SPURIOUS: The generator does not explicitly cite this chunk index/anchor, or uses generic reasoning that could have been produced from the input text alone or model pre-training.
• [1] YES: The justification explicitly cites this chunk's index (e.g., [Chunk X], [I-15]) AND derives its arguments from facts, figures, or claims unique to this chunk.

--- EXECUTION INSTRUCTIONS ---
You must evaluate step-by-step. Return a single valid JSON object. Do not include markdown code fences or conversational text.

Required JSON Structure:
{
  "input_core_subject": "<Specific policy or issue debated in the input text>",
  "chunk_core_subject": "<Specific policy or issue debated in the retrieved chunk>",
  "topic_overlap_rationale": "<Step-by-step reason distinguishing Level 2 (broad domain) from Level 4 (direct policy overlap)>",
  "ideological_rationale": "<Step-by-step reason distinguishing Level 4 (parliamentary stance) from Level 5 (formal doctrine)>",
  "information_utility_rationale": "<Evaluation of whether the chunk clarifies the input text; apply R_top <= 2 constraint>",
  "attribution_check": "<Verification of explicit citation and unique fact usage in generator justification>",
  "R_top": <int 1-5>,
  "R_ideo": <int 1-5>,
  "N_info": <int 1-3>,
  "A_caus": <int 0-1>
}
"""

# ---------------------------------------------------------------------------
# Prompt & message building
# ---------------------------------------------------------------------------


def build_user_message(record: dict, chunk: dict, include_anchors: bool = True) -> str:
    """Construct a cleanly delineated, un-biased user message for the LLM judge."""
    meta = chunk.get("chunk_metadata", {})
    chunk_idx_1based = chunk.get("chunk_index", 0) + 1
    rag = record.get("source_context", {}).get("rag", {})
    justification = rag.get("justification", "").strip() or "No justification provided."

    # Format metadata cleanly without leaking similarity/retrieval scores
    metadata_lines = [
        f"- Chunk Identifier: [{chunk_idx_1based}] (or 'Chunk {chunk_idx_1based}')",
        f"- Target Party: {meta.get('party', 'Unknown')}",
        f"- Speaker: {meta.get('speaker', 'Unknown')}",
        f"- Source Document: {meta.get('source', 'Unknown')}",
    ]
    if meta.get("date"):
        metadata_lines.append(f"- Date: {meta.get('date')}")

    metadata_block = "\n".join(metadata_lines)

    user_message = f"""### RETRIEVED CHUNK METADATA
    {metadata_block}

    ### INPUT TEXT (Target of Bias Classification)

    {record.get("input_text", "").strip()}

    ### RETRIEVED PASSAGE (Chunk [{chunk_idx_1based}])

    {chunk.get("chunk_text", "").strip()}

    ### GENERATOR JUSTIFICATION (To evaluate for A_caus)

    {justification}

    ### SCORING TASK & CRITICAL REMINDERS

    Judge Chunk [{chunk_idx_1based}] against the Input Text using the 4 dimensions.

    * R_top Check: If the chunk debates a different policy, bill, or fiscal year than the input text, R_top MUST NOT exceed 2.
    * Hard Gating: If R_top <= 2, N_info MUST be 1.
    * A_caus Check: Assign 1 ONLY if the justification explicitly references [{chunk_idx_1based}] or 'Chunk {chunk_idx_1based}' AND uses facts unique to this passage.

    Output ONLY the raw JSON object conforming to the required schema."""

    return user_message


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------


def _judge_structured(
    client: OpenAI, model: str, system_prompt: str, user_message: str
) -> dict:
    """Single attempt with response_format json_schema."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=1500,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "chunk_judge",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "R_top": {"type": "integer"},
                        "R_ideo": {"type": "integer"},
                        "N_info": {"type": "integer"},
                        "A_caus": {"type": "integer"},
                    },
                    "required": ["R_top", "R_ideo", "N_info", "A_caus"],
                    "additionalProperties": False,
                },
            },
        },
        extra_headers={
            "HTTP-Referer": "https://github.com/Jamo197/political-bias-detection",
            "X-Title": "LLM Judge Calibration",
        },
    )
    raw = resp.choices[0].message.content.strip()
    return json.loads(raw)


def _judge_plain(
    client: OpenAI, model: str, system_prompt: str, user_message: str
) -> dict:
    """Fallback attempt without structured response format."""
    user_message = (
        user_message
        + "\n\nIMPORTANT: Your response must be ONLY the JSON object, with no markdown code fences and no extra text."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=1500,
        extra_headers={
            "HTTP-Referer": "https://github.com/Jamo197/political-bias-detection",
            "X-Title": "LLM Judge Calibration",
        },
    )
    raw = resp.choices[0].message.content.strip()

    # --- evaluator-style robust JSON extraction ---
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw.rsplit("\n", 1)[0]
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    return json.loads(raw)


def judge_chunk(
    client: OpenAI, model: str, system_prompt: str, user_message: str
) -> dict:
    """Judge a chunk with retry and fallback."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            if attempt == 0:
                return _judge_structured(client, model, system_prompt, user_message)
            else:
                return _judge_plain(client, model, system_prompt, user_message)
        except Exception as e:  # noqa: BLE001
            last_error = e
            # BadRequest usually means response_format or temp not supported
            if isinstance(e, (ValueError, json.JSONDecodeError)):
                # Retry on JSON parse failures (transient formatting issues)
                time.sleep(2**attempt)
                continue
            if hasattr(e, "status_code") and getattr(e, "status_code", None) == 400:
                # 400 errors → retry with plain fallback immediately
                continue
            # Everything else (transient 5xx, connection, timeout) → backoff
            time.sleep(2**attempt)
            continue
    raise (
        last_error
        if last_error is not None
        else RuntimeError("Judge failed after retries")
    )


def _validate_and_clean(raw: dict) -> dict:
    """Clamp values to valid ranges and ensure types."""
    result = {}
    for key, lo, hi in [
        ("R_top", 1, 5),
        ("R_ideo", 1, 5),
        ("N_info", 1, 3),
        ("A_caus", 0, 1),
    ]:
        val = raw.get(key)
        if val is None:
            result[key] = None
            continue
        try:
            val = int(val)
        except TypeError, ValueError:
            result[key] = None
            continue
        result[key] = max(lo, min(hi, val))
    return result


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_judging(
    args: argparse.Namespace, run_snapshot_path: Path | None = None
) -> tuple[list[dict], int, str]:
    """Load annotations, judge chunks, write log, return results."""
    records = load_existing_annotations(args.annotations)
    if not records:
        print("No human annotations found. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Flatten to (record, chunk) tasks
    tasks = []
    for _text_index, record in records.items():
        for chunk in record.get("chunk_annotations", []):
            tasks.append((record, chunk))

    if args.limit:
        tasks = tasks[: args.limit]

    # Prompt setup
    if args.prompt_file:
        system_prompt = Path(args.prompt_file).read_text(encoding="utf-8")
        prompt_file_path = args.prompt_file
    else:
        system_prompt = JUDGE_SYSTEM_PROMPT
        prompt_file_path = None

    prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]

    # OpenRouter client
    api_key = os.getenv("OPENROUTER_API_KEY_ME", "")
    if not api_key:
        print("Error: OPENROUTER_API_KEY_ME not set in .env.local", file=sys.stderr)
        sys.exit(1)
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Process tasks
    def _process(record: dict, chunk: dict) -> dict:
        chunk_idx = chunk["chunk_index"]
        msg = build_user_message(record, chunk)
        try:
            raw = judge_chunk(client, args.model, system_prompt, msg)
            cleaned = _validate_and_clean(raw)
            return {
                "text_index": record["text_index"],
                "chunk_index": chunk_idx,
                "model": args.model,
                "prompt_hash": prompt_hash,
                "prompt_file": prompt_file_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "judge": cleaned,
                "human": {
                    "R_top": chunk["R_top"],
                    "R_ideo": chunk["R_ideo"],
                    "N_info": chunk["N_info"],
                    "A_caus": chunk["A_caus"],
                },
            }
        except Exception as e:  # noqa: BLE001
            return {
                "text_index": record["text_index"],
                "chunk_index": chunk_idx,
                "model": args.model,
                "prompt_hash": prompt_hash,
                "prompt_file": prompt_file_path,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "human": {
                    "R_top": chunk["R_top"],
                    "R_ideo": chunk["R_ideo"],
                    "N_info": chunk["N_info"],
                    "A_caus": chunk["A_caus"],
                },
            }

    results: list[dict] = []
    failed = 0

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {executor.submit(_process, r, c): (r, c) for r, c in tasks}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Judging chunks"
            ):
                res = future.result()
                results.append(res)
                if "error" in res:
                    failed += 1
    else:
        for record, chunk in tqdm(tasks, desc="Judging chunks"):
            res = _process(record, chunk)
            results.append(res)
            if "error" in res:
                failed += 1

    # Append to output log
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for res in results:
            handle.write(json.dumps(res, ensure_ascii=False) + "\n")

    # Write per-run snapshot into the run output folder
    if run_snapshot_path:
        run_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with run_snapshot_path.open("w", encoding="utf-8") as handle:
            for res in results:
                handle.write(json.dumps(res, ensure_ascii=False) + "\n")

    return results, failed, prompt_hash


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _safe_kappa(y1, y2, labels=[1, 2, 3, 4, 5], weights=None):
    """Compute Cohen's kappa, handling the all-agree edge case."""
    k = cohen_kappa_score(y1, y2, labels=labels, weights=weights)
    if np.isnan(k):
        if len(y1) > 0 and np.all(y1 == y2):
            return 1.0
        return None
    return float(k)


def compute_metrics(results: list[dict]) -> dict:
    """Compute agreement metrics per dimension."""
    dims = {
        "R_top": {"ordinal": True, "labels": [1, 2, 3, 4, 5]},
        "R_ideo": {"ordinal": True, "labels": [1, 2, 3, 4, 5]},
        "N_info": {"ordinal": False, "labels": [1, 2, 3]},
        "A_caus": {"ordinal": False, "labels": [0, 1]},
    }
    per_dim = {}
    for dim, cfg in dims.items():
        h_vals = []
        j_vals = []
        for r in results:
            if "error" in r or "judge" not in r:
                continue
            h = r["human"][dim]
            j = r["judge"].get(dim)
            if j is None or h is None:
                continue
            h_vals.append(int(h))
            j_vals.append(int(j))

        if not h_vals:
            per_dim[dim] = None
            continue

        h_arr = np.array(h_vals)
        j_arr = np.array(j_vals)
        n = len(h_arr)
        agreement = float(np.mean(h_arr == j_arr))
        mean_diff = float(np.mean(j_arr - h_arr))

        fixed_labels = cfg["labels"]
        cm = confusion_matrix(h_arr, j_arr, labels=fixed_labels)

        qwk = None
        kappa = None
        if cfg["ordinal"]:
            qwk = _safe_kappa(h_arr, j_arr, weights="quadratic")
            metric_val = qwk
        else:
            kappa = _safe_kappa(h_arr, j_arr, weights=None)
            metric_val = kappa

        below = metric_val is not None and metric_val < 0.70

        per_dim[dim] = {
            "n": int(n),
            "exact_agreement": agreement,
            "qwk": qwk if cfg["ordinal"] else None,
            "kappa": kappa if not cfg["ordinal"] else None,
            "mean_diff": mean_diff,
            "labels": fixed_labels,
            "confusion_matrix": cm.tolist(),
            "below_threshold": below,
        }

    return per_dim


def get_worst_disagreements(results: list[dict], n: int = 5) -> list[dict]:
    """Return the top-N largest |diff| disagreements for ordinal dims."""
    disagreements = []
    for r in results:
        if "error" in r or "judge" not in r:
            continue
        for dim in ["R_top", "R_ideo"]:
            h = r["human"][dim]
            j = r["judge"].get(dim)
            if j is None:
                continue
            diff = abs(int(j) - int(h))
            if diff > 0:
                disagreements.append(
                    {
                        "text_index": r["text_index"],
                        "chunk_index": r["chunk_index"],
                        "dimension": dim,
                        "human": int(h),
                        "judge": int(j),
                        "diff": diff,
                    }
                )
    disagreements.sort(key=lambda x: (-x["diff"], x["text_index"], x["chunk_index"]))
    return disagreements[:n]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_and_save_report(
    results: list[dict],
    per_dim: dict,
    worst: list[dict],
    model: str,
    prompt_hash: str,
    report_path: str,
) -> None:
    """Print a concise report to stdout and write a JSON report to disk."""
    judged = len([r for r in results if "judge" in r])
    total = len(results)

    print("\n=== LLM Judge Agreement Report ===")
    print(f"Model: {model}")
    print(f"Prompt hash: {prompt_hash}")
    print(f"Judged chunks: {judged} / {total}")

    print("\nPer-dimension metrics:")
    header = f"{'Dim':<12} {'n':>6} {'Agree%':>7} {'κ/κ_w':>7} {'MeanΔ':>7}"
    print(header)
    print("-" * len(header))
    for dim in ["R_top", "R_ideo", "N_info", "A_caus"]:
        d = per_dim[dim]
        if d is None:
            print(f"{dim:<12} {'—':>6} {'—':>7} {'—':>7} {'—':>7}")
            continue
        metric_val = d["qwk"] if dim in ("R_top", "R_ideo") else d["kappa"]
        metric_label = "κ_w" if dim in ("R_top", "R_ideo") else "κ"
        flag = " ⚠️ below 0.70" if d.get("below_threshold") else ""
        metric_str = f"{metric_val:>6.3f}" if metric_val is not None else "   N/A"
        print(
            f"{dim:<12} {d['n']:>6} {d['exact_agreement']*100:>6.1f}% "
            f"{metric_str} {d['mean_diff']:>+6.2f}{flag}"
        )

    print("\nConfusion matrices:")
    for dim in ["R_top", "R_ideo", "N_info", "A_caus"]:
        d = per_dim[dim]
        if d is None:
            continue
        print(f"\n{dim} (labels: {d['labels']})")
        for row, label in zip(d["confusion_matrix"], d["labels"]):
            print(f"  {label}: {row}")

    if worst:
        print(f"\nTop {len(worst)} disagreements (|diff| ≥ 1):")
        for wd in worst:
            print(
                f"  sample {wd['text_index']} chunk {wd['chunk_index']} "
                f"{wd['dimension']}: human={wd['human']} judge={wd['judge']} diff={wd['diff']}"
            )

    report = {
        "model": model,
        "prompt_hash": prompt_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_chunks": total,
        "judged_chunks": judged,
        "per_dimension": per_dim,
        "worst_disagreements": worst,
    }
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with report_file.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {report_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-judge against human chunk annotations and compute agreement metrics."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model ID")
    parser.add_argument(
        "--annotations",
        default=DEFAULT_ANNOTATIONS_PATH,
        help="Path to human annotation JSONL",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder for per-run results (a run subfolder is created inside).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Name of the output subfolder (default: timestamp).",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_LOG,
        help="Append-only log for LLM annotations",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path for the JSON agreement report (default: <output-root>/<run-id>/llm_judge_report.json)",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Override the embedded system prompt with a file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Judge only the first N chunks (smoke test)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="ThreadPoolExecutor workers"
    )
    parser.add_argument(
        "--show-disagreements",
        type=int,
        default=5,
        help="Print top N worst disagreements for R_top / R_ideo",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    run_snapshot_path = output_dir / "llm_judge_annotations.jsonl"
    report_path = (
        Path(args.report) if args.report else output_dir / "llm_judge_report.json"
    )

    results, failed, prompt_hash = run_judging(args, run_snapshot_path)
    per_dim = compute_metrics(results)
    worst = get_worst_disagreements(results, n=args.show_disagreements)
    print_and_save_report(results, per_dim, worst, args.model, prompt_hash, report_path)

    if failed:
        print(f"\nWarning: {failed} chunk(s) failed judging.")


if __name__ == "__main__":
    main()
