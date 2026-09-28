"""Jev (TypeSafe System One) judge calibration script.

Runs the Jev decision model through the TypeSafe Python SDK over the
human-annotated chunks saved by ``run_streamlit.py``, computes agreement
metrics (QWK for ordinal R_top/R_ideo, standard Cohen's κ for nominal
A_caus), and produces a calibration report.

By default the SDK talks to the native TypeSafe API using ``TYPESAFE_API_KEY``
from ``.env.local``. When that key is absent it falls back to the
OpenRouter-compatible endpoint with ``OPENROUTER_API_KEY_ME``. The model id is
pinned per transport (see ``TYPESAFE_JEV_MODEL`` / ``OPENROUTER_JEV_MODEL``);
the response's ``model`` field records the version that actually answered.

Each (record, chunk) pair becomes one System One request whose four typed
questions are answered in parallel and independently:

* ``R_top``  — Score, 5 ordered levels (topical relevance)
* ``R_ideo`` — Score, 5 ordered levels (ideological specificity)
* ``A_caus`` — Noul, probability that the justification relied on this chunk

Jev answers are typed values with probabilities, not generated text, so there
is no prompt parsing or JSON-repair fallback. The raw answers (per-level
probabilities, probability-weighted score, confidence, usage/cost) are stored
alongside the derived integer labels, so thresholds and label derivation can
be re-tuned offline from a previous run's JSONL with ``--rederive-from``
without new API calls.

Retries, backoff, and ``Retry-After`` handling are delegated to the SDK's
``RetryPolicy`` (see ``RETRY_POLICY``), so no hand-rolled HTTP client is
needed.

Usage (from the project root):

    python src/run_llm_judge.py --limit 5
    python src/run_llm_judge.py --concurrency 8
    python src/run_llm_judge.py --a-caus-threshold 0.6 \
        --rederive-from "RAG Analysis/<run_id>/llm_judge_annotations.jsonl"

Each run writes a per-run snapshot (annotations JSONL) and the JSON agreement
report into a timestamped subfolder under ``RAG Analysis/<run_id>/`` (similar
to how ``evaluate_metrics.py`` organizes results). The append-only log still
accumulates across runs.

For calibration iterations, edit the JEV_QUESTIONS constant at the top of
this file or pass ``--questions-file`` (a JSON file with the same shape).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from tqdm import tqdm
from typesafe_sdk import RetryPolicy, TypeSafeClient

from run_streamlit import load_existing_annotations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env.local")

# Native TypeSafe API (TYPESAFE_API_KEY). Pinned to a version id so confidence
# thresholds stay tuned against one model version; the response's `model` field
# records the version that actually answered.
TYPESAFE_JEV_MODEL = os.getenv("TYPESAFE_JEV_MODEL", "jev-1.13.0")
# OpenRouter-compatible endpoint (OPENROUTER_API_KEY_ME), used only when no
# TYPESAFE_API_KEY is present.
OPENROUTER_BASE_URL = "https://openrouter.ai/api"
OPENROUTER_JEV_MODEL = os.getenv("OPENROUTER_JEV_MODEL", "~typesafe/jev-latest")

DEFAULT_ANNOTATIONS_PATH = "RAG Analysis/qualitative_annotations.jsonl"
DEFAULT_OUTPUT_ROOT = "RAG Analysis"
DEFAULT_OUTPUT_LOG = "RAG Analysis/llm_judge_annotations.jsonl"

DEFAULT_A_CAUS_THRESHOLD = 0.5
DEFAULT_SCORE_LABEL_MODE = "argmax"  # "argmax" | "round"
LOW_CONFIDENCE_FLOOR = 0.5
NEAR_THRESHOLD_MARGIN = 0.15  # |noul - a_caus_threshold| below this is flagged
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT_SECONDS = 180  # Jev calls can take ~60s; keep headroom

# The SDK owns backoff, jitter, and Retry-After handling. `timeout=None` keeps
# the retry budget unbounded so a slow first attempt does not suppress retries;
# each individual attempt is still capped by the client timeout above.
RETRY_POLICY = RetryPolicy(
    max_retries=MAX_ATTEMPTS - 1,
    backoff_initial=1.0,
    backoff_max=30.0,
    timeout=None,
)

EVALUATION_TASK = (
    "Evaluating evidence quality in a RAG pipeline for political stance and "
    "bias classification. `input_text` is a German political text (social "
    "media post, speech snippet, or quote) that was classified for political "
    "bias by a generator. `retrieved_chunk` is the passage the generator "
    "retrieved from a corpus of German parliamentary speeches as evidence, "
    "and `generator_justification` is the rationale it produced for its "
    "classification. Judge whether the retrieved chunk is valid, actionable, "
    "and causally utilized evidence for classifying the input text. The "
    "question rubrics are written in English; the texts themselves are German."
)

# Rubric level texts ported from the human annotation guidelines, restructured
# per the TypeSafe docs:
#   * instructions are objects: "question" + "compare"/"inspect" + "focus", and
#     reference state fields via backticked paths (e.g. `retrieved_chunk.text`)
#   * Score levels are {"what", "examples"} objects — the documented remedy when
#     the model splits probability between neighbouring levels; same field names
#     on every level so the model can compare like with like
# Score criteria are ordered ascending: criteria[0] is level 1 (Jev levels are
# 0-indexed; derive_labels() shifts back to the 1-based human scales).
JEV_QUESTIONS = {
    "R_top": {
        "type": "score",
        "instructions": {
            "question": "How topically relevant is the retrieved chunk to the input text?",
            "compare": ["`input_text`", "`retrieved_chunk.text`"],
            "focus": (
                "Judge strictly by specific policy target and legislative "
                "mechanism, not by broad theme."
            ),
        },
        "criteria": [
            {
                "what": "IRRELEVANT — completely different topic, domain, or societal sphere.",
                "examples": [
                    "The input text debates pension reform; the chunk is about fisheries quotas."
                ],
            },
            {
                "what": (
                    "BROAD THEMATIC DOMAIN ONLY — shares a high-level domain "
                    "(e.g. public finance, social welfare, environment), but "
                    "targets different policies, different debates, or "
                    "different eras."
                ),
                "examples": [
                    "The input text is a generic complaint about taxes or "
                    "costs; the chunk debates a specific federal budget or the "
                    "Schuldenbremse."
                ],
            },
            {
                "what": (
                    "RELATED SUB-ISSUE — same sub-policy domain, but different "
                    "specific mechanisms, opposing bills, or non-overlapping "
                    "details."
                ),
                "examples": [
                    "Both texts concern labour-market policy, but the input "
                    "text debates the minimum wage while the chunk debates "
                    "unemployment-benefit sanctions."
                ],
            },
            {
                "what": (
                    "DIRECT POLICY OVERLAP — both texts debate the exact same "
                    "policy measure, bill, or specific institutional "
                    "mechanism, differing only in rhetorical framing."
                ),
                "examples": [
                    "Both texts specifically debate the Schuldenbremse or the "
                    "Bürgergeld rates."
                ],
            },
            {
                "what": (
                    "IDENTICAL TARGET — the exact same legislative bill, "
                    "motion, entity, or specific quote is referenced in both "
                    "texts."
                ),
                "examples": [
                    "Both texts reference the same named amendment or the same "
                    "quoted statement."
                ],
            },
        ],
    },
    "R_ideo": {
        "type": "score",
        "instructions": {
            "question": (
                "How ideologically specific is the retrieved chunk on its own "
                "merits, as evidence about political positions?"
            ),
            "inspect": "`retrieved_chunk.text`",
            "focus": (
                "Judge the chunk's own content, not the input text and not "
                "the generator justification."
            ),
        },
        "criteria": [
            {
                "what": (
                    "PROCEDURAL / NEUTRAL — bureaucratic announcements, "
                    "committee schedules, uncontroversial administrative "
                    "statements."
                ),
                "examples": [
                    "A speaker announcing the next agenda item or a vote " "schedule."
                ],
            },
            {
                "what": (
                    "DESCRIPTIVE REPORTING — mentions political conflict "
                    "objectively without taking an ideological stance."
                ),
                "examples": [
                    "A neutral summary that two parties disagree about a bill."
                ],
            },
            {
                "what": (
                    "VALUE-LADEN RHETORIC — polarized adjectives, general "
                    "critique, or emotional appeals without a clear "
                    "programmatic ideology."
                ),
                "examples": [
                    "Calling a policy 'a disgrace' or 'common sense' without "
                    "any stated programme."
                ],
            },
            {
                "what": (
                    "DISTINCT IDEOLOGICAL POSITION — a clear, partisan policy "
                    "stance (e.g. fiscal hawkishness, deregulation, welfare "
                    "expansion) expressed in parliamentary debate or public "
                    "speech."
                ),
                "examples": [
                    "A speaker declaring 'Not with us as CDU/CSU!' on the "
                    "parliament floor — debate rhetoric, not formal doctrine."
                ],
            },
            {
                "what": (
                    "BINDING MANIFESTO / FORMAL PROGRAMME — explicit reference "
                    "to official party programmes, election manifestos, formal "
                    "coalition agreements, or binding roll-call votes."
                ),
                "examples": [
                    "Quoting the coalition agreement or a party's election "
                    "manifesto by name."
                ],
            },
        ],
    },
    "A_caus": {
        "type": "noul",
        "instructions": {
            "question": (
                "Did the generator justification demonstrably rely on this "
                "specific retrieved chunk?"
            ),
            "compare": ["`generator_justification`", "`retrieved_chunk.text`"],
            "focus": (
                "Require an explicit citation of this chunk's identifier "
                "(`retrieved_chunk.citation_forms`) and arguments derived "
                "from facts, figures, or claims unique to this chunk."
            ),
        },
        "criteria": {
            "true": (
                "The justification explicitly cites this chunk by one of "
                "`retrieved_chunk.citation_forms` AND derives its arguments "
                "from facts, figures, or claims unique to this chunk."
            ),
            "false": (
                "The justification does not explicitly cite this chunk's "
                "identifier, or uses generic reasoning that could have been "
                "produced from the input text alone or from model "
                "pre-training."
            ),
        },
    },
}

# Expected shape of the four questions (used to validate --questions-file and
# the answers that come back).
_EXPECTED_QUESTIONS = {
    "R_top": {"type": "score", "levels": 5},
    "R_ideo": {"type": "score", "levels": 5},
    "A_caus": {"type": "noul"},
}
SCORE_LEVELS = {
    name: cfg["levels"] for name, cfg in _EXPECTED_QUESTIONS.items() if "levels" in cfg
}


# ---------------------------------------------------------------------------
# Question & state building
# ---------------------------------------------------------------------------


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_valid_instructions(value: object) -> bool:
    """Instructions may be a string or a structured object with a question."""
    if _is_nonempty_string(value):
        return True
    return isinstance(value, dict) and _is_nonempty_string(value.get("question"))


def _is_valid_score_level(value: object) -> bool:
    """A Score level may be a string or a structured object with a 'what'."""
    if _is_nonempty_string(value):
        return True
    return isinstance(value, dict) and _is_nonempty_string(value.get("what"))


def _validate_questions(questions: dict) -> None:
    """Ensure the question set matches the metrics' fixed label scales."""
    if not isinstance(questions, dict):
        raise ValueError(
            "Questions definition must be a JSON object keyed by question name"
        )
    for name, expected in _EXPECTED_QUESTIONS.items():
        q = questions.get(name)
        if not isinstance(q, dict):
            raise ValueError(f"Question '{name}' missing from questions definition")
        if q.get("type") != expected["type"]:
            raise ValueError(
                f"Question '{name}' must be of type '{expected['type']}', "
                f"got {q.get('type')!r}"
            )
        if not _is_valid_instructions(q.get("instructions")):
            raise ValueError(
                f"Question '{name}' needs 'instructions' as a non-empty string "
                "or an object with a non-empty 'question' field"
            )
        criteria = q.get("criteria")
        if expected["type"] == "score":
            n_levels = expected["levels"]
            if not isinstance(criteria, list) or len(criteria) != n_levels:
                raise ValueError(
                    f"Question '{name}' needs a criteria list with exactly "
                    f"{n_levels} level descriptions (ascending from level 1)"
                )
            if not all(_is_valid_score_level(c) for c in criteria):
                raise ValueError(
                    f"Question '{name}' criteria must be non-empty strings or "
                    "objects with a non-empty 'what' field"
                )
        else:  # noul
            if not isinstance(criteria, dict):
                raise ValueError(f"Question '{name}' needs criteria.true/false")
            for side in ("true", "false"):
                if not _is_nonempty_string(criteria.get(side)):
                    raise ValueError(
                        f"Question '{name}' needs non-empty criteria.{side}"
                    )


def build_questions(questions_file: str | None) -> tuple[dict, str | None]:
    """Return the question set (default or from file) after validation."""
    if questions_file:
        questions = json.loads(Path(questions_file).read_text(encoding="utf-8"))
    else:
        questions = json.loads(json.dumps(JEV_QUESTIONS))  # deep copy
    _validate_questions(questions)
    return questions, questions_file


def build_state(record: dict, chunk: dict) -> dict:
    """Build the Jev (System One) state for one (record, chunk) pair.

    State is a JSON object; each rubric references these fields by backticked
    path (e.g. ``retrieved_chunk.text``). The rubric lives in the questions;
    the state carries the material to judge. No similarity/retrieval scores
    are included.
    """
    meta = chunk.get("chunk_metadata") or {}
    chunk_idx_1based = chunk.get("chunk_index", 0) + 1
    rag = (record.get("source_context") or {}).get("rag") or {}
    justification = (
        rag.get("justification") or ""
    ).strip() or "No justification provided."

    retrieved_chunk = {
        "chunk_identifier": f"[{chunk_idx_1based}]",
        "citation_forms": f'"[{chunk_idx_1based}]" or "Chunk {chunk_idx_1based}"',
        "party": meta.get("party", "Unknown"),
        "speaker": meta.get("speaker", "Unknown"),
        "source": meta.get("source", "Unknown"),
    }
    if meta.get("date"):
        retrieved_chunk["date"] = str(meta.get("date"))
    retrieved_chunk["text"] = (chunk.get("chunk_text") or "").strip()

    return {
        "evaluation_task": EVALUATION_TASK,
        "input_text": (record.get("input_text") or "").strip(),
        "retrieved_chunk": retrieved_chunk,
        "generator_justification": justification,
    }


# ---------------------------------------------------------------------------
# Jev client (TypeSafe Python SDK)
# ---------------------------------------------------------------------------


def build_client(model: str | None) -> tuple[TypeSafeClient, str]:
    """Create the Jev client and resolve the model id to send.

    Prefers the native TypeSafe API (``TYPESAFE_API_KEY``); falls back to the
    OpenRouter-compatible endpoint (``OPENROUTER_API_KEY_ME``) when no native
    key is present. Returns the client and the resolved model id so callers can
    record which model was requested; the response's ``model`` field records
    the version that actually answered.
    """
    typesafe_key = os.getenv("TYPESAFE_API_KEY", "").strip()
    if typesafe_key:
        resolved = model or TYPESAFE_JEV_MODEL
        return (
            TypeSafeClient(
                api_key=typesafe_key,
                model=resolved,
                timeout=REQUEST_TIMEOUT_SECONDS,
                retry=RETRY_POLICY,
            ),
            resolved,
        )

    openrouter_key = os.getenv("OPENROUTER_API_KEY_ME", "").strip()
    if openrouter_key:
        resolved = model or OPENROUTER_JEV_MODEL
        return (
            TypeSafeClient(
                api_key=openrouter_key,
                base_url=OPENROUTER_BASE_URL,
                model=resolved,
                timeout=REQUEST_TIMEOUT_SECONDS,
                retry=RETRY_POLICY,
            ),
            resolved,
        )

    raise SystemExit(
        "Error: set TYPESAFE_API_KEY or OPENROUTER_API_KEY_ME in .env.local"
    )


def judge_chunk(
    client: TypeSafeClient,
    state: dict,
    questions: dict,
    model: str,
) -> dict:
    """Judge one chunk via the SDK; returns {'answers', 'usage', 'model'}.

    The SDK already validates the response against its answer schema; the
    checks below only guard the fields the metrics rely on. Answers are
    converted to plain dicts so the raw values can be stored in the JSONL and
    re-derived offline.
    """
    response = client.system_one(state=state, questions=questions, model=model)

    validated: dict[str, dict] = {}
    for name, expected in _EXPECTED_QUESTIONS.items():
        ans = response.answers.get(name)
        if ans is None or ans.type != expected["type"]:
            raise RuntimeError(f"Missing or malformed answer for '{name}': {ans!r}")
        validated[name] = ans.model_dump()

    for name, expected in _EXPECTED_QUESTIONS.items():
        ans = validated[name]
        if expected["type"] == "score":
            score = ans.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise RuntimeError(f"Answer '{name}' has no numeric score")
            probs = ans.get("probabilities")
            if probs is not None:
                if not isinstance(probs, dict):
                    raise RuntimeError(f"Answer '{name}' has malformed probabilities")
                for key, value in probs.items():
                    if (
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not 0 <= value <= 1
                    ):
                        raise RuntimeError(
                            f"Answer '{name}' has invalid probability {key}={value!r}"
                        )
        else:  # noul
            noul = ans.get("noul")
            if (
                not isinstance(noul, (int, float))
                or isinstance(noul, bool)
                or not 0 <= noul <= 1
            ):
                raise RuntimeError(f"Answer '{name}' has no valid noul probability")

    usage = response.usage.model_dump()
    # The native TypeSafe API reports tokens only; OpenRouter also reports cost.
    raw_usage = (response.raw_http_response.json() or {}).get("usage") or {}
    if raw_usage.get("cost") is not None:
        usage["cost"] = raw_usage["cost"]

    return {"answers": validated, "usage": usage, "model": response.model}


# ---------------------------------------------------------------------------
# Label derivation (raw probabilities -> integer labels on the human scales)
# ---------------------------------------------------------------------------


def _score_label_from_probabilities(answer: dict, n_levels: int) -> int | None:
    """Most probable level index (0-based) from a score answer, or None."""
    probs = answer.get("probabilities")
    if not isinstance(probs, dict) or not probs:
        return None
    best_idx: int | None = None
    best_p = -1.0
    for key, value in probs.items():
        try:
            idx = int(key)
            p = float(value)
        except TypeError, ValueError:
            continue
        if not 0 <= idx < n_levels:
            continue
        if p > best_p:
            best_p = p
            best_idx = idx
    return best_idx


def derive_score_label(answer: dict, mode: str, n_levels: int) -> int:
    """Integer label (1-based) from a Jev score answer.

    Jev score answers are 0-indexed: criteria[0] is level 1. 'argmax' picks
    the most probable level; 'round' uses the rounded probability-weighted
    position. Both are shifted by +1 onto the human annotation scale.
    """
    if mode == "argmax":
        idx = _score_label_from_probabilities(answer, n_levels)
        if idx is not None:
            return idx + 1
    score = float(answer.get("score", 0.0))
    idx = int(score + 0.5)  # round half up, 0-based weighted position
    return max(0, min(n_levels - 1, idx)) + 1


def derive_labels(
    raw_answers: dict,
    score_label_mode: str = DEFAULT_SCORE_LABEL_MODE,
    a_caus_threshold: float = DEFAULT_A_CAUS_THRESHOLD,
) -> dict:
    """Turn raw Jev answers into integer labels on the human scales.

    The rubric's hard gating rule is enforced
    deterministically here, because System One questions are answered
    independently and cannot see each other's answers.
    """
    labels = {}
    for dim, n_levels in SCORE_LEVELS.items():
        labels[dim] = derive_score_label(raw_answers[dim], score_label_mode, n_levels)
    noul = float(raw_answers["A_caus"].get("noul", 0.0))
    labels["A_caus"] = 1 if noul >= a_caus_threshold else 0
    return labels


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_judging(
    args: argparse.Namespace, run_snapshot_path: Path | None = None
) -> tuple[list[dict], int, str]:
    """Load annotations, judge chunks with Jev, write log, return results."""
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

    # Question setup
    questions, questions_file = build_questions(args.questions_file)
    questions_hash = hashlib.sha256(
        json.dumps(questions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]

    # Jev client: the native TypeSafe API when TYPESAFE_API_KEY is set,
    # otherwise the OpenRouter-compatible endpoint.
    client, args.model = build_client(args.model)

    print(f"Judge model: {args.model}")
    print(
        f"Questions hash: {questions_hash}"
        + (f" ({questions_file})" if questions_file else " (embedded JEV_QUESTIONS)")
    )
    print(f"Judging {len(tasks)} chunk(s) at concurrency {args.concurrency}")

    def _human(chunk: dict) -> dict:
        return {k: chunk.get(k) for k in ("R_top", "R_ideo", "A_caus")}

    def _process(record: dict, chunk: dict) -> dict:
        base = {
            "text_index": record.get("text_index"),
            "chunk_index": chunk.get("chunk_index"),
            "model": args.model,
            "questions_hash": questions_hash,
            "questions_file": questions_file,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            state = build_state(record, chunk)
            judged = judge_chunk(client, state, questions, args.model)
            labels = derive_labels(
                judged["answers"], args.score_label_mode, args.a_caus_threshold
            )
            return {
                **base,
                "response_model": judged["model"],
                "raw_answers": judged["answers"],
                "derived_labels": labels,
                "label_config": {
                    "score_label_mode": args.score_label_mode,
                    "a_caus_threshold": args.a_caus_threshold,
                },
                "usage": judged["usage"],
                "human": _human(chunk),
            }
        except Exception as e:  # noqa: BLE001
            return {**base, "error": str(e), "human": _human(chunk)}

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

    client.close()
    return results, failed, questions_hash


# ---------------------------------------------------------------------------
# Offline re-derivation (no API calls)
# ---------------------------------------------------------------------------


def load_results_from_jsonl(path: str) -> list[dict]:
    """Load previous run records (log or snapshot) from JSONL."""
    results = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                results.append(record)
    return results


def rederive_labels_in_place(
    results: list[dict], score_label_mode: str, a_caus_threshold: float
) -> list[dict]:
    """Recompute derived_labels from stored raw_answers under new settings."""
    updated = []
    for r in results:
        if "error" in r or "raw_answers" not in r:
            updated.append(r)
            continue
        r = dict(r)
        r["derived_labels"] = derive_labels(
            r["raw_answers"], score_label_mode, a_caus_threshold
        )
        r["label_config"] = {
            "score_label_mode": score_label_mode,
            "a_caus_threshold": a_caus_threshold,
        }
        updated.append(r)
    return updated


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
    """Compute agreement metrics per dimension on the derived labels."""
    dims = {
        "R_top": {"ordinal": True, "labels": [1, 2, 3, 4, 5]},
        "R_ideo": {"ordinal": True, "labels": [1, 2, 3, 4, 5]},
        "A_caus": {"ordinal": False, "labels": [0, 1]},
    }
    per_dim = {}
    for dim, cfg in dims.items():
        h_vals = []
        j_vals = []
        conf_vals = []
        for r in results:
            if "error" in r or "derived_labels" not in r:
                continue
            h = (r.get("human") or {}).get(dim)
            j = r["derived_labels"].get(dim)
            if j is None or h is None:
                continue
            h_vals.append(int(h))
            j_vals.append(int(j))
            conf = ((r.get("raw_answers") or {}).get(dim) or {}).get("confidence")
            if conf is not None:
                conf_vals.append(float(conf))

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
            "mean_confidence": float(np.mean(conf_vals)) if conf_vals else None,
            "labels": fixed_labels,
            "confusion_matrix": cm.tolist(),
            "below_threshold": below,
        }

    return per_dim


def get_worst_disagreements(results: list[dict], n: int = 5) -> list[dict]:
    """Return the top-N largest |diff| disagreements for ordinal dims."""
    disagreements = []
    for r in results:
        if "error" in r or "derived_labels" not in r:
            continue
        for dim in ["R_top", "R_ideo"]:
            h = r["human"][dim]
            j = r["derived_labels"].get(dim)
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


def get_low_confidence(
    results: list[dict],
    floor: float = LOW_CONFIDENCE_FLOOR,
    a_caus_threshold: float = DEFAULT_A_CAUS_THRESHOLD,
    near_threshold_margin: float = NEAR_THRESHOLD_MARGIN,
) -> list[dict]:
    """Flag judgments the model itself was unsure about (candidates for
    manual review). Score answers carry a confidence; Noul answers do not,
    so A_caus is flagged when p(yes) sits near the decision threshold."""
    flagged = []
    for r in results:
        if "error" in r or "raw_answers" not in r:
            continue
        issues = []
        for dim in ("R_top", "R_ideo"):
            conf = (r["raw_answers"].get(dim) or {}).get("confidence")
            if conf is not None and conf < floor:
                issues.append({"dimension": dim, "confidence": float(conf)})
        noul = (r["raw_answers"].get("A_caus") or {}).get("noul")
        if (
            noul is not None
            and abs(float(noul) - a_caus_threshold) <= near_threshold_margin
        ):
            issues.append(
                {
                    "dimension": "A_caus",
                    "noul": float(noul),
                    "threshold": a_caus_threshold,
                }
            )
        if issues:
            flagged.append(
                {
                    "text_index": r.get("text_index"),
                    "chunk_index": r.get("chunk_index"),
                    "issues": issues,
                }
            )
    flagged.sort(key=lambda x: (str(x["text_index"]), x["chunk_index"]))
    return flagged


def compute_usage_totals(results: list[dict]) -> dict:
    """Sum usage across judged records (Jev reports cost per response)."""
    input_tokens = 0
    output_tokens = 0
    cost = 0.0
    for r in results:
        usage = r.get("usage")
        if not isinstance(usage, dict):
            continue
        input_tokens += usage.get("input_tokens") or 0
        output_tokens += usage.get("output_tokens") or 0
        cost += usage.get("cost") or 0.0
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cost": float(cost),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_and_save_report(
    results: list[dict],
    per_dim: dict,
    worst: list[dict],
    low_conf: list[dict],
    model: str,
    questions_hash: str,
    report_path: str,
    *,
    score_label_mode: str,
    a_caus_threshold: float,
    rederived_from: str | None = None,
) -> None:
    """Print a concise report to stdout and write a JSON report to disk."""
    judged = len([r for r in results if "derived_labels" in r])
    total = len(results)
    usage_totals = compute_usage_totals(results)

    print("\n=== Jev Judge Agreement Report ===")
    print(f"Model: {model}")
    print(f"Questions hash: {questions_hash}")
    if rederived_from:
        print(f"Rederived from: {rederived_from} (no new API calls)")
    print(
        f"Label derivation: score_label_mode={score_label_mode}, "
        f"a_caus_threshold={a_caus_threshold}"
    )
    print(f"Judged chunks: {judged} / {total}")

    print("\nPer-dimension metrics:")
    header = f"{'Dim':<12} {'n':>6} {'Agree%':>7} {'κ/κ_w':>7} {'MeanΔ':>7} {'Conf':>6}"
    print(header)
    print("-" * len(header))
    for dim in ["R_top", "R_ideo", "A_caus"]:
        d = per_dim[dim]
        if d is None:
            print(f"{dim:<12} {'—':>6} {'—':>7} {'—':>7} {'—':>7} {'—':>6}")
            continue
        metric_val = d["qwk"] if dim in ("R_top", "R_ideo") else d["kappa"]
        flag = " ⚠️ below 0.70" if d.get("below_threshold") else ""
        metric_str = f"{metric_val:>6.3f}" if metric_val is not None else "   N/A"
        conf = d.get("mean_confidence")
        conf_str = f"{conf:>6.2f}" if conf is not None else "     —"
        print(
            f"{dim:<12} {d['n']:>6} {d['exact_agreement']*100:>6.1f}% "
            f"{metric_str} {d['mean_diff']:>+6.2f} {conf_str}{flag}"
        )

    print("\nConfusion matrices:")
    for dim in ["R_top", "R_ideo", "A_caus"]:
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

    if low_conf:
        print(
            f"\nLow-confidence judgments (confidence < {LOW_CONFIDENCE_FLOOR} "
            f"or A_caus noul within {NEAR_THRESHOLD_MARGIN} of the threshold): "
            f"{len(low_conf)}"
        )
        for lc in low_conf[:20]:
            parts = []
            for issue in lc["issues"]:
                if "confidence" in issue:
                    parts.append(f"{issue['dimension']} conf={issue['confidence']:.2f}")
                else:
                    parts.append(f"{issue['dimension']} noul={issue['noul']:.2f}")
            print(
                f"  sample {lc['text_index']} chunk {lc['chunk_index']}: "
                + ", ".join(parts)
            )
        if len(low_conf) > 20:
            print(f"  … and {len(low_conf) - 20} more")

    print(
        f"\nUsage: {usage_totals['input_tokens']:,} input tokens, "
        f"{usage_totals['output_tokens']:,} output tokens (free), "
        f"total cost ${usage_totals['cost']:.6f}"
    )

    report = {
        "model": model,
        "questions_hash": questions_hash,
        "label_config": {
            "score_label_mode": score_label_mode,
            "a_caus_threshold": a_caus_threshold,
        },
        "rederived_from": rederived_from,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_chunks": total,
        "judged_chunks": judged,
        "usage_totals": usage_totals,
        "per_dimension": per_dim,
        "low_confidence": low_conf,
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
        description="Run the Jev (TypeSafe System One) judge via the TypeSafe "
        "Python SDK against human chunk annotations and compute agreement metrics."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id to send (default: native TypeSafe "
        f"{TYPESAFE_JEV_MODEL} when TYPESAFE_API_KEY is set, else OpenRouter "
        f"{OPENROUTER_JEV_MODEL}). The response records the version that answered.",
    )
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
        help="Append-only log for Jev judgments",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path for the JSON agreement report (default: <output-root>/<run-id>/llm_judge_report.json)",
    )
    parser.add_argument(
        "--questions-file",
        default=None,
        help="JSON file overriding the embedded Jev question set (instructions + criteria)",
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
    parser.add_argument(
        "--a-caus-threshold",
        type=float,
        default=DEFAULT_A_CAUS_THRESHOLD,
        help="Noul probability >= threshold -> A_caus = 1",
    )
    parser.add_argument(
        "--score-label-mode",
        choices=["argmax", "round"],
        default=DEFAULT_SCORE_LABEL_MODE,
        help="Turn a score answer into an integer label: most probable level "
        "(argmax) or rounded probability-weighted position (round)",
    )
    parser.add_argument(
        "--rederive-from",
        default=None,
        help="Skip the API and recompute metrics from a previous run's JSONL "
        "(stored raw answers) with the current thresholds and label mode",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    args.run_id = run_id  # resolved run id, used for per-run output paths
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")

    report_path = (
        Path(args.report) if args.report else output_dir / "llm_judge_report.json"
    )

    if args.rederive_from:
        results = load_results_from_jsonl(args.rederive_from)
        if not results:
            print(
                f"No result records found in {args.rederive_from}. Exiting.",
                file=sys.stderr,
            )
            sys.exit(1)
        results = rederive_labels_in_place(
            results, args.score_label_mode, args.a_caus_threshold
        )
        failed = sum(1 for r in results if "error" in r)
        questions_hash = next(
            (r.get("questions_hash") for r in results if r.get("questions_hash")),
            "unknown",
        )
        if not args.model:
            # Recover the model from the stored records when rederiving offline.
            args.model = next(
                (
                    r.get("response_model") or r.get("model")
                    for r in results
                    if r.get("response_model") or r.get("model")
                ),
                "unknown",
            )
        rederived_from = args.rederive_from
        print(
            f"Rederiving labels for {len(results)} record(s) from "
            f"{args.rederive_from} (no API calls)."
        )
    else:
        run_snapshot_path = output_dir / "llm_judge_annotations.jsonl"
        results, failed, questions_hash = run_judging(args, run_snapshot_path)
        rederived_from = None

    per_dim = compute_metrics(results)
    worst = get_worst_disagreements(results, n=args.show_disagreements)
    low_conf = get_low_confidence(results, a_caus_threshold=args.a_caus_threshold)
    print_and_save_report(
        results,
        per_dim,
        worst,
        low_conf,
        args.model,
        questions_hash,
        report_path,
        score_label_mode=args.score_label_mode,
        a_caus_threshold=args.a_caus_threshold,
        rederived_from=rederived_from,
    )

    if failed:
        print(f"\nWarning: {failed} chunk(s) failed judging.")


if __name__ == "__main__":
    main()
