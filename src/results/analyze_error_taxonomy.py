import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGS_DIR = Path("logs/batch_runs")
OUTPUT_DIR = Path("results/qualitative")


def _model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    if "qwen" in name_lower:
        return "Qwen"
    elif "ministral" in name_lower or "mistral" in name_lower:
        return "Mistral"
    elif "llama" in name_lower:
        return "Llama"
    return "Other"


def parse_record(entry: Dict[str, Any]) -> Dict[str, Any] | None:
    """Extracts core metrics and qualitative fields from a log entry."""
    try:
        llm = entry["parameters"].get("llm", "unknown_llm")
        embedding = entry["parameters"].get("embedding_model", "unknown_emb")
        retrieval_mode = entry["parameters"].get("retrieval_mode", "unknown_mode")

        pred_bias = float(entry["output"]["bias"])
        true_bias = float(entry["ground_truth"]["label_ideology"])
        abs_error = abs(pred_bias - true_bias)
        signed_error = pred_bias - true_bias

        # Extract retrieved context summary
        chunks_summary = []
        for i, chunk in enumerate(entry["inputs"].get("retrieved_chunks", []), 1):
            chunks_summary.append(
                {
                    "rank": i,
                    "party": chunk.get("party"),
                    "speaker": chunk.get("speaker"),
                    "score": chunk.get("score"),
                    "date": chunk.get("date"),
                    "snippet": (
                        chunk.get("text", "")[:180] + "..."
                        if len(chunk.get("text", "")) > 180
                        else chunk.get("text", "")
                    ),
                }
            )

        return {
            "run_id": entry.get("run_id"),
            "llm": llm,
            "embedding_model": embedding,
            "retrieval_mode": retrieval_mode,
            "text_index": entry.get("input_metadata", {}).get("text_index"),
            "target_party": entry.get("input_metadata", {}).get("party"),
            "target_speaker": entry.get("input_metadata", {}).get("speaker"),
            "input_text": entry.get("inputs", {}).get("text"),
            "pred_bias": pred_bias,
            "true_bias": true_bias,
            "abs_error": round(abs_error, 4),
            "signed_error": round(signed_error, 4),
            "justification": entry.get("output", {}).get("justification"),
            "retrieved_chunks": chunks_summary,
            "raw_entry": entry,  # Retain original payload
        }
    except (KeyError, ValueError, TypeError) as e:
        return None


def generate_markdown_report(
    group_key: str, best_items: List[Dict], worst_items: List[Dict], out_path: Path
):
    """Generates an easily readable Markdown report for qualitative inspection."""
    with open(out_path, "w", encoding="utf-8") as md:
        md.write(f"# Qualitative Evaluation Report\n")
        md.write(f"**Group:** `{group_key}`\n\n---\n\n")

        for section_title, records in [
            ("Worst 25 Predictions (Highest Error)", worst_items),
            ("Best 25 Predictions (Lowest Error)", best_items),
        ]:
            md.write(f"## {section_title}\n\n")
            for idx, r in enumerate(records, 1):
                md.write(
                    f"### #{idx} | Error: {r['abs_error']} (Pred: {r['pred_bias']} vs Truth: {r['true_bias']})\n"
                )
                md.write(
                    f"- **Metadata:** Speaker: `{r['target_speaker']}` | Party: `{r['target_party']}` | Text ID: `{r['text_index']}`\n"
                )
                md.write(f"- **Input Text:**\n> {r['input_text']}\n\n")
                md.write(f"- **Model Justification:**\n> {r['justification']}\n\n")
                md.write(f"**Retrieved Chunks (Top {len(r['retrieved_chunks'])}):**\n")
                for c in r["retrieved_chunks"]:
                    md.write(
                        f"  - `[Rank {c['rank']}]` **{c['speaker']}** ({c['party']}, {c['date']}) | Score: `{c['score']}`\n"
                    )
                    md.write(f"    - *Snippet:* {c['snippet']}\n")
                md.write("\n---\n\n")


def process_eval_logs(logs_dir: Path, output_dir: Path, top_k: int = 25):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped_data = defaultdict(list)
    total_parsed = 0

    # Find all JSONL files recursively in logs_dir
    log_files = sorted(logs_dir.rglob("party_label_*.jsonl"))
    if not log_files:
        print(f"No JSONL files found in {logs_dir}")
        return

    print(f"Scanning {logs_dir} for JSONL logs...")
    for log_file in log_files:
        print(f"  Reading {log_file}...")
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw_json = json.loads(line)
                    parsed = parse_record(raw_json)
                    if parsed:
                        key = f"{parsed['llm']}__{parsed['embedding_model']}".replace(
                            "/", "_"
                        )
                        grouped_data[key].append(parsed)
                        total_parsed += 1
                except json.JSONDecodeError:
                    continue

    print(
        f"Loaded {total_parsed} valid evaluations across {len(grouped_data)} (LLM, Embedding) configurations.\n"
    )

    for group_key, records in grouped_data.items():
        # Sort by absolute error ascending
        records_sorted = sorted(records, key=lambda x: x["abs_error"])

        best_25 = records_sorted[:top_k]
        worst_25 = list(reversed(records_sorted[-top_k:]))  # Largest error first

        # Determine model family and create path: outcomes/{family}/{llm}__{emb}/
        llm_name = records[0]["llm"].split("/")[-1]  # clean model name
        family = _model_family(llm_name)
        group_subpath = out_dir / "outcomes" / family / group_key
        group_subpath.mkdir(parents=True, exist_ok=True)

        # 1. Save raw JSONL files
        with open(group_subpath / "best_25.jsonl", "w", encoding="utf-8") as f_best:
            for item in best_25:
                f_best.write(json.dumps(item["raw_entry"], ensure_ascii=False) + "\n")

        with open(group_subpath / "worst_25.jsonl", "w", encoding="utf-8") as f_worst:
            for item in worst_25:
                f_worst.write(json.dumps(item["raw_entry"], ensure_ascii=False) + "\n")

        # 2. Save Human-Readable Markdown
        generate_markdown_report(
            group_key, best_25, worst_25, group_subpath / "qualitative_review.md"
        )

        print(
            f"[{group_key}] -> Saved best/worst {top_k} records to `{group_subpath}/`"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract best/worst RAG evaluations per model group."
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=LOGS_DIR,
        help="Root directory containing JSONL log files (searches recursively)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for reports",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=25,
        help="Number of best/worst records to extract (default: 25)",
    )

    args = parser.parse_args()
    process_eval_logs(args.logs_dir, output_dir=args.output_dir, top_k=args.k)
