"""
Analyze directional bias in RAG evaluation logs.

Computes signed errors (pred - true) to detect whether each model configuration
systematically leans left, right, or stays centric.

Scale: 1.0 (extreme left) → 7.0 (extreme right)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import scipy.stats as stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOGS_DIR = Path("logs/batch_runs")
OUTPUT_DIR = Path("results/bias_direction")
CENTRIC_THRESHOLD = 0.5  # |signed_error| <= 0.5 counts as centric


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    if "qwen" in name_lower:
        return "Qwen"
    elif "ministral" in name_lower or "mistral" in name_lower:
        return "Mistral"
    elif "llama" in name_lower:
        return "Llama"
    return "Other"


def _bin_ideology(score: float) -> str:
    if score < 3.0:
        return "Left"
    elif score > 5.0:
        return "Right"
    return "Center"


def parse_record(entry: Dict[str, Any]) -> Dict[str, Any] | None:
    """Extract core fields, supporting both old and new log formats."""
    try:
        # --- New format ---
        if "ground_truth" in entry and "label_ideology" in entry["ground_truth"]:
            pred_bias = float(entry["output"]["bias"])
            true_bias = float(entry["ground_truth"]["label_ideology"])
        # --- Old format ---
        elif "ches_ground_truth" in entry and entry["ches_ground_truth"] is not None:
            pred_bias = float(entry["output"]["score"])
            true_bias = float(entry["ches_ground_truth"])
        else:
            return None

        llm = entry["parameters"].get("llm", "unknown_llm")
        embedding = entry["parameters"].get("embedding_model", "unknown_emb")
        retrieval_mode = entry["parameters"].get("retrieval_mode", "unknown_mode")

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
            "ideology_bin": _bin_ideology(true_bias),
            "justification": entry.get("output", {}).get("justification"),
            "retrieved_chunks": chunks_summary,
            "raw_entry": entry,
        }
    except (KeyError, ValueError, TypeError):
        return None


def _describe_direction(signed_error: float) -> str:
    if signed_error < -CENTRIC_THRESHOLD:
        return "Left-leaning"
    elif signed_error > CENTRIC_THRESHOLD:
        return "Right-leaning"
    return "Centric"


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

def generate_json_report(
    group_key: str,
    records: List[Dict],
    top_k: int,
    out_path: Path,
):
    """Save structured JSON with all statistics."""
    signed_errors = [r["signed_error"] for r in records]
    n = len(signed_errors)

    # Direction counts
    left_count = sum(1 for s in signed_errors if s < -CENTRIC_THRESHOLD)
    centric_count = sum(1 for s in signed_errors if abs(s) <= CENTRIC_THRESHOLD)
    right_count = sum(1 for s in signed_errors if s > CENTRIC_THRESHOLD)

    # T-test against 0
    t_stat, p_value = stats.ttest_1samp(signed_errors, popmean=0)

    # Skewness
    skewness = stats.skew(signed_errors)

    # Asymmetry by true ideology bin
    bin_stats = defaultdict(lambda: {"count": 0, "mean_signed_error": 0.0, "mean_abs_error": 0.0})
    for r in records:
        b = r["ideology_bin"]
        bin_stats[b]["count"] += 1
        bin_stats[b]["mean_signed_error"] += r["signed_error"]
        bin_stats[b]["mean_abs_error"] += r["abs_error"]

    for b in bin_stats:
        c = bin_stats[b]["count"]
        bin_stats[b]["mean_signed_error"] = round(bin_stats[b]["mean_signed_error"] / c, 4)
        bin_stats[b]["mean_abs_error"] = round(bin_stats[b]["mean_abs_error"] / c, 4)

    # Top-K extremes
    by_signed = sorted(records, key=lambda x: x["signed_error"])
    most_left = by_signed[:top_k]
    most_right = list(reversed(by_signed[-top_k:]))
    most_centric = sorted(records, key=lambda x: abs(x["signed_error"]))[:top_k]

    report = {
        "group": group_key,
        "n_total": n,
        "signed_error": {
            "mean": round(sum(signed_errors) / n, 4),
            "median": round(sorted(signed_errors)[n // 2], 4),
            "std": round(stats.tstd(signed_errors), 4),
            "min": round(min(signed_errors), 4),
            "max": round(max(signed_errors), 4),
            "skewness": round(skewness, 4),
        },
        "direction_counts": {
            "left_leaning": {"count": left_count, "pct": round(left_count / n * 100, 2)},
            "centric": {"count": centric_count, "pct": round(centric_count / n * 100, 2)},
            "right_leaning": {"count": right_count, "pct": round(right_count / n * 100, 2)},
        },
        "t_test_against_zero": {
            "t_statistic": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "significant_at_05": bool(p_value < 0.05),
        },
        "by_true_ideology_bin": dict(bin_stats),
        "examples": {
            "most_left_leaning": [_strip_raw(r) for r in most_left],
            "most_right_leaning": [_strip_raw(r) for r in most_right],
            "most_centric": [_strip_raw(r) for r in most_centric],
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def _strip_raw(record: Dict) -> Dict:
    """Return record dict without the bulky raw_entry for reports."""
    return {k: v for k, v in record.items() if k != "raw_entry"}


def generate_markdown_report(
    group_key: str,
    records: List[Dict],
    top_k: int,
    out_path: Path,
):
    """Human-readable Markdown report with tables and interpretation."""
    signed_errors = [r["signed_error"] for r in records]
    n = len(signed_errors)
    mean_err = sum(signed_errors) / n
    median_err = sorted(signed_errors)[n // 2]
    std_err = stats.tstd(signed_errors)
    skewness = stats.skew(signed_errors)
    t_stat, p_value = stats.ttest_1samp(signed_errors, popmean=0)

    left_count = sum(1 for s in signed_errors if s < -CENTRIC_THRESHOLD)
    centric_count = sum(1 for s in signed_errors if abs(s) <= CENTRIC_THRESHOLD)
    right_count = sum(1 for s in signed_errors if s > CENTRIC_THRESHOLD)

    # Bin stats
    bin_stats = defaultdict(lambda: {"count": 0, "sum_se": 0.0, "sum_ae": 0.0})
    for r in records:
        b = r["ideology_bin"]
        bin_stats[b]["count"] += 1
        bin_stats[b]["sum_se"] += r["signed_error"]
        bin_stats[b]["sum_ae"] += r["abs_error"]

    by_signed = sorted(records, key=lambda x: x["signed_error"])
    most_left = by_signed[:top_k]
    most_right = list(reversed(by_signed[-top_k:]))
    most_centric = sorted(records, key=lambda x: abs(x["signed_error"]))[:top_k]

    with open(out_path, "w", encoding="utf-8") as md:
        md.write(f"# Directional Bias Analysis\n\n")
        md.write(f"**Group:** `{group_key}`  \n")
        md.write(f"**Total Evaluations:** {n}  \n")
        md.write(f"**Centric Threshold:** ±{CENTRIC_THRESHOLD}  \n\n")
        md.write("---\n\n")

        # Overall signed-error summary
        md.write("## Signed-Error Summary\n\n")
        md.write(f"| Statistic | Value |\n|-----------|-------|\n")
        md.write(f"| Mean | {round(mean_err, 4)} |\n")
        md.write(f"| Median | {round(median_err, 4)} |\n")
        md.write(f"| Std. Dev | {round(std_err, 4)} |\n")
        md.write(f"| Skewness | {round(skewness, 4)} |\n")
        md.write(f"| Min | {round(min(signed_errors), 4)} |\n")
        md.write(f"| Max | {round(max(signed_errors), 4)} |\n")
        md.write("\n")

        # Interpretation
        if p_value < 0.05:
            direction = "**left**" if mean_err < 0 else "**right**"
            md.write(
                f"> **Interpretation:** The mean signed error is significantly different from zero "
                f"(t = {round(t_stat, 3)}, p = {round(p_value, 4)}). "
                f"This indicates a systematic {direction} bias in predictions.\n\n"
            )
        else:
            md.write(
                f"> **Interpretation:** No significant systematic bias detected "
                f"(t = {round(t_stat, 3)}, p = {round(p_value, 4)}). "
                f"Errors appear roughly centered around zero.\n\n"
            )

        # Direction counts
        md.write("## Direction of Errors\n\n")
        md.write("| Direction | Count | Percentage |\n|-----------|-------|------------|\n")
        md.write(
            f"| Left-leaning (error < -{CENTRIC_THRESHOLD}) | {left_count} | {round(left_count / n * 100, 2)}% |\n"
        )
        md.write(
            f"| Centric (|error| ≤ {CENTRIC_THRESHOLD}) | {centric_count} | {round(centric_count / n * 100, 2)}% |\n"
        )
        md.write(
            f"| Right-leaning (error > +{CENTRIC_THRESHOLD}) | {right_count} | {round(right_count / n * 100, 2)}% |\n"
        )
        md.write("\n")

        # By ideology bin
        md.write("## Mean Signed Error by True Ideology Bin\n\n")
        md.write("| True Ideology | Count | Mean Signed Error | Mean Abs Error |\n")
        md.write("|----------------|-------|-------------------|----------------|\n")
        for b in ("Left", "Center", "Right"):
            s = bin_stats[b]
            c = s["count"]
            if c:
                mse = round(s["sum_se"] / c, 4)
                mae = round(s["sum_ae"] / c, 4)
            else:
                mse = mae = "N/A"
            md.write(f"| {b} | {c} | {mse} | {mae} |\n")
        md.write("\n")
        md.write(
            "> **Note:** A positive mean signed error for a bin means the model "
            "overrates (predicted more right than true) texts with that true ideology.\n\n"
        )

        # Examples
        md.write("---\n\n")
        for title, items in [
            (f"Most Left-Leaning Errors (Top {top_k})", most_left),
            (f"Most Right-Leaning Errors (Top {top_k})", most_right),
            (f"Most Centric Predictions (Top {top_k})", most_centric),
        ]:
            md.write(f"## {title}\n\n")
            for idx, r in enumerate(items, 1):
                md.write(
                    f"### #{idx} | Signed Error: {r['signed_error']} "
                    f"(Pred: {r['pred_bias']} vs Truth: {r['true_bias']})\n"
                )
                md.write(
                    f"- **Metadata:** Speaker: `{r['target_speaker']}` | Party: `{r['target_party']}` | Text ID: `{r['text_index']}`\n"
                )
                md.write(f"- **Input Text:**\n> {r['input_text']}\n\n")
                md.write(f"- **Model Justification:**\n> {r['justification']}\n\n")
                if r["retrieved_chunks"]:
                    md.write(f"**Retrieved Chunks (Top {len(r['retrieved_chunks'])}):**\n")
                    for c in r["retrieved_chunks"]:
                        md.write(
                            f"  - `[Rank {c['rank']}]` **{c['speaker']}** ({c['party']}, {c['date']}) | Score: `{c['score']}`\n"
                        )
                        md.write(f"    - *Snippet:* {c['snippet']}\n")
                md.write("\n---\n\n")


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

def process_eval_logs(logs_dir: Path, output_dir: Path, top_k: int = 25):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped_data = defaultdict(list)
    total_parsed = 0

    log_files = sorted(logs_dir.rglob("*.jsonl"))
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
        llm_name = records[0]["llm"].split("/")[-1]
        family = _model_family(llm_name)
        group_subpath = out_dir / family / group_key
        group_subpath.mkdir(parents=True, exist_ok=True)

        generate_json_report(
            group_key, records, top_k, group_subpath / "bias_analysis.json"
        )
        generate_markdown_report(
            group_key, records, top_k, group_subpath / "bias_analysis.md"
        )

        print(f"[{group_key}] -> Saved bias analysis to `{group_subpath}/`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze directional (left/right) bias in RAG evaluation logs."
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
        help="Number of extreme examples to extract (default: 25)",
    )

    args = parser.parse_args()
    process_eval_logs(args.logs_dir, output_dir=args.output_dir, top_k=args.k)
