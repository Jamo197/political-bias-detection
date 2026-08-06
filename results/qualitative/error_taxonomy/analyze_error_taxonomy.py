#!/usr/bin/env python3
"""analyze_error_taxonomy.py

Isolates extreme prediction errors from RAG batch-run JSONL logs to build a
qualitative failure taxonomy for thesis write-up.

Produces (per run):
  1. extreme_cases.jsonl — structured extreme cases (worst 10% / best 10%).
  2. extreme_cases.csv   — flat table for spreadsheet review.
  3. summary.json        — per-run MAE and bin counts.

Outputs are saved to results/error_taxonomy/<YYYY-MM-DD>/.

Usage:
    python results/analyze_error_taxonomy.py
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("logs/batch_runs")
RESULTS_DIR = Path("results/analysis")
OUTPUT_SUBDIR = "error_taxonomy"

TARGET_COL = "label_ideology"
PRED_COL = "bias"

WORST_THRESHOLD = 1.5
BEST_THRESHOLD = 0.2
WORST_PCT = 0.10
BEST_PCT = 0.10

CHUNK_TEXT_TRUNCATE = 500

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def discover_rag_jsonl_files(base_dir: Path) -> List[Path]:
    """Find all party_label_*.jsonl files under base_dir."""
    return sorted(base_dir.rglob("party_label_*.jsonl"))


def load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    """Stream-read a JSONL file."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def extract_ground_truth(record: Dict[str, Any]) -> float:
    """Safely extract label_ideology as float."""
    gt = record.get("ground_truth", {})
    val = gt.get(TARGET_COL)
    if val is None:
        return float("nan")
    return float(val)


def extract_prediction(record: Dict[str, Any]) -> float:
    """Safely extract output.bias as float."""
    out = record.get("output", {})
    val = out.get(PRED_COL)
    if val is None:
        return float("nan")
    return float(val)


def extract_run_key(record: Dict[str, Any]) -> str:
    """Build a unique run key from parameters."""
    params = record.get("parameters", {})
    parts = [
        params.get("llm", "unknown"),
        params.get("embedding_model", "none"),
        params.get("retrieval_mode", "unknown"),
        record.get("run_id", "unknown"),
    ]
    return "|".join(str(p) for p in parts)


def parse_run_key(run_key: str) -> Dict[str, str]:
    """Reverse extract_run_key."""
    parts = run_key.split("|")
    return {
        "model": parts[0] if len(parts) > 0 else "unknown",
        "embedding": parts[1] if len(parts) > 1 else "none",
        "strategy": parts[2] if len(parts) > 2 else "unknown",
        "run_id": parts[3] if len(parts) > 3 else "unknown",
    }


def truncate_text(text: str, limit: int = CHUNK_TEXT_TRUNCATE) -> str:
    """Truncate long text with ellipsis marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + " [truncated]"


def format_retrieved_chunks(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract and lightly truncate retrieved chunks."""
    inputs = record.get("inputs", {})
    chunks = inputs.get("retrieved_chunks", [])
    formatted = []
    for idx, chunk in enumerate(chunks, start=1):
        formatted.append(
            {
                "rank": idx,
                "party": chunk.get("party", "unknown"),
                "speaker": chunk.get("speaker", "unknown"),
                "date": chunk.get("date", "unknown"),
                "speech_id": chunk.get("speech_id", "unknown"),
                "score": chunk.get("score", None),
                "text": truncate_text(chunk.get("text", "")),
            }
        )
    return formatted


def flatten_retrieved_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten chunk list into CSV columns."""
    flat = {}
    for chunk in chunks:
        prefix = f"retrieved_chunk_{chunk['rank']}"
        flat[f"{prefix}_party"] = chunk["party"]
        flat[f"{prefix}_speaker"] = chunk["speaker"]
        flat[f"{prefix}_date"] = chunk["date"]
        flat[f"{prefix}_speech_id"] = chunk["speech_id"]
        flat[f"{prefix}_score"] = chunk["score"]
        flat[f"{prefix}_text"] = chunk["text"]
    return flat


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def process_all_rag_runs(base_dir: Path) -> pd.DataFrame:
    """Load all RAG party_label JSONLs and return a flat DataFrame."""
    files = discover_rag_jsonl_files(base_dir)
    all_rows = []

    for file_path in files:
        records = load_jsonl_records(file_path)
        for rec in records:
            params = rec.get("parameters", {})
            if not params.get("is_rag", False):
                continue

            gt = extract_ground_truth(rec)
            pred = extract_prediction(rec)
            if pd.isna(gt) or pd.isna(pred):
                continue

            residual = abs(pred - gt)
            meta = rec.get("input_metadata", {})
            inputs = rec.get("inputs", {})
            output = rec.get("output", {})

            row = {
                "file_path": str(file_path),
                "run_id": rec.get("run_id", "unknown"),
                "model": params.get("llm", "unknown"),
                "embedding": params.get("embedding_model", "none"),
                "strategy": params.get("retrieval_mode", "unknown"),
                "llm_region": params.get("llm_region", "unknown"),
                "k_chunks": params.get("k_chunks", 0),
                "text_index": meta.get("text_index", "unknown"),
                "target_party": meta.get("party", "unknown"),
                "target_speaker": meta.get("speaker", "unknown"),
                "target_source": meta.get("source", "unknown"),
                "target_text": inputs.get("text", ""),
                "predicted_bias": pred,
                "ground_truth_ideology": gt,
                "residual": residual,
                "justification": output.get("justification", ""),
                "retrieved_chunks": format_retrieved_chunks(rec),
                "manual_failure_mode": None,
                "manual_notes": None,
            }
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    return df


def bin_residuals_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """Assign worst_10 / best_10 bins per run."""
    df["bin"] = None

    run_keys = df[["run_id", "model", "embedding", "strategy"]].drop_duplicates()
    for _, run in run_keys.iterrows():
        mask = (
            (df["run_id"] == run["run_id"])
            & (df["model"] == run["model"])
            & (df["embedding"] == run["embedding"])
            & (df["strategy"] == run["strategy"])
        )
        run_df = df.loc[mask]
        n = len(run_df)
        if n == 0:
            continue

        # Worst 10%
        worst_n = max(1, int(n * WORST_PCT))
        worst_indices = run_df["residual"].nlargest(worst_n).index
        # Best 10%
        best_n = max(1, int(n * BEST_PCT))
        best_indices = run_df["residual"].nsmallest(best_n).index

        # Apply threshold override: if residual > 1.5, force worst_10
        # (already captured by percentile, but ensure any >1.5 is included)
        extra_worst = run_df[run_df["residual"] > WORST_THRESHOLD].index
        worst_indices = worst_indices.union(extra_worst)

        # Apply threshold override: if residual < 0.2, force best_10
        extra_best = run_df[run_df["residual"] < BEST_THRESHOLD].index
        best_indices = best_indices.union(extra_best)

        df.loc[worst_indices, "bin"] = "worst_10"
        df.loc[best_indices, "bin"] = "best_10"

    return df


def build_extreme_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows tagged worst_10 or best_10."""
    extreme = df[df["bin"].isin(["worst_10", "best_10"])].copy()
    return extreme


def export_extreme_jsonl(extreme_df: pd.DataFrame, out_path: Path) -> None:
    """Write extreme_cases.jsonl."""
    records = extreme_df.to_dict(orient="records")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_extreme_csv(extreme_df: pd.DataFrame, out_path: Path) -> None:
    """Write extreme_cases.csv with flattened chunks."""
    rows = []
    for _, rec in extreme_df.iterrows():
        flat = {
            "file_path": rec["file_path"],
            "run_id": rec["run_id"],
            "model": rec["model"],
            "embedding": rec["embedding"],
            "strategy": rec["strategy"],
            "llm_region": rec["llm_region"],
            "k_chunks": rec["k_chunks"],
            "text_index": rec["text_index"],
            "target_party": rec["target_party"],
            "target_speaker": rec["target_speaker"],
            "target_source": rec["target_source"],
            "target_text": truncate_text(rec["target_text"]),
            "predicted_bias": rec["predicted_bias"],
            "ground_truth_ideology": rec["ground_truth_ideology"],
            "residual": rec["residual"],
            "bin": rec["bin"],
            "justification": rec["justification"],
            "manual_failure_mode": rec["manual_failure_mode"],
            "manual_notes": rec["manual_notes"],
        }
        flat.update(flatten_retrieved_chunks(rec["retrieved_chunks"]))
        rows.append(flat)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")


def build_summary(df: pd.DataFrame, extreme_df: pd.DataFrame) -> Dict[str, Any]:
    """Build per-run summary statistics."""
    summary = {}

    run_keys = df[["run_id", "model", "embedding", "strategy"]].drop_duplicates()
    for _, run in run_keys.iterrows():
        mask = (
            (df["run_id"] == run["run_id"])
            & (df["model"] == run["model"])
            & (df["embedding"] == run["embedding"])
            & (df["strategy"] == run["strategy"])
        )
        run_df = df.loc[mask]
        n = len(run_df)
        mae = mean_absolute_error(
            run_df["ground_truth_ideology"], run_df["predicted_bias"]
        )

        worst_mask = extreme_df[
            (extreme_df["run_id"] == run["run_id"])
            & (extreme_df["model"] == run["model"])
            & (extreme_df["embedding"] == run["embedding"])
            & (extreme_df["strategy"] == run["strategy"])
            & (extreme_df["bin"] == "worst_10")
        ]
        best_mask = extreme_df[
            (extreme_df["run_id"] == run["run_id"])
            & (extreme_df["model"] == run["model"])
            & (extreme_df["embedding"] == run["embedding"])
            & (extreme_df["strategy"] == run["strategy"])
            & (extreme_df["bin"] == "best_10")
        ]

        key = (
            f"{run['model']} | {run['embedding']} | {run['strategy']} | {run['run_id']}"
        )
        summary[key] = {
            "total_samples": int(n),
            "mae": round(float(mae), 4),
            "worst_10_count": int(len(worst_mask)),
            "worst_10_pct": round(len(worst_mask) / n * 100, 2) if n > 0 else 0.0,
            "best_10_count": int(len(best_mask)),
            "best_10_pct": round(len(best_mask) / n * 100, 2) if n > 0 else 0.0,
            "worst_10_preview_text_indices": worst_mask["text_index"].tolist()[:20],
            "best_10_preview_text_indices": best_mask["text_index"].tolist()[:20],
        }

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Error taxonomy analysis for RAG batch runs."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Root directory containing batch_runs subfolders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=RESULTS_DIR / OUTPUT_SUBDIR,
        help="Directory to write taxonomy outputs.",
    )
    args = parser.parse_args()

    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    out_dir = args.out_dir / today_str
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Discovering and loading RAG party_label JSONL files …")
    df = process_all_rag_runs(args.base_dir)
    print(
        f"       Loaded {len(df)} RAG-evaluated samples from {df['file_path'].nunique()} files."
    )

    if df.empty:
        print("No RAG records found. Exiting.")
        return

    print("[2/5] Binning residuals per run (worst 10% / best 10%) …")
    df = bin_residuals_per_run(df)

    print("[3/5] Extracting extreme cases …")
    extreme_df = build_extreme_cases(df)
    print(f"       {len(extreme_df)} extreme cases selected.")

    print("[4/5] Writing extreme_cases.jsonl …")
    jsonl_path = out_dir / "extreme_cases.jsonl"
    export_extreme_jsonl(extreme_df, jsonl_path)
    print(f"       → {jsonl_path}")

    print("[5/5] Writing extreme_cases.csv and summary.json …")
    csv_path = out_dir / "extreme_cases.csv"
    export_extreme_csv(extreme_df, csv_path)
    print(f"       → {csv_path}")

    summary = build_summary(df, extreme_df)
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"       → {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
