#!/usr/bin/env python3
"""analyze_error_taxonomy.py

Isolates extreme prediction errors from RAG batch-run JSONL logs to build a
qualitative failure taxonomy for thesis write-up.

Binning strategy
----------------
Each unique (run_id, model, embedding_model, retrieval_mode) group is binned
independently:
  * worst_10 — top 10% by absolute residual (plus any residual > 1.5)
  * best_10  — bottom 10% by absolute residual (plus any residual < 0.2)

Outputs
-------
Every execution creates a new timestamped folder:

  results/qualitative/error_taxonomy/<YYYYMMDD_HHMMSS>/

containing the extreme-case data (JSONL, CSV), per-run summary (JSON), per-query
CSV, and distribution plots.

Usage:
    python src/results/analyze_error_taxonomy.py
    python src/results/analyze_error_taxonomy.py --batch-run 2026-08-04_eval_matrix_20260804_191413
    python src/results/analyze_error_taxonomy.py --embedding-models bge,jina
    python src/results/analyze_error_taxonomy.py --retrieval-modes simple,twostage
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("logs/batch_runs")
OUTPUT_ROOT = Path("results/qualitative/error_taxonomy")

TARGET_COL = "label_ideology"
PRED_COL = "bias"

WORST_THRESHOLD = 1.5
BEST_THRESHOLD = 0.2
WORST_PCT = 0.10
BEST_PCT = 0.10

CHUNK_TEXT_TRUNCATE = 500
DISPLAY_TEXT_TRUNCATE = 300

MAIN_PARTIES = ["SPD", "CDU/CSU", "AfD", "FDP", "Greens", "Left"]

# ---------------------------------------------------------------------------
# Text Helpers
# ---------------------------------------------------------------------------


def truncate_text(text: str, limit: int = CHUNK_TEXT_TRUNCATE) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " [truncated]"


def format_retrieved_chunks(record: Dict[str, Any]) -> List[Dict[str, Any]]:
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
# Data Loading
# ---------------------------------------------------------------------------


def load_rag_logs(
    base_dir: Path,
    batch_run: Optional[str] = None,
    embedding_models: Optional[List[str]] = None,
    retrieval_modes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load party_label_*.jsonl files and keep only RAG runs with chunks."""
    records: List[Dict] = []
    jsonl_files = sorted(base_dir.rglob("party_label_*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            f"No party_label_*.jsonl files found under '{base_dir}'"
        )

    for fp in jsonl_files:
        run_folder = fp.relative_to(base_dir).parts[0]
        if batch_run and run_folder != batch_run:
            continue

        with open(fp, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    log = json.loads(line)
                except json.JSONDecodeError:
                    continue

                params = log.get("parameters", {})
                emb = params.get("embedding_model", "none")
                is_rag = params.get("is_rag", False)
                retrieval_mode = params.get("retrieval_mode", "unknown")

                if not is_rag or emb in (None, "none", ""):
                    continue
                if embedding_models and emb not in embedding_models:
                    continue
                if retrieval_modes and retrieval_mode not in retrieval_modes:
                    continue

                chunks = log.get("inputs", {}).get("retrieved_chunks", [])
                if not chunks:
                    continue

                meta = log.get("input_metadata", {})
                gt = log.get("ground_truth", {})
                output = log.get("output", {})

                gt_val = gt.get(TARGET_COL)
                pred_val = output.get(PRED_COL)
                if gt_val is None or pred_val is None:
                    continue

                gt_f = float(gt_val)
                pred_f = float(pred_val)

                records.append(
                    {
                        "file_path": str(fp),
                        "run_id": log.get("run_id", "unknown"),
                        "batch_run": run_folder,
                        "model": _clean_model_name(params.get("llm", "unknown")),
                        "embedding_model": emb,
                        "retrieval_mode": retrieval_mode,
                        "condition": f"{emb}/{retrieval_mode}",
                        "llm_region": params.get("llm_region", "unknown"),
                        "k_chunks": params.get("k_chunks", 0),
                        "text_index": meta.get("text_index", ""),
                        "target_party": meta.get("party", "unknown"),
                        "target_speaker": meta.get("speaker", "unknown"),
                        "target_source": meta.get("source", "unknown"),
                        "target_text": log.get("inputs", {}).get("text", ""),
                        "predicted_bias": pred_f,
                        "label_ideology": gt_f,
                        "residual": abs(pred_f - gt_f),
                        "justification": output.get("justification", ""),
                        "retrieved_chunks": format_retrieved_chunks(log),
                        "n_chunks": len(chunks),
                        "manual_failure_mode": None,
                        "manual_notes": None,
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        print("No matching RAG records found (check your filters).")
        return df

    print(f"Loaded {len(df)} RAG records from {len(jsonl_files)} JSONL files.")
    print(f"  Batch runs:       {sorted(df['batch_run'].unique())}")
    print(f"  Embedding models: {sorted(df['embedding_model'].unique())}")
    print(f"  Retrieval modes:  {sorted(df['retrieval_mode'].unique())}")
    print(f"  Conditions:       {sorted(df['condition'].unique())}")
    return df


def _clean_model_name(llm_path: str) -> str:
    if not llm_path or llm_path == "unknown":
        return "Unknown"
    return llm_path.split("/")[-1]


# ---------------------------------------------------------------------------
# Residual Binning
# ---------------------------------------------------------------------------


def bin_residuals_per_run(df: pd.DataFrame) -> pd.DataFrame:
    """Assign worst_10 / best_10 bins per (run_id, model, emb, retrieval_mode)."""
    df = df.copy()
    df["bin"] = None

    run_keys = df[["run_id", "model", "embedding_model", "retrieval_mode"]].drop_duplicates()
    for _, run in run_keys.iterrows():
        mask = (
            (df["run_id"] == run["run_id"])
            & (df["model"] == run["model"])
            & (df["embedding_model"] == run["embedding_model"])
            & (df["retrieval_mode"] == run["retrieval_mode"])
        )
        run_df = df.loc[mask]
        n = len(run_df)
        if n == 0:
            continue

        worst_n = max(1, int(n * WORST_PCT))
        worst_indices = run_df["residual"].nlargest(worst_n).index
        best_n = max(1, int(n * BEST_PCT))
        best_indices = run_df["residual"].nsmallest(best_n).index

        extra_worst = run_df[run_df["residual"] > WORST_THRESHOLD].index
        worst_indices = worst_indices.union(extra_worst)
        extra_best = run_df[run_df["residual"] < BEST_THRESHOLD].index
        best_indices = best_indices.union(extra_best)

        df.loc[worst_indices, "bin"] = "worst_10"
        df.loc[best_indices, "bin"] = "best_10"

    return df


def build_extreme_cases(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["bin"].isin(["worst_10", "best_10"])].copy()


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


def export_extreme_jsonl(extreme_df: pd.DataFrame, out_path: Path) -> None:
    records = extreme_df.to_dict(orient="records")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_extreme_csv(extreme_df: pd.DataFrame, out_path: Path) -> None:
    rows = []
    for _, rec in extreme_df.iterrows():
        flat = {
            "file_path": rec["file_path"],
            "run_id": rec["run_id"],
            "batch_run": rec["batch_run"],
            "model": rec["model"],
            "embedding_model": rec["embedding_model"],
            "retrieval_mode": rec["retrieval_mode"],
            "condition": rec["condition"],
            "llm_region": rec["llm_region"],
            "k_chunks": rec["k_chunks"],
            "text_index": rec["text_index"],
            "target_party": rec["target_party"],
            "target_speaker": rec["target_speaker"],
            "target_source": rec["target_source"],
            "target_text": truncate_text(rec["target_text"], DISPLAY_TEXT_TRUNCATE),
            "predicted_bias": rec["predicted_bias"],
            "label_ideology": rec["label_ideology"],
            "residual": rec["residual"],
            "n_chunks": rec["n_chunks"],
            "bin": rec["bin"],
            "justification": rec["justification"],
            "manual_failure_mode": rec["manual_failure_mode"],
            "manual_notes": rec["manual_notes"],
        }
        flat.update(flatten_retrieved_chunks(rec["retrieved_chunks"]))
        rows.append(flat)

    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")


def build_summary(df: pd.DataFrame, extreme_df: pd.DataFrame) -> Dict[str, Any]:
    summary = {}
    run_keys = df[["run_id", "model", "embedding_model", "retrieval_mode"]].drop_duplicates()
    for _, run in run_keys.iterrows():
        mask = (
            (df["run_id"] == run["run_id"])
            & (df["model"] == run["model"])
            & (df["embedding_model"] == run["embedding_model"])
            & (df["retrieval_mode"] == run["retrieval_mode"])
        )
        run_df = df.loc[mask]
        n = len(run_df)
        mae = mean_absolute_error(run_df["label_ideology"], run_df["predicted_bias"])

        extreme_sub = extreme_df[
            (extreme_df["run_id"] == run["run_id"])
            & (extreme_df["model"] == run["model"])
            & (extreme_df["embedding_model"] == run["embedding_model"])
            & (extreme_df["retrieval_mode"] == run["retrieval_mode"])
        ]
        worst_mask = extreme_sub[extreme_sub["bin"] == "worst_10"]
        best_mask = extreme_sub[extreme_sub["bin"] == "best_10"]

        cond = f"{run['embedding_model']}/{run['retrieval_mode']}"
        key = f"{run['model']} | {cond} | {run['run_id']}"
        summary[key] = {
            "total_samples": int(n),
            "mae": round(float(mae), 4),
            "condition": cond,
            "worst_10_count": int(len(worst_mask)),
            "worst_10_pct": round(len(worst_mask) / n * 100, 2) if n > 0 else 0.0,
            "best_10_count": int(len(best_mask)),
            "best_10_pct": round(len(best_mask) / n * 100, 2) if n > 0 else 0.0,
            "worst_10_preview_text_indices": worst_mask["text_index"].tolist()[:20],
            "best_10_preview_text_indices": best_mask["text_index"].tolist()[:20],
        }
    return summary


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_residual_distribution(df: pd.DataFrame, output_dir: Path):
    """Overall histogram + per-condition residual distribution plots."""
    valid = df.dropna(subset=["residual"]).copy()

    # Overall histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, valid["residual"].max() + 0.1, 60)
    ax.hist(valid["residual"], bins=bins, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(BEST_THRESHOLD, color="#2ecc71", linestyle="--", linewidth=1.5,
               label=f"Best threshold ({BEST_THRESHOLD})")
    ax.axvline(WORST_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Worst threshold ({WORST_THRESHOLD})")

    pct90 = valid["residual"].quantile(0.90)
    pct10 = valid["residual"].quantile(0.10)
    ax.axvline(pct90, color="#c0392b", linestyle=":", linewidth=1,
               label=f"90th pctl ({pct90:.2f})")
    ax.axvline(pct10, color="#27ae60", linestyle=":", linewidth=1,
               label=f"10th pctl ({pct10:.2f})")

    ax.set_xlabel("Absolute Residual |predicted − ground truth|", fontsize=12)
    ax.set_ylabel("Number of Queries", fontsize=12)
    ax.set_title("Residual Distribution (All Conditions)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, output_dir / "residual_distribution_overall.png")

    # Per-condition small multiples
    conditions = sorted(valid["condition"].unique())
    n_cols = 3
    n_rows = (len(conditions) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, cond in enumerate(conditions):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = valid[valid["condition"] == cond]
        ax.hist(sub["residual"], bins=bins, color="#3498db", edgecolor="white", alpha=0.8)
        ax.axvline(BEST_THRESHOLD, color="#2ecc71", linestyle="--", linewidth=1)
        ax.axvline(WORST_THRESHOLD, color="#e74c3c", linestyle="--", linewidth=1)
        ax.set_title(cond, fontsize=9, fontweight="bold")
        ax.set_xlim(0, valid["residual"].max() + 0.1)

    for idx in range(len(conditions), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle("Residual Distribution by Condition", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, output_dir / "residual_distribution_by_condition.png")


def plot_extreme_cases_by_condition(df: pd.DataFrame, output_dir: Path):
    """Grouped bar chart: worst_10 vs best_10 count per condition."""
    valid = df.dropna(subset=["bin"]).copy()
    counts = (
        valid.groupby(["condition", "bin"]).size()
        .reset_index(name="count")
    )

    pivot = counts.pivot_table(
        index="condition", columns="bin", values="count", fill_value=0
    )
    for col in ["worst_10", "best_10"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.sort_index()

    x = np.arange(len(pivot))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width / 2, pivot["worst_10"], width, label="Worst 10%",
           color="#e74c3c", edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, pivot["best_10"], width, label="Best 10%",
           color="#2ecc71", edgecolor="black", linewidth=0.5)

    for idx, (_, row) in enumerate(pivot.iterrows()):
        total = row["worst_10"] + row["best_10"]
        ymax = max(row["worst_10"], row["best_10"])
        ax.text(idx, ymax + 2, str(int(total)), ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)

    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Extreme Cases", fontsize=12)
    ax.set_title("Extreme Cases by Condition", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, output_dir / "extreme_cases_by_condition.png")


def plot_mae_by_condition(df: pd.DataFrame, output_dir: Path):
    """Bar chart of overall MAE per condition, colored by embedding model."""
    cond_mae = (
        df.groupby("condition")["residual"]
        .mean()
        .reset_index()
        .sort_values("residual", ascending=False)
    )
    total_mae = df["residual"].mean()

    emb_colors = {"e5": "#1f77b4", "jina": "#ff7f0e", "qwen3": "#2ca02c", "bge": "#d62728"}
    bar_colors = [
        emb_colors.get(c.split("/")[0], "#7f7f7f") for c in cond_mae["condition"]
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(
        cond_mae["condition"],
        cond_mae["residual"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(total_mae, color="gray", linestyle="--", linewidth=1,
               label=f"Overall MAE = {total_mae:.3f}")

    for bar, val in zip(bars, cond_mae["residual"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01 * cond_mae["residual"].max(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xticks(range(len(cond_mae)))
    ax.set_xticklabels(cond_mae["condition"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Absolute Error", fontsize=12)
    ax.set_title("MAE by Condition", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, output_dir / "mae_by_condition.png")


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    base_dir: Path = BASE_DIR,
    output_root: Path = OUTPUT_ROOT,
    run_id: Optional[str] = None,
    batch_run: Optional[str] = None,
    embedding_models: Optional[List[str]] = None,
    retrieval_modes: Optional[List[str]] = None,
):
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Error Taxonomy Analysis — Extreme Case Extraction")
    print("=" * 80)
    print(f"Output directory: {output_dir.resolve()}\n")

    print("[1/6] Loading RAG party_label JSONL files …")
    df = load_rag_logs(base_dir, batch_run, embedding_models, retrieval_modes)
    if df.empty:
        return

    # Per-query CSV (before binning)
    per_query_cols = [
        "file_path", "run_id", "batch_run", "model", "embedding_model",
        "retrieval_mode", "condition", "llm_region", "k_chunks",
        "text_index", "target_party", "target_speaker", "target_source",
        "n_chunks", "predicted_bias", "label_ideology", "residual",
    ]
    per_query_path = output_dir / "error_taxonomy_per_query.csv"
    df[per_query_cols].to_csv(per_query_path, index=False)
    print(f"\nSaved per-query CSV: {per_query_path}")

    print("[2/6] Binning residuals per run (worst 10% / best 10%) …")
    df = bin_residuals_per_run(df)

    bin_counts = df["bin"].value_counts()
    print(f"       {bin_counts.get('worst_10', 0)} worst, {bin_counts.get('best_10', 0)} best")

    print("[3/6] Extracting extreme cases …")
    extreme_df = build_extreme_cases(df)
    print(f"       {len(extreme_df)} extreme cases total.")

    print("[4/6] Writing extreme_cases.jsonl …")
    jsonl_path = output_dir / "extreme_cases.jsonl"
    export_extreme_jsonl(extreme_df, jsonl_path)
    print(f"       → {jsonl_path}")

    print("[5/6] Writing extreme_cases.csv and summary.json …")
    csv_path = output_dir / "extreme_cases.csv"
    export_extreme_csv(extreme_df, csv_path)
    print(f"       → {csv_path}")

    summary = build_summary(df, extreme_df)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"       → {summary_path}")

    print("[6/6] Generating visualizations …")
    plot_residual_distribution(df, output_dir)
    plot_extreme_cases_by_condition(df, output_dir)
    plot_mae_by_condition(df, output_dir)

    print(f"\nAll outputs written to: {output_dir.resolve()}")
    print("Done.")


def _comma_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Error taxonomy analysis — isolate extreme prediction errors.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Root directory with batch run JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for results; a <run-id> subfolder is created per execution.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Name of the output subfolder (default: current timestamp YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--batch-run",
        type=str,
        default=None,
        help="Restrict analysis to a single batch run folder under --base-dir.",
    )
    parser.add_argument(
        "--embedding-models",
        type=_comma_list,
        default=None,
        help="Comma-separated embedding models to include (default: all found).",
    )
    parser.add_argument(
        "--retrieval-modes",
        type=_comma_list,
        default=None,
        help="Comma-separated retrieval modes to include (default: all found).",
    )
    args = parser.parse_args()
    main(
        base_dir=args.base_dir,
        output_root=args.output_root,
        run_id=args.run_id,
        batch_run=args.batch_run,
        embedding_models=args.embedding_models,
        retrieval_modes=args.retrieval_modes,
    )
