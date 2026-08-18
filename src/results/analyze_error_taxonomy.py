#!/usr/bin/env python3
"""analyze_error_taxonomy.py

Analyzes batch-run RAG logs, computes performance metrics, and isolates extreme
prediction errors (best 10% and worst 10%) per LLM, embedding model, and retrieval setup.

Outputs (under results/qualitative/error_taxonomy/<run_id>/):
  - extreme_cases.jsonl / extreme_cases.csv
  - all_predictions_evaluated.csv
  - summary_metrics.json
  - Visualizations (Residuals, MAE by LLM/Embedding, Retrieval Bias Diagnostics)
"""

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

# ---------------------------------------------------------------------------
# Defaults & Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path("logs/batch_runs")
OUTPUT_ROOT = Path("results/qualitative/error_taxonomy")

DEFAULT_TARGET_COL = "label_ideology"
DEFAULT_PRED_COL = "bias"

DEFAULT_PERCENTILE = 0.10
WORST_RESIDUAL_THRESHOLD = 1.5
BEST_RESIDUAL_THRESHOLD = 0.2

CHUNK_TEXT_TRUNCATE = 500
DISPLAY_TEXT_TRUNCATE = 300

sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Text & Metadata Helpers
# ---------------------------------------------------------------------------


def truncate_text(text: str, limit: int = CHUNK_TEXT_TRUNCATE) -> str:
    if not isinstance(text, str):
        return ""
    return text if len(text) <= limit else text[:limit] + " [truncated]"


def extract_retrieval_diagnostics(
    chunks: List[Dict[str, Any]], target_party: str
) -> Dict[str, Any]:
    """Computes RAG context metrics to evaluate context bias and retrieval quality."""
    if not chunks:
        return {
            "n_chunks": 0,
            "retrieved_party_match_rate": 0.0,
            "dominant_retrieved_party": "None",
            "retrieved_parties_list": "",
            "mean_retrieval_score": np.nan,
            "score_spread": np.nan,
        }

    parties = [c.get("party", "unknown") for c in chunks]
    scores = [
        c.get("score") for c in chunks if isinstance(c.get("score"), (int, float))
    ]

    party_counts = Counter(parties)
    dominant_party = party_counts.most_common(1)[0][0] if party_counts else "unknown"
    match_count = sum(1 for p in parties if p.lower() == str(target_party).lower())
    match_rate = round(match_count / len(chunks), 4)

    return {
        "n_chunks": len(chunks),
        "retrieved_party_match_rate": match_rate,
        "dominant_retrieved_party": dominant_party,
        "retrieved_parties_list": ", ".join(parties),
        "mean_retrieval_score": round(float(np.mean(scores)), 4) if scores else np.nan,
        "score_spread": (
            round(float(max(scores) - min(scores)), 4) if len(scores) > 1 else 0.0
        ),
    }


def format_retrieved_chunks(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunks = record.get("inputs", {}).get("retrieved_chunks", [])
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
        prefix = f"chunk_{chunk['rank']}"
        flat[f"{prefix}_party"] = chunk["party"]
        flat[f"{prefix}_speaker"] = chunk["speaker"]
        flat[f"{prefix}_date"] = chunk["date"]
        flat[f"{prefix}_speech_id"] = chunk["speech_id"]
        flat[f"{prefix}_score"] = chunk["score"]
        flat[f"{prefix}_text"] = chunk["text"]
    return flat


def clean_model_name(llm_path: str) -> str:
    if not llm_path or llm_path == "unknown":
        return "Unknown"
    return llm_path.split("/")[-1]


# ---------------------------------------------------------------------------
# Data Ingestion
# ---------------------------------------------------------------------------


def load_rag_logs(
    base_dir: Path,
    file_glob: str = "**/*.jsonl",
    batch_run: Optional[str] = None,
    embedding_models: Optional[List[str]] = None,
    retrieval_modes: Optional[List[str]] = None,
    target_col: str = DEFAULT_TARGET_COL,
    pred_col: str = DEFAULT_PRED_COL,
) -> pd.DataFrame:
    records: List[Dict] = []
    jsonl_files = sorted(base_dir.rglob(file_glob))

    if not jsonl_files:
        raise FileNotFoundError(
            f"No JSONL files found matching pattern '{file_glob}' in '{base_dir}'"
        )

    for fp in jsonl_files:
        run_folder = fp.relative_to(base_dir).parts[0]
        if batch_run and run_folder != batch_run:
            continue

        with open(fp, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    log = json.loads(line)
                except json.JSONDecodeError:
                    continue

                params = log.get("parameters", {})
                emb = params.get("embedding_model", "none")
                retrieval_mode = params.get("retrieval_mode", "unknown")
                is_rag = params.get("is_rag", True)

                if embedding_models and emb not in embedding_models:
                    continue
                if retrieval_modes and retrieval_mode not in retrieval_modes:
                    continue

                gt = log.get("ground_truth", {})
                output = log.get("output", {})
                meta = log.get("input_metadata", {})
                inputs = log.get("inputs", {})

                gt_val = gt.get(target_col)
                pred_val = output.get(pred_col)
                if gt_val is None or pred_val is None:
                    continue

                try:
                    gt_f = float(gt_val)
                    pred_f = float(pred_val)
                except ValueError, TypeError:
                    continue

                raw_chunks = inputs.get("retrieved_chunks", [])
                target_party = meta.get("party", "unknown")
                diagnostics = extract_retrieval_diagnostics(raw_chunks, target_party)
                formatted_chunks = format_retrieved_chunks(log)

                records.append(
                    {
                        "file_path": str(fp),
                        "run_id": log.get("run_id", "unknown"),
                        "timestamp": log.get("timestamp", ""),
                        "batch_run": run_folder,
                        "model": clean_model_name(params.get("llm", "unknown")),
                        "embedding_model": emb,
                        "retrieval_mode": retrieval_mode,
                        "condition": f"{emb}/{retrieval_mode}",
                        "hybrid": params.get("hybrid", False),
                        "is_rag": is_rag,
                        "llm_region": params.get("llm_region", "unknown"),
                        "k_chunks": params.get("k_chunks", len(raw_chunks)),
                        "text_index": str(meta.get("text_index", "")),
                        "target_party": target_party,
                        "target_speaker": meta.get("speaker", "unknown"),
                        "target_source": meta.get("source", "unknown"),
                        "target_text": inputs.get("text", ""),
                        "predicted_val": pred_f,
                        "ground_truth_val": gt_f,
                        "signed_error": round(pred_f - gt_f, 4),
                        "residual": round(abs(pred_f - gt_f), 4),
                        "justification": output.get("justification", ""),
                        "retrieved_chunks": formatted_chunks,
                        **diagnostics,
                        "manual_failure_mode": None,
                        "manual_notes": None,
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        print("No valid records found matching the criteria.")
        return df

    print(f"Loaded {len(df)} records across {len(jsonl_files)} file(s).")
    print(f"  Models:           {sorted(df['model'].unique())}")
    print(f"  Embedding models: {sorted(df['embedding_model'].unique())}")
    print(f"  Retrieval modes:  {sorted(df['retrieval_mode'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Quantile Binning
# ---------------------------------------------------------------------------


def bin_extremes(
    df: pd.DataFrame,
    group_cols: List[str],
    percentile: float = DEFAULT_PERCENTILE,
    strict_quantiles: bool = False,
    worst_threshold: float = WORST_RESIDUAL_THRESHOLD,
    best_threshold: float = BEST_RESIDUAL_THRESHOLD,
) -> pd.DataFrame:
    """Bins worst and best performing predictions per group."""
    df = df.copy()
    df["bin"] = "nominal"

    grouped = df.groupby(group_cols)
    for _, group_df in grouped:
        n = len(group_df)
        if n == 0:
            continue

        n_extreme = max(1, int(np.ceil(n * percentile)))

        worst_idx = group_df["residual"].nlargest(n_extreme).index
        best_idx = group_df["residual"].nsmallest(n_extreme).index

        if not strict_quantiles:
            extra_worst = group_df[group_df["residual"] >= worst_threshold].index
            extra_best = group_df[group_df["residual"] <= best_threshold].index
            worst_idx = worst_idx.union(extra_worst)
            best_idx = best_idx.union(extra_best)

        # Ensure no overlap
        best_idx = best_idx.difference(worst_idx)

        df.loc[worst_idx, "bin"] = "worst_10"
        df.loc[best_idx, "bin"] = "best_10"

    return df


# ---------------------------------------------------------------------------
# Metric Summary
# ---------------------------------------------------------------------------


def generate_summary_metrics(df: pd.DataFrame, group_cols: List[str]) -> Dict[str, Any]:
    summary = {}
    grouped = df.groupby(group_cols)

    for group_key, grp in grouped:
        key_str = " | ".join(
            f"{col}={val}"
            for col, val in zip(
                group_cols, group_key if isinstance(group_key, tuple) else [group_key]
            )
        )

        y_true = grp["ground_truth_val"]
        y_pred = grp["predicted_val"]
        residuals = grp["residual"]
        signed = grp["signed_error"]

        worst_sub = grp[grp["bin"] == "worst_10"]
        best_sub = grp[grp["bin"] == "best_10"]

        summary[key_str] = {
            "count": int(len(grp)),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse": round(float(root_mean_squared_error(y_true, y_pred)), 4),
            "mean_signed_bias": round(float(signed.mean()), 4),
            "std_residual": round(float(residuals.std()), 4),
            "worst_10_count": int(len(worst_sub)),
            "best_10_count": int(len(best_sub)),
            "worst_10_indices": worst_sub["text_index"].tolist(),
            "best_10_indices": best_sub["text_index"].tolist(),
        }

    return summary


# ---------------------------------------------------------------------------
# Visualization Suite
# ---------------------------------------------------------------------------


def generate_visualizations(df: pd.DataFrame, out_dir: Path) -> None:
    """Generates comparative analytical plots for model error and retrieval impact."""

    # 1. MAE Comparison by Model & Embedding
    fig, ax = plt.subplots(figsize=(10, 5))
    mae_df = df.groupby(["model", "embedding_model"])["residual"].mean().reset_index()
    sns.barplot(
        data=mae_df,
        x="model",
        y="residual",
        hue="embedding_model",
        ax=ax,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title(
        "Mean Absolute Error (MAE) by LLM and Embedding Model", fontweight="bold"
    )
    ax.set_ylabel("MAE (|Predicted - Ground Truth|)")
    ax.set_xlabel("LLM")
    plt.xticks(rotation=20, ha="right")
    plt.legend(title="Embedding Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    _save_fig(fig, out_dir / "mae_model_x_embedding.png")

    # 2. Residual Distribution by Model
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        data=df,
        x="model",
        y="residual",
        hue="model",
        palette="Set2",
        legend=False,
        ax=ax,
    )
    ax.axhline(
        WORST_RESIDUAL_THRESHOLD,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Worst Threshold (≥ {WORST_RESIDUAL_THRESHOLD})",
    )
    ax.axhline(
        BEST_RESIDUAL_THRESHOLD,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Best Threshold (≤ {BEST_RESIDUAL_THRESHOLD})",
    )
    ax.set_title("Residual Distribution per LLM", fontweight="bold")
    ax.set_ylabel("Absolute Residual")
    ax.set_xlabel("LLM")
    ax.legend(loc="upper right")
    plt.tight_layout()
    _save_fig(fig, out_dir / "residual_distribution_by_llm.png")

    # 3. Directional Error / Signed Bias Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df,
        x="model",
        y="signed_error",
        hue="model",
        palette="Pastel1",
        legend=False,
        ax=ax,
        inner="quartile",
    )
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_title(
        "Systematic Ideological Drift (Signed Error: Predicted - Ground Truth)",
        fontweight="bold",
    )
    ax.set_ylabel("Signed Error (>0: Leaning Right / <0: Leaning Left)")
    ax.set_xlabel("LLM")
    plt.tight_layout()
    _save_fig(fig, out_dir / "directional_bias_violin.png")

    # 4. Context Alignment vs. Residual
    if "retrieved_party_match_rate" in df.columns:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.regplot(
            data=df,
            x="retrieved_party_match_rate",
            y="residual",
            scatter_kws={"alpha": 0.3, "color": "#34495e"},
            line_kws={"color": "#e74c3c"},
            ax=ax,
        )
        ax.set_title(
            "Retrieval Party Alignment vs. Prediction Error", fontweight="bold"
        )
        ax.set_xlabel("Retrieved Chunks Party Match Rate (0.0 to 1.0)")
        ax.set_ylabel("Absolute Residual")
        plt.tight_layout()
        _save_fig(fig, out_dir / "retrieval_alignment_vs_residual.png")


def _save_fig(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exporters
# ---------------------------------------------------------------------------


def export_extreme_dataset(extreme_df: pd.DataFrame, out_dir: Path) -> None:
    # 1. JSONL Export
    jsonl_path = out_dir / "extreme_cases.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for row in extreme_df.to_dict(orient="records"):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 2. Flattened CSV Export
    csv_rows = []
    for _, rec in extreme_df.iterrows():
        base_dict = {k: v for k, v in rec.items() if k != "retrieved_chunks"}
        base_dict["target_text"] = truncate_text(
            base_dict.get("target_text", ""), DISPLAY_TEXT_TRUNCATE
        )
        base_dict.update(flatten_retrieved_chunks(rec["retrieved_chunks"]))
        csv_rows.append(base_dict)

    csv_path = out_dir / "extreme_cases.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Isolate and analyze top 10% best/worst RAG evaluations."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Root directory containing JSONL logs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Directory to store analysis outputs.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Custom folder name for the execution run.",
    )
    parser.add_argument(
        "--batch-run",
        type=str,
        default=None,
        help="Filter to a single batch run folder.",
    )
    parser.add_argument(
        "--embedding-models",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=None,
    )
    parser.add_argument(
        "--retrieval-modes",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=None,
    )
    parser.add_argument(
        "--group-by",
        type=lambda s: [x.strip() for x in s.split(",")],
        default=["model", "embedding_model", "retrieval_mode"],
        help="Comma-separated columns to group by when calculating top/bottom 10%% (e.g., 'model' or 'model,embedding_model').",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=DEFAULT_TARGET_COL,
        help="Target ground truth key.",
    )
    parser.add_argument(
        "--pred-col", type=str, default=DEFAULT_PRED_COL, help="Prediction output key."
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PERCENTILE,
        help="Percentile threshold (e.g. 0.10 for 10%%).",
    )
    parser.add_argument(
        "--strict-quantiles",
        action="store_true",
        help="Enforce exact quantiles without adding threshold-based extra cases.",
    )

    args = parser.parse_args()

    run_name = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print(f"RAG Error Taxonomy & Extreme Cases Extraction [{run_name}]")
    print("=" * 75)

    df = load_rag_logs(
        base_dir=args.base_dir,
        batch_run=args.batch_run,
        embedding_models=args.embedding_models,
        retrieval_modes=args.retrieval_modes,
        target_col=args.target_col,
        pred_col=args.pred_col,
    )

    if df.empty:
        return

    # Export full evaluated per-query baseline CSV
    full_csv_path = out_dir / "all_predictions_evaluated.csv"
    export_df = df.drop(columns=["retrieved_chunks"])
    export_df.to_csv(full_csv_path, index=False, encoding="utf-8-sig")
    print(f"Per-query predictions exported to: {full_csv_path}")

    # Bin extreme cases
    print(
        f"Binning best/worst cases grouped by: {args.group_by} (p={args.percentile}) ..."
    )
    df = bin_extremes(
        df,
        group_cols=args.group_by,
        percentile=args.percentile,
        strict_quantiles=args.strict_quantiles,
    )

    extreme_df = df[df["bin"].isin(["worst_10", "best_10"])].copy()
    counts = extreme_df["bin"].value_counts()
    print(
        f"Isolated {counts.get('worst_10', 0)} worst cases and {counts.get('best_10', 0)} best cases."
    )

    # Export extremes
    export_extreme_dataset(extreme_df, out_dir)
    print(f"Extreme cases written to: {out_dir / 'extreme_cases.csv'}")

    # Generate Summary Metrics
    summary = generate_summary_metrics(df, group_cols=args.group_by)
    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary metrics written to: {out_dir / 'summary_metrics.json'}")

    # Visualizations
    print("Rendering diagnostic plots ...")
    generate_visualizations(df, out_dir)
    print(f"All plots and data stored in: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
