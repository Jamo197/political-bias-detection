#!/usr/bin/env python3
"""analyze_retrieval_scores.py

Analyzes retrieval-score dynamics and score-to-error calibration
from batch-run JSONL logs.

Loads party_label_*.jsonl files from logs/batch_runs, computes per-query
retrieval-score statistics (range, variance, mean), and correlates them
with prediction error (absolute bias error).

Score types
-----------
Different retrieval strategies produce scores on incompatible scales:

  * cosine        — bi-encoder cosine similarity (~[0.4, 1.0]);
                    modes: simple, hyde, hyde_hybrid
  * rrf_fusion    — Qdrant dense+sparse RRF fusion (bge-m3 only);
                    modes: simple_hybrid
  * cross_encoder — unbounded reranker logits (~[-9, 10]);
                    modes: twostage, twostage_hybrid

Raw scores are therefore only compared *within* a score type. For
cross-condition comparison, scores are min-max normalized per
(embedding_model, retrieval_mode) condition.

Outputs
-------
Every execution creates a new timestamped folder:

  results/qualitative/retrieval_scores/<YYYYMMDD_HHMMSS>/

containing CSV tables (per-query, summary, correlations) and box/scatter
plots. Use --run-id to override the folder name.

Usage:
    python src/results/analyze_retrieval_scores.py
    python src/results/analyze_retrieval_scores.py --batch-run 2026-08-04_eval_matrix_20260804_191413
    python src/results/analyze_retrieval_scores.py --embedding-models bge,jina
    python src/results/analyze_retrieval_scores.py --retrieval-modes twostage,twostage_hybrid
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("logs/batch_runs")
OUTPUT_ROOT = Path("results/qualitative/retrieval_scores")

SCORE_TYPE_LABELS = {
    "cosine": "Cosine Similarity",
    "rrf_fusion": "RRF Fusion Score",
    "cross_encoder": "Cross-Encoder Logit",
}


def classify_score_type(retrieval_mode: str) -> str:
    """Map a retrieval mode to the type of score it produces."""
    if retrieval_mode.startswith("twostage"):
        return "cross_encoder"
    if retrieval_mode == "simple_hybrid":
        return "rrf_fusion"
    return "cosine"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_rag_logs(
    base_dir: Path,
    batch_run: Optional[str] = None,
    embedding_models: Optional[List[str]] = None,
    retrieval_modes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load party_label_*.jsonl files and keep RAG runs with scored chunks."""
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
                is_rag = params.get("is_rag", True)
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

                scores = [float(c["score"]) for c in chunks if "score" in c]
                if not scores:
                    continue

                gt = log.get("ground_truth", {})
                pred = log.get("output", {}).get("bias")
                meta = log.get("input_metadata", {})

                records.append(
                    {
                        "run_id": log.get("run_id", "unknown"),
                        "batch_run": run_folder,
                        "model": _clean_model_name(params.get("llm", "unknown")),
                        "embedding_model": emb,
                        "retrieval_mode": retrieval_mode,
                        "score_type": classify_score_type(retrieval_mode),
                        "text_index": meta.get("text_index", ""),
                        "party": meta.get("party", ""),
                        "scores": scores,
                        "n_chunks": len(scores),
                        "predicted_bias": pred,
                        "label_ideology": gt.get("label_ideology"),
                        "label_economic": gt.get("label_economic"),
                        "label_galtan": gt.get("label_galtan"),
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
    print(f"  Score types:      {sorted(df['score_type'].unique())}")
    return df


def _clean_model_name(llm_path: str) -> str:
    if not llm_path or llm_path == "unknown":
        return "Unknown"
    return llm_path.split("/")[-1]


# ---------------------------------------------------------------------------
# Feature Engineering (raw + normalized)
# ---------------------------------------------------------------------------


def compute_score_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-query score statistics (raw + per-condition normalized)."""
    df = df.copy()

    def _features(scores: List[float]) -> Tuple[float, float, float, float, float]:
        arr = np.asarray(scores, dtype=float)
        top1 = float(arr[0])
        top5 = float(arr[-1])  # last retrieved chunk (rank k)
        mean = float(np.mean(arr))
        var = float(np.var(arr, ddof=1)) if len(arr) >= 2 else 0.0
        return top1, top5, mean, var, top1 - top5

    feats = df["scores"].apply(
        lambda s: pd.Series(
            _features(s),
            index=[
                "score_top1",
                "score_top5",
                "score_mean",
                "score_variance",
                "score_range",
            ],
        )
    )
    df = pd.concat([df.reset_index(drop=True), feats.reset_index(drop=True)], axis=1)

    # Per-condition min-max normalization puts cross-encoder logits, RRF
    # fusion scores, and cosine similarities on a common [0,1] scale.
    df["score_mean_normalized"] = np.nan
    df["score_range_normalized"] = np.nan
    df["score_variance_normalized"] = np.nan
    for _, g in df.groupby(["embedding_model", "retrieval_mode"]):
        # Mean-score normalization (min/max over per-query means)
        smin, smax = g["score_mean"].min(), g["score_mean"].max()
        sdenom = smax - smin
        df.loc[g.index, "score_mean_normalized"] = (
            (g["score_mean"] - smin) / sdenom if sdenom > 0 else 0.5
        )

        # Range/variance normalization (min/max over all raw chunk scores)
        all_scores = np.concatenate(g["scores"].values)
        cmin, cmax = float(all_scores.min()), float(all_scores.max())
        denom = cmax - cmin
        for idx, scores in g["scores"].items():
            if denom <= 0:
                df.at[idx, "score_range_normalized"] = 0.0
                df.at[idx, "score_variance_normalized"] = 0.0
            else:
                norm = [(s - cmin) / denom for s in scores]
                df.at[idx, "score_range_normalized"] = max(norm) - min(norm)
                df.at[idx, "score_variance_normalized"] = (
                    float(np.var(norm, ddof=1)) if len(norm) >= 2 else 0.0
                )

    # Absolute prediction error
    valid = df.dropna(subset=["predicted_bias", "label_ideology"]).copy()
    valid["predicted_bias"] = pd.to_numeric(valid["predicted_bias"], errors="coerce")
    valid["label_ideology"] = pd.to_numeric(valid["label_ideology"], errors="coerce")
    valid["abs_error"] = (valid["predicted_bias"] - valid["label_ideology"]).abs()
    df = df.merge(
        valid[
            ["run_id", "text_index", "embedding_model", "retrieval_mode", "abs_error"]
        ],
        on=["run_id", "text_index", "embedding_model", "retrieval_mode"],
        how="left",
    )
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics per (embedding_model, retrieval_mode)."""
    groups = []
    for (emb, mode), g in df.groupby(["embedding_model", "retrieval_mode"]):
        groups.append(
            {
                "embedding_model": emb,
                "retrieval_mode": mode,
                "score_type": g["score_type"].iloc[0],
                "n_queries": len(g),
                "mean_top1": round(g["score_top1"].mean(), 4),
                "mean_top5": round(g["score_top5"].mean(), 4),
                "mean_range": round(g["score_range"].mean(), 4),
                "median_range": round(g["score_range"].median(), 4),
                "std_range": round(g["score_range"].std(), 4),
                "mean_variance": round(g["score_variance"].mean(), 4),
                "median_variance": round(g["score_variance"].median(), 4),
                "std_variance": round(g["score_variance"].std(), 4),
                "mean_abs_error": round(g["abs_error"].mean(), 4),
                "median_abs_error": round(g["abs_error"].median(), 4),
                "std_abs_error": round(g["abs_error"].std(), 4),
                # Normalized stats
                "mean_range_norm": round(g["score_range_normalized"].mean(), 4),
                "median_range_norm": round(g["score_range_normalized"].median(), 4),
                "std_range_norm": round(g["score_range_normalized"].std(), 4),
                "mean_variance_norm": round(g["score_variance_normalized"].mean(), 4),
                "median_variance_norm": round(
                    g["score_variance_normalized"].median(), 4
                ),
                "std_variance_norm": round(g["score_variance_normalized"].std(), 4),
            }
        )
    return pd.DataFrame(groups).sort_values(["embedding_model", "retrieval_mode"])


# ---------------------------------------------------------------------------
# Correlation (Score-to-Error Calibration)
# ---------------------------------------------------------------------------


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson & Spearman between mean score and absolute error.

    Raw scores are only correlated per condition, because raw scales are
    not comparable across score types (cosine vs. RRF vs. cross-encoder
    logits). Normalized scores additionally allow global and
    per-score-type correlations.
    """
    valid = df.dropna(subset=["score_mean", "abs_error"]).copy()
    rows = []

    def _corr(sub: pd.DataFrame, label: str, score_col: str) -> Dict:
        if len(sub) < 3:
            return {
                "group": label,
                "score_column": score_col,
                "n": len(sub),
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
            }
        pr, pp = pearsonr(sub[score_col], sub["abs_error"])
        sr, sp = spearmanr(sub[score_col], sub["abs_error"])
        return {
            "group": label,
            "score_column": score_col,
            "n": len(sub),
            "pearson_r": round(float(pr), 4),
            "pearson_p": round(float(pp), 4),
            "spearman_r": round(float(sr), 4),
            "spearman_p": round(float(sp), 4),
        }

    # Raw scores: per condition only
    for (emb, mode), g in valid.groupby(["embedding_model", "retrieval_mode"]):
        rows.append(_corr(g, f"{emb}/{mode}", "score_mean"))

    # Normalized scores: global, per score type, per condition
    rows.append(_corr(valid, "Global", "score_mean_normalized"))
    for st, g in valid.groupby("score_type"):
        rows.append(_corr(g, f"score_type={st}", "score_mean_normalized"))
    for (emb, mode), g in valid.groupby(["embedding_model", "retrieval_mode"]):
        rows.append(_corr(g, f"{emb}/{mode}", "score_mean_normalized"))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.close(fig)


def flatten_scores(df: pd.DataFrame, normalized: bool = False) -> pd.DataFrame:
    """Long-format DataFrame with one row per individual chunk score."""
    rows = []
    for (emb, mode), g in df.groupby(
        ["embedding_model", "retrieval_mode"], sort=False
    ):
        cond = f"{emb}/{mode}"
        score_type = g["score_type"].iloc[0]
        if normalized:
            all_scores = np.concatenate(g["scores"].values)
            cmin, cmax = float(all_scores.min()), float(all_scores.max())
            denom = cmax - cmin
        for scores in g["scores"]:
            if normalized:
                vals = (
                    [(s - cmin) / denom for s in scores]
                    if denom > 0
                    else [0.5] * len(scores)
                )
            else:
                vals = scores
            for v in vals:
                rows.append(
                    {"condition": cond, "score_type": score_type, "score": v}
                )
    return pd.DataFrame(rows)


def plot_boxplot_scores(
    plot_df: pd.DataFrame, title: str, xlabel: str, out_path: Path
):
    """Box plot of individual chunk scores across conditions."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(
        figsize=(12, max(6, len(plot_df["condition"].unique()) * 0.5))
    )
    sns.boxplot(
        data=plot_df,
        x="score",
        y="condition",
        ax=ax,
        palette="mako",
        hue="condition",
        legend=False,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Condition (Embedding / Strategy)", fontsize=12)
    _save(fig, out_path)


def plot_boxplot_metric(
    df: pd.DataFrame, metric: str, title: str, xlabel: str, out_path: Path
):
    """Generic box plot of a per-query metric across conditions."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(6, len(df["condition"].unique()) * 0.5)))
    sns.boxplot(
        data=df,
        x=metric,
        y="condition",
        ax=ax,
        palette="mako",
        hue="condition",
        legend=False,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Condition (Embedding / Strategy)", fontsize=12)
    _save(fig, out_path)


def plot_score_vs_error(df: pd.DataFrame, out_path: Path, use_normalized: bool = False):
    """Scatter plot: mean retrieval score vs absolute error, with regression & correlations."""
    score_col = "score_mean_normalized" if use_normalized else "score_mean"
    valid = df.dropna(subset=[score_col, "abs_error"]).copy()
    if valid.empty:
        print(
            f"Skipping {'normalized ' if use_normalized else ''}"
            "score-vs-error plot — no valid data."
        )
        return

    conditions = sorted(valid["condition"].unique())
    n_cols = min(len(conditions), 3)
    n_rows = (len(conditions) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False
    )

    for idx, cond in enumerate(conditions):
        ax = axes[idx // n_cols][idx % n_cols]
        sub = valid[valid["condition"] == cond]
        if len(sub) < 3:
            ax.set_visible(False)
            continue

        sns.scatterplot(
            data=sub, x=score_col, y="abs_error", alpha=0.5, ax=ax, edgecolor=None
        )
        sns.regplot(
            data=sub,
            x=score_col,
            y="abs_error",
            scatter=False,
            ax=ax,
            color="red",
            line_kws={"linewidth": 1.5},
        )

        pr, pp = pearsonr(sub[score_col], sub["abs_error"])
        sr, sp = spearmanr(sub[score_col], sub["abs_error"])

        if use_normalized:
            xlabel = "Normalized Mean Score [0,1]"
        else:
            xlabel = f"Mean {SCORE_TYPE_LABELS[sub['score_type'].iloc[0]]}"

        ax.set_title(cond, fontsize=11, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Absolute Error |pred − true|", fontsize=10)
        ax.text(
            0.05,
            0.95,
            f"Pearson r={pr:.3f} (p={pp:.3f})\nSpearman ρ={sr:.3f} (p={sp:.3f})",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Hide unused axes
    for idx in range(len(conditions), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    title = (
        "Score-to-Error Calibration: "
        + ("Normalized " if use_normalized else "")
        + "Mean Retrieval Score vs. Absolute Error"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, out_path)


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
    print("Retrieval Score Dynamics & Score-to-Error Calibration Analysis")
    print("=" * 80)
    print(f"Output directory: {output_dir.resolve()}\n")

    df = load_rag_logs(base_dir, batch_run, embedding_models, retrieval_modes)
    if df.empty:
        return
    df = compute_score_features(df)
    df["condition"] = df["embedding_model"] + "/" + df["retrieval_mode"]
    score_types = sorted(df["score_type"].unique())

    # Save per-query CSV
    per_query_cols = [
        "run_id",
        "batch_run",
        "model",
        "embedding_model",
        "retrieval_mode",
        "score_type",
        "condition",
        "text_index",
        "party",
        "n_chunks",
        "score_top1",
        "score_top5",
        "score_mean",
        "score_variance",
        "score_range",
        "score_mean_normalized",
        "score_range_normalized",
        "score_variance_normalized",
        "predicted_bias",
        "label_ideology",
        "abs_error",
    ]
    per_query_path = output_dir / "retrieval_score_per_query.csv"
    df[per_query_cols].to_csv(per_query_path, index=False)
    print(f"\nSaved per-query CSV: {per_query_path}")

    # Aggregate summary
    summary = aggregate_by_condition(df)
    summary_path = output_dir / "retrieval_score_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\nAggregate Summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary CSV: {summary_path}")

    # Correlations
    corr_df = compute_correlations(df)
    corr_path = output_dir / "retrieval_score_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print("\nScore-to-Error Correlations (mean score vs abs_error):")
    print(corr_df.to_string(index=False))
    print(f"\nSaved correlations CSV: {corr_path}")

    # Plots — raw, one figure per score type (scales are not comparable)
    print("\nGenerating raw-score plots …")
    flat_raw = flatten_scores(df, normalized=False)
    for st in score_types:
        label = SCORE_TYPE_LABELS[st]
        plot_boxplot_scores(
            flat_raw[flat_raw["score_type"] == st],
            f"Distribution of Retrieval Scores (All Chunks) — {label}",
            label,
            output_dir / f"boxplot_all_scores_{st}.png",
        )
        sub = df[df["score_type"] == st]
        plot_boxplot_metric(
            sub,
            "score_range",
            f"Score Range (Top1 − Last) per Query — {label}",
            "Score Range",
            output_dir / f"boxplot_score_range_{st}.png",
        )
        plot_boxplot_metric(
            sub,
            "score_variance",
            f"Score Variance per Query — {label}",
            "Variance",
            output_dir / f"boxplot_score_variance_{st}.png",
        )
    plot_score_vs_error(
        df, output_dir / "scatter_score_vs_error.png", use_normalized=False
    )

    # Plots — normalized (cross-condition comparable)
    print("\nGenerating normalized-score plots …")
    plot_boxplot_scores(
        flatten_scores(df, normalized=True),
        "Distribution of Retrieval Scores (All Chunks) — Normalized",
        "Normalized Score [0,1]",
        output_dir / "boxplot_all_scores_normalized.png",
    )
    plot_boxplot_metric(
        df,
        "score_range_normalized",
        "Normalized Score Range per Query",
        "Normalized Range [0,1]",
        output_dir / "boxplot_score_range_normalized.png",
    )
    plot_boxplot_metric(
        df,
        "score_variance_normalized",
        "Normalized Score Variance per Query",
        "Normalized Variance [0,1]",
        output_dir / "boxplot_score_variance_normalized.png",
    )
    plot_score_vs_error(
        df, output_dir / "scatter_score_vs_error_normalized.png", use_normalized=True
    )

    print(f"\nAll outputs written to: {output_dir.resolve()}")
    print("Done.")


def _comma_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze retrieval score dynamics.",
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
