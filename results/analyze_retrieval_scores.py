#!/usr/bin/env python3
"""analyze_retrieval_scores.py

Analyzes retrieval-score dynamics and score-to-error calibration
from batch-run JSONL logs.

Loads party_label_*.jsonl files from logs/batch_runs, computes per-query
retrieval-score statistics (range, variance, mean), and correlates them
with prediction error (MAE). Generates box plots and calibration scatter
plots saved to src/results/qualitative/.

Usage:
    python src/results/analyze_retrieval_scores.py
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("logs/batch_runs")
OUTPUT_DIR = Path("src/results/qualitative")
EMBEDDING_MODELS = {"e5", "jina", "qwen3"}

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_rag_logs(base_dir: Path) -> pd.DataFrame:
    """Load party_label_*.jsonl files and keep only RAG runs with target embeddings."""
    records: List[Dict] = []
    jsonl_files = sorted(base_dir.rglob("party_label_*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            f"No party_label_*.jsonl files found under '{base_dir}'"
        )

    for fp in jsonl_files:
        with open(fp, "r", encoding="utf-8") as fh:
            for line_idx, line in enumerate(fh, 1):
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

                if not is_rag or emb not in EMBEDDING_MODELS:
                    continue

                chunks = log.get("inputs", {}).get("retrieved_chunks", [])
                if not chunks:
                    continue

                scores = [float(c.get("score", 0)) for c in chunks if "score" in c]
                if not scores:
                    continue

                gt = log.get("ground_truth", {})
                pred = log.get("output", {}).get("bias")
                meta = log.get("input_metadata", {})

                records.append(
                    {
                        "run_id": log.get("run_id", "unknown"),
                        "model": _clean_model_name(params.get("llm", "unknown")),
                        "embedding_model": emb,
                        "retrieval_mode": retrieval_mode,
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
    print(f"Loaded {len(df)} RAG records from {len(jsonl_files)} JSONL files.")
    print(f"  Embedding models: {sorted(df['embedding_model'].unique())}")
    print(f"  Retrieval modes:  {sorted(df['retrieval_mode'].unique())}")
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
        arr = np.array(scores, dtype=float)
        top1 = float(arr[0])
        top5 = float(arr[-1]) if len(arr) >= 5 else float(arr[-1])
        mean = float(np.mean(arr))
        var = float(np.var(arr, ddof=1)) if len(arr) >= 2 else 0.0
        rng = top1 - top5
        return top1, top5, mean, var, rng

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

    # Per-condition min-max normalization so twostage (cross-encoder) scores
    # are on the same [0,1] scale as cosine similarities.
    df["score_mean_normalized"] = np.nan
    for (emb, mode), g in df.groupby(["embedding_model", "retrieval_mode"]):
        cond_min = g["score_mean"].min()
        cond_max = g["score_mean"].max()
        denom = cond_max - cond_min
        if denom > 0:
            df.loc[g.index, "score_mean_normalized"] = (
                g["score_mean"] - cond_min
            ) / denom
        else:
            df.loc[g.index, "score_mean_normalized"] = 0.5

    # Normalized range & variance (per-condition min-max of raw scores)
    df["score_range_normalized"] = np.nan
    df["score_variance_normalized"] = np.nan
    for (emb, mode), g in df.groupby(["embedding_model", "retrieval_mode"]):
        all_scores = np.concatenate(g["scores"].values)
        cond_min = float(all_scores.min())
        cond_max = float(all_scores.max())
        denom = cond_max - cond_min
        if denom <= 0:
            df.loc[g.index, "score_range_normalized"] = 0.0
            df.loc[g.index, "score_variance_normalized"] = 0.0
            continue
        # Normalize each query's scores, then re-compute range & variance
        for idx, row in g.iterrows():
            norm = [(s - cond_min) / denom for s in row["scores"]]
            df.at[idx, "score_range_normalized"] = max(norm) - min(norm)
            df.at[idx, "score_variance_normalized"] = (
                float(np.var(norm, ddof=1)) if len(norm) >= 2 else 0.0
            )

    # Absolute error
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
    """Pearson & Spearman between score_mean and abs_error globally and per group."""
    valid = df.dropna(subset=["score_mean", "abs_error"]).copy()
    rows = []

    def _corr(sub: pd.DataFrame, label: str, score_col: str = "score_mean") -> Dict:
        if len(sub) < 3:
            return {
                "group": label,
                "score_type": score_col,
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
            "score_type": score_col,
            "n": len(sub),
            "pearson_r": round(float(pr), 4),
            "pearson_p": round(float(pp), 4),
            "spearman_r": round(float(sr), 4),
            "spearman_p": round(float(sp), 4),
        }

    # Raw score correlations
    rows.append(_corr(valid, "Global", "score_mean"))
    for (emb, mode), g in valid.groupby(["embedding_model", "retrieval_mode"]):
        rows.append(_corr(g, f"{emb}/{mode}", "score_mean"))

    # Normalized score correlations
    rows.append(_corr(valid, "Global", "score_mean_normalized"))
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


def plot_boxplot_all_scores(df: pd.DataFrame, out_path: Path, normalized: bool = False):
    """Box plot of every individual chunk score across conditions."""
    rows = []
    for _, r in df.iterrows():
        cond = f"{r['embedding_model']}/{r['retrieval_mode']}"
        scores = r["scores"]
        if normalized:
            # Per-condition normalization
            all_scores = np.concatenate(df[df["condition"] == cond]["scores"].values)
            cond_min, cond_max = float(all_scores.min()), float(all_scores.max())
            denom = cond_max - cond_min
            if denom > 0:
                scores = [(s - cond_min) / denom for s in scores]
            else:
                scores = [0.5] * len(scores)
        for s in scores:
            rows.append({"condition": cond, "score": s})
    plot_df = pd.DataFrame(rows)

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
    title = "Distribution of Retrieval Scores (All Chunks)" + (
        " — Normalized" if normalized else ""
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(
        "Normalized Score [0,1]" if normalized else "Cosine Similarity Score",
        fontsize=12,
    )
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
            f"Skipping {'normalized ' if use_normalized else ''}score-vs-error plot — no valid data."
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

        ax.set_title(cond, fontsize=11, fontweight="bold")
        ax.set_xlabel(
            "Normalized Mean Score [0,1]" if use_normalized else "Mean Retrieval Score",
            fontsize=10,
        )
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
    fig.tight_layout()
    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(base_dir: Path = BASE_DIR, output_dir: Path = OUTPUT_DIR):
    print("=" * 80)
    print("Retrieval Score Dynamics & Score-to-Error Calibration Analysis")
    print("=" * 80)

    df = load_rag_logs(base_dir)
    df = compute_score_features(df)
    df["condition"] = df["embedding_model"] + "/" + df["retrieval_mode"]

    # Save per-query CSV
    per_query_cols = [
        "run_id",
        "model",
        "embedding_model",
        "retrieval_mode",
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
    print("\nScore-to-Error Correlations (score_mean vs abs_error):")
    print(corr_df.to_string(index=False))
    print(f"\nSaved correlations CSV: {corr_path}")

    # Plots — raw
    print("\nGenerating raw-score plots …")
    plot_boxplot_all_scores(df, output_dir / "boxplot_all_scores.png", normalized=False)
    plot_boxplot_metric(
        df,
        "score_range",
        "Score Range (Top1 − Top5) per Query",
        "Score Range",
        output_dir / "boxplot_score_range.png",
    )
    plot_boxplot_metric(
        df,
        "score_variance",
        "Score Variance per Query",
        "Variance",
        output_dir / "boxplot_score_variance.png",
    )
    plot_score_vs_error(
        df, output_dir / "scatter_score_vs_error.png", use_normalized=False
    )

    # Plots — normalized
    print("\nGenerating normalized-score plots …")
    plot_boxplot_all_scores(
        df, output_dir / "boxplot_all_scores_normalized.png", normalized=True
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
        "Normalized Variance",
        output_dir / "boxplot_score_variance_normalized.png",
    )
    plot_score_vs_error(
        df, output_dir / "scatter_score_vs_error_normalized.png", use_normalized=True
    )

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze retrieval score dynamics.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Root directory with batch run JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to write results.",
    )
    args = parser.parse_args()
    main(base_dir=args.base_dir, output_dir=args.output_dir)
