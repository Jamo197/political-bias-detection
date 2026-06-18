"""
evaluate_metrics.py
-------------------
Loads batch-run JSONL logs from logs/batch_runs/, computes evaluation metrics
(MAE, RMSE, Pearson r, Spearman rho) per model × condition, and produces:
  - evaluation_metrics.csv          — full metrics table
  - results/mae_comparison.png      — grouped bar chart (MAE)
  - results/all_metrics.png         — 2x2 grid for all four metrics
  - results/rag_delta_heatmap.png   — RAG-vs-Baseline delta per metric

Directory layout expected:
  logs/batch_runs/<date>/<condition>/<model>_*.jsonl
  where <condition> is e.g. "Simple" (RAG) or "no_rag" (Baseline).
"""

import json
import os
import glob
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map folder names to human-readable condition labels
CONDITION_MAP = {
    "simple":  "RAG (Simple)",
    "no_rag":  "Baseline (No RAG)",
    "twostage": "RAG (TwoStage)",
    "hyde":    "RAG (HyDE)",
}

# Short model name extracted from the start of the filename before the first "_"
# e.g. "deepseek_simple_evaluation_logs.jsonl" -> "deepseek"

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _condition_label(folder_name: str) -> str:
    """Normalise a subfolder name to a human-readable condition label."""
    return CONDITION_MAP.get(folder_name.lower(), folder_name)


def _model_name_from_file(file_path: str) -> str:
    """Extract a short model name from the filename stem."""
    stem = Path(file_path).stem           # e.g. "deepseek_simple_evaluation_logs"
    return stem.split("_")[0].capitalize() # e.g. "Deepseek"


def load_logs_from_directory(base_dir: str) -> pd.DataFrame:
    """
    Walks *base_dir* recursively, finds all .jsonl files, and flattens
    each log entry into a row. The experimental condition is inferred from
    the immediate parent folder of each file (not from chunk presence).

    Expected structure:
      base_dir/<date>/<condition>/<model>_*.jsonl
    """
    records = []
    jsonl_files = glob.glob(os.path.join(base_dir, "**", "*.jsonl"), recursive=True)

    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found under '{base_dir}'")

    for file_path in sorted(jsonl_files):
        condition_folder = Path(file_path).parent.name
        condition = _condition_label(condition_folder)
        model_name = _model_name_from_file(file_path)

        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                log = json.loads(line)

                records.append({
                    "model":          model_name,
                    "condition":      condition,
                    "party":          log["input_metadata"]["party"],
                    "speaker":        log["input_metadata"]["speaker"],
                    "year":           log["input_metadata"]["year"],
                    "predicted_bias": log["output"]["bias"],
                    "actual_lrgen":   log["ches_ground_truth"]["lrgen"],
                    "actual_lrecon":  log["ches_ground_truth"]["lrecon"],
                    "actual_galtan":  log["ches_ground_truth"]["galtan"],
                    "llm_id":         log["parameters"]["llm"],
                    "retrieval_mode": log["parameters"]["retrieval_mode"],
                    "n_chunks":       len(log["inputs"].get("retrieved_chunks", [])),
                })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {len(jsonl_files)} files.")
    print(f"  Models:     {sorted(df['model'].unique())}")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    print(f"  Total rows: {len(df)}\n")
    return df


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------

def compute_evaluation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes MAE, RMSE, Pearson r, and Spearman rho grouped by Model × Condition.
    Uses actual_lrgen as the ground-truth target.
    """
    results = []

    for (model, condition), group in df.groupby(["model", "condition"]):
        y_true = group["actual_lrgen"].values
        y_pred = group["predicted_bias"].values

        mae   = mean_absolute_error(y_true, y_pred)
        rmse  = np.sqrt(mean_squared_error(y_true, y_pred))

        if len(group) >= 3:
            pearson_r,  p_pearson  = pearsonr(y_true, y_pred)
            spearman_r, p_spearman = spearmanr(y_true, y_pred)
        else:
            pearson_r = spearman_r = p_pearson = p_spearman = float("nan")

        results.append({
            "Model":        model,
            "Condition":    condition,
            "N_samples":    len(group),
            "MAE":          round(mae, 4),
            "RMSE":         round(rmse, 4),
            "Pearson_r":    round(pearson_r, 4),
            "Spearman_rho": round(spearman_r, 4),
        })

    results_df = pd.DataFrame(results).sort_values(["Model", "Condition"])
    return results_df


def compute_rag_delta(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the per-model delta (RAG minus Baseline) for each metric.
    Negative delta for MAE/RMSE means RAG is better; positive for Pearson/Spearman.
    """
    pivot_mae   = metrics_df.pivot(index="Model", columns="Condition", values="MAE")
    pivot_rmse  = metrics_df.pivot(index="Model", columns="Condition", values="RMSE")
    pivot_pear  = metrics_df.pivot(index="Model", columns="Condition", values="Pearson_r")
    pivot_spear = metrics_df.pivot(index="Model", columns="Condition", values="Spearman_rho")

    # Identify baseline and RAG column names (robust to naming variations).
    # "Baseline" is preferred over a plain "RAG" match to avoid collisions.
    def _find_col(pivot, *keywords):
        for kw in keywords:
            matches = [c for c in pivot.columns if kw.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    baseline_col = _find_col(pivot_mae, "baseline", "no_rag", "no rag")
    # RAG column = any column that is NOT the baseline column
    rag_col = next(
        (c for c in pivot_mae.columns if c != baseline_col),
        None,
    )

    if baseline_col is None or rag_col is None:
        print("Warning: Could not compute RAG delta — expected both a Baseline and a RAG condition.")
        return pd.DataFrame()

    delta = pd.DataFrame(index=pivot_mae.index)
    delta["ΔMAE (RAG-Base)"]          = pivot_mae[rag_col]   - pivot_mae[baseline_col]
    delta["ΔRMSE (RAG-Base)"]         = pivot_rmse[rag_col]  - pivot_rmse[baseline_col]
    delta["ΔPearson_r (RAG-Base)"]    = pivot_pear[rag_col]  - pivot_pear[baseline_col]
    delta["ΔSpearman_rho (RAG-Base)"] = pivot_spear[rag_col] - pivot_spear[baseline_col]
    return delta.reset_index()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_metric_comparison(metrics_df: pd.DataFrame, metric: str, output_path: Path):
    """Grouped horizontal bar chart for a single metric."""
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.barplot(
        data=metrics_df,
        x=metric,
        y="Model",
        hue="Condition",
        ax=ax,
        palette="mako",
    )

    lower_better = metric in ("MAE", "RMSE")
    direction = "Lower is Better" if lower_better else "Higher is Better"
    ax.set_title(f"{metric} by Model and Condition", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{metric} ({direction})", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.legend(title="Condition", bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, output_path)


def plot_all_metrics_grid(metrics_df: pd.DataFrame, output_path: Path):
    """2×2 grid showing MAE, RMSE, Pearson r, Spearman rho side by side."""
    sns.set_theme(style="whitegrid")
    metrics  = ["MAE", "RMSE", "Pearson_r", "Spearman_rho"]
    titles   = ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)",
                "Pearson r", "Spearman ρ"]
    xlabels  = ["MAE (lower=better)", "RMSE (lower=better)",
                "Pearson r (higher=better)", "Spearman ρ (higher=better)"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, metric, title, xlabel in zip(axes, metrics, titles, xlabels):
        sns.barplot(
            data=metrics_df,
            x=metric,
            y="Model",
            hue="Condition",
            ax=ax,
            palette="mako",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Model", fontsize=10)
        ax.legend(title="Condition", fontsize=8)

    fig.suptitle("Model Evaluation: All Metrics", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


def plot_rag_delta_heatmap(delta_df: pd.DataFrame, output_path: Path):
    """
    Heatmap of RAG-minus-Baseline deltas per model.
    Red = RAG is worse, Blue/Green = RAG is better (metric-aware colouring).
    """
    if delta_df.empty:
        print("Skipping RAG delta heatmap — no delta data available.")
        return

    sns.set_theme(style="white")
    delta_plot = delta_df.set_index("Model")

    # For MAE/RMSE negative = better (RAG improved); for correlations positive = better.
    # We flip MAE/RMSE so that in ALL columns positive = improvement.
    display = delta_plot.copy()
    display["ΔMAE (RAG-Base)"]  = -display["ΔMAE (RAG-Base)"]
    display["ΔRMSE (RAG-Base)"] = -display["ΔRMSE (RAG-Base)"]
    display.columns = [
        "ΔMAE\n(+good)", "ΔRMSE\n(+good)",
        "ΔPearson r\n(+good)", "ΔSpearman ρ\n(+good)"
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(
        display,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("RAG vs. Baseline: Improvement Δ per Model\n(positive = RAG is better)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Model")
    ax.set_xlabel("Metric")
    _save(fig, output_path)


def plot_scatter_predicted_vs_actual(df: pd.DataFrame, output_path: Path):
    """
    Scatter plot of predicted bias vs. actual lrgen, faceted by condition,
    coloured by model.
    """
    sns.set_theme(style="whitegrid")
    conditions = sorted(df["condition"].unique())
    n_cols = len(conditions)
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 6), sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        sub = df[df["condition"] == cond]
        sns.scatterplot(
            data=sub, x="actual_lrgen", y="predicted_bias",
            hue="model", alpha=0.6, ax=ax, palette="tab10",
        )
        # Identity line
        lims = [
            min(sub["actual_lrgen"].min(), sub["predicted_bias"].min()) - 0.5,
            max(sub["actual_lrgen"].max(), sub["predicted_bias"].max()) + 0.5,
        ]
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"Condition: {cond}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual lrgen (CHES)", fontsize=11)
        ax.set_ylabel("Predicted Bias" if ax == axes[0] else "", fontsize=11)
        ax.legend(title="Model", fontsize=8)

    fig.suptitle("Predicted Bias vs. Ground-Truth lrgen", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main(base_dir: str = "logs/batch_runs"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = load_logs_from_directory(base_dir)

    # 2. Compute metrics
    metrics_df = compute_evaluation_metrics(df)
    delta_df   = compute_rag_delta(metrics_df)

    # 3. Print tables
    print("=" * 60)
    print("EVALUATION METRICS (sorted by Model, Condition)")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print()

    if not delta_df.empty:
        print("=" * 60)
        print("RAG vs. BASELINE DELTA (positive = RAG is better)")
        print("  Note: ΔMAE / ΔRMSE are shown as RAG-Base, so negative = RAG is better")
        print("=" * 60)
        print(delta_df.to_string(index=False))
        print()

    # 4. Export CSVs
    metrics_csv = RESULTS_DIR / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved: {metrics_csv}")

    if not delta_df.empty:
        delta_csv = RESULTS_DIR / "rag_delta.csv"
        delta_df.to_csv(delta_csv, index=False)
        print(f"Saved: {delta_csv}")

    # 5. Visualise
    plot_metric_comparison(metrics_df, "MAE",
                           RESULTS_DIR / "mae_comparison.png")
    plot_metric_comparison(metrics_df, "RMSE",
                           RESULTS_DIR / "rmse_comparison.png")
    plot_all_metrics_grid(metrics_df,
                          RESULTS_DIR / "all_metrics.png")
    plot_rag_delta_heatmap(delta_df,
                           RESULTS_DIR / "rag_delta_heatmap.png")
    plot_scatter_predicted_vs_actual(df,
                                     RESULTS_DIR / "scatter_predicted_vs_actual.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate batch-run LLM bias prediction logs.")
    parser.add_argument(
        "--base-dir",
        default="logs/batch_runs",
        help="Root directory containing batch run subdirectories (default: logs/batch_runs)",
    )
    args = parser.parse_args()
    main(base_dir=args.base_dir)
