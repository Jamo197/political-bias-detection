"""evaluate_metrics.py

Loads batch-run JSONL logs from logs/batch_runs/, computes evaluation metrics
(MAE, RMSE, Pearson r, Spearman rho) per LLM model × embedding model × retrieval
strategy, and produces:
  - evaluation_metrics.csv          — full metrics table
  - rag_delta.csv                   — RAG-vs-Baseline delta per condition
  - results/mae_comparison.png      — grouped bar chart (MAE)
  - results/all_metrics.png         — 2x2 grid for all four metrics
  - results/rag_delta_heatmap.png   — RAG-vs-Baseline delta per condition

Log directory layout (new):
  logs/batch_runs/<date>_<runid>/<embedding_model>/<retrieval_mode>/<filename>.jsonl

Log entry parameters (new):
  embedding_model, retrieval_mode, hybrid, is_rag, k_chunks
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

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _model_name_from_file(file_path: str) -> str:
    stem = Path(file_path).stem
    return stem.split("_")[0].capitalize()


def load_logs_from_directory(base_dir: str) -> pd.DataFrame:
    """Walks *base_dir* recursively, finds all .jsonl files, and flattens
    each log entry into a row. Condition is read from log parameters.
    """
    records = []
    jsonl_files = glob.glob(os.path.join(base_dir, "**", "*.jsonl"), recursive=True)

    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found under '{base_dir}'")

    for file_path in sorted(jsonl_files):
        condition_folder = Path(file_path).parent.name
        model_name = _model_name_from_file(file_path)

        with open(file_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                log = json.loads(line)

                params = log.get("parameters", {})
                embedding_model = params.get("embedding_model", "unknown")
                retrieval_mode = params.get("retrieval_mode", condition_folder)
                hybrid = params.get("hybrid", False)
                is_rag = params.get("is_rag", True)

                if not is_rag or embedding_model == "none":
                    condition = "no_rag"
                else:
                    condition = f"{embedding_model}/{retrieval_mode}"

                gt = log.get("ground_truth", log.get("ches_ground_truth", {}))
                label_ideology = gt.get("label_ideology", gt.get("lrgen"))
                label_economic = gt.get("label_economic", gt.get("lrecon"))
                label_galtan = gt.get("label_galtan", gt.get("galtan"))

                meta = log.get("input_metadata", {})
                records.append({
                    "model": model_name,
                    "embedding_model": embedding_model,
                    "retrieval_mode": retrieval_mode,
                    "hybrid": hybrid,
                    "is_rag": is_rag,
                    "condition": condition,
                    "party": meta.get("party", ""),
                    "speaker": meta.get("speaker", ""),
                    "source": meta.get("source", meta.get("year", "")),
                    "predicted_bias": log["output"]["bias"],
                    "actual_ideology": label_ideology,
                    "actual_economic": label_economic,
                    "actual_galtan": label_galtan,
                    "llm_id": params.get("llm", ""),
                    "k_chunks": params.get("k_chunks", 0),
                    "n_chunks": len(log["inputs"].get("retrieved_chunks", [])),
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
    """Computes MAE, RMSE, Pearson r, and Spearman rho grouped by
    LLM Model × Condition (embedding_model/retrieval_mode)."""
    results = []

    for (model, condition), group in df.groupby(["model", "condition"]):
        y_true = group["actual_ideology"].values
        y_pred = group["predicted_bias"].values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        if len(group) >= 3:
            pearson_r, p_pearson = pearsonr(y_true, y_pred)
            spearman_r, p_spearman = spearmanr(y_true, y_pred)
        else:
            pearson_r = spearman_r = p_pearson = p_spearman = float("nan")

        results.append({
            "Model": model,
            "Condition": condition,
            "N_samples": len(group),
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "Pearson_r": round(pearson_r, 4),
            "Spearman_rho": round(spearman_r, 4),
        })

    results_df = pd.DataFrame(results).sort_values(["Model", "Condition"])
    return results_df


def compute_rag_delta(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Computes per-model delta (RAG condition minus no_rag baseline)
    for each metric. Negative delta for MAE/RMSE means RAG is better;
    positive for Pearson/Spearman means RAG is better.
    """
    baseline = metrics_df[metrics_df["Condition"] == "no_rag"]
    rag = metrics_df[metrics_df["Condition"] != "no_rag"]

    if baseline.empty:
        print("Warning: No no_rag baseline found — cannot compute RAG delta.")
        return pd.DataFrame()

    baseline = baseline.set_index("Model")
    deltas = []

    for _, row in rag.iterrows():
        model = row["Model"]
        if model not in baseline.index:
            continue
        base = baseline.loc[model]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]

        deltas.append({
            "Model": model,
            "Condition": row["Condition"],
            "ΔMAE (RAG-Base)": row["MAE"] - base["MAE"],
            "ΔRMSE (RAG-Base)": row["RMSE"] - base["RMSE"],
            "ΔPearson_r (RAG-Base)": row["Pearson_r"] - base["Pearson_r"],
            "ΔSpearman_rho (RAG-Base)": row["Spearman_rho"] - base["Spearman_rho"],
        })

    return pd.DataFrame(deltas).sort_values(["Model", "Condition"])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_metric_comparison(metrics_df: pd.DataFrame, metric: str, output_path: Path):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, max(6, len(metrics_df) * 0.3)))

    sns.barplot(
        data=metrics_df,
        x=metric,
        y="Condition",
        hue="Model",
        ax=ax,
        palette="mako",
    )

    lower_better = metric in ("MAE", "RMSE")
    direction = "Lower is Better" if lower_better else "Higher is Better"
    ax.set_title(f"{metric} by Condition and Model", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{metric} ({direction})", fontsize=12)
    ax.set_ylabel("Condition", fontsize=12)
    ax.legend(title="LLM Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, output_path)


def plot_all_metrics_grid(metrics_df: pd.DataFrame, output_path: Path):
    sns.set_theme(style="whitegrid")
    metrics = ["MAE", "RMSE", "Pearson_r", "Spearman_rho"]
    titles = ["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)",
              "Pearson r", "Spearman ρ"]
    xlabels = ["MAE (lower=better)", "RMSE (lower=better)",
               "Pearson r (higher=better)", "Spearman ρ (higher=better)"]

    n_conditions = len(metrics_df["Condition"].unique())
    fig_height = max(10, n_conditions * 0.4)
    fig, axes = plt.subplots(2, 2, figsize=(18, fig_height))
    axes = axes.flatten()

    for ax, metric, title, xlabel in zip(axes, metrics, titles, xlabels):
        sns.barplot(
            data=metrics_df,
            x=metric,
            y="Condition",
            hue="Model",
            ax=ax,
            palette="mako",
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("Condition", fontsize=10)
        ax.legend(title="Model", fontsize=8)

    fig.suptitle("Model Evaluation: All Metrics", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


def plot_rag_delta_heatmap(delta_df: pd.DataFrame, output_path: Path):
    if delta_df.empty:
        print("Skipping RAG delta heatmap — no delta data available.")
        return

    sns.set_theme(style="white")

    models = sorted(delta_df["Model"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(10 * n_models, max(6, len(delta_df) * 0.2)), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = delta_df[delta_df["Model"] == model].set_index("Condition")
        display = sub.copy()
        display["ΔMAE (RAG-Base)"] = -display["ΔMAE (RAG-Base)"]
        display["ΔRMSE (RAG-Base)"] = -display["ΔRMSE (RAG-Base)"]
        display.columns = [
            "ΔMAE\n(+good)", "ΔRMSE\n(+good)",
            "ΔPearson r\n(+good)", "ΔSpearman ρ\n(+good)",
        ]

        sns.heatmap(
            display,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title(f"{model}\n(positive = RAG is better)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Condition")
        ax.set_xlabel("Metric")

    fig.suptitle("RAG vs. Baseline: Improvement Δ per Condition", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


def plot_scatter_predicted_vs_actual(df: pd.DataFrame, output_path: Path):
    sns.set_theme(style="whitegrid")
    conditions = sorted(df["condition"].unique())
    n_cols = min(len(conditions), 4)
    n_rows = (len(conditions) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 6 * n_rows), sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, (ax, cond) in enumerate(zip(axes.flatten(), conditions)):
        sub = df[df["condition"] == cond]
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.scatterplot(
            data=sub, x="actual_ideology", y="predicted_bias",
            hue="model", alpha=0.6, ax=ax, palette="tab10",
        )
        lims = [
            min(sub["actual_ideology"].min(), sub["predicted_bias"].min()) - 0.5,
            max(sub["actual_ideology"].max(), sub["predicted_bias"].max()) + 0.5,
        ]
        ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(f"Condition: {cond}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Actual ideology (ground truth)", fontsize=11)
        if idx % n_cols == 0:
            ax.set_ylabel("Predicted Bias", fontsize=11)
        ax.legend(title="Model", fontsize=8)

    for idx in range(len(conditions), n_rows * n_cols):
        axes.flatten()[idx].set_visible(False)

    fig.suptitle("Predicted Bias vs. Ground-Truth Ideology", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main(base_dir: str = "logs/batch_runs"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_logs_from_directory(base_dir)
    metrics_df = compute_evaluation_metrics(df)
    delta_df = compute_rag_delta(metrics_df)

    print("=" * 80)
    print("EVALUATION METRICS (sorted by Model, Condition)")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print()

    if not delta_df.empty:
        print("=" * 80)
        print("RAG vs. BASELINE DELTA")
        print("  Note: ΔMAE / ΔRMSE shown as RAG-Base (negative = RAG is better)")
        print("        ΔPearson / ΔSpearman (positive = RAG is better)")
        print("=" * 80)
        print(delta_df.to_string(index=False))
        print()

    metrics_csv = RESULTS_DIR / "evaluation_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved: {metrics_csv}")

    if not delta_df.empty:
        delta_csv = RESULTS_DIR / "rag_delta.csv"
        delta_df.to_csv(delta_csv, index=False)
        print(f"Saved: {delta_csv}")

    plot_metric_comparison(metrics_df, "MAE", RESULTS_DIR / "mae_comparison.png")
    plot_metric_comparison(metrics_df, "RMSE", RESULTS_DIR / "rmse_comparison.png")
    plot_all_metrics_grid(metrics_df, RESULTS_DIR / "all_metrics.png")
    plot_rag_delta_heatmap(delta_df, RESULTS_DIR / "rag_delta_heatmap.png")
    plot_scatter_predicted_vs_actual(df, RESULTS_DIR / "scatter_predicted_vs_actual.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate batch-run LLM bias prediction logs.")
    parser.add_argument(
        "--base-dir",
        default="logs/batch_runs",
        help="Root directory containing batch run subdirectories (default: logs/batch_runs)",
    )
    args = parser.parse_args()
    main(base_dir=args.base_dir)
