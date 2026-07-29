"""evaluate_metrics.py

Loads batch-run JSONL logs, computes evaluation metrics
(MAE, RMSE, Pearson r, Spearman rho) per LLM model × embedding model × retrieval
strategy, and produces CSV summary tables and plot figures in results/.

Log directory layout:
  logs/batch_runs/<run_id>/<embedding_model>/<condition>/<filename>.jsonl

Usage:
  python evaluate_metrics.py --base-dir logs/batch_runs --target label_ideology
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------


def _clean_model_name(llm_path: str) -> str:
    """Extracts a clean, short model name from LLM provider slug path.
    Example: 'mistralai/Ministral-3-8B-Instruct-2512' -> 'Ministral-3-8B-Instruct-2512'
    """
    if not llm_path or llm_path == "unknown":
        return "Unknown"
    return llm_path.split("/")[-1]


def load_logs_from_directory(base_dir: str) -> pd.DataFrame:
    """Walks base_dir recursively, reads JSONL entries, and flattens them into a DataFrame."""
    records = []
    jsonl_files = glob.glob(
        os.path.join(base_dir, "**", "party_label_*.jsonl"), recursive=True
    )

    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found under '{base_dir}'")

    for file_path in sorted(jsonl_files):
        folder_condition = Path(file_path).parent.name

        with open(file_path, "r", encoding="utf-8") as fh:
            for line_idx, line in enumerate(fh, 1):
                line = line.strip()
                if not line or line.startswith("//"):
                    continue

                try:
                    log = json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"Warning: Skipping malformed JSON in {file_path}:{line_idx} - {e}"
                    )
                    continue

                params = log.get("parameters", {})
                llm_raw = params.get("llm", "unknown")
                model_name = _clean_model_name(llm_raw)

                embedding_model = params.get("embedding_model", "none")
                retrieval_mode = params.get("retrieval_mode", folder_condition)
                hybrid = params.get("hybrid", False)
                is_rag = params.get("is_rag", True)

                # Format condition string cleanly
                if (
                    not is_rag
                    or embedding_model == "none"
                    or retrieval_mode == "no_rag"
                ):
                    condition = "no_rag"
                else:
                    emb_str = f"{embedding_model}-hybrid" if hybrid else embedding_model
                    condition = f"{emb_str}/{retrieval_mode}"

                gt = log.get("ground_truth", {})
                output = log.get("output", {})
                meta = log.get("input_metadata", {})
                inputs = log.get("inputs", {})

                records.append(
                    {
                        "run_id": log.get("run_id", "unknown"),
                        "model": model_name,
                        "llm_full": llm_raw,
                        "llm_region": params.get("llm_region", "unknown"),
                        "embedding_model": embedding_model,
                        "retrieval_mode": retrieval_mode,
                        "hybrid": hybrid,
                        "is_rag": is_rag,
                        "condition": condition,
                        "text_index": meta.get("text_index", ""),
                        "party": meta.get("party", ""),
                        "speaker": meta.get("speaker", ""),
                        "source": meta.get("source", ""),
                        "predicted_bias": output.get(
                            "bias"
                        ),  # May be None if LLM error
                        "label_ideology": gt.get("label_ideology"),
                        "label_economic": gt.get("label_economic"),
                        "label_galtan": gt.get("label_galtan"),
                        "k_chunks": params.get("k_chunks", 0),
                        "n_chunks_retrieved": len(inputs.get("retrieved_chunks", [])),
                    }
                )

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {len(jsonl_files)} JSONL files.")
    print(f"  Models found:     {sorted(df['model'].unique())}")
    print(f"  Conditions found: {sorted(df['condition'].unique())}")
    print(f"  Total records:    {len(df)}\n")
    return df


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


def compute_evaluation_metrics(
    df: pd.DataFrame, target_col: str = "label_ideology"
) -> pd.DataFrame:
    """Computes MAE, RMSE, Pearson r, and Spearman rho grouped by Model x Condition."""
    results = []

    # Drop missing predictions or missing ground truth labels
    valid_df = df.dropna(subset=["predicted_bias", target_col]).copy()
    valid_df["predicted_bias"] = pd.to_numeric(valid_df["predicted_bias"])
    valid_df[target_col] = pd.to_numeric(valid_df[target_col])

    dropped_count = len(df) - len(valid_df)
    if dropped_count > 0:
        print(
            f"Note: Omitted {dropped_count} rows with missing predictions/labels for target '{target_col}'."
        )

    for (model, condition), group in valid_df.groupby(["model", "condition"]):
        y_true = group[target_col].values
        y_pred = group["predicted_bias"].values

        if len(group) == 0:
            continue

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        if len(group) >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            pearson_r, _ = pearsonr(y_true, y_pred)
            spearman_r, _ = spearmanr(y_true, y_pred)
        else:
            pearson_r = spearman_r = float("nan")

        results.append(
            {
                "Target": target_col,
                "Model": model,
                "Condition": condition,
                "N_samples": len(group),
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
                "Pearson_r": (
                    round(float(pearson_r), 4) if not np.isnan(pearson_r) else np.nan
                ),
                "Spearman_rho": (
                    round(float(spearman_r), 4) if not np.isnan(spearman_r) else np.nan
                ),
            }
        )

    results_df = pd.DataFrame(results).sort_values(["Model", "Condition"])
    return results_df


def compute_rag_delta(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Computes per-model delta relative to 'no_rag' baseline.
    For MAE/RMSE: negative delta = RAG improvement.
    For Pearson/Spearman: positive delta = RAG improvement.
    """
    baseline = metrics_df[metrics_df["Condition"] == "no_rag"]
    rag = metrics_df[metrics_df["Condition"] != "no_rag"]

    if baseline.empty:
        print(
            "Warning: No 'no_rag' baseline found in metrics — skipping delta calculation."
        )
        return pd.DataFrame()

    baseline_map = baseline.set_index("Model")
    deltas = []

    for _, row in rag.iterrows():
        model = row["Model"]
        if model not in baseline_map.index:
            continue
        base = baseline_map.loc[model]
        if isinstance(base, pd.DataFrame):
            base = base.iloc[0]

        deltas.append(
            {
                "Target": row.get("Target", "label_ideology"),
                "Model": model,
                "Condition": row["Condition"],
                "ΔMAE (RAG-Base)": round(row["MAE"] - base["MAE"], 4),
                "ΔRMSE (RAG-Base)": round(row["RMSE"] - base["RMSE"], 4),
                "ΔPearson_r (RAG-Base)": (
                    round(row["Pearson_r"] - base["Pearson_r"], 4)
                    if not np.isnan(row["Pearson_r"])
                    else np.nan
                ),
                "ΔSpearman_rho (RAG-Base)": (
                    round(row["Spearman_rho"] - base["Spearman_rho"], 4)
                    if not np.isnan(row["Spearman_rho"])
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(deltas).sort_values(["Model", "Condition"])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.close(fig)


def plot_metric_comparison(
    metrics_df: pd.DataFrame, metric: str, target_name: str, output_path: Path
):
    if metrics_df.empty:
        return
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, max(6, len(metrics_df) * 0.35)))

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
    ax.set_title(
        f"{metric} by Condition and Model (Target: {target_name})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(f"{metric} ({direction})", fontsize=12)
    ax.set_ylabel("Condition", fontsize=12)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, output_path)


def plot_all_metrics_grid(
    metrics_df: pd.DataFrame, target_name: str, output_path: Path
):
    if metrics_df.empty:
        return
    sns.set_theme(style="whitegrid")
    metrics = ["MAE", "RMSE", "Pearson_r", "Spearman_rho"]
    titles = [
        "Mean Absolute Error (MAE)",
        "Root Mean Squared Error (RMSE)",
        "Pearson r",
        "Spearman ρ",
    ]
    xlabels = [
        "MAE (lower=better)",
        "RMSE (lower=better)",
        "Pearson r (higher=better)",
        "Spearman ρ (higher=better)",
    ]

    n_conditions = len(metrics_df["Condition"].unique())
    fig_height = max(10, n_conditions * 0.45)
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

    fig.suptitle(
        f"Model Evaluation Summary — Target: {target_name}",
        fontsize=16,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout()
    _save(fig, output_path)


def plot_rag_delta_heatmap(delta_df: pd.DataFrame, output_path: Path):
    if delta_df.empty:
        print("Skipping RAG delta heatmap — no delta data available.")
        return

    sns.set_theme(style="white")
    models = sorted(delta_df["Model"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(9 * n_models, max(6, len(delta_df) * 0.3)), sharey=True
    )
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = delta_df[delta_df["Model"] == model].set_index("Condition")
        if sub.empty:
            continue
        display = sub[
            [
                "ΔMAE (RAG-Base)",
                "ΔRMSE (RAG-Base)",
                "ΔPearson_r (RAG-Base)",
                "ΔSpearman_rho (RAG-Base)",
            ]
        ].copy()
        # Invert MAE & RMSE so positive = improvement across all 4 columns
        display["ΔMAE (RAG-Base)"] = -display["ΔMAE (RAG-Base)"]
        display["ΔRMSE (RAG-Base)"] = -display["ΔRMSE (RAG-Base)"]
        display.columns = [
            "ΔMAE\n(+good)",
            "ΔRMSE\n(+good)",
            "ΔPearson r\n(+good)",
            "ΔSpearman ρ\n(+good)",
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
        ax.set_title(
            f"Model: {model}\n(Positive values = RAG outperforms Baseline)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("Condition")
        ax.set_xlabel("Metric Delta")

    fig.suptitle(
        "RAG Improvement vs. Baseline (no_rag)", fontsize=14, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, output_path)


def plot_scatter_predicted_vs_actual(
    df: pd.DataFrame, target_col: str, output_path: Path
):
    valid_df = df.dropna(subset=["predicted_bias", target_col]).copy()
    if valid_df.empty:
        return

    valid_df["predicted_bias"] = pd.to_numeric(valid_df["predicted_bias"])
    valid_df[target_col] = pd.to_numeric(valid_df[target_col])

    sns.set_theme(style="whitegrid")
    conditions = sorted(valid_df["condition"].unique())
    n_cols = min(len(conditions), 4)
    n_rows = (len(conditions) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows), sharey=True
    )
    axes_flat = np.array(axes).flatten()

    for idx, cond in enumerate(conditions):
        ax = axes_flat[idx]
        sub = valid_df[valid_df["condition"] == cond]
        if sub.empty:
            ax.set_visible(False)
            continue

        sns.scatterplot(
            data=sub,
            x=target_col,
            y="predicted_bias",
            hue="model",
            alpha=0.7,
            ax=ax,
            palette="tab10",
        )

        min_val = min(sub[target_col].min(), sub["predicted_bias"].min()) - 0.5
        max_val = max(sub[target_col].max(), sub["predicted_bias"].max()) + 0.5
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "k--",
            linewidth=1,
            label="Identity (Perfect)",
        )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(f"Condition: {cond}", fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Ground Truth ({target_col})", fontsize=10)
        ax.set_ylabel("Predicted Bias Score", fontsize=10)
        ax.legend(title="Model", fontsize=8)

    # Hide extra subplot axes
    for idx in range(len(conditions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"Predicted Bias vs. Ground Truth ({target_col})",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main(base_dir: str = "logs/batch_runs", target_col: str = "label_ideology"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_logs_from_directory(base_dir)
    metrics_df = compute_evaluation_metrics(df, target_col=target_col)
    delta_df = compute_rag_delta(metrics_df)

    print("=" * 80)
    print(f"EVALUATION METRICS (Target: {target_col})")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print()

    if not delta_df.empty:
        print("=" * 80)
        print("RAG vs. BASELINE DELTA")
        print("  Note: ΔMAE / ΔRMSE shown as RAG - Baseline (negative = RAG is better)")
        print("        ΔPearson / ΔSpearman (positive = RAG is better)")
        print("=" * 80)
        print(delta_df.to_string(index=False))
        print()

    # Save CSV outputs
    metrics_csv = RESULTS_DIR / f"evaluation_metrics_{target_col}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved CSV: {metrics_csv}")

    if not delta_df.empty:
        delta_csv = RESULTS_DIR / f"rag_delta_{target_col}.csv"
        delta_df.to_csv(delta_csv, index=False)
        print(f"Saved CSV: {delta_csv}")

    # Generate and save plots
    plot_metric_comparison(
        metrics_df, "MAE", target_col, RESULTS_DIR / f"mae_comparison_{target_col}.png"
    )
    plot_metric_comparison(
        metrics_df,
        "RMSE",
        target_col,
        RESULTS_DIR / f"rmse_comparison_{target_col}.png",
    )
    plot_all_metrics_grid(
        metrics_df, target_col, RESULTS_DIR / f"all_metrics_{target_col}.png"
    )
    plot_rag_delta_heatmap(
        delta_df, RESULTS_DIR / f"rag_delta_heatmap_{target_col}.png"
    )
    plot_scatter_predicted_vs_actual(
        df, target_col, RESULTS_DIR / f"scatter_predicted_vs_actual_{target_col}.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate batch-run LLM bias prediction logs."
    )
    parser.add_argument(
        "--base-dir",
        default="logs/batch_runs",
        help="Root directory containing batch run JSONL files (default: logs/batch_runs)",
    )
    parser.add_argument(
        "--target",
        default="label_ideology",
        choices=["label_ideology", "label_economic", "label_galtan"],
        help="Ground-truth label column to evaluate against (default: label_ideology)",
    )
    args = parser.parse_args()
    main(base_dir=args.base_dir, target_col=args.target)
