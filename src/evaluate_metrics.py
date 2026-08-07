"""evaluate_metrics.py

Loads batch-run JSONL logs, computes evaluation metrics
(MAE, RMSE, Pearson r, Spearman rho) per LLM model x embedding model x retrieval
strategy, and produces CSV summary tables and plot figures.

Non-RAG records (is_rag=False or retrieval_mode="no_rag") are included as a
"no_rag" baseline condition and flagged at the top of every output for easy
comparison. A RAG-vs-baseline delta table and heatmap quantify the improvement
(or regression) from adding retrieval.

Outputs
-------
Every execution creates a new timestamped folder:

  results/evaluation/<YYYYMMDD_HHMMSS>/

Usage:
    python src/evaluate_metrics.py
    python src/evaluate_metrics.py --target label_economic
    python src/evaluate_metrics.py --batch-run 2026-08-04_eval_matrix_20260804_191413
    python src/evaluate_metrics.py --embedding-models bge,jina
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("logs/batch_runs")
OUTPUT_ROOT = Path("results/evaluation")

# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------


def _clean_model_name(llm_path: str) -> str:
    if not llm_path or llm_path == "unknown":
        return "Unknown"
    return llm_path.split("/")[-1]


def load_logs_from_directory(
    base_dir: Path,
    batch_run: Optional[str] = None,
    embedding_models: Optional[List[str]] = None,
    retrieval_modes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Walk base_dir recursively, read JSONL entries, flatten into a DataFrame."""
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
                retrieval_mode = params.get("retrieval_mode", "unknown")
                is_rag = params.get("is_rag", True)

                # Determine condition
                if (
                    not is_rag
                    or emb in (None, "none", "")
                    or retrieval_mode == "no_rag"
                ):
                    condition = "no_rag"
                    emb = "none"
                    retrieval_mode = "no_rag"
                else:
                    condition = f"{emb}/{retrieval_mode}"

                if embedding_models and emb not in embedding_models:
                    continue
                if retrieval_modes and retrieval_mode not in retrieval_modes:
                    continue

                meta = log.get("input_metadata", {})
                gt = log.get("ground_truth", {})
                output = log.get("output", {})
                inputs = log.get("inputs", {})

                records.append(
                    {
                        "run_id": log.get("run_id", "unknown"),
                        "batch_run": run_folder,
                        "model": _clean_model_name(params.get("llm", "unknown")),
                        "llm_full": params.get("llm", "unknown"),
                        "llm_region": params.get("llm_region", "unknown"),
                        "embedding_model": emb,
                        "retrieval_mode": retrieval_mode,
                        "is_rag": is_rag,
                        "condition": condition,
                        "text_index": meta.get("text_index", ""),
                        "party": meta.get("party", ""),
                        "speaker": meta.get("speaker", ""),
                        "source": meta.get("source", ""),
                        "predicted_bias": output.get("bias"),
                        "label_ideology": gt.get("label_ideology"),
                        "label_economic": gt.get("label_economic"),
                        "label_galtan": gt.get("label_galtan"),
                        "k_chunks": params.get("k_chunks", 0),
                        "n_chunks_retrieved": len(inputs.get("retrieved_chunks", [])),
                    }
                )

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from {len(jsonl_files)} JSONL files.")
    print(f"  Batch runs:       {sorted(df['batch_run'].unique())}")
    print(f"  Embedding models: {sorted(df['embedding_model'].unique())}")
    print(f"  Retrieval modes:  {sorted(df['retrieval_mode'].unique())}")
    print(f"  Conditions:       {sorted(df['condition'].unique())}")
    print(f"  Models:           {sorted(df['model'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Condition ordering (no_rag first for easy comparison)
# ---------------------------------------------------------------------------


def _condition_order(conditions: List[str]) -> List[str]:
    """Return conditions sorted with 'no_rag' always first."""
    others = sorted(c for c in conditions if c != "no_rag")
    return ["no_rag"] + others


def _apply_condition_order(df: pd.DataFrame, col: str = "condition") -> pd.DataFrame:
    """Convert condition column to categorical with no_rag first."""
    order = _condition_order(df[col].unique().tolist())
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


def compute_evaluation_metrics(
    df: pd.DataFrame, target_col: str = "label_ideology"
) -> pd.DataFrame:
    """MAE, RMSE, Pearson r, Spearman rho grouped by Model x Condition."""
    results = []

    valid_df = df.dropna(subset=["predicted_bias", target_col]).copy()
    valid_df["predicted_bias"] = pd.to_numeric(valid_df["predicted_bias"])
    valid_df[target_col] = pd.to_numeric(valid_df[target_col])

    dropped_count = len(df) - len(valid_df)
    if dropped_count > 0:
        print(
            f"Note: Omitted {dropped_count} rows with missing predictions/labels "
            f"for target '{target_col}'."
        )

    for (model, condition), group in valid_df.groupby(["model", "condition"]):
        y_true = group[target_col].values
        y_pred = group["predicted_bias"].values

        if len(group) == 0:
            continue

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        if len(group) >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
            pr, _ = pearsonr(y_true, y_pred)
            sr, _ = spearmanr(y_true, y_pred)
        else:
            pr = sr = float("nan")

        results.append(
            {
                "Target": target_col,
                "Model": model,
                "Condition": condition,
                "N_samples": len(group),
                "MAE": round(float(mae), 4),
                "RMSE": round(float(rmse), 4),
                "Pearson_r": (round(float(pr), 4) if not np.isnan(pr) else np.nan),
                "Spearman_rho": (round(float(sr), 4) if not np.isnan(sr) else np.nan),
            }
        )

    results_df = pd.DataFrame(results)
    # Sort: model alphabetically, then no_rag first within each model
    results_df = _apply_condition_order(results_df, "Condition")
    results_df = results_df.sort_values(["Model", "Condition"]).reset_index(drop=True)
    return results_df


def compute_rag_delta(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Per-model delta relative to 'no_rag' baseline.
    For MAE/RMSE: negative delta = RAG improvement.
    For Pearson/Spearman: positive delta = RAG improvement.
    """
    baseline = metrics_df[metrics_df["Condition"] == "no_rag"]
    rag = metrics_df[metrics_df["Condition"] != "no_rag"]

    if baseline.empty:
        print("Warning: No 'no_rag' baseline found — skipping delta calculation.")
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

    delta_df = pd.DataFrame(deltas)
    if not delta_df.empty:
        delta_df = _apply_condition_order(delta_df, "Condition")
        delta_df = delta_df.sort_values(["Model", "Condition"]).reset_index(drop=True)
    return delta_df


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

    lower_better = metric in ("MAE", "RMSE")
    direction = "lower = better" if lower_better else "higher = better"

    fig, ax = plt.subplots(figsize=(12, max(6, len(metrics_df) * 0.35)))

    # Use categorical ordering so no_rag sits at top
    plot_df = _apply_condition_order(metrics_df.copy(), "Condition")

    sns.barplot(
        data=plot_df,
        x=metric,
        y="Condition",
        hue="Model",
        ax=ax,
        palette="mako",
        order=_condition_order(plot_df["Condition"].unique()),
    )

    ax.set_title(
        f"{metric} by Condition and Model  |  Target: {target_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(f"{metric}  ({direction})", fontsize=12)
    ax.set_ylabel("Condition", fontsize=12)
    ax.legend(title="Model", bbox_to_anchor=(1.01, 1), loc="upper left")
    _save(fig, output_path)


def plot_all_metrics_grid(
    metrics_df: pd.DataFrame, target_name: str, output_path: Path
):
    if metrics_df.empty:
        return
    sns.set_theme(style="whitegrid")

    plot_df = _apply_condition_order(metrics_df.copy(), "Condition")
    cond_order = _condition_order(plot_df["Condition"].unique())

    metrics = ["MAE", "RMSE", "Pearson_r", "Spearman_rho"]
    titles = [
        "Mean Absolute Error (MAE)",
        "Root Mean Squared Error (RMSE)",
        "Pearson r",
        "Spearman ρ",
    ]
    xlabels = [
        "MAE (lower = better)",
        "RMSE (lower = better)",
        "Pearson r (higher = better)",
        "Spearman ρ (higher = better)",
    ]

    n_conditions = len(plot_df["Condition"].unique())
    fig_height = max(10, n_conditions * 0.45)
    fig, axes = plt.subplots(2, 2, figsize=(18, fig_height))
    axes = axes.flatten()

    for ax, metric, title, xlabel in zip(axes, metrics, titles, xlabels):
        sns.barplot(
            data=plot_df,
            x=metric,
            y="Condition",
            hue="Model",
            ax=ax,
            palette="mako",
            order=cond_order,
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
    plot_df = _apply_condition_order(delta_df.copy(), "Condition")

    models = sorted(plot_df["Model"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(9 * n_models, max(6, len(plot_df) * 0.3)), sharey=True
    )
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = plot_df[plot_df["Model"] == model].set_index("Condition")
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
        # Invert MAE & RMSE so positive = improvement across all columns
        display["ΔMAE (RAG-Base)"] = -display["ΔMAE (RAG-Base)"]
        display["ΔRMSE (RAG-Base)"] = -display["ΔRMSE (RAG-Base)"]
        display.columns = [
            "ΔMAE\n(+good)",
            "ΔRMSE\n(+good)",
            "ΔPearson r\n(+good)",
            "ΔSpearman ρ\n(+good)",
        ]

        # Sort so no_rag is not applicable here (it's the baseline, not in delta)
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
            f"Model: {model}\n(Positive values → RAG outperforms Baseline)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_ylabel("Condition")
        ax.set_xlabel("Metric Delta (positive = RAG is better)")

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
    # no_rag first in the grid
    conditions = _condition_order(sorted(valid_df["condition"].unique()))
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


def main(
    base_dir: Path = BASE_DIR,
    output_root: Path = OUTPUT_ROOT,
    run_id: Optional[str] = None,
    target_col: str = "label_ideology",
    batch_run: Optional[str] = None,
    embedding_models: Optional[List[str]] = None,
    retrieval_modes: Optional[List[str]] = None,
):
    run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Model Evaluation Metrics")
    print("=" * 80)
    print(f"Output directory: {output_dir.resolve()}\n")

    print("[1/4] Loading batch-run JSONL files …")
    df = load_logs_from_directory(
        base_dir, batch_run, embedding_models, retrieval_modes
    )
    if df.empty:
        return

    print("[2/4] Computing evaluation metrics …")
    metrics_df = compute_evaluation_metrics(df, target_col=target_col)
    delta_df = compute_rag_delta(metrics_df)

    print()
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

    print("[3/4] Saving CSV outputs …")
    metrics_csv = output_dir / f"evaluation_metrics_{target_col}.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"  → {metrics_csv}")

    if not delta_df.empty:
        delta_csv = output_dir / f"rag_delta_{target_col}.csv"
        delta_df.to_csv(delta_csv, index=False)
        print(f"  → {delta_csv}")

    print("[4/4] Generating plots …")
    plot_metric_comparison(
        metrics_df, "MAE", target_col, output_dir / f"mae_comparison_{target_col}.png"
    )
    plot_metric_comparison(
        metrics_df,
        "RMSE",
        target_col,
        output_dir / f"rmse_comparison_{target_col}.png",
    )
    plot_all_metrics_grid(
        metrics_df, target_col, output_dir / f"all_metrics_{target_col}.png"
    )
    plot_rag_delta_heatmap(delta_df, output_dir / f"rag_delta_heatmap_{target_col}.png")
    plot_scatter_predicted_vs_actual(
        df, target_col, output_dir / f"scatter_predicted_vs_actual_{target_col}.png"
    )

    print(f"\nAll outputs written to: {output_dir.resolve()}")
    print("Done.")


def _comma_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate batch-run LLM bias prediction logs.",
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
        "--target",
        default="label_ideology",
        choices=["label_ideology", "label_economic", "label_galtan"],
        help="Ground-truth label to evaluate against (default: label_ideology).",
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
        target_col=args.target,
        batch_run=args.batch_run,
        embedding_models=args.embedding_models,
        retrieval_modes=args.retrieval_modes,
    )
