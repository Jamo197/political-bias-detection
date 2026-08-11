"""evaluate_metrics.py

Loads batch-run JSONL logs, deduplicates records, computes evaluation metrics
(MAE, RMSE, Pearson r, Spearman rho) per condition, and calculates pairwise
deltas against the no_rag baseline on matching text samples.

Outputs CSV summary tables and visualization plots.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Helper Functions & Utilities
# ---------------------------------------------------------------------------


def _clean_model_name(llm_path: str) -> str:
    if not llm_path or llm_path == "unknown":
        return "Unknown"
    return llm_path.split("/")[-1]


def _model_family(model_name: str) -> str:
    name_lower = model_name.lower()
    if "qwen" in name_lower:
        return "Qwen"
    elif "ministral" in name_lower or "mistral" in name_lower:
        return "Mistral"
    elif "llama" in name_lower:
        return "Llama"
    return "Other"


def _model_size_key(model_name: str) -> int:
    name_lower = model_name.lower()
    if "large" in name_lower:
        return 999
    match = re.search(r"(\d+)b", name_lower)
    return int(match.group(1)) if match else 999


def _condition_order(conditions: List[str]) -> List[str]:
    """Return conditions sorted with 'no_rag' always first."""
    others = sorted(c for c in conditions if c != "no_rag")
    return ["no_rag"] + others


def _apply_condition_order(df: pd.DataFrame, col: str = "Condition") -> pd.DataFrame:
    """Convert condition column to categorical with no_rag first."""
    if col not in df.columns or df.empty:
        return df
    order = _condition_order(df[col].unique().tolist())
    df = df.copy()
    df[col] = pd.Categorical(df[col], categories=order, ordered=True)
    return df


def _calculate_array_stats(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float, float]:
    """Computes MAE, RMSE, Pearson r, and Spearman rho safely."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    if len(y_true) >= 3 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pr, _ = pearsonr(y_true, y_pred)
        sr, _ = spearmanr(y_true, y_pred)
    else:
        pr, sr = np.nan, np.nan

    return float(mae), float(rmse), float(pr), float(sr)


# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------


def load_logs_from_directory(
    base_dir: Path,
    batch_run: Optional[str] = None,
    embedding_models: Optional[List[str]] = None,
    retrieval_modes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Walk base_dir recursively, read JSONL entries, flatten into a DataFrame,

    and deduplicate re-run entries to prevent sample size inflation.
    """
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
                retrieval_mode = params.get("retrieval_mode", "unknown")
                is_rag = params.get("is_rag", True)

                # Determine condition name
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
                        "text_index": str(meta.get("text_index", "")),
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
    if df.empty:
        print("Warning: Loaded DataFrame is empty.")
        return df

    # Deduplicate runs to prevent sample-size inflation from duplicated executions
    initial_len = len(df)
    df = df.drop_duplicates(subset=["model", "condition", "text_index"], keep="last")
    print(
        f"Loaded {len(df)} unique records (deduplicated from {initial_len}) across {len(jsonl_files)} files."
    )
    print(f"  Batch runs:       {sorted(df['batch_run'].unique())}")
    print(f"  Embedding models: {sorted(df['embedding_model'].unique())}")
    print(f"  Retrieval modes:  {sorted(df['retrieval_mode'].unique())}")
    print(f"  Conditions:       {sorted(df['condition'].unique())}")
    print(f"  Models:           {sorted(df['model'].unique())}")
    return df


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------


def compute_evaluation_metrics(
    df: pd.DataFrame, target_col: str = "label_ideology"
) -> pd.DataFrame:
    """Computes MAE, RMSE, Pearson r, and Spearman rho on all deduplicated valid

    predictions per condition (maximizes dataset size N per condition).
    """
    valid_df = df.dropna(subset=["predicted_bias", target_col]).copy()
    valid_df["predicted_bias"] = pd.to_numeric(valid_df["predicted_bias"])
    valid_df[target_col] = pd.to_numeric(valid_df[target_col])

    dropped_count = len(df) - len(valid_df)
    if dropped_count > 0:
        print(
            f"Note: Omitted {dropped_count} rows with missing/non-numeric predictions "
            f"for target '{target_col}'."
        )

    results = []
    for (model, condition), group in valid_df.groupby(["model", "condition"]):
        y_true = group[target_col].values
        y_pred = group["predicted_bias"].values

        if len(group) == 0:
            continue

        mae, rmse, pr, sr = _calculate_array_stats(y_true, y_pred)

        results.append(
            {
                "Target": target_col,
                "Model": model,
                "Condition": condition,
                "N_samples": len(group),
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "Pearson_r": round(pr, 4) if not np.isnan(pr) else np.nan,
                "Spearman_rho": round(sr, 4) if not np.isnan(sr) else np.nan,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = _apply_condition_order(results_df, "Condition")
    return results_df.sort_values(["Model", "Condition"]).reset_index(drop=True)


def compute_pairwise_rag_delta(
    df: pd.DataFrame, target_col: str = "label_ideology"
) -> pd.DataFrame:
    """Computes RAG vs no_rag deltas strictly on the paired intersection

    between EACH RAG condition and the no_rag baseline individually.
    """
    valid_df = df.dropna(subset=["predicted_bias", target_col]).copy()
    valid_df["predicted_bias"] = pd.to_numeric(valid_df["predicted_bias"])
    valid_df[target_col] = pd.to_numeric(valid_df[target_col])

    deltas = []

    for model, model_df in valid_df.groupby("model"):
        baseline_df = model_df[model_df["condition"] == "no_rag"]
        if baseline_df.empty:
            continue

        rag_conditions = [c for c in model_df["condition"].unique() if c != "no_rag"]

        for cond in rag_conditions:
            cond_df = model_df[model_df["condition"] == cond]

            # Merge strictly on matching text_index items
            merged = pd.merge(
                baseline_df[["text_index", target_col, "predicted_bias"]],
                cond_df[["text_index", "predicted_bias"]],
                on="text_index",
                suffixes=("_base", "_rag"),
            )

            if len(merged) < 3:
                continue

            y_true = merged[target_col].values
            y_base = merged["predicted_bias_base"].values
            y_rag = merged["predicted_bias_rag"].values

            mae_base, rmse_base, pr_base, sr_base = _calculate_array_stats(
                y_true, y_base
            )
            mae_rag, rmse_rag, pr_rag, sr_rag = _calculate_array_stats(y_true, y_rag)

            delta_pr = (
                round(pr_rag - pr_base, 4)
                if not (np.isnan(pr_rag) or np.isnan(pr_base))
                else np.nan
            )
            delta_sr = (
                round(sr_rag - sr_base, 4)
                if not (np.isnan(sr_rag) or np.isnan(sr_base))
                else np.nan
            )

            deltas.append(
                {
                    "Target": target_col,
                    "Model": model,
                    "Condition": cond,
                    "N_paired": len(merged),
                    "ΔMAE (RAG-Base)": round(mae_rag - mae_base, 4),
                    "ΔRMSE (RAG-Base)": round(rmse_rag - rmse_base, 4),
                    "ΔPearson_r (RAG-Base)": delta_pr,
                    "ΔSpearman_rho (RAG-Base)": delta_sr,
                }
            )

    delta_df = pd.DataFrame(deltas)
    if not delta_df.empty:
        delta_df = _apply_condition_order(delta_df, "Condition")
        delta_df = delta_df.sort_values(["Model", "Condition"]).reset_index(drop=True)
    return delta_df


# ---------------------------------------------------------------------------
# Visualization Functions
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

    plot_df = _apply_condition_order(metrics_df.copy(), "Condition")
    cond_order = _condition_order(plot_df["Condition"].unique().tolist())

    fig, ax = plt.subplots(figsize=(12, max(6, len(cond_order) * 0.35)))

    sns.barplot(
        data=plot_df,
        x=metric,
        y="Condition",
        hue="Model",
        ax=ax,
        palette="mako",
        order=cond_order,
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


def plot_mae_rmse_combined(
    metrics_df: pd.DataFrame, target_name: str, output_path: Path
):
    """Plots MAE and RMSE side-by-side in a single figure for easy error comparison."""
    if metrics_df.empty:
        return
    sns.set_theme(style="whitegrid")

    plot_df = _apply_condition_order(metrics_df.copy(), "Condition")
    cond_order = _condition_order(plot_df["Condition"].unique().tolist())

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, max(6, len(cond_order) * 0.45)), sharey=True
    )

    sns.barplot(
        data=plot_df,
        x="MAE",
        y="Condition",
        hue="Model",
        ax=ax1,
        palette="mako",
        order=cond_order,
    )
    ax1.set_title("Mean Absolute Error (MAE)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("MAE (lower = better)", fontsize=11)
    ax1.set_ylabel("Condition", fontsize=11)
    ax1.legend(title="Model", fontsize=9)

    sns.barplot(
        data=plot_df,
        x="RMSE",
        y="Condition",
        hue="Model",
        ax=ax2,
        palette="mako",
        order=cond_order,
    )
    ax2.set_title("Root Mean Squared Error (RMSE)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("RMSE (lower = better)", fontsize=11)
    ax2.set_ylabel("")
    ax2.legend(title="Model", fontsize=9)

    fig.suptitle(
        f"Error Metric Comparison (MAE & RMSE)  |  Target: {target_name}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    _save(fig, output_path)


def plot_all_metrics_grid(
    metrics_df: pd.DataFrame, target_name: str, output_path: Path
):
    if metrics_df.empty:
        return
    sns.set_theme(style="whitegrid")

    plot_df = _apply_condition_order(metrics_df.copy(), "Condition")
    cond_order = _condition_order(plot_df["Condition"].unique().tolist())

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

    n_conditions = len(cond_order)
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


def plot_model_family_heatmaps(
    delta_df: pd.DataFrame, target_name: str, output_dir: Path
):
    """Generates heatmaps per model family showing pairwise RAG deltas.

    Handles missing models and missing RAG conditions gracefully.
    Green always indicates RAG improvement across all 4 metric tiles.
    """
    if delta_df.empty:
        print("Skipping RAG delta heatmap — no delta data available.")
        return

    sns.set_theme(style="white")
    plot_df = delta_df.copy()
    plot_df["Family"] = plot_df["Model"].apply(_model_family)

    delta_metrics = [
        ("ΔMAE (RAG-Base)", "ΔMAE (Lower is Better)", True),
        ("ΔRMSE (RAG-Base)", "ΔRMSE (Lower is Better)", True),
        ("ΔPearson_r (RAG-Base)", "ΔPearson r (Higher is Better)", False),
        ("ΔSpearman_rho (RAG-Base)", "ΔSpearman ρ (Higher is Better)", False),
    ]

    families = sorted(plot_df["Family"].unique())

    for family in families:
        sub = plot_df[plot_df["Family"] == family]
        if sub.empty:
            continue

        model_variants = sorted(sub["Model"].unique(), key=_model_size_key)
        conditions = sorted(sub["Condition"].unique())

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(
                max(8, 2.5 * len(model_variants) + 2),
                max(7, 1.8 * len(conditions) + 1),
            ),
        )
        axes = axes.flatten()

        for ax, (col, title, invert) in zip(axes, delta_metrics):
            if col not in sub.columns or sub[col].dropna().empty:
                ax.text(0.5, 0.5, "No Data Available", ha="center", va="center")
                ax.set_title(title, fontsize=11, fontweight="bold")
                continue

            pivot = sub.pivot_table(
                index="Condition", columns="Model", values=col, aggfunc="first"
            )
            pivot = pivot.reindex(index=conditions, columns=model_variants)

            # Invert values for error metrics so negative delta displays green (improvement)
            display_data = -pivot if invert else pivot

            # Safe formatting for string matrix with NaNs
            map_func = pivot.map if hasattr(pivot, "map") else pivot.applymap
            annot_matrix = map_func(lambda x: f"{x:.3f}" if pd.notnull(x) else "N/A")

            sns.heatmap(
                display_data,
                annot=annot_matrix,
                fmt="",
                cmap="RdYlGn",
                center=0,
                linewidths=0.5,
                ax=ax,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_facecolor("#f0f0f0")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Condition", fontsize=10)
            ax.set_xlabel("Model Variant", fontsize=10)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=30, ha="right", fontsize=9
            )
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        fig.suptitle(
            f"{family} Family — Pairwise RAG Delta vs. Baseline\nTarget: {target_name}",
            fontsize=14,
            fontweight="bold",
        )
        fig.tight_layout()
        save_path = output_dir / f"rag_delta_heatmap_{family.lower()}_{target_name}.png"
        _save(fig, save_path)


def plot_scatter_predicted_vs_actual(
    df: pd.DataFrame, target_col: str, output_path: Path
):
    valid_df = df.dropna(subset=["predicted_bias", target_col]).copy()
    if valid_df.empty:
        return

    valid_df["predicted_bias"] = pd.to_numeric(valid_df["predicted_bias"])
    valid_df[target_col] = pd.to_numeric(valid_df[target_col])

    sns.set_theme(style="whitegrid")
    conditions = _condition_order(sorted(valid_df["condition"].unique().tolist()))
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
        print("No valid records found. Exiting.")
        return

    print("[2/4] Computing evaluation metrics & pairwise deltas …")
    metrics_df = compute_evaluation_metrics(df, target_col=target_col)
    delta_df = compute_pairwise_rag_delta(df, target_col=target_col)

    print()
    print("=" * 80)
    print(f"EVALUATION METRICS (Target: {target_col})")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    print()

    if not delta_df.empty:
        print("=" * 80)
        print("PAIRWISE RAG vs. BASELINE DELTA")
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
        delta_csv = output_dir / f"rag_delta_pairwise_{target_col}.csv"
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
    plot_mae_rmse_combined(
        metrics_df, target_col, output_dir / f"mae_rmse_combined_{target_col}.png"
    )
    plot_all_metrics_grid(
        metrics_df, target_col, output_dir / f"all_metrics_{target_col}.png"
    )
    plot_model_family_heatmaps(delta_df, target_col, output_dir)
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
        help="Root directory for results.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Name of output subfolder (default: timestamp).",
    )
    parser.add_argument(
        "--target",
        default="label_ideology",
        choices=["label_ideology", "label_economic", "label_galtan"],
        help="Ground-truth label column.",
    )
    parser.add_argument(
        "--batch-run",
        type=str,
        default=None,
        help="Restrict analysis to a single batch run folder.",
    )
    parser.add_argument(
        "--embedding-models",
        type=_comma_list,
        default=None,
        help="Comma-separated embedding models to filter.",
    )
    parser.add_argument(
        "--retrieval-modes",
        type=_comma_list,
        default=None,
        help="Comma-separated retrieval modes to filter.",
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
