#!/usr/bin/env python3
"""analyze_retrieval_party_alignment.py

Analyzes retrieval composition and party alignment from batch-run JSONL logs.

Score types
-----------
Different retrieval strategies produce scores on incompatible scales
(cosine, RRF fusion, cross-encoder logits), but party proportions are
scale-free and directly comparable across all conditions.

Analyses
--------
1. Stacked bar chart of retrieved-party proportions conditioned on the
   target text's party (overall + per-condition breakdown).
2. In-Party vs. Out-Party retrieval impact: MAE and Pearson/Spearman r
   when at least one retrieved chunk matches the target party vs. when
   all chunks are from opposing parties (overall + per condition).
3. In-Party ratio (fraction of same-party chunks) vs. absolute error
   correlations and scatter plots.

Outputs
-------
Every execution creates a new timestamped folder:

  results/qualitative/retrieval_party_alignment/<YYYYMMDD_HHMMSS>/

Use --run-id to override the folder name.

Usage:
    python src/results/analyze_retrieval_party_alignment.py
    python src/results/analyze_retrieval_party_alignment.py --batch-run 2026-08-04_eval_matrix_20260804_191413
    python src/results/analyze_retrieval_party_alignment.py --embedding-models bge,jina
    python src/results/analyze_retrieval_party_alignment.py --retrieval-modes simple,twostage
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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("logs/batch_runs")
OUTPUT_ROOT = Path("results/qualitative/retrieval_party_alignment")

MAIN_PARTIES = ["SPD", "CDU/CSU", "AfD", "FDP", "Greens", "Left"]
PARTY_ORDER = MAIN_PARTIES + ["Other"]

PARTY_COLORS = {
    "SPD": "#E3000F",
    "CDU/CSU": "#000000",
    "AfD": "#0489DB",
    "FDP": "#FFEF00",
    "Greens": "#1AA024",
    "Left": "#BE3075",
    "Other": "#808080",
}

# ---------------------------------------------------------------------------
# Party Normalization
# ---------------------------------------------------------------------------


def normalize_party(party_name: Optional[str]) -> str:
    """Normalize raw party strings to canonical names."""
    if not party_name:
        return "Other"

    p = party_name.strip()
    for ws in ("\xa0", "\n", "\t", "\r", "\u200b"):
        p = p.replace(ws, " ")
    while "  " in p:
        p = p.replace("  ", " ")
    p = p.strip().upper()

    if p == "SPD":
        return "SPD"
    if p in ("CDU", "CSU", "CDU/CSU"):
        return "CDU/CSU"
    if p in ("AFD", "ALTERNATIVE FÜR DEUTSCHLAND"):
        return "AfD"
    if p == "FDP":
        return "FDP"
    if "GRÜN" in p or "GRUN" in p or p in ("B90GRUNE", "B90/GRÜNE"):
        return "Greens"
    if p in (
        "DIE LINKE",
        "LINKSPARTEI",
        "PDS",
        "LINKE",
        "DIE LINKSPARTEI",
        "DIE LINKE.PDS",
    ) or ("LINKE" in p and "DIE" in p):
        return "Left"

    return "Other"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------


def load_rag_logs(
    base_dir: Path,
    batch_run: Optional[str] = None,
    embedding_models: Optional[List[str]] = None,
    retrieval_modes: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load party_label_*.jsonl files and keep RAG runs with retrieved chunks."""
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

                meta = log.get("input_metadata", {})
                gt = log.get("ground_truth", {})
                output = log.get("output", {})

                target_party = normalize_party(meta.get("party"))
                retrieved_parties = [normalize_party(c.get("party")) for c in chunks]

                records.append(
                    {
                        "run_id": log.get("run_id", "unknown"),
                        "batch_run": run_folder,
                        "model": _clean_model_name(params.get("llm", "unknown")),
                        "embedding_model": emb,
                        "retrieval_mode": retrieval_mode,
                        "condition": f"{emb}/{retrieval_mode}",
                        "text_index": meta.get("text_index", ""),
                        "target_party": target_party,
                        "retrieved_parties": retrieved_parties,
                        "n_chunks": len(retrieved_parties),
                        "predicted_bias": output.get("bias"),
                        "label_ideology": gt.get("label_ideology"),
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
# Feature Engineering
# ---------------------------------------------------------------------------


def compute_alignment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-query party-alignment and error columns."""
    df = df.copy()

    df["predicted_bias"] = pd.to_numeric(df["predicted_bias"], errors="coerce")
    df["label_ideology"] = pd.to_numeric(df["label_ideology"], errors="coerce")
    df["abs_error"] = (df["predicted_bias"] - df["label_ideology"]).abs()

    df["n_inparty"] = df.apply(
        lambda r: sum(1 for p in r["retrieved_parties"] if p == r["target_party"]),
        axis=1,
    )
    df["inparty_ratio"] = df["n_inparty"] / df["n_chunks"]
    df["has_inparty"] = df["n_inparty"] > 0
    df["retrieval_type"] = df["has_inparty"].map(
        {True: "In-Party Match", False: "Out-Party Noise"}
    )

    return df


# ---------------------------------------------------------------------------
# Analysis 1: Party Distribution per Target Party
# ---------------------------------------------------------------------------


def _explode_parties(df_main: pd.DataFrame) -> pd.DataFrame:
    """Long-format DataFrame: one row per (query, retrieved_chunk)."""
    long = df_main[["condition", "target_party", "retrieved_parties"]].explode(
        "retrieved_parties"
    )
    return long.rename(columns={"retrieved_parties": "retrieved_party"}).reset_index(
        drop=True
    )


def _crosstab_party(long_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation of target_party x retrieved_party (row-normalized)."""
    ct = pd.crosstab(
        long_df["target_party"], long_df["retrieved_party"], normalize="index"
    )
    ct = ct.reindex(columns=PARTY_ORDER, fill_value=0)
    ct = ct.reindex(index=MAIN_PARTIES, fill_value=0)
    return ct


def analyze_party_distribution(df_main: pd.DataFrame, output_dir: Path):
    """Compute and plot party distribution overall + per condition."""
    output_dir.mkdir(parents=True, exist_ok=True)

    long = _explode_parties(df_main)

    # Overall
    ct = _crosstab_party(long)
    overall_path = output_dir / "party_distribution_overall.csv"
    ct.round(4).to_csv(overall_path)
    print(f"\nSaved overall party distribution CSV: {overall_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ct.plot(
        kind="bar",
        stacked=True,
        color=[PARTY_COLORS[c] for c in ct.columns],
        ax=ax,
        width=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(
        "Retrieved Party Distribution by Target Party (All Conditions)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Target Party", fontsize=12)
    ax.set_ylabel("Proportion of Retrieved Chunks", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Retrieved Party", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, output_dir / "party_distribution_stacked.png")

    # Per-condition long CSV
    rows = []
    for cond in sorted(long["condition"].unique()):
        sub_long = long[long["condition"] == cond]
        ct = _crosstab_party(sub_long)
        for target in MAIN_PARTIES:
            for retrieved in PARTY_ORDER:
                rows.append(
                    {
                        "condition": cond,
                        "target_party": target,
                        "retrieved_party": retrieved,
                        "proportion": round(ct.loc[target, retrieved], 4),
                    }
                )
    cond_csv_path = output_dir / "party_distribution_by_condition.csv"
    pd.DataFrame(rows).to_csv(cond_csv_path, index=False)
    print(f"Saved per-condition party distribution CSV: {cond_csv_path}")

    # Per-condition grid plot
    conditions = sorted(long["condition"].unique())
    n_cond = len(conditions)
    n_cols = 3
    n_rows = (n_cond + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
    )

    for idx, cond in enumerate(conditions):
        ax = axes[idx // n_cols][idx % n_cols]
        sub_long = long[long["condition"] == cond]
        ct = _crosstab_party(sub_long)
        ct.plot(
            kind="bar",
            stacked=True,
            color=[PARTY_COLORS[c] for c in ct.columns],
            ax=ax,
            width=0.7,
            edgecolor="white",
            linewidth=0.5,
            legend=False,
        )
        ax.set_title(cond, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("")
        ax.set_xlabel("")

    for idx in range(n_cond, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.legend(
        labels=PARTY_ORDER,
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=PARTY_COLORS[p]) for p in PARTY_ORDER
        ],
        loc="lower center",
        ncol=7,
        fontsize=9,
    )
    fig.suptitle(
        "Retrieved Party Distribution by Target Party — Per Condition",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    _save(fig, output_dir / "party_distribution_by_condition.png")

    # In-party ratio bar chart per condition
    _plot_inparty_ratio_by_condition(df_main, output_dir)


def _plot_inparty_ratio_by_condition(df_main: pd.DataFrame, output_dir: Path):
    """Bar chart of mean in-party ratio per condition."""
    cond_ratio = (
        df_main.groupby("condition")["inparty_ratio"]
        .mean()
        .reset_index()
        .sort_values("inparty_ratio", ascending=False)
    )

    emb_colors = {
        "e5": "#1f77b4",
        "jina": "#ff7f0e",
        "qwen3": "#2ca02c",
        "bge": "#d62728",
    }
    bar_colors = [
        emb_colors.get(c.split("/")[0], "#7f7f7f") for c in cond_ratio["condition"]
    ]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(
        cond_ratio["condition"],
        cond_ratio["inparty_ratio"],
        color=bar_colors,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, val in zip(bars, cond_ratio["inparty_ratio"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.005,
            f"{val:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(
        "Mean In-Party Retrieval Ratio by Condition",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Mean In-Party Ratio (higher = more same-party chunks)", fontsize=11)
    ax.set_xticks(range(len(cond_ratio)))
    ax.set_xticklabels(cond_ratio["condition"], rotation=45, ha="right", fontsize=9)
    ax.set_ylim(0, cond_ratio["inparty_ratio"].max() * 1.3)
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, output_dir / "inparty_ratio_by_condition.png")


# ---------------------------------------------------------------------------
# Analysis 2: In-Party vs. Out-Party Retrieval Impact
# ---------------------------------------------------------------------------


def _compute_metrics(sub_df: pd.DataFrame) -> Dict:
    """Compute MAE, Pearson r, Spearman r for a subset."""
    valid = sub_df.dropna(subset=["predicted_bias", "label_ideology"]).copy()
    n = len(valid)
    if n == 0:
        return {"n": 0, "MAE": np.nan, "Pearson_r": np.nan, "Spearman_r": np.nan}

    mae = float(valid["abs_error"].mean())
    pr = np.nan
    sr = np.nan
    if (
        n >= 3
        and valid["predicted_bias"].std() > 0
        and valid["label_ideology"].std() > 0
    ):
        pr, _ = pearsonr(valid["predicted_bias"], valid["label_ideology"])
        sr, _ = spearmanr(valid["predicted_bias"], valid["label_ideology"])

    return {
        "n": n,
        "MAE": round(mae, 4),
        "Pearson_r": round(float(pr), 4),
        "Spearman_r": round(float(sr), 4),
    }


def analyze_inparty_outparty(df_main: pd.DataFrame, output_dir: Path):
    """Classify queries and compare performance metrics (overall + per condition)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for rtype in ["In-Party Match", "Out-Party Noise"]:
        sub = df_main[df_main["retrieval_type"] == rtype]
        metrics = _compute_metrics(sub)
        results.append({"condition": "Overall", "retrieval_type": rtype, **metrics})

    for cond in sorted(df_main["condition"].unique()):
        for rtype in ["In-Party Match", "Out-Party Noise"]:
            sub = df_main[
                (df_main["condition"] == cond) & (df_main["retrieval_type"] == rtype)
            ]
            metrics = _compute_metrics(sub)
            results.append({"condition": cond, "retrieval_type": rtype, **metrics})

    metrics_df = pd.DataFrame(results)
    csv_path = output_dir / "inparty_outparty_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    print(f"\nSaved in-party/out-party metrics CSV: {csv_path}")
    print(metrics_df[metrics_df["condition"] == "Overall"].to_string(index=False))

    _plot_overall_inparty_outparty(metrics_df, output_dir)
    _plot_inparty_outparty_by_condition(metrics_df, output_dir)


def _plot_overall_inparty_outparty(metrics_df: pd.DataFrame, output_dir: Path):
    """Overall MAE and Pearson bar plots."""
    overall = metrics_df[metrics_df["condition"] == "Overall"].copy()
    if overall.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in zip(
        axes,
        ["MAE", "Pearson_r"],
        ["MAE (lower is better)", "Pearson r (higher is better)"],
    ):
        colors = [
            "#2ecc71" if rt == "In-Party Match" else "#e74c3c"
            for rt in overall["retrieval_type"]
        ]
        bars = ax.bar(
            overall["retrieval_type"],
            overall[metric],
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=11)

        vals = overall[metric].dropna()
        if len(vals):
            margin = max(vals.max() * 0.15, 0.05)
            ymin = min(vals.min() - margin if metric == "Pearson_r" else 0, -0.05)
            ax.set_ylim(ymin, vals.max() + margin)

        for bar, row in zip(bars, overall.itertuples()):
            height = bar.get_height()
            if np.isnan(height):
                continue
            if metric == "Pearson_r" and height < 0:
                offset, va = -0.03, "top"
            else:
                offset, va = 0.02, "bottom"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f"{height:.3f}\n(n={row.n})",
                ha="center",
                va=va,
                fontsize=9,
            )
        sns.despine(ax=ax)

    fig.suptitle(
        "In-Party vs. Out-Party Retrieval Impact (Overall, Aggregated)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, output_dir / "inparty_vs_outparty_metrics.png")


def _plot_inparty_outparty_by_condition(metrics_df: pd.DataFrame, output_dir: Path):
    """Per-condition grouped bar chart of MAE and Pearson r."""
    per_cond = metrics_df[metrics_df["condition"] != "Overall"].copy()
    if per_cond.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, metric in enumerate(["MAE", "Pearson_r"]):
        ax = axes[ax_idx]
        pivot = per_cond.pivot_table(
            index="condition", columns="retrieval_type", values=metric
        ).sort_index()

        x = np.arange(len(pivot))
        width = 0.35
        ax.bar(
            x - width / 2,
            pivot["In-Party Match"],
            width,
            label="In-Party Match",
            color="#2ecc71",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.bar(
            x + width / 2,
            pivot["Out-Party Noise"],
            width,
            label="Out-Party Noise",
            color="#e74c3c",
            edgecolor="black",
            linewidth=0.5,
        )

        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=8)
        ax.legend(fontsize=10)
        sns.despine(ax=ax)

    fig.suptitle(
        "In-Party vs. Out-Party Metrics by Condition",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, output_dir / "inparty_outparty_by_condition.png")


# ---------------------------------------------------------------------------
# Analysis 3: In-Party Ratio vs. Error
# ---------------------------------------------------------------------------


def compute_inparty_correlations(df_main: pd.DataFrame) -> pd.DataFrame:
    """Pearson & Spearman between inparty_ratio and abs_error."""
    valid = df_main.dropna(subset=["inparty_ratio", "abs_error"]).copy()
    rows = []

    def _corr(sub: pd.DataFrame, label: str) -> Dict:
        if len(sub) < 3:
            return {
                "group": label,
                "n": len(sub),
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
            }
        pr, pp = pearsonr(sub["inparty_ratio"], sub["abs_error"])
        sr, sp = spearmanr(sub["inparty_ratio"], sub["abs_error"])
        return {
            "group": label,
            "n": len(sub),
            "pearson_r": round(float(pr), 4),
            "pearson_p": round(float(pp), 4),
            "spearman_r": round(float(sr), 4),
            "spearman_p": round(float(sp), 4),
        }

    rows.append(_corr(valid, "Global"))
    for (emb, mode), g in valid.groupby(["embedding_model", "retrieval_mode"]):
        rows.append(_corr(g, f"{emb}/{mode}"))

    return pd.DataFrame(rows)


def plot_inparty_ratio_vs_error(df_main: pd.DataFrame, output_dir: Path):
    """Scatter grid: in-party ratio vs absolute error per condition."""
    valid = df_main.dropna(subset=["inparty_ratio", "abs_error"]).copy()
    conditions = sorted(valid["condition"].unique())
    n_cols = 3
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
            data=sub,
            x="inparty_ratio",
            y="abs_error",
            alpha=0.5,
            ax=ax,
            edgecolor=None,
        )
        sns.regplot(
            data=sub,
            x="inparty_ratio",
            y="abs_error",
            scatter=False,
            ax=ax,
            color="red",
            line_kws={"linewidth": 1.5},
        )

        pr, pp = pearsonr(sub["inparty_ratio"], sub["abs_error"])
        sr, sp = spearmanr(sub["inparty_ratio"], sub["abs_error"])

        ax.set_title(cond, fontsize=11, fontweight="bold")
        ax.set_xlabel("In-Party Retrieval Ratio", fontsize=10)
        ax.set_ylabel("Absolute Error |pred − true|", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.text(
            0.05,
            0.95,
            f"Pearson r={pr:.3f} (p={pp:.3f})\nSpearman ρ={sr:.3f} (p={sp:.3f})",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    for idx in range(len(conditions), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.suptitle(
        "In-Party Retrieval Ratio vs. Absolute Error",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, output_dir / "scatter_inparty_ratio_vs_error.png")


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
    run_id = run_id or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Retrieval Composition & Party Alignment Analysis")
    print("=" * 80)
    print(f"Output directory: {output_dir.resolve()}\n")

    df = load_rag_logs(base_dir, batch_run, embedding_models, retrieval_modes)
    if df.empty:
        return

    df = compute_alignment_features(df)

    # Per-query CSV
    per_query_cols = [
        "run_id",
        "batch_run",
        "model",
        "embedding_model",
        "retrieval_mode",
        "condition",
        "text_index",
        "target_party",
        "n_chunks",
        "n_inparty",
        "inparty_ratio",
        "has_inparty",
        "retrieval_type",
        "predicted_bias",
        "label_ideology",
        "abs_error",
    ]
    per_query_path = output_dir / "party_alignment_per_query.csv"
    df[per_query_cols].to_csv(per_query_path, index=False)
    print(f"\nSaved per-query CSV: {per_query_path}")

    # Filter to main parties for analysis
    df_main = df[df["target_party"].isin(MAIN_PARTIES)].copy()
    if df_main.empty:
        print("No records with main target parties — nothing to analyze.")
        return

    print(f"\nFocusing on target parties: {MAIN_PARTIES}")
    print(f"Records with main target parties: {len(df_main)}")
    print("Target party distribution:")
    print(df_main["target_party"].value_counts().to_string())

    # Analysis 1: Party distribution
    analyze_party_distribution(df_main, output_dir)

    # Analysis 2: In-party vs out-party metrics
    analyze_inparty_outparty(df_main, output_dir)

    # Analysis 3: In-party ratio vs error correlations + scatter
    corr_df = compute_inparty_correlations(df_main)
    corr_path = output_dir / "inparty_ratio_correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print("\nIn-Party Ratio vs Error Correlations:")
    print(corr_df.to_string(index=False))
    print(f"\nSaved correlations CSV: {corr_path}")

    plot_inparty_ratio_vs_error(df_main, output_dir)

    print(f"\nAll outputs written to: {output_dir.resolve()}")
    print("Done.")


def _comma_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze retrieval party alignment.",
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
