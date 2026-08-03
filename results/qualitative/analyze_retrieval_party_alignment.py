#!/usr/bin/env python3
"""analyze_retrieval_party_alignment.py

Analyzes retrieval composition and party alignment from batch-run JSONL logs.

Produces:
  1. Stacked bar chart of retrieved-party proportions conditioned on the target
     text's party (SPD, CDU/CSU, AfD, FDP, Greens, Left).
  2. In-Party vs. Out-Party retrieval impact: MAE and Pearson r when at least
     one retrieved chunk matches the target party vs. when all chunks are from
     opposing parties.

Outputs are saved to results/retrieval_party_alignment/.

Usage:
    python results/analyze_retrieval_party_alignment.py
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("logs/batch_runs")
OUTPUT_DIR = Path("results/analysis/retrieval_party_alignment")
TARGET_COL = "label_ideology"

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


def load_rag_logs(base_dir: Path) -> pd.DataFrame:
    """Load party_label_*.jsonl files and keep only RAG runs with retrieved chunks."""
    records: List[Dict] = []
    jsonl_files = sorted(base_dir.rglob("party_label_*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(
            f"No party_label_*.jsonl files found under '{base_dir}'"
        )

    for fp in jsonl_files:
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
                is_rag = params.get("is_rag", True)
                emb = params.get("embedding_model", "none")
                retrieval_mode = params.get("retrieval_mode", "unknown")

                if not is_rag:
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
                        "label_economic": gt.get("label_economic"),
                        "label_galtan": gt.get("label_galtan"),
                    }
                )

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} RAG records from {len(jsonl_files)} JSONL files.")
    print(f"  Conditions: {sorted(df['condition'].unique())}")
    return df


def _clean_model_name(llm_path: str) -> str:
    if not llm_path or llm_path == "unknown":
        return "Unknown"
    return llm_path.split("/")[-1]


# ---------------------------------------------------------------------------
# Analysis 1: Party Distribution per Target Party
# ---------------------------------------------------------------------------


def analyze_party_distribution(df: pd.DataFrame, output_dir: Path):
    """Compute and plot the proportion of retrieved parties per target party."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df_main = df[df["target_party"].isin(MAIN_PARTIES)].copy()
    if df_main.empty:
        print("No records with main target parties — skipping party distribution.")
        return

    rows = []
    for _, r in df_main.iterrows():
        for rp in r["retrieved_parties"]:
            rows.append({"target_party": r["target_party"], "retrieved_party": rp})
    long_df = pd.DataFrame(rows)

    crosstab = pd.crosstab(
        long_df["target_party"], long_df["retrieved_party"], normalize="index"
    )
    crosstab = crosstab.reindex(columns=PARTY_ORDER, fill_value=0)
    crosstab = crosstab.reindex(index=MAIN_PARTIES, fill_value=0)

    csv_path = output_dir / "party_distribution.csv"
    crosstab.round(4).to_csv(csv_path)
    print(f"\nSaved party distribution CSV: {csv_path}")

    fig, ax = plt.subplots(figsize=(10, 6))
    crosstab.plot(
        kind="bar",
        stacked=True,
        color=[PARTY_COLORS[c] for c in crosstab.columns],
        ax=ax,
        width=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(
        "Retrieved Party Distribution by Target Party", fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Target Party", fontsize=12)
    ax.set_ylabel("Proportion of Retrieved Chunks", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Retrieved Party", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    sns.despine(ax=ax)
    fig.tight_layout()
    _save(fig, output_dir / "party_distribution_stacked.png")


# ---------------------------------------------------------------------------
# Analysis 2: In-Party vs. Out-Party Retrieval Impact
# ---------------------------------------------------------------------------


def _compute_metrics(sub_df: pd.DataFrame, target_col: str = TARGET_COL) -> Dict:
    """Compute MAE and Pearson r for a subset."""
    valid = sub_df.dropna(subset=["predicted_bias", target_col]).copy()
    valid["predicted_bias"] = pd.to_numeric(valid["predicted_bias"], errors="coerce")
    valid[target_col] = pd.to_numeric(valid[target_col], errors="coerce")
    valid = valid.dropna(subset=["predicted_bias", target_col])

    n = len(valid)
    if n == 0:
        return {"n": 0, "MAE": np.nan, "Pearson_r": np.nan}

    mae = mean_absolute_error(valid[target_col], valid["predicted_bias"])
    if n >= 3 and np.std(valid[target_col]) > 0 and np.std(valid["predicted_bias"]) > 0:
        pr, _ = pearsonr(valid[target_col], valid["predicted_bias"])
    else:
        pr = np.nan

    return {"n": n, "MAE": round(float(mae), 4), "Pearson_r": round(float(pr), 4)}


def analyze_inparty_outparty(df: pd.DataFrame, output_dir: Path):
    """Classify queries and compare performance metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df_main = df[df["target_party"].isin(MAIN_PARTIES)].copy()
    if df_main.empty:
        print(
            "No records with main target parties — skipping in-party/out-party analysis."
        )
        return

    df_main["has_inparty"] = df_main.apply(
        lambda r: r["target_party"] in r["retrieved_parties"], axis=1
    )
    df_main["retrieval_type"] = df_main["has_inparty"].map(
        {True: "In-Party Match", False: "Out-Party Noise"}
    )

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

    overall = metrics_df[metrics_df["condition"] == "Overall"].copy()
    if overall.empty:
        print("No overall metrics to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    colors_mae = [
        "#2ecc71" if rtype == "In-Party Match" else "#e74c3c"
        for rtype in overall["retrieval_type"]
    ]
    bars = ax.bar(
        overall["retrieval_type"],
        overall["MAE"],
        color=colors_mae,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title("Mean Absolute Error (MAE)", fontsize=13, fontweight="bold")
    ax.set_ylabel("MAE (lower is better)", fontsize=11)
    ax.set_ylim(0, max(overall["MAE"].dropna().max() * 1.2, 0.1))
    for bar, row in zip(bars, overall.itertuples()):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.02,
                f"{height:.3f}\n(n={row.n})",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    sns.despine(ax=ax)

    ax = axes[1]
    colors_pearson = [
        "#2ecc71" if rtype == "In-Party Match" else "#e74c3c"
        for rtype in overall["retrieval_type"]
    ]
    bars = ax.bar(
        overall["retrieval_type"],
        overall["Pearson_r"],
        color=colors_pearson,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_title("Pearson Correlation (r)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Pearson r (higher is better)", fontsize=11)
    ymin = overall["Pearson_r"].dropna()
    ax.set_ylim(
        min(ymin.min() * 1.2 if not ymin.empty else -0.5, -0.1),
        max(ymin.max() * 1.2 if not ymin.empty else 0.1, 0.5),
    )
    for bar, row in zip(bars, overall.itertuples()):
        height = bar.get_height()
        if not np.isnan(height):
            offset = 0.02 if height >= 0 else -0.05
            va = "bottom" if height >= 0 else "top"
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


def _save(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(base_dir: Path = BASE_DIR, output_dir: Path = OUTPUT_DIR):
    print("=" * 80)
    print("Retrieval Composition & Party Alignment Analysis")
    print("=" * 80)

    df = load_rag_logs(base_dir)

    print(f"\nFocusing on target parties: {MAIN_PARTIES}")
    print("Target party distribution in loaded records:")
    print(df["target_party"].value_counts().to_string())

    analyze_party_distribution(df, output_dir)
    analyze_inparty_outparty(df, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze retrieval party alignment.")
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
