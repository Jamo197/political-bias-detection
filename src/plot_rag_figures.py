"""plot_rag_figures.py

Figure for the thesis section 'RAG vs. no-RAG' (RQ1b). Reads the paired results
written by ``evaluate_metrics.py --rag-vs-norag`` and writes PDF + PNG to
results/thesis/02_rag_vs_no_rag/:

  fig_rag_dmae   paired ΔMAE (RAG − no-RAG, 95% CI) per model; one sub-row per
                 RAG condition, filled marker = Holm-significant (Wilcoxon).
                 Right margin: count of significantly better / worse conditions.

Usage:  python src/plot_rag_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_metrics as em  # noqa: E402
from plot_baseline_figures import (  # noqa: E402  (shared style + rcParams)
    FAMILIES, FAMILY_COLOR, FAMILY_LABEL, FAMILY_MARKER, GRID, INK, INK_2, INK_3, short_name,
)

OUT_DIR = em.THESIS_ROOT / "02_rag_vs_no_rag"
TARGET = "ches_lrgen"
DELTA_CSV = OUT_DIR / f"rag_vs_no_rag_{TARGET}.csv"

# Cells excluded from the figure (failed retrieval). Empty since the re-run of
# Ministral-3-8B bge/twostage and bge/twostage_hybrid on 10 November 2026.
INVALID_CELLS: set = set()


def _save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  → {OUT_DIR / name}.pdf / .png")


def plot_dmae(delta: pd.DataFrame):
    conditions = sorted(delta["Condition"].unique())
    n_cond = len(conditions)
    band = 0.8  # vertical extent used by a model's sub-rows
    offset = {c: band / 2 - i * band / (n_cond - 1) for i, c in enumerate(conditions)}

    models, ys, fam_mid, y = [], [], {}, 0.0
    for fam in FAMILIES:
        fam_models = (delta[delta["Family"] == fam]
                      .drop_duplicates("Model").sort_values("Params_B")["Model"].tolist())
        start = y
        for m in fam_models:
            models.append(m)
            ys.append(y)
            y -= 1
        fam_mid[fam] = (start + y + 1) / 2
        y -= 0.5

    fig, ax = plt.subplots(figsize=(5.4, 5.6))
    ax.axvline(0, color=INK_3, linewidth=0.8, zorder=1)
    for m, y0 in zip(models, ys):
        sub = delta[delta["Model"] == m]
        fam = sub["Family"].iloc[0]
        col = FAMILY_COLOR[fam]
        ax.axhspan(y0 - band / 2 - 0.08, y0 + band / 2 + 0.08, color=GRID, alpha=0.35,
                   linewidth=0, zorder=0)
        for _, r in sub.iterrows():
            yy = y0 + offset[r["Condition"]]
            ax.plot([r["dMAE_lo"], r["dMAE_hi"]], [yy, yy], color=col, linewidth=0.8,
                    alpha=0.8, zorder=2)
            ax.scatter([r["dMAE"]], [yy], s=9, marker=FAMILY_MARKER[fam], zorder=3,
                       color=col if r["significant"] else "white", edgecolors=col,
                       linewidths=0.8)
        better = int((sub["direction"] == "RAG better").sum())
        worse = int((sub["direction"] == "RAG worse").sum())
        ax.text(1.02, y0, f"{better:>2} / {worse:>2}  (of {len(sub)})",
                transform=ax.get_yaxis_transform(), va="center", fontsize=6.8, color=INK_2)

    ax.text(1.02, ys[0] + 0.75, "better / worse", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=6.8, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([short_name(m).split()[1] for m in models])
    for fam, mid in fam_mid.items():
        ax.text(-0.11, mid, FAMILY_LABEL[fam], transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=8, color=INK)
    ax.set_ylim(ys[-1] - 0.65, ys[0] + 1.0)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("ΔMAE = MAE(RAG) − MAE(no-RAG), 95% CI   (negative = RAG better)")
    ax.text(0.0, ys[0] + 0.62, "← RAG better   RAG worse →", ha="center", va="bottom",
            fontsize=6.8, color=INK_2)
    handles = [
        Line2D([], [], marker="o", linestyle="", color=INK_2, markersize=4,
               label="Holm-significant (Wilcoxon, p < 0.05)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=INK_2, markersize=4, label="not significant"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=2, fontsize=6.8)
    fig.tight_layout()
    _save(fig, "fig_rag_dmae")


def main():
    delta = pd.read_csv(DELTA_CSV)
    bad = delta.apply(lambda r: (r["Model"], r["Condition"]) in INVALID_CELLS, axis=1)
    delta = delta[~bad]
    print(f"Excluded {int(bad.sum())} invalid cells (failed retrieval).")
    print("Plotting RAG vs no-RAG figures …")
    plot_dmae(delta)


if __name__ == "__main__":
    main()
