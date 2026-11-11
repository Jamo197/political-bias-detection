"""plot_baseline_figures.py

Figures for the thesis section 'Baseline performance (No-RAG)'.

Reads the point metrics written by ``evaluate_metrics.py --baseline`` and the
post-level no_rag predictions from the logs (same filters: party_label_* logs,
no k-ablation files, target = exact CHES lrgen). Writes PDF + PNG to
results/thesis/01_baseline_no_rag/:

  fig_baseline_calibration  small multiples, one panel per model (families in rows):
                            box per party, jittered posts, identity line, fitted
                            calibration line and its fixed point c = a / (1 - b)
  fig_baseline_spread       distribution of predicted scores per model vs. target
  fig_baseline_anchor       all ten calibration lines in one panel, with identity,
                            sample mean of the target and scale midpoint 4
  fig_baseline_mae          MAE dot plot (95% CI) per model, grouped by family,
                            with constant-prediction references (median, mean, 4)
  fig_baseline_size         MAE (95% CI) against log2 parameters, one line per
                            family, reference = constant prediction at the median

Usage:  python src/plot_baseline_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_metrics as em  # noqa: E402

OUT_DIR = em.THESIS_ROOT / "01_baseline_no_rag"
TARGET = "ches_lrgen"
METRICS_CSV = OUT_DIR / f"point_metrics_no_rag_{TARGET}.csv"

FAMILIES = ["Llama", "Qwen", "Mistral"]  # same order as the thesis tables
FAMILY_LABEL = {"Llama": "Llama", "Qwen": "Qwen", "Mistral": "Ministral"}  # thesis naming
FAMILY_COLOR = {"Llama": "#2a78d6", "Mistral": "#eb6834", "Qwen": "#1baf7a"}
FAMILY_MARKER = {"Llama": "o", "Mistral": "s", "Qwen": "D"}
INK, INK_2, INK_3, GRID = "#1f1f1e", "#5b5a55", "#8a8980", "#e6e5e0"
SCALE = (1, 7)
VIEW = (-0.1, 8.1)  # some predictions fall outside 1-7 (Llama 3B down to 0, Ministral 8B up to 7.9)
SEED = 42
# Tick-label alignment for close party positions (Grüne/SPD, FDP/Union).
LABEL_HA = {"Greens": "right", "SPD": "left", "FDP": "right"}

PARTY_SHORT = {
    "DIE LINKE": "DIE LINKE", "BÜNDNIS 90/DIE GRÜNEN": "Greens", "SPD": "SPD",
    "FDP": "FDP", "CDU": "CDU", "CDU/CSU": "CDU", "CSU": "CSU", "AfD": "AfD",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": INK_3,
    "axes.labelcolor": INK,
    "axes.titlesize": 9,
    "axes.linewidth": 0.6,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "savefig.bbox": "tight",
})


def short_name(model: str) -> str:
    meta = em._meta_for(model)
    return f"{FAMILY_LABEL.get(meta['family'], meta['family'])} {int(meta['params_b'])}B"


def load_data():
    metrics = pd.read_csv(METRICS_CSV)
    df = em.load_logs_from_directory(
        em.BASE_DIR, retrieval_modes=["no_rag"], exclude_file_patterns=("k-ablation",)
    )
    posts = em.prepare_target_frame(df, TARGET)
    posts["party_short"] = posts["party"].map(PARTY_SHORT)
    metrics["short"] = metrics["Model"].map(short_name)
    metrics["c"] = metrics["Calibration_intercept"] / (1 - metrics["Calibration_slope"])
    return metrics, posts


def target_stats(posts: pd.DataFrame):
    t = posts.drop_duplicates("text_index")["true"]
    return t, float(t.mean()), float(t.median())


def party_positions(posts: pd.DataFrame) -> pd.Series:
    """target value -> party label (CDU includes posts labelled CDU/CSU)."""
    return posts.groupby("true")["party_short"].first().sort_index()


def models_by_family(metrics: pd.DataFrame):
    return {
        fam: metrics[metrics["Family"] == fam].sort_values("Params_B")
        for fam in FAMILIES
    }


def _shade_outside(ax, axis="y"):
    span = ax.axhspan if axis == "y" else ax.axvspan
    span(VIEW[0], SCALE[0], color=GRID, alpha=0.6, linewidth=0, zorder=0)
    span(SCALE[1], VIEW[1], color=GRID, alpha=0.6, linewidth=0, zorder=0)


def _style_scale_axes(ax, positions: pd.Series, shade=True):
    ax.set_xlim(1.3, 7.0)
    ax.set_ylim(*VIEW)
    ax.set_yticks(range(1, 8))
    ax.set_xticks(positions.index)
    ax.set_xticklabels(positions.values, rotation=90, fontsize=7.5)
    for lab in ax.get_xticklabels():
        lab.set_horizontalalignment(LABEL_HA.get(lab.get_text(), "center"))
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    if shade:
        _shade_outside(ax, "y")


def _save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  → {OUT_DIR / name}.pdf / .png")


# ---------------------------------------------------------------------------
# fig:baseline-calibration
# ---------------------------------------------------------------------------


def plot_calibration(metrics, posts):
    positions = party_positions(posts)
    fams = models_by_family(metrics)
    rng = np.random.default_rng(SEED)
    fig, axes = plt.subplots(3, 4, figsize=(6.3, 6.2), sharex=True, sharey=True)
    x_line = np.array([1.3, 7.0])

    for r, fam in enumerate(FAMILIES):
        rows = fams[fam]
        color = FAMILY_COLOR[fam]
        for col in range(4):
            ax = axes[r, col]
            if col >= len(rows):
                ax.axis("off")
                continue
            m = rows.iloc[col]
            sub = posts[posts["model"] == m["Model"]]
            groups = [sub.loc[sub["true"] == x, "pred"].to_numpy() for x in positions.index]
            jitter = rng.uniform(-0.07, 0.07, len(sub))
            ax.scatter(sub["true"] + jitter, sub["pred"], s=2, color=color,
                       alpha=0.18, linewidths=0, rasterized=True, zorder=2)
            ax.boxplot(
                groups, positions=positions.index, widths=0.18, showfliers=False,
                patch_artist=True, manage_ticks=False, zorder=3,
                boxprops=dict(facecolor="none", edgecolor=INK, linewidth=0.7),
                medianprops=dict(color=INK, linewidth=1.0),
                whiskerprops=dict(color=INK, linewidth=0.6),
                capprops=dict(color=INK, linewidth=0.6),
            )
            ax.plot(x_line, x_line, color=INK_3, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
            a, b, c = m["Calibration_intercept"], m["Calibration_slope"], m["c"]
            ax.plot(x_line, a + b * x_line, color=color, linewidth=1.6, zorder=4)
            if SCALE[0] <= c <= SCALE[1]:
                ax.scatter([c], [c], s=22, color=color, edgecolors="white",
                           linewidths=1.0, zorder=5)
            ax.set_title(m["short"], loc="left", color=INK, pad=12, fontsize=9.5)
            ax.text(0, 1.02, f"MAE {m['MAE']:.2f}  β {b:.2f}  c {c:.2f}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=7.2, color=INK_2)
            _style_scale_axes(ax, positions)
            ax.tick_params(labelbottom=r == 2)

    fig.supxlabel("Target (CHES 2024 lrgen of the author's party, 1–7)", fontsize=9, color=INK)
    fig.supylabel("Predicted score (1–7)", fontsize=9, color=INK)
    handles = [
        Line2D([], [], color=INK_3, linestyle=(0, (3, 2)), linewidth=0.8, label="Identity"),
        Line2D([], [], color=INK_2, linewidth=1.6, label="Calibration line"),
        Line2D([], [], marker="o", linestyle="", color=INK_2, markeredgecolor="white",
               markersize=5, label="c = α / (1 − β)"),
    ]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.6, label="Outside 1–7 scale"))
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    axes[0, 3].legend(handles=handles, loc="center left", fontsize=7.5)
    _save(fig, "fig_baseline_calibration")


# ---------------------------------------------------------------------------
# fig:baseline-spread
# ---------------------------------------------------------------------------


def plot_spread(metrics, posts):
    order = [m for fam in FAMILIES for m in models_by_family(metrics)[fam]["Model"]]
    target, _, _ = target_stats(posts)
    labels = ["Target"] + [short_name(m) for m in order]
    data = [target.to_numpy()] + [posts.loc[posts["model"] == m, "pred"].to_numpy() for m in order]
    colors = [INK_3] + [FAMILY_COLOR[em._meta_for(m)["family"]] for m in order]

    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    ypos = np.arange(len(labels))[::-1]
    parts = ax.violinplot(data, positions=ypos, vert=False, widths=0.85,
                          showextrema=False, bw_method=0.15, points=400)
    for body, col in zip(parts["bodies"], colors):
        body.set_facecolor(col)
        body.set_edgecolor(col)
        body.set_alpha(0.55)
        body.set_linewidth(0.6)
    for y, d in zip(ypos, data):
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        ax.plot([q1, q3], [y, y], color=INK, linewidth=2.0, solid_capstyle="butt")
        ax.scatter([med], [y], s=10, color="white", edgecolors=INK, linewidths=0.7, zorder=3)
    ax.axhline(ypos[0] - 0.5, color=GRID, linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlim(*VIEW)
    ax.set_xticks(range(0, 9))
    _shade_outside(ax, "x")
    for x in (SCALE[0] - 0.55, SCALE[1] + 0.55):
        ax.text(x, ypos[-1] - 0.45, "outside\nscale", ha="center", va="bottom", fontsize=6.5, color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Score (1 = extreme left, 7 = extreme right)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    _save(fig, "fig_baseline_spread")


# ---------------------------------------------------------------------------
# fig:baseline-anchor
# ---------------------------------------------------------------------------


def plot_anchor(metrics, posts):
    positions = party_positions(posts)
    _, t_mean, _ = target_stats(posts)
    lo, hi = positions.index.min(), positions.index.max()
    x = np.array([lo, hi])

    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot([1, 7], [1, 7], color=INK_3, linewidth=0.8, linestyle=(0, (3, 2)))
    # Vertical lines: a calibration line that crosses the identity to the right of
    # the target mean has a positive mean signed error (e_bar = (1 - b)(c - mean)).
    ax.axvline(4.0, color=INK_3, linewidth=0.6, linestyle=":")
    ax.axvline(t_mean, color=INK_2, linewidth=0.6, linestyle="-.")
    ends = []

    for fam in FAMILIES:
        rows = models_by_family(metrics)[fam]
        n = len(rows)
        for i, (_, m) in enumerate(rows.iterrows()):
            alpha_line = 0.45 + 0.55 * (i + 1) / n  # larger model = more opaque
            y = m["Calibration_intercept"] + m["Calibration_slope"] * x
            ax.plot(x, y, color=FAMILY_COLOR[fam], linewidth=1.6, alpha=alpha_line)
            ax.scatter([x[1]], [y[1]], marker=FAMILY_MARKER[fam], s=14,
                       color=FAMILY_COLOR[fam], alpha=alpha_line, zorder=3)
            c = m["c"]
            if 1 <= c <= 7:
                ax.scatter([c], [c], s=20, color=FAMILY_COLOR[fam], edgecolors="white",
                           linewidths=0.8, alpha=alpha_line, zorder=4)
            ends.append([y[1], m["short"], y[1]])

    # direct labels at the right end, pushed apart to avoid overlaps
    ends.sort()
    gap = 0.2
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    shift = max(0.0, ends[-1][0] - 6.9)
    for y_lab, name, y_end in ends:
        ax.plot([hi + 0.06, hi + 0.2], [y_end, y_lab - shift], color=INK_3, linewidth=0.5)
        ax.text(hi + 0.24, y_lab - shift, name, fontsize=7, color=INK_2, va="center")

    _style_scale_axes(ax, positions, shade=False)
    ax.set_ylim(1, 7)
    ax.set_xlim(1.3, 7.75)
    # numeric ticks at the bottom, party positions on a secondary axis at the top
    top = ax.secondary_xaxis("top")
    top.set_xticks(positions.index)
    top.set_xticklabels(positions.values, rotation=90, fontsize=7)
    top.tick_params(length=2, colors=INK_2)
    top.spines["top"].set_visible(False)
    ax.set_xticks(range(2, 8))
    ax.set_xticklabels([str(t) for t in range(2, 8)], rotation=0, fontsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_xlabel("Target (CHES 2024 lrgen, 1–7)")
    ax.set_ylabel("Fitted prediction α + β·target")
    handles = [Line2D([], [], color=FAMILY_COLOR[f], marker=FAMILY_MARKER[f], linewidth=1.6,
                      markersize=4, label=FAMILY_LABEL[f]) for f in FAMILIES]
    handles += [
        Line2D([], [], color=INK_3, linewidth=0.8, linestyle=(0, (3, 2)), label="Identity"),
        Line2D([], [], marker="o", linestyle="", color=INK_2, markeredgecolor="white",
               markersize=5, label="c = α / (1 − β)"),
        Line2D([], [], color=INK_2, linewidth=0.6, linestyle="-.", label=f"Target mean {t_mean:.2f}"),
        Line2D([], [], color=INK_3, linewidth=0.6, linestyle=":", label="Scale midpoint 4"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_baseline_anchor")


# ---------------------------------------------------------------------------
# fig:baseline-size
# ---------------------------------------------------------------------------


def plot_size(metrics, posts):
    target, _, t_median = target_stats(posts)
    ref_mae = float((target - t_median).abs().mean())

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.axhline(ref_mae, color=INK_3, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.text(np.log2(2.6), ref_mae - 0.02,
            f"constant prediction at\ntarget median {t_median:.2f}: MAE {ref_mae:.2f}",
            fontsize=6.5, color=INK_2, va="top")
    offsets = {"Llama": -0.05, "Mistral": 0.0, "Qwen": 0.05}
    for fam in FAMILIES:
        rows = models_by_family(metrics)[fam]
        xs = np.log2(rows["Params_B"].astype(float)) + offsets[fam]
        col = FAMILY_COLOR[fam]
        ax.errorbar(xs, rows["MAE"],
                    yerr=[rows["MAE"] - rows["MAE_lo"], rows["MAE_hi"] - rows["MAE"]],
                    color=col, marker=FAMILY_MARKER[fam], markersize=5, linewidth=1.6,
                    elinewidth=1.0, capsize=2, markeredgecolor="white", markeredgewidth=0.8)
        ax.annotate(FAMILY_LABEL[fam], (xs.iloc[-1], rows["MAE"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=7.5, color=INK, va="center")
    ticks = [4, 8, 16, 32, 64]
    ax.set_xticks(np.log2(ticks))
    ax.set_xticklabels([f"{t}B" for t in ticks])
    ax.set_xlim(np.log2(2.5), np.log2(110))
    ax.set_ylim(0.6, 2.0)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Parameters (log₂ scale)")
    ax.set_ylabel("MAE (scale points, 95% CI)")
    fig.tight_layout()
    _save(fig, "fig_baseline_size")


# ---------------------------------------------------------------------------
# fig:baseline-mae
# ---------------------------------------------------------------------------


def constant_references(posts: pd.DataFrame):
    """MAE of a constant prediction at the target median, target mean and the
    scale midpoint 4 (label, constant, MAE)."""
    target, t_mean, t_median = target_stats(posts)
    return [
        (f"median {t_median:.2f}", t_median, float((target - t_median).abs().mean())),
        (f"mean {t_mean:.2f}", t_mean, float((target - t_mean).abs().mean())),
        ("midpoint 4", 4.0, float((target - 4.0).abs().mean())),
    ]


def plot_mae(metrics, posts):
    refs = constant_references(posts)
    best_ref = min(r[2] for r in refs)
    fams = models_by_family(metrics)

    rows, y, ys, fam_mid = [], 0.0, [], {}
    for fam in FAMILIES:
        start = y
        for _, m in fams[fam].iterrows():
            rows.append(m)
            ys.append(y)
            y -= 1
        fam_mid[fam] = (start + y + 1) / 2
        y -= 0.6  # gap between families

    fig, ax = plt.subplots(figsize=(5.0, 4.1))
    ref_styles = [(0, (3, 2)), "-.", ":"]
    for i, ((label, _, mae), ls) in enumerate(zip(refs, ref_styles)):
        ax.axvline(mae, color=INK_3, linewidth=0.8, linestyle=ls, zorder=1)
        ax.text(mae, ys[0] + 0.65 + 0.6 * i, f"{label}: {mae:.2f}", ha="center",
                va="bottom", fontsize=7, color=INK_2,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

    for yy, m in zip(ys, rows):
        col = FAMILY_COLOR[m["Family"]]
        beats = m["MAE_hi"] < best_ref
        ax.plot([m["MAE_lo"], m["MAE_hi"]], [yy, yy], color=col, linewidth=1.6,
                solid_capstyle="round", zorder=2)
        ax.scatter([m["MAE"]], [yy], marker=FAMILY_MARKER[m["Family"]], s=34, zorder=3,
                   color=col if beats else "white", edgecolors=col, linewidths=1.4)
        ax.text(1.01, yy, f"{m['MAE']:.2f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=6.8, color=INK_2)

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{int(m['Params_B'])}B" for m in rows])
    for fam, mid in fam_mid.items():
        ax.text(-0.13, mid, FAMILY_LABEL[fam], transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=8, color=INK)
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 2.5)
    ax.set_xlim(0.6, 2.05)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("MAE (scale points, 95% CI)")
    handles = [
        Line2D([], [], marker="o", linestyle="", color=INK_2, markersize=5,
               label=f"CI below best constant reference ({best_ref:.2f})"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="white",
               markeredgecolor=INK_2, markersize=5, label="CI overlaps or above"),
    ]
    for (label, _, mae), ls in zip(refs, ref_styles):
        handles.append(Line2D([], [], color=INK_3, linewidth=0.8, linestyle=ls,
                              label=f"Constant at {label}"))
    ax.text(1.01, ys[0] + 0.7, "MAE", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=6.8, color=INK)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              ncol=2, fontsize=7)
    fig.tight_layout()
    _save(fig, "fig_baseline_mae")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics, posts = load_data()
    print("Plotting baseline figures …")
    plot_calibration(metrics, posts)
    plot_spread(metrics, posts)
    plot_anchor(metrics, posts)
    plot_size(metrics, posts)
    plot_mae(metrics, posts)


if __name__ == "__main__":
    main()
