"""plot_config_figures.py

Figures for the thesis section 'Retrieval configurations' (RQ1b, part 1).
Reads the paired chapter-02 CSV and the tables written by
``retrieval_config_analysis.py``; writes PDF + PNG to
results/thesis/03_retrieval_configs/:

  fig_config_heatmap  MAE per model (rows) x condition (columns), no-RAG first
  fig_config_effects  per-model effect of each embedder / strategy / hybrid
                      (deviation from the model mean), one dot per model,
                      bands = run-to-run SD with fixed chunks and for HyDE reruns
  fig_config_m3       M3 coefficients (abs_error): design factors next to one
                      doubling of model size and the family terms
  fig_config_k        MAE against k (e5/simple), no-RAG as horizontal lines;
                      left all available posts, right posts common to all k

Usage:  python src/plot_config_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_metrics as em  # noqa: E402
from plot_baseline_figures import (  # noqa: E402  (shared style + rcParams)
    FAMILIES, FAMILY_COLOR, FAMILY_LABEL, FAMILY_MARKER, GRID, INK, INK_2, INK_3, short_name,
)

OUT_DIR = em.THESIS_ROOT / "03_retrieval_configs"
TARGET = "ches_lrgen"
PAIRED_CSV = em.THESIS_ROOT / "02_rag_vs_no_rag" / f"rag_vs_no_rag_{TARGET}.csv"
BASELINE_CSV = em.THESIS_ROOT / "01_baseline_no_rag" / f"point_metrics_no_rag_{TARGET}.csv"
EMBEDDERS = ["bge", "e5", "jina", "qwen3"]
STRATEGIES = ["simple", "hyde", "twostage"]
CONDITIONS = [f"{e}/{s}" for e in EMBEDDERS for s in STRATEGIES]
CONDITIONS = CONDITIONS[:3] + ["bge/simple_hybrid", "bge/twostage_hybrid"] + CONDITIONS[3:]
STRAT_LABEL = {"simple": "simple", "hyde": "HyDE", "twostage": "two-stage",
               "simple_hybrid": "simple+hyb", "twostage_hybrid": "two-st.+hyb"}
# single-hue sequential ramp (light -> dark), low MAE = light
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f7fb", "#a9c6ea", "#2a78d6", "#123d73"])


def _save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  → {OUT_DIR / name}.pdf / .png")


def model_order(paired: pd.DataFrame):
    out = []
    for fam in FAMILIES:
        out += (paired[paired["Family"] == fam].drop_duplicates("Model")
                .sort_values("Params_B")["Model"].tolist())
    return out


def noise_sd() -> dict:
    """Run-to-run SDs: fixed context (LLM sampling only) and HyDE (pooled)."""
    s = pd.read_csv(OUT_DIR / "noise_floor_summary.csv").set_index("Kind")["SD_run_to_run"]
    return {"fixed": float(s["fixed context"]), "hyde": float(s["HyDE (pooled)"])}


# ---------------------------------------------------------------------------

def plot_heatmap(paired: pd.DataFrame):
    models = model_order(paired)
    mat = paired.pivot(index="Model", columns="Condition", values="MAE_rag").reindex(
        index=models, columns=CONDITIONS)
    # full no-RAG baseline (all posts, as in tab:baseline-metrics)
    base = pd.read_csv(BASELINE_CSV).set_index("Model")["MAE"].reindex(models)
    full = pd.concat([base.rename("no_rag"), mat], axis=1)
    vmin, vmax = np.nanmin(full.values), np.nanmax(full.values)

    fig, ax = plt.subplots(figsize=(6.3, 3.9))
    gap = 0.6
    xs = [0] + [1 + gap + i for i in range(len(CONDITIONS))]
    for j, (col, x) in enumerate(zip(full.columns, xs)):
        for i, m in enumerate(models):
            v = full.loc[m, col]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((x - .5, i - .5), 1, 1, facecolor="white",
                                           edgecolor=GRID, hatch="////", linewidth=0))
                continue
            c = SEQ((v - vmin) / (vmax - vmin))
            ax.add_patch(plt.Rectangle((x - .48, i - .48), .96, .96, facecolor=c, linewidth=0))
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            ax.text(x, i, f"{v:.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if lum < 0.55 else INK)
    ax.set_xlim(-0.6, xs[-1] + 0.6)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_name(m) for m in models], fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(["no-RAG"] + [STRAT_LABEL[c.split("/")[1]] for c in CONDITIONS],
                       rotation=60, ha="right", fontsize=6.5)
    # embedder group labels above the columns
    start = 1
    for e in EMBEDDERS:
        n = sum(c.startswith(e + "/") for c in CONDITIONS)
        x0, x1 = xs[start], xs[start + n - 1]
        ax.plot([x0 - .4, x1 + .4], [-0.75, -0.75], color=INK_3, linewidth=0.6, clip_on=False)
        ax.text((x0 + x1) / 2, -0.9, e, ha="center", va="bottom", fontsize=7.5, color=INK)
        start += n
    fams = [em._meta_for(m)["family"] for m in models]
    for k in range(1, len(fams)):  # family separators
        if fams[k] != fams[k - 1]:
            ax.axhline(k - 0.5, color="white", linewidth=2.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(vmin, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("MAE (lower = better)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    _save(fig, "fig_config_heatmap")


def plot_effects(effects: pd.DataFrame, sd: dict):
    panels = [("embedder", EMBEDDERS, "Embedder"), ("strategy", STRATEGIES, "Retrieval strategy"),
              ("hybrid", ["simple_hybrid − simple", "twostage_hybrid − twostage"], "Hybrid (bge)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.8), sharey=True,
                             gridspec_kw={"width_ratios": [4, 3, 2]})
    rng = np.random.default_rng(1)
    for ax, (factor, levels, title) in zip(axes, panels):
        sub = effects[effects["Factor"] == factor]
        ax.axhspan(-sd["hyde"], sd["hyde"], color=GRID, alpha=0.45, linewidth=0, zorder=0)
        ax.axhspan(-sd["fixed"], sd["fixed"], color="#cfcdc4", alpha=0.8, linewidth=0, zorder=0)
        ax.axhline(0, color=INK_3, linewidth=0.7, zorder=1)
        for i, lev in enumerate(levels):
            d = sub[sub["Level"] == lev]
            for fam in FAMILIES:
                f = d[d["Family"] == fam]
                jit = rng.uniform(-0.18, 0.18, len(f))
                ax.scatter(i + jit, f["Effect"], s=14, marker=FAMILY_MARKER[fam],
                           facecolor=FAMILY_COLOR[fam], edgecolor="white", linewidth=0.5, zorder=3)
            ax.plot([i - 0.28, i + 0.28], [d["Effect"].mean()] * 2, color=INK, linewidth=1.4, zorder=4)
        lab = {"simple_hybrid − simple": "simple", "twostage_hybrid − twostage": "two-stage"}
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([lab.get(l, STRAT_LABEL.get(l, l)) for l in levels], fontsize=7.5)
        ax.set_xlim(-0.6, len(levels) - 0.4)
        ax.set_title(title, fontsize=8.5, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ΔMAE vs. model mean\n(negative = better)", fontsize=7.5)
    axes[2].set_ylabel("")
    axes[2].text(0.5, -0.2, "ΔMAE = hybrid − dense", transform=axes[2].transAxes,
                 ha="center", fontsize=6.5, color=INK_2)
    handles = [Line2D([], [], marker=FAMILY_MARKER[f], linestyle="", markerfacecolor=FAMILY_COLOR[f],
                      markeredgecolor="white", markersize=5, label=FAMILY_LABEL[f]) for f in FAMILIES]
    handles += [Line2D([], [], color=INK, linewidth=1.4, label="mean over models"),
                plt.Rectangle((0, 0), 1, 1, color="#cfcdc4", alpha=0.8,
                              label=f"± run-to-run SD, same chunks ({sd['fixed']:.3f})"),
                plt.Rectangle((0, 0), 1, 1, color=GRID, alpha=0.45,
                              label=f"± run-to-run SD, HyDE reruns ({sd['hyde']:.3f})")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6.8, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, "fig_config_effects")


def plot_m3(coef: pd.DataFrame):
    c = coef[(coef["Outcome"] == "abs_error") & (coef["Spec"] == "M3_rag_design")].set_index("Term")

    def find(sub):
        return [t for t in c.index if sub in t]

    rows = [("Model size ×2", find("size_c")[0], "size")]
    for t in find("[T.Mistral]") + find("[T.Qwen]"):
        lab = "Ministral" if "Mistral" in t else "Qwen"
        rows.append((f"{lab} (vs Llama)", t, "family"))
    for t in sorted(find("embedding_model")):
        rows.append((f"{t.split('[T.')[1].rstrip(']')} (vs bge)", t, "design"))
    for t in sorted(find("strategy")):
        lab = {"hyde": "HyDE", "twostage": "two-stage"}[t.split("[T.")[1].rstrip("]")]
        rows.append((f"{lab} (vs simple)", t, "design"))
    rows.append(("hybrid (vs dense)", "hybrid", "design"))
    color = {"size": INK, "family": INK_2, "design": "#2a78d6"}

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axvline(0, color=INK_3, linewidth=0.7)
    for i, (lab, term, grp) in enumerate(rows):
        r = c.loc[term]
        ax.plot([r["CI_lo"], r["CI_hi"]], [i, i], color=color[grp], linewidth=1.2)
        ax.scatter([r["Estimate"]], [i], s=18, color=color[grp], zorder=3,
                   facecolor=color[grp] if r["p"] < 0.05 else "white", linewidth=1)
        ax.text(max(r["CI_hi"], 0) + 0.01, i, f"{r['Estimate']:+.3f}", va="center", fontsize=6.5,
                color=INK_2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for k in (0.5, 2.5):
        ax.axhline(k, color=GRID, linewidth=0.6)
    ax.set_xlabel("Change in absolute error (scale points), 95% CI")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    _save(fig, "fig_config_m3")


def _k_panel(ax, d: pd.DataFrame, mark_incomplete: bool):
    for model, g in d.groupby("Model"):
        fam = em._meta_for(model)["family"]
        col = FAMILY_COLOR[fam]
        base = g[g["K"] == 0].iloc[0]
        ax.axhline(base["MAE"], color=col, linewidth=0.9, linestyle="--", alpha=0.9)
        r = g[g["K"] > 0].sort_values("K")
        ax.fill_between(r["K"], r["MAE_lo"], r["MAE_hi"], color=col, alpha=0.12, linewidth=0)
        ax.plot(r["K"], r["MAE"], color=col, linewidth=1.6)
        full = (r["N"] >= 700) | (not mark_incomplete)
        ax.scatter(r.loc[full, "K"], r.loc[full, "MAE"], s=22, marker=FAMILY_MARKER[fam],
                   color=col, edgecolor="white", linewidth=0.6, zorder=3)
        ax.scatter(r.loc[~full, "K"], r.loc[~full, "MAE"], s=22, marker=FAMILY_MARKER[fam],
                   facecolor="white", edgecolor=col, linewidth=1, zorder=3)
        for _, rr in r[~full].iterrows():
            ax.annotate(f"N = {int(rr['N'])}", (rr["K"], rr["MAE"]), textcoords="offset points",
                        xytext=(-4, -11), ha="right", fontsize=6, color=INK_2)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 5, 10, 20])
    ax.set_xticklabels(["1", "3", "5", "10", "20"])
    ax.minorticks_off()
    ax.set_xlim(0.8, 25)
    ax.set_xlabel("k (retrieved chunks, e5/simple)")
    ax.grid(color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def plot_k(k: pd.DataFrame):
    """Left: all available posts per k. Right: posts shared by every k of a
    model (incl. no-RAG), so the baseline line is on the same posts."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, subset, mark in ((axes[0], "all_available", True), (axes[1], "common_texts", False)):
        d = k[k["Subset"] == subset]
        _k_panel(ax, d, mark)
        if subset == "all_available":
            ax.set_title("All available posts", fontsize=8.5, color=INK)
        else:
            ns = ", ".join(f"{short_name(m)}: {int(g['N'].iloc[0])}"
                           for m, g in d.groupby("Model"))
            ax.set_title(f"Common posts ({ns})", fontsize=8.5, color=INK)
    axes[0].set_ylabel("MAE (95% CI)")
    models = k["Model"].unique()
    handles = [Line2D([], [], color=FAMILY_COLOR[em._meta_for(m)["family"]], linewidth=1.6,
                      marker=FAMILY_MARKER[em._meta_for(m)["family"]], markersize=4,
                      label=short_name(m)) for m in models]
    handles += [Line2D([], [], color=INK_2, linewidth=0.9, linestyle="--", label="no-RAG (same model, same posts)"),
                Line2D([], [], marker="s", linestyle="", markerfacecolor="white",
                       markeredgecolor=INK_2, markersize=4, label="incomplete run")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "fig_config_k")


def main():
    paired = pd.read_csv(PAIRED_CSV)
    print("Plotting retrieval-config figures …")
    plot_heatmap(paired)
    plot_effects(pd.read_csv(OUT_DIR / "config_effects.csv"), noise_sd())
    m3 = OUT_DIR / "m3_coefficients.csv"
    if m3.exists():
        plot_m3(pd.read_csv(m3))
    plot_k(pd.read_csv(OUT_DIR / "k_ablation.csv"))


if __name__ == "__main__":
    main()
