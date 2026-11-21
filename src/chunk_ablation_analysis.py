"""chunk_ablation_analysis.py

Evaluation of the chunk ablations in ``logs/chunk_ablation`` against the
matching main-matrix conditions (no-RAG and e5/simple, k=5, same model).
Target: CHES lrgen (1-7), as in the other thesis chapters.

* Random-chunk run (``e5pool/random5``, all 775 posts): compared with no-RAG and
  e5/simple on all posts (files ``chunk_ablation_*`` and ``fig_chunk_ablation_*``).
* R_top runs (``e5/rtop4plus``, only the ~100 judged posts of a model, of which
  only those with at least one chunk rated R_top >= 4 are predicted): evaluated
  on exactly those posts, with no-RAG, e5/simple and random5 restricted to the
  same posts (files ``chunk_ablation_rtop_*`` and ``fig_chunk_ablation_rtop_*``).
  The judged posts are a quadrant sample chosen by how e5/simple performed, so
  random5 on the same posts is the control against regression to the mean.

Writes to results/thesis/05_chunk_ablation/:
  chunk_ablation_metrics.csv     point metrics per model x condition (own N)
  chunk_ablation_paired.csv      paired deltas (ablation minus reference) with CIs
  chunk_ablation_party.csv       signed error per author party
  fig_chunk_ablation_mae.*       MAE per model and condition
  fig_chunk_ablation_delta.*     paired dMAE vs no-RAG and vs e5/simple
  fig_chunk_ablation_party.*     signed error per party and condition
  chunk_ablation.md              summary tables

Usage:  python src/chunk_ablation_analysis.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_metrics as em  # noqa: E402
from plot_baseline_figures import (
    GRID,
    INK,
    INK_2,
    INK_3,
)  # noqa: E402  (shared rcParams)

TARGET = "ches_lrgen"
OUT_DIR = em.THESIS_ROOT / "05_chunk_ablation"
ABL_DIR = Path("logs/chunk_ablation")
REF = ["no_rag", "e5/simple"]
COLORS = {"no_rag": "#8a8f98", "e5/simple": "#2a78d6"}
ABL_COLOR = "#d9622b"
ABL_LABEL = {
    "e5pool/random5": "5 random chunks",
    "e5/rtop4plus": "only R_top ≥ 4",
    "e5/rtoplt4": "only R_top < 4",
}
LABEL = {"no_rag": "no RAG", "e5/simple": "e5/simple (retrieved)", **ABL_LABEL}
PARTY_SHORT = {
    "BÜNDNIS 90/DIE GRÜNEN": "Greens",
    "DIE LINKE": "Linke",
    "CDU": "CDU",
    "CSU": "CSU",
    "CDU/CSU": "CDU/CSU",
    "SPD": "SPD",
    "FDP": "FDP",
    "AfD": "AfD",
}


def short(model: str) -> str:
    meta = em._meta_for(model)
    return f"{meta['family']} {int(meta['params_b'])}B"


def load() -> pd.DataFrame:
    abl = em.load_logs_from_directory(ABL_DIR)
    models = sorted(abl["model"].unique())
    ref = em.load_logs_from_directory(
        em.BASE_DIR, models=models, exclude_file_patterns=("k-ablation",)
    )
    ref = ref[ref["condition"].isin(REF)]
    df = pd.concat([ref, abl], ignore_index=True)
    return em.prepare_target_frame(df, TARGET)


def paired(df: pd.DataFrame, abl_cond: str, ref_cond: str) -> pd.DataFrame:
    """ΔMAE etc. of ``abl_cond`` minus ``ref_cond``, paired per post."""
    sub = df[df["condition"].isin([abl_cond, ref_cond])].copy()
    sub["condition"] = sub["condition"].replace(
        {ref_cond: "no_rag"}
    )  # reference plays the baseline role
    out = em.compute_pairwise_rag_delta(sub, target_col=TARGET)
    out["Reference"] = ref_cond
    out["Condition"] = out["Condition"].astype(str)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    all_abl = sorted(c for c in df["condition"].unique() if c not in REF)
    rtop_conds = [c for c in all_abl if "rtop" in c]
    abl_conds = [c for c in all_abl if c not in rtop_conds]
    print("Ablation conditions:", abl_conds, "| R_top conditions:", rtop_conds)
    df_rtop = df[df["condition"].isin(rtop_conds)]
    df = df[~df["condition"].isin(rtop_conds)]

    metrics = em.compute_evaluation_metrics(df, target_col=TARGET)
    metrics["Condition"] = metrics["Condition"].astype(str)
    metrics.to_csv(OUT_DIR / "chunk_ablation_metrics.csv", index=False)

    deltas = pd.concat(
        [paired(df, a, r) for a in abl_conds for r in REF], ignore_index=True
    )
    deltas.to_csv(OUT_DIR / "chunk_ablation_paired.csv", index=False)

    party = (
        df.assign(party_short=df["party_group"].map(lambda p: PARTY_SHORT.get(p, p)))
        .groupby(["model", "condition", "party_short"])
        .agg(
            n=("signed_error", "size"),
            true=("true", "mean"),
            mean_pred=("pred", "mean"),
            signed_error=("signed_error", "mean"),
            mae=("abs_error", "mean"),
        )
        .reset_index()
    )
    party.to_csv(OUT_DIR / "chunk_ablation_party.csv", index=False)

    models = sorted(
        df["model"].unique(),
        key=lambda m: (em._meta_for(m)["family"], em._meta_for(m)["params_b"]),
    )
    conds = REF + abl_conds

    # ---- fig 1: MAE per model x condition -----------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    w = 0.8 / len(conds)
    for j, c in enumerate(conds):
        for i, m in enumerate(models):
            r = metrics[(metrics["Model"] == m) & (metrics["Condition"] == c)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i + (j - (len(conds) - 1) / 2) * w
            col = COLORS.get(c, ABL_COLOR)
            ax.bar(
                x,
                r["MAE"],
                w * 0.9,
                color=col,
                label=LABEL.get(c, c) if i == 0 else None,
            )
            ax.errorbar(
                x,
                r["MAE"],
                yerr=[[r["MAE"] - r["MAE_lo"]], [r["MAE_hi"] - r["MAE"]]],
                color=INK,
                lw=0.8,
                capsize=2,
            )
            ax.text(
                x,
                r["MAE_hi"] + 0.03,
                f"{r['MAE']:.2f}",
                ha="center",
                fontsize=7,
                color=INK_2,
            )
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([short(m) for m in models])
    ax.set_ylabel("MAE vs CHES lrgen (95% CI)")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=len(conds), fontsize=7.5
    )
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_chunk_ablation_mae.{ext}", dpi=300)
    plt.close(fig)

    # ---- fig 2: paired delta MAE ---------------------------------------------
    rows = deltas.sort_values(["Model", "Reference", "Condition"]).reset_index(
        drop=True
    )
    fig, ax = plt.subplots(figsize=(5.6, 0.55 * len(rows) + 1.0))
    for y, (_, r) in enumerate(rows.iterrows()):
        col = COLORS[r["Reference"]]
        ax.errorbar(
            r["ΔMAE (RAG-Base)"],
            y,
            xerr=[
                [r["ΔMAE (RAG-Base)"] - r["ΔMAE (RAG-Base)_lo"]],
                [r["ΔMAE (RAG-Base)_hi"] - r["ΔMAE (RAG-Base)"]],
            ],
            fmt="o",
            color=col,
            ecolor=col,
            capsize=2.5,
            ms=5,
        )
        ax.text(
            r["ΔMAE (RAG-Base)_hi"] + 0.02,
            y,
            f"{r['ΔMAE (RAG-Base)']:+.2f}  (n={r['N_paired']})",
            va="center",
            fontsize=7,
            color=INK_2,
        )
    ax.axvline(0, color=INK_3, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [
            f"{short(r['Model'])}: {LABEL.get(r['Condition'], r['Condition'])}\nvs {LABEL[r['Reference']]}"
            for _, r in rows.iterrows()
        ],
        fontsize=7.5,
    )
    ax.invert_yaxis()
    ax.set_xlabel("paired ΔMAE (ablation − reference), 95% CI\n< 0 = ablation better")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.35)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_chunk_ablation_delta.{ext}", dpi=300)
    plt.close(fig)

    # ---- fig 3: signed error by author party ----------------------------------
    order = [
        p
        for p in ["Linke", "Greens", "SPD", "FDP", "CDU", "CSU", "CDU/CSU", "AfD"]
        if p in set(party["party_short"])
    ]
    fig, axes = plt.subplots(
        1, len(models), figsize=(3.6 * len(models), 3.2), sharey=True, squeeze=False
    )
    for ax, m in zip(axes[0], models):
        for j, c in enumerate(conds):
            sub = (
                party[(party["model"] == m) & (party["condition"] == c)]
                .set_index("party_short")
                .reindex(order)
            )
            x = np.arange(len(order)) + (j - (len(conds) - 1) / 2) * 0.25
            ax.bar(
                x,
                sub["signed_error"],
                0.23,
                color=COLORS.get(c, ABL_COLOR),
                label=LABEL.get(c, c),
            )
        ax.axhline(0, color=INK_3, lw=0.8)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7.5)
        ax.set_title(short(m))
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("mean signed error (pred − CHES lrgen)")
    axes[0][-1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_chunk_ablation_party.{ext}", dpi=300)
    plt.close(fig)

    # ---- markdown -------------------------------------------------------------
    cols = [
        "Model",
        "Condition",
        "N_samples",
        "MAE",
        "RMSE",
        "Pearson_r",
        "Mean_signed_error",
        "Calibration_slope",
    ]
    dcols = [
        "Model",
        "Condition",
        "Reference",
        "N_paired",
        "ΔMAE (RAG-Base)",
        "ΔMAE (RAG-Base)_lo",
        "ΔMAE (RAG-Base)_hi",
        "ΔMAE_p_holm",
        "ΔPearson_r (RAG-Base)",
        "Δ|Bias| (RAG-Base)",
    ]
    with open(OUT_DIR / "chunk_ablation.md", "w", encoding="utf-8") as f:
        f.write("# Chunk ablation (target: CHES lrgen)\n\n## Point metrics\n\n")
        f.write(
            em._md_table(metrics[cols])
            + "\n\n## Paired deltas (ablation minus reference)\n\n"
        )
        f.write(em._md_table(deltas[dcols]) + "\n")
    print(metrics[cols].to_string(index=False))
    print(deltas[dcols].to_string(index=False))
    if rtop_conds:
        rtop_analysis(df, df_rtop, abl_conds, rtop_conds, cols, dcols)


QUAD_LABEL = {"Q1_RAG_Win": "RAG win", "Q2_RAG_Distraction": "RAG distraction",
              "Q3_Joint_Failure": "joint failure", "Q4_Baseline_Sufficiency": "baseline sufficiency"}
FAMILY_FILE = {"Llama-3.1-8B-Instruct": "llama_8b", "Ministral-3-8B-Instruct-2512": "mistral_8b"}


def quadrants(model: str) -> pd.DataFrame:
    """text_index -> quadrant of the judged sample the model's R_top run is based on."""
    import json
    fam = FAMILY_FILE.get(model)
    summary = json.loads((Path("RAG Analysis/LLM judge results/families/summary.json")).read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in open(summary[fam]["source_file"], encoding="utf-8")]
    return pd.DataFrame({"text_index": [str(r["text_index"]) for r in rows],
                         "quadrant": [r["quadrant"] for r in rows]})


def rtop_analysis(df_full, df_rtop, rand_conds, rtop_conds, cols, dcols):
    """Every condition restricted to the posts the R_top run predicted."""
    sub_frames, quad_rows = [], []
    for m in sorted(df_rtop["model"].unique()):
        for rc in rtop_conds:
            posts = set(df_rtop[(df_rtop["model"] == m) & (df_rtop["condition"] == rc)]["text_index"])
            keep = df_full[(df_full["model"] == m) & df_full["text_index"].isin(posts)]
            keep = keep[keep["condition"].isin(REF + rand_conds)]
            sub_frames.append(keep)
            sub_frames.append(df_rtop[(df_rtop["model"] == m) & (df_rtop["condition"] == rc)])
            q = quadrants(m)
            q = q[q["text_index"].isin(posts)]
            for quad, n in q["quadrant"].value_counts().items():
                quad_rows.append({"model": m, "quadrant": QUAD_LABEL.get(quad, quad), "n_posts": int(n)})
    sub = pd.concat(sub_frames, ignore_index=True)

    metrics = em.compute_evaluation_metrics(sub, target_col=TARGET)
    metrics["Condition"] = metrics["Condition"].astype(str)
    metrics.to_csv(OUT_DIR / "chunk_ablation_rtop_metrics.csv", index=False)
    quad_df = pd.DataFrame(quad_rows)
    quad_df.to_csv(OUT_DIR / "chunk_ablation_rtop_quadrants.csv", index=False)

    # chunk count per post for the R_top condition
    kc = (df_rtop.groupby(["model", "condition"])["n_chunks_retrieved"]
          .agg(["count", "mean", "min", "max"]).reset_index())
    kc.to_csv(OUT_DIR / "chunk_ablation_rtop_chunks_per_post.csv", index=False)

    refs = REF + rand_conds
    deltas = pd.concat([paired(sub, rc, r) for rc in rtop_conds for r in refs], ignore_index=True)
    # random5 vs e5/simple on the same posts: how much of an e5/simple difference is selection?
    deltas = pd.concat([deltas] + [paired(sub, a, r) for a in rand_conds for r in REF], ignore_index=True)
    deltas.to_csv(OUT_DIR / "chunk_ablation_rtop_paired.csv", index=False)

    models = sorted(sub["model"].unique(), key=lambda m: (em._meta_for(m)["family"], em._meta_for(m)["params_b"]))
    conds = REF + rand_conds + rtop_conds
    cols_c = {**COLORS, **{c: "#e0a03a" for c in rand_conds}, **{c: ABL_COLOR for c in rtop_conds}}

    # ---- fig: MAE on the R_top posts ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    w = 0.8 / len(conds)
    for j, c in enumerate(conds):
        for i, m in enumerate(models):
            r = metrics[(metrics["Model"] == m) & (metrics["Condition"] == c)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i + (j - (len(conds) - 1) / 2) * w
            ax.bar(x, r["MAE"], w * 0.9, color=cols_c[c], label=LABEL.get(c, c) if i == 0 else None)
            ax.errorbar(x, r["MAE"], yerr=[[r["MAE"] - r["MAE_lo"]], [r["MAE_hi"] - r["MAE"]]],
                        color=INK, lw=0.8, capsize=2)
            ax.text(x, r["MAE_hi"] + 0.03, f"{r['MAE']:.2f}", ha="center", fontsize=7, color=INK_2)
            ax.text(x, 0.03, f"n={int(r['N_samples'])}", ha="center", fontsize=6, color="white", rotation=90, va="bottom")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([short(m) for m in models])
    ax.set_ylabel("MAE vs CHES lrgen (95% CI)")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=len(conds), fontsize=7, columnspacing=0.9)
    ax.set_title("Posts with at least one chunk rated R_top ≥ 4 (judged quadrant sample)", fontsize=8, y=1.12)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_chunk_ablation_rtop_mae.{ext}", dpi=300)
    plt.close(fig)

    # ---- fig: paired deltas of the R_top condition ----------------------------
    rows = deltas[deltas["Condition"].isin(rtop_conds)].sort_values(["Model", "Reference"]).reset_index(drop=True)
    ref_col = {**COLORS, **{c: "#e0a03a" for c in rand_conds}}
    fig, ax = plt.subplots(figsize=(6.0, 0.5 * len(rows) + 1.1))
    d, lo, hi = "ΔMAE (RAG-Base)", "ΔMAE (RAG-Base)_lo", "ΔMAE (RAG-Base)_hi"
    for y, (_, r) in enumerate(rows.iterrows()):
        col = ref_col[r["Reference"]]
        ax.errorbar(r[d], y, xerr=[[r[d] - r[lo]], [r[hi] - r[d]]], fmt="o", color=col, ecolor=col, capsize=2.5, ms=5)
        ax.text(r[hi] + 0.03, y, f"{r[d]:+.2f}  (n={r['N_paired']})", va="center", fontsize=7, color=INK_2)
    ax.axvline(0, color=INK_3, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{short(r['Model'])}: only R_top ≥ 4\nvs {LABEL[r['Reference']]}" for _, r in rows.iterrows()], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("paired ΔMAE (only R_top ≥ 4 − reference), 95% CI\n< 0 = relevant chunks better")
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.4)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_chunk_ablation_rtop_delta.{ext}", dpi=300)
    plt.close(fig)

    with open(OUT_DIR / "chunk_ablation_rtop.md", "w", encoding="utf-8") as f:
        f.write("# R_top >= 4 chunk ablation (target: CHES lrgen)\n\n"
                "All conditions are restricted to the posts for which at least one logged e5/simple chunk was rated "
                "R_top >= 4 by the LLM judge. Only those chunks are shown to the model (variable k, 1-5).\n\n"
                "## Point metrics on these posts\n\n")
        f.write(em._md_table(metrics[cols]) + "\n\n## Paired deltas (first condition minus reference)\n\n")
        f.write(em._md_table(deltas[dcols]) + "\n\n## Quadrant of the posts\n\n")
        f.write(em._md_table(quad_df) + "\n\n## Chunks per post (R_top condition)\n\n")
        f.write(em._md_table(kc) + "\n")
    print("\n=== R_top subset ===")
    print(metrics[cols].to_string(index=False))
    print(deltas[dcols].to_string(index=False))
    print(quad_df.to_string(index=False))


if __name__ == "__main__":
    main()
