"""cluster_bootstrap.py

Account-level (cluster) bootstrap companion to evaluate_metrics.py.

Resampling unit: author account (``speaker``, 72 accounts). Each replicate draws
72 accounts with replacement and takes all their posts. The same account draws
are used for every model / condition (so paired comparisons are paired).
Writes new ``*_cluster*`` files next to the existing outputs; nothing is overwritten.

Part A  baseline (no-RAG) intervals, reference comparison, paired model differences
Part B  RAG vs. no-RAG (paired dMAE), compared with the post-level result
Part C  dependence diagnostics (cluster sizes, ICC, design effect)
"""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import evaluate_metrics as em  # noqa: E402

TARGET = "ches_lrgen"
OUT_A = em.THESIS_ROOT / "01_baseline_no_rag"
OUT_B = em.THESIS_ROOT / "02_rag_vs_no_rag"
B = em.DEFAULT_N_BOOT
SEED = em.DEFAULT_SEED
METRICS = ["MAE", "RMSE", "Pearson_r", "CCC", "Mean_signed_error", "Calibration_slope"]


def pct(x):
    return em._percentile_ci(np.asarray(x, float))


def acct_moments(df: pd.DataFrame, acct: pd.Series, n_acct: int) -> np.ndarray:
    """(n_acct, 9) summed moments per account (see em._row_moments)."""
    M = em._row_moments(df["true"].to_numpy(float), df["pred"].to_numpy(float))
    S = np.zeros((n_acct, M.shape[1]))
    np.add.at(S, acct.to_numpy(), M)
    return S


def draw(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return S[idx].sum(1)  # (B, 9)


def fmt_ci(lo, hi):
    return f"[{lo:+.3f}, {hi:+.3f}]"


def main():
    valid = em.prepare_target_frame(em.load_main_matrix(), TARGET)
    accounts = sorted(valid["speaker"].unique())
    amap = {a: i for i, a in enumerate(accounts)}
    valid["acct"] = valid["speaker"].map(amap)
    A = len(accounts)
    print(f"{A} accounts; {valid.text_index.nunique()} posts")

    rng = np.random.default_rng(SEED)
    idx_acct = rng.integers(0, A, size=(B, A))  # shared cluster draws

    base = valid[valid["condition"] == "no_rag"].copy()
    models = sorted(base["model"].unique(), key=lambda m: (em._meta_for(m)["family"], em._meta_for(m)["params_b"]))

    # ---- post-level reference (existing helpers, same seed as thesis) ----
    post_metrics = em.compute_evaluation_metrics(base, TARGET, B, SEED).set_index("Model")
    old_csv = em.THESIS_ROOT / "01_baseline_no_rag" / f"point_metrics_no_rag_{TARGET}.csv"
    if old_csv.exists():
        old = pd.read_csv(old_csv).set_index("Model")
        diff = max(abs(old.loc[m, "MAE"] - post_metrics.loc[m, "MAE"]) for m in models)
        print(f"Max |MAE(existing CSV) - MAE(current)| = {diff:.4f}")

    t_median, _ = em.constant_reference_mae(valid)
    print(f"constant reference: median {t_median:.4f}")

    # ================= Part A =================
    rows: List[Dict] = []
    mom = {}
    for m in models:
        g = base[base["model"] == m]
        S = acct_moments(g, g["acct"], A)
        mom[m] = (g, S)
        pt = em._metrics_from_moments(S.sum(0))
        bs = em._metrics_from_moments(draw(S, idx_acct))
        for k in METRICS:
            lo, hi = pct(bs[k])
            rows.append(dict(
                Model=m, N=len(g), Metric=k, Estimate=em._round(float(pt[k][0])),
                Post_lo=post_metrics.loc[m, f"{k}_lo"], Post_hi=post_metrics.loc[m, f"{k}_hi"],
                Cluster_lo=em._round(lo), Cluster_hi=em._round(hi),
            ))
    ci = pd.DataFrame(rows)
    ci["Width_ratio"] = ((ci.Cluster_hi - ci.Cluster_lo) / (ci.Post_hi - ci.Post_lo)).round(3)
    ci.to_csv(OUT_A / f"point_metrics_no_rag_{TARGET}_cluster.csv", index=False)

    # A2: MAE - reference MAE (fixed median, recomputed on resampled posts);
    # post-level counterpart with its own post-level draws (seed SEED)
    rng_p = np.random.default_rng(SEED)
    rows = []
    for m in models:
        g, S = mom[m]
        ref_abs = (g["true"] - t_median).abs().to_numpy()
        Sref = np.zeros(A)
        np.add.at(Sref, g["acct"].to_numpy(), ref_abs)
        n_a = S[:, 0]
        mae_pt = S[:, 1].sum() / n_a.sum()
        ref_pt = ref_abs.mean()
        d_c = draw(S, idx_acct)[:, 1] / draw(S, idx_acct)[:, 0] - Sref[idx_acct].sum(1) / n_a[idx_acct].sum(1)
        n = len(g)
        ip = rng_p.integers(0, n, size=(B, n))
        ae = (g["pred"] - g["true"]).abs().to_numpy()
        d_p = ae[ip].mean(1) - ref_abs[ip].mean(1)
        cl, ch = pct(d_c)
        pl, ph = pct(d_p)
        rows.append(dict(
            Model=m, N=n, MAE=em._round(mae_pt), Ref_MAE=em._round(ref_pt),
            Diff=em._round(mae_pt - ref_pt),
            Post_lo=em._round(pl), Post_hi=em._round(ph), Post_excludes_0=bool(pl > 0 or ph < 0),
            Cluster_lo=em._round(cl), Cluster_hi=em._round(ch), Cluster_excludes_0=bool(cl > 0 or ch < 0),
            Cluster_direction=("better than ref" if ch < 0 else "worse than ref" if cl > 0 else "n.s."),
        ))
    pd.DataFrame(rows).to_csv(OUT_A / f"mae_vs_reference_{TARGET}_cluster.csv", index=False)
    ref_df = pd.DataFrame(rows)

    # A4: paired MAE differences on shared posts
    def paired(m1, m2):
        a = base[base.model == m1][["text_index", "acct", "true", "pred"]]
        b = base[base.model == m2][["text_index", "pred"]]
        mm = a.merge(b, on="text_index", suffixes=("_1", "_2"))
        e1 = (mm.pred_1 - mm.true).abs().to_numpy()
        e2 = (mm.pred_2 - mm.true).abs().to_numpy()
        ac = mm.acct.to_numpy()
        n_a = np.bincount(ac, minlength=A).astype(float)
        s1 = np.bincount(ac, weights=e1, minlength=A)
        s2 = np.bincount(ac, weights=e2, minlength=A)
        d_c = (s1[idx_acct].sum(1) - s2[idx_acct].sum(1)) / n_a[idx_acct].sum(1)
        n = len(mm)
        ip = np.random.default_rng(SEED).integers(0, n, size=(B, n))
        d_p = e1[ip].mean(1) - e2[ip].mean(1)
        cl, ch = pct(d_c)
        pl, ph = pct(d_p)
        return dict(
            Comparison=f"{em._meta_for(m1)['family']} {int(em._meta_for(m1)['params_b'])}B minus "
                       f"{em._meta_for(m2)['family']} {int(em._meta_for(m2)['params_b'])}B",
            N_paired=n, dMAE=em._round(e1.mean() - e2.mean()),
            Post_lo=em._round(pl), Post_hi=em._round(ph), Post_excludes_0=bool(pl > 0 or ph < 0),
            Cluster_lo=em._round(cl), Cluster_hi=em._round(ch), Cluster_excludes_0=bool(cl > 0 or ch < 0),
        )

    L70 = "Meta-Llama-3.1-70B-Instruct-FP8"
    Q72 = "Qwen2.5-72B-Instruct-FP8-dynamic"
    M14 = "Ministral-3-14B-Instruct-2512"
    pair_df = pd.DataFrame([paired(L70, Q72), paired(M14, L70)])
    pair_df.to_csv(OUT_A / f"paired_mae_differences_{TARGET}_cluster.csv", index=False)

    # ================= Part C =================
    u = base.drop_duplicates("text_index")
    per_acct = u.groupby("acct").size()
    party_share = u["party_group"].value_counts(normalize=True)
    top3 = party_share.head(3)
    diag = {
        "n_posts": len(u), "n_accounts": int(per_acct.size),
        "posts_per_account_min": int(per_acct.min()),
        "posts_per_account_median": float(per_acct.median()),
        "posts_per_account_max": int(per_acct.max()),
        "share_top3_parties": round(float(top3.sum()), 4),
        "top3_parties": "; ".join(f"{p} {s:.3f}" for p, s in top3.items()),
        "share_largest_account": round(float(per_acct.max() / len(u)), 4),
    }
    pd.DataFrame([diag]).to_csv(OUT_A / f"dependence_diagnostics_{TARGET}_cluster.csv", index=False)

    icc_rows = []
    for m in models:
        g, S = mom[m]
        ae = (g["pred"] - g["true"]).abs().to_numpy()
        ac = g["acct"].to_numpy()
        n_i = np.bincount(ac, minlength=A).astype(float)
        keep = n_i > 0
        k = keep.sum()
        N = n_i.sum()
        s_i = np.bincount(ac, weights=ae, minlength=A)
        mean_i = np.where(keep, s_i / np.where(keep, n_i, 1), 0)
        grand = ae.mean()
        ssb = (n_i[keep] * (mean_i[keep] - grand) ** 2).sum()
        ssw = ((ae - mean_i[ac]) ** 2).sum()
        msb, msw = ssb / (k - 1), ssw / (N - k)
        n0 = (N - (n_i[keep] ** 2).sum() / N) / (k - 1)
        icc = (msb - msw) / (msb + (n0 - 1) * msw)
        mbar = (n_i[keep] ** 2).sum() / N  # size-weighted mean cluster size
        deff = 1 + (mbar - 1) * icc
        # empirical variance inflation of the MAE (cluster vs post bootstrap)
        pl, ph = post_metrics.loc[m, "MAE_lo"], post_metrics.loc[m, "MAE_hi"]
        cl = ci[(ci.Model == m) & (ci.Metric == "MAE")].iloc[0]
        wr = (cl.Cluster_hi - cl.Cluster_lo) / (ph - pl)
        icc_rows.append(dict(
            Model=m, N=int(N), Mean_abs_error=em._round(grand), ICC=em._round(icc),
            Mean_cluster_size_weighted=round(float(mbar), 2), Design_effect=round(float(deff), 2),
            SE_inflation_sqrt_deff=round(float(np.sqrt(max(deff, 0))), 2),
            Empirical_CI_width_ratio=round(float(wr), 2),
        ))
    icc_df = pd.DataFrame(icc_rows)
    icc_df.to_csv(OUT_A / f"icc_design_effect_{TARGET}_cluster.csv", index=False)

    # ================= Part B =================
    post_delta = em.compute_rag_vs_norag(valid, B, SEED, ref_mae=np.nan)
    rows = []
    for model, mdf in valid.groupby("model"):
        b = mdf[mdf.condition == "no_rag"][["text_index", "acct", "true", "pred"]]
        for cond in sorted(c for c in mdf.condition.unique() if c != "no_rag"):
            r = mdf[mdf.condition == cond][["text_index", "pred"]]
            mm = b.merge(r, on="text_index", suffixes=("_b", "_r"))
            if len(mm) < 3:
                continue
            ac = mm.acct.to_numpy()
            n_a = np.bincount(ac, minlength=A).astype(float)
            eb = (mm.pred_b - mm.true).abs().to_numpy()
            er = (mm.pred_r - mm.true).abs().to_numpy()
            sb = np.bincount(ac, weights=eb, minlength=A)
            sr = np.bincount(ac, weights=er, minlength=A)
            d = (sr[idx_acct].sum(1) - sb[idx_acct].sum(1)) / n_a[idx_acct].sum(1)
            lo, hi = pct(d)
            rows.append(dict(
                Model=model, Condition=cond, N_paired=len(mm),
                dMAE_cluster=em._round(er.mean() - eb.mean()),
                Cluster_lo=em._round(lo), Cluster_hi=em._round(hi),
                p_boot_cluster=em._boot_p_two_sided(d),
            ))
    cl_df = pd.DataFrame(rows)
    keep_cols = ["Model", "Family", "Params_B", "Condition", "N_paired", "dMAE", "dMAE_lo", "dMAE_hi",
                 "p_holm", "direction"]
    out = post_delta[keep_cols].rename(columns={
        "dMAE": "dMAE_post", "dMAE_lo": "Post_lo", "dMAE_hi": "Post_hi",
        "p_holm": "p_holm_wilcoxon_post", "direction": "Post_wilcoxon_holm"})
    out = out.merge(cl_df.drop(columns="N_paired"), on=["Model", "Condition"], how="outer", validate="1:1")

    def cls(lo, hi):
        return "RAG better" if hi < 0 else "RAG worse" if lo > 0 else "n.s."

    out["Post_CI_direction"] = [cls(l, h) for l, h in zip(out.Post_lo, out.Post_hi)]
    out["Cluster_CI_direction"] = [cls(l, h) for l, h in zip(out.Cluster_lo, out.Cluster_hi)]
    out["Width_ratio"] = ((out.Cluster_hi - out.Cluster_lo) / (out.Post_hi - out.Post_lo)).round(2)
    out["Changed_vs_post_CI"] = out.Post_CI_direction != out.Cluster_CI_direction
    out["Changed_vs_post_wilcoxon_holm"] = out.Post_wilcoxon_holm != out.Cluster_CI_direction
    out = out.sort_values(["Family", "Params_B", "Condition"]).reset_index(drop=True)
    out.to_csv(OUT_B / f"rag_vs_no_rag_{TARGET}_cluster.csv", index=False)

    summ = out.groupby(["Family", "Params_B", "Model"]).agg(
        N_conditions=("Condition", "size"),
        Post_holm_better=("Post_wilcoxon_holm", lambda s: int((s == "RAG better").sum())),
        Post_holm_worse=("Post_wilcoxon_holm", lambda s: int((s == "RAG worse").sum())),
        Post_CI_better=("Post_CI_direction", lambda s: int((s == "RAG better").sum())),
        Post_CI_worse=("Post_CI_direction", lambda s: int((s == "RAG worse").sum())),
        Cluster_CI_better=("Cluster_CI_direction", lambda s: int((s == "RAG better").sum())),
        Cluster_CI_worse=("Cluster_CI_direction", lambda s: int((s == "RAG worse").sum())),
    ).reset_index()
    tot = summ.select_dtypes("number").drop(columns="Params_B").sum()
    summ.loc[len(summ)] = {"Family": "", "Params_B": np.nan, "Model": "TOTAL", **tot.to_dict()}
    summ.to_csv(OUT_B / f"rag_vs_no_rag_summary_{TARGET}_cluster.csv", index=False)

    # ================= text summary =================
    pd.set_option("display.width", 250, "display.max_columns", 50)
    lines = []
    P = lines.append
    P("== (1) MAE: post-level vs cluster 95% intervals ==")
    mae = ci[ci.Metric == "MAE"].merge(icc_df[["Model", "ICC", "Design_effect"]], on="Model")
    for _, r in mae.iterrows():
        P(f"{r.Model:36s} MAE {r.Estimate:.3f}  post [{r.Post_lo:.3f}, {r.Post_hi:.3f}]  "
          f"cluster [{r.Cluster_lo:.3f}, {r.Cluster_hi:.3f}]  width x{r.Width_ratio:.2f}")
    P("\n== MAE minus constant-median reference ==")
    P(ref_df.to_string(index=False))
    P("\n== Mean signed error ==")
    P(ci[ci.Metric == "Mean_signed_error"].to_string(index=False))
    P("\n== Paired MAE differences ==")
    P(pair_df.to_string(index=False))
    P("\n== Dependence diagnostics ==")
    P(pd.Series(diag).to_string())
    P(icc_df.to_string(index=False))
    P("\n== RAG vs no-RAG ==")
    P(summ.to_string(index=False))
    ch = out[out.Changed_vs_post_wilcoxon_holm | out.Changed_vs_post_CI]
    P(f"\nChanged vs post-level Wilcoxon/Holm: {int(out.Changed_vs_post_wilcoxon_holm.sum())}; "
      f"changed vs post-level CI: {int(out.Changed_vs_post_CI.sum())}")
    P(ch[["Model", "Condition", "dMAE_post", "Post_lo", "Post_hi", "Post_wilcoxon_holm", "Post_CI_direction",
          "Cluster_lo", "Cluster_hi", "Cluster_CI_direction"]].to_string(index=False))
    txt = "\n".join(lines)
    (OUT_B / f"cluster_bootstrap_summary_{TARGET}.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
