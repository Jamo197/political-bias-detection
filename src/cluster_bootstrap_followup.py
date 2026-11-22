"""Follow-up to cluster_bootstrap.py (same data, seed 42, 2000 draws, account draws).

1 fixed-reference MAE difference, 2 mean-signed-error intervals, 3 Table 5.1 point values,
4 sensitivity without the largest account, 5 Llama-8B qwen3/hyde + slope/offset changes.
Writes *_cluster_followup* files; nothing existing is overwritten.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import evaluate_metrics as em  # noqa: E402

T = "ches_lrgen"
OUT_A, OUT_B = em.THESIS_ROOT / "01_baseline_no_rag", em.THESIS_ROOT / "02_rag_vs_no_rag"
B, SEED = em.DEFAULT_N_BOOT, em.DEFAULT_SEED
L70, Q72, M14 = ("Meta-Llama-3.1-70B-Instruct-FP8", "Qwen2.5-72B-Instruct-FP8-dynamic",
                 "Ministral-3-14B-Instruct-2512")
pct = lambda x: em._percentile_ci(np.asarray(x, float))
r3 = lambda x: round(float(x), 3)


def short(m):
    return f"{em._meta_for(m)['family']}-{int(em._meta_for(m)['params_b'])}B"


def sums(ac, A, **cols):
    return {k: np.bincount(ac, weights=v, minlength=A) for k, v in cols.items()}


def mom(df, A):
    S = np.zeros((A, 9))
    np.add.at(S, df["acct"].to_numpy(), em._row_moments(df["true"].to_numpy(float), df["pred"].to_numpy(float)))
    return S


def analyse(valid, accounts, tag, t_median):
    """Part A + B on the given frame, cluster draws over `accounts`."""
    amap = {a: i for i, a in enumerate(accounts)}
    v = valid[valid.speaker.isin(accounts)].copy()
    v["acct"] = v.speaker.map(amap)
    A = len(accounts)
    idx = np.random.default_rng(SEED).integers(0, A, size=(B, A))
    base = v[v.condition == "no_rag"]
    models = sorted(base.model.unique(), key=lambda m: (em._meta_for(m)["family"], em._meta_for(m)["params_b"]))
    res = {"models": models, "v": v, "idx": idx, "A": A}

    rows = []
    for m in models:
        g = base[base.model == m]
        S = mom(g, A)
        D = S[idx].sum(1)
        bs = em._metrics_from_moments(D)
        pt = em._metrics_from_moments(S.sum(0))
        ref = (g["true"] - t_median).abs().to_numpy()
        Sref = np.bincount(g.acct.to_numpy(), weights=ref, minlength=A)
        diff = D[:, 1] / D[:, 0] - Sref[idx].sum(1) / D[:, 0]  # fixed constant, same draws
        row = dict(Model=m, N=len(g), MAE=r3(pt["MAE"][0]))
        row["MAE_lo"], row["MAE_hi"] = map(r3, pct(bs["MAE"]))
        row["MSE"] = r3(pt["Mean_signed_error"][0])
        row["MSE_lo"], row["MSE_hi"] = map(r3, pct(bs["Mean_signed_error"]))
        row["Ref_MAE"] = r3(ref.mean())
        row["Diff"] = r3(pt["MAE"][0] - ref.mean())
        row["Diff_lo"], row["Diff_hi"] = map(r3, pct(diff))
        row["Share_draws_diff_neg"] = round(float((diff < 0).mean()), 3)
        row["Ref_MAE_lo"], row["Ref_MAE_hi"] = map(r3, pct(Sref[idx].sum(1) / D[:, 0]))
        rows.append(row)
    res["base"] = pd.DataFrame(rows)

    def paired(m1, m2):
        a = base[base.model == m1][["text_index", "acct", "true", "pred"]]
        mm = a.merge(base[base.model == m2][["text_index", "pred"]], on="text_index", suffixes=("_1", "_2"))
        e1, e2 = (mm.pred_1 - mm.true).abs().to_numpy(), (mm.pred_2 - mm.true).abs().to_numpy()
        n_a = np.bincount(mm.acct, minlength=A).astype(float)
        s = sums(mm.acct.to_numpy(), A, a=e1, b=e2)
        d = (s["a"][idx].sum(1) - s["b"][idx].sum(1)) / n_a[idx].sum(1)
        lo, hi = pct(d)
        return dict(Comparison=f"{short(m1)} minus {short(m2)}", N=len(mm), dMAE=r3(e1.mean() - e2.mean()),
                    lo=r3(lo), hi=r3(hi), excludes_0=bool(lo > 0 or hi < 0))
    res["paired"] = pd.DataFrame([paired(L70, Q72), paired(M14, L70)])

    rows = []
    for model, mdf in v.groupby("model"):
        b = mdf[mdf.condition == "no_rag"][["text_index", "acct", "true", "pred"]]
        for cond in sorted(c for c in mdf.condition.unique() if c != "no_rag"):
            mm = b.merge(mdf[mdf.condition == cond][["text_index", "pred"]], on="text_index", suffixes=("_b", "_r"))
            if len(mm) < 3:
                continue
            Sb = np.zeros((A, 9)); Sr = np.zeros((A, 9))
            t = mm.true.to_numpy(float)
            np.add.at(Sb, mm.acct.to_numpy(), em._row_moments(t, mm.pred_b.to_numpy(float)))
            np.add.at(Sr, mm.acct.to_numpy(), em._row_moments(t, mm.pred_r.to_numpy(float)))
            pb, pr = em._metrics_from_moments(Sb.sum(0)), em._metrics_from_moments(Sr.sum(0))
            bb, br = em._metrics_from_moments(Sb[idx].sum(1)), em._metrics_from_moments(Sr[idx].sum(1))
            ib = float(em._vec_metrics(t, mm.pred_b.to_numpy(float))["Calibration_intercept"][()])
            ir = float(em._vec_metrics(t, mm.pred_r.to_numpy(float))["Calibration_intercept"][()])
            # intercept = mean(p) - slope*mean(t): from moments per draw
            def icpt(S):
                w = S[:, 0]; mt, mp = S[:, 4] / w, S[:, 5] / w
                return mp - em._metrics_from_moments(S)["Calibration_slope"] * mt
            d_int = icpt(Sr[idx].sum(1)) - icpt(Sb[idx].sum(1))
            d = {
                "dMAE": (br["MAE"] - bb["MAE"], pr["MAE"][0] - pb["MAE"][0]),
                "dSlope": (br["Calibration_slope"] - bb["Calibration_slope"],
                           pr["Calibration_slope"][0] - pb["Calibration_slope"][0]),
                "dIntercept": (d_int, ir - ib),
                "dAbsMSE": (np.abs(br["Mean_signed_error"]) - np.abs(bb["Mean_signed_error"]),
                            abs(pr["Mean_signed_error"][0]) - abs(pb["Mean_signed_error"][0])),
            }
            row = dict(Model=model, Condition=cond, N_paired=len(mm), Slope_base=r3(pb["Calibration_slope"][0]),
                       Slope_rag=r3(pr["Calibration_slope"][0]))
            for k, (samp, pt) in d.items():
                lo, hi = pct(samp)
                row[k], row[k + "_lo"], row[k + "_hi"] = r3(pt), r3(lo), r3(hi)
            row["Cluster_CI_direction"] = ("RAG better" if row["dMAE_hi"] < 0 else
                                           "RAG worse" if row["dMAE_lo"] > 0 else "n.s.")
            rows.append(row)
    res["rag"] = pd.DataFrame(rows)
    # post-level Holm/interval counts on the same frame
    post = em.compute_rag_vs_norag(v, B, SEED, np.nan)
    res["post"] = post
    return res


def main():
    pd.set_option("display.width", 250, "display.max_columns", 60)
    valid = em.prepare_target_frame(em.load_main_matrix(), T)
    t_median, _ = em.constant_reference_mae(valid)  # full-data median, fixed
    accounts = sorted(valid.speaker.unique())
    sizes = valid[valid.condition == "no_rag"].drop_duplicates("text_index").speaker.value_counts()
    big = sizes.index[0]
    print(f"\nfixed reference {t_median:.4f}; largest account {big!r}: {sizes.iloc[0]} posts "
          f"(party {valid[valid.speaker == big].party.iloc[0]}, target "
          f"{valid[valid.speaker == big].true.iloc[0]:.3f})")
    full = analyse(valid, accounts, "full", t_median)
    sens = analyse(valid, [a for a in accounts if a != big], "sens", t_median)

    out = []
    P = out.append
    P(f"Fixed reference = {t_median:.4f} (median of the 775 unique posts, same as the first run; it was NOT "
      "recomputed per draw before either). Largest account: "
      f"{big} ({sizes.iloc[0]} posts, party {valid[valid.speaker == big].party.iloc[0]}, "
      f"target {valid[valid.speaker == big].true.iloc[0]:.3f}).")
    b = full["base"]
    b.to_csv(OUT_A / f"mae_vs_fixed_reference_{T}_cluster_followup.csv", index=False)
    P("\n== 1/2. Cluster intervals: MAE, MAE - fixed ref, mean signed error ==")
    P(b.to_string(index=False))
    P(f"MSE intervals excluding zero: {int(((b.MSE_lo > 0) | (b.MSE_hi < 0)).sum())} of 10 -> "
      + ", ".join(short(m) for m, l, h in zip(b.Model, b.MSE_lo, b.MSE_hi) if l > 0 or h < 0))

    # 3. Table 5.1 point values
    pm = em.compute_evaluation_metrics(full["v"][full["v"].condition == "no_rag"], T, 0, SEED)
    dirn = em.compute_bias_direction(full["v"][full["v"].condition == "no_rag"], pm, T)
    t51 = pm[["Model", "N_samples"] + em.METRIC_NAMES].copy()
    t51["c"] = t51["Calibration_intercept"] / (1 - t51["Calibration_slope"])
    t51 = t51.merge(dirn[["Model", "Pct_centric"]], on="Model")
    t51["o"] = t51.Model
    t51["Model"] = t51.Model.map(short)
    t51 = t51.drop(columns="o").round(3)
    t51.to_csv(OUT_A / f"table51_point_values_{T}_cluster_followup.csv", index=False)
    P("\n== 3. Table 5.1 point values (current runs, no_rag) ==")
    P(t51.to_string(index=False))

    # 4. sensitivity
    P("\n== 4a. Sensitivity (without largest account): MAE and diff to fixed ref ==")
    sb = sens["base"].merge(b[["Model", "MAE_lo", "MAE_hi", "Diff_lo", "Diff_hi", "MSE_lo", "MSE_hi"]],
                            on="Model", suffixes=("", "_full"))
    sb.to_csv(OUT_A / f"sensitivity_without_largest_account_baseline_{T}_cluster_followup.csv", index=False)
    P(sb[["Model", "N", "MAE", "MAE_lo", "MAE_hi", "MAE_lo_full", "MAE_hi_full", "Diff", "Diff_lo", "Diff_hi",
          "Share_draws_diff_neg", "MSE", "MSE_lo", "MSE_hi"]].to_string(index=False))
    P("\n== 4b. Paired differences: full | without largest account ==")
    pp = full["paired"].merge(sens["paired"], on="Comparison", suffixes=("_full", "_sens"))
    pp.to_csv(OUT_A / f"sensitivity_without_largest_account_paired_{T}_cluster_followup.csv", index=False)
    P(pp.to_string(index=False))
    P("\n== 4c. Part B counts (better / worse) ==")
    def cnt(s, v): return int((s == v).sum())
    rows = []
    for name, r in [("full (72 accounts)", full), ("without largest (71)", sens)]:
        post = r["post"]
        ci_post = np.where(post.dMAE_hi < 0, "RAG better", np.where(post.dMAE_lo > 0, "RAG worse", "n.s."))
        rows.append(dict(Sample=name, N_cond=len(r["rag"]),
                         Holm_better=cnt(post.direction, "RAG better"), Holm_worse=cnt(post.direction, "RAG worse"),
                         PostCI_better=cnt(pd.Series(ci_post), "RAG better"), PostCI_worse=cnt(pd.Series(ci_post), "RAG worse"),
                         ClusterCI_better=cnt(r["rag"].Cluster_CI_direction, "RAG better"),
                         ClusterCI_worse=cnt(r["rag"].Cluster_CI_direction, "RAG worse")))
    P(pd.DataFrame(rows).to_string(index=False))
    pm_ = sens["rag"].groupby("Model").Cluster_CI_direction.agg(
        better=lambda s: cnt(s, "RAG better"), worse=lambda s: cnt(s, "RAG worse"))
    pf_ = full["rag"].groupby("Model").Cluster_CI_direction.agg(
        better=lambda s: cnt(s, "RAG better"), worse=lambda s: cnt(s, "RAG worse"))
    per = pf_.join(pm_, lsuffix="_full", rsuffix="_sens")
    per.index = [short(m) for m in per.index]
    per.to_csv(OUT_B / f"sensitivity_without_largest_account_rag_counts_{T}_cluster_followup.csv")
    P(per.to_string())
    ch = full["rag"][["Model", "Condition", "Cluster_CI_direction"]].merge(
        sens["rag"][["Model", "Condition", "dMAE", "dMAE_lo", "dMAE_hi", "Cluster_CI_direction"]],
        on=["Model", "Condition"], suffixes=("_full", "_sens"))
    ch = ch[ch.Cluster_CI_direction_full != ch.Cluster_CI_direction_sens]
    ch.to_csv(OUT_B / f"sensitivity_without_largest_account_rag_changes_{T}_cluster_followup.csv", index=False)
    P(f"\nConditions whose cluster-CI conclusion changes when the account is dropped: {len(ch)}")
    P(ch.to_string(index=False))

    # 5
    r = full["rag"]
    r.to_csv(OUT_B / f"rag_vs_no_rag_slope_offset_{T}_cluster_followup.csv", index=False)
    P("\n== 5a. Llama-3.1-8B qwen3/hyde ==")
    P(r[(r.Model == "Llama-3.1-8B-Instruct") & (r.Condition == "qwen3/hyde")].to_string(index=False))
    l8 = r[r.Model == "Llama-3.1-8B-Instruct"]
    P(f"Llama-3.1-8B dMAE range over its conditions: {l8.dMAE.min():.3f} to {l8.dMAE.max():.3f}; "
      f"min at {l8.loc[l8.dMAE.idxmin(), 'Condition']} (n={int(l8.loc[l8.dMAE.idxmin(), 'N_paired'])})")
    P("\n== 5b. Slope / offset changes (RAG - no-RAG), cluster 95% CI ==")
    for m in ["Ministral-3-8B-Instruct-2512", M14, "Qwen2.5-7B-Instruct"]:
        P(f"\n{short(m)}")
        P(r[r.Model == m][["Condition", "N_paired", "Slope_base", "Slope_rag", "dSlope", "dSlope_lo", "dSlope_hi",
                          "dIntercept", "dIntercept_lo", "dIntercept_hi", "dAbsMSE", "dAbsMSE_lo",
                          "dAbsMSE_hi"]].to_string(index=False))
    txt = "\n".join(out)
    (OUT_B / f"cluster_bootstrap_followup_summary_{T}.txt").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
