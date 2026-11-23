"""cluster_ch54_55_effects.py

Account-level inference for the retrieval diagnostics (5.4) and the effect of
retrieval on the predictions (5.5). The point estimates are those of
``why_rag_analysis.py``, ``RAG Analysis/analyze_chunk_influence.py`` and
``chunk_ablation_analysis.py``. Intervals, p values and classifications are recomputed
with the shared account draws of ``cluster_common`` (72 accounts, 2000 draws,
seed 42, the same draws as for 5.1/5.2).

Subcommands (item numbers of the re-analysis request):
  composition  3  e5/simple same-party / same-bloc / opposite-side share, mean
                  position and distance per author party: account CIs,
                  account-block permutation null, comparison to the random-chunk
                  reference; rank 1 vs rank 5
  evidence     5  R ~ B + E per model, pooled, by subset (+ interaction
                  contrasts), cited/uncited and relevant/other: SEs clustered by
                  account, account bootstrap CI, account-block permutation
  direction    6  overall shift statistics, movement, per-party changes,
                  compress/expand/shift classes
  ablation     7  random-chunk and R_top runs: paired deltas
  weights      9  quadrant sampling weights (check only)
  all          everything above. Items 5-7 are also re-run without the largest
               account (*_cluster_sens.csv).

Account-block permutation (items 3 and 5): the accounts are permuted, and each
post of account a receives the evidence (chunk list or E value) of a post of
the donor account pi(a). The donor's posts are drawn without replacement, or
with replacement when the donor has fewer posts. Evidence that comes from one
account therefore stays together, and the link between an author and the
evidence retrieved for their posts is broken.

Usage:  python src/cluster_ch54_55_effects.py all
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cluster_common as cc  # noqa: E402
import why_rag_analysis as w  # noqa: E402  (chdir to repo root)
import evaluate_metrics as em  # noqa: E402

aci = w.aci
OUT4 = w.ROOT / em.THESIS_ROOT / "04_why_rag"
OUT5 = w.ROOT / em.THESIS_ROOT / "05_chunk_ablation"
OUT_CI = OUT4 / "chunk_influence_all"
N_PERM = 1000
LO_C, HI_C = w.SIDE_LO, w.SIDE_HI


FEW_ACCOUNTS = 10  # cluster inference on fewer accounts is flagged as not interpretable
STRATUM = 6  # accounts per size stratum in the size-stratified block permutation


def block_perm(codes: np.ndarray, rng, stratum: int = STRATUM) -> np.ndarray:
    """Index array: row i takes the evidence of a row of donor account pi(acct_i).
    Accounts are permuted within strata of ``stratum`` accounts of similar size
    (sorted by number of rows), so donors and recipients have similar sizes; the
    donor's rows are drawn without replacement (with replacement only if the
    donor is smaller)."""
    acc, n = np.unique(codes, return_counts=True)
    order = acc[np.argsort(n, kind="stable")]
    donor = {}
    for s in range(0, len(order), stratum):
        blk = order[s:s + stratum]
        donor.update(dict(zip(blk, rng.permutation(blk))))
    members = {a: np.flatnonzero(codes == a) for a in acc}
    out = np.empty(len(codes), dtype=int)
    for a in acc:
        rec, don = members[a], members[donor[a]]
        out[rec] = rng.choice(don, size=len(rec), replace=len(rec) > len(don))
    return out


def fmt_ci(e, lo, hi, nd=2):
    return f"{e:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


# ===========================================================================
# item 3: composition
# ===========================================================================

def random5_arrays(groups):
    import random_chunk_composition as rcc
    fp = [f for f in rcc.RANDOM_LOGS if "llama" in f.as_posix().lower()] or rcc.RANDOM_LOGS
    import json
    seen = {}
    for line in fp[0].read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("//"):
            rec = json.loads(line)
            seen[str((rec.get("input_metadata") or {}).get("text_index"))] = rec
    rows = []
    for t, rec in seen.items():
        a = w._raw_to_canon((rec.get("input_metadata") or {}).get("party"), "author")
        apos = w.POSITIONS[a]
        for rank, c in enumerate((rec.get("inputs") or {}).get("retrieved_chunks") or [], 1):
            g = w.party_group(w._raw_to_canon(c.get("party"), "chunk"))
            pos = w.group_position(g) if g else np.nan
            rows.append({"text_index": t, "author_canon": a, "author_group": w.party_group(a), "author_pos": apos,
                         "author_bloc": em._bloc(apos), "rank": rank, "chunk_group": g or w.NO_POSITION,
                         "chunk_pos": pos, "chunk_bloc": em._bloc(pos) if g else -1})
    ret = pd.DataFrame(rows)
    if ret.groupby("text_index").size().ne(5).any():
        raise SystemExit("random5: not 5 chunks per post")
    return w.RetrievalArrays(ret, groups), fp[0].name


def per_text_authors(arr, a, a_bloc, a_pos, opp):
    """RetrievalArrays.per_text with the author attributes given explicitly."""
    G, P, Bl, S = arr.G, arr.P, arr.B, arr.S
    valid = G >= 0
    nv = valid.sum(1).astype(float)
    nv[nv == 0] = np.nan
    with np.errstate(invalid="ignore"):
        return {
            "same_party": (G == a[:, None]).sum(1) / nv,
            "same_bloc": ((Bl == a_bloc[:, None]) & valid).sum(1) / nv,
            "mean_position": np.nanmean(np.where(valid, P, np.nan), 1),
            "distance": np.nanmean(np.where(valid, np.abs(P - a_pos[:, None]), np.nan), 1),
            "opposite_bloc": np.where(opp != 0, ((S == opp[:, None]) & valid).sum(1) / nv, np.nan),
        }


def label_perm_null(arr, codes, rng, n=N_PERM):
    """Null by permuting the author attributes (party, bloc, position) across
    ACCOUNTS: every account keeps its posts and their chunk lists and takes the
    author party of a randomly assigned other account. Returns, per permutation,
    the per-text statistics and the permuted author group (for the party scopes)."""
    acc = np.unique(codes)
    first = {a: np.flatnonzero(codes == a)[0] for a in acc}
    attrs = {k: getattr(arr, k) for k in ("a", "a_bloc", "a_pos", "opp", "a_group")}
    for k, v in attrs.items():
        per_acct = {a: v[first[a]] for a in acc}
        if any(len(set(v[codes == a])) != 1 for a in acc):
            raise SystemExit(f"author attribute {k} is not constant within an account")
        attrs[k] = per_acct
    out = []
    for _ in range(n):
        pi = dict(zip(acc, rng.permutation(acc)))
        src = np.array([pi[c] for c in codes])
        a, ab, ap, op, ag = (np.array([attrs[k][s] for s in src]) for k in ("a", "a_bloc", "a_pos", "opp", "a_group"))
        out.append((per_text_authors(arr, a, ab, ap, op), ag))
    return out


def composition(D):
    kb = w.load_kb(w.DEFAULT_KB if w.DEFAULT_KB.exists() else w.KB_SPEECHES)
    ret = w.load_retrieval()
    groups = w.group_order([g for g in set(ret["chunk_group"]) | set(kb["group"]) | set(ret["author_group"])
                            if g != w.NO_POSITION])
    arr = w.RetrievalArrays(ret, groups)
    codes = D.codes(arr.texts)
    obs, exp = arr.per_text(), w.kb_expectation(arr, kb)
    rng = np.random.default_rng(cc.SEED)
    null = label_perm_null(arr, codes, rng)
    old = pd.read_csv(OUT4 / "retrieval_chance_level.csv")
    rows = []
    for scope, mask in w.author_scopes(arr):
        grp = None if scope == "All posts" else arr.a_group[mask][0]
        for stat, label in w.STAT_LABELS.items():
            o = obs[stat][mask]
            if np.all(np.isnan(o)):
                continue
            e = exp[stat][mask]
            ok = ~np.isnan(o) & ~np.isnan(e)
            c = codes[mask]
            bs = D.boot_mean(c, o)
            lo, hi = cc.pct(bs)
            bd = D.boot_mean(c[ok], o[ok] - e[ok])
            null_means = np.array([np.nanmean(nl[stat] if grp is None else nl[stat][ag == grp])
                                   for nl, ag in null])
            if scope == "All posts" and stat == "mean_position":
                null_means[:] = np.nan
            centre = np.nanmean(null_means) if not np.all(np.isnan(null_means)) else np.nan
            mean = float(np.nanmean(o))
            rows.append({
                "Author party": scope, "Statistic": label, "Posts": int(ok.sum()),
                "Accounts": D.n_accounts(arr.texts[mask]),
                "Observed": mean, "Cluster CI low": lo, "Cluster CI high": hi,
                "Knowledge-base expectation": float(np.mean(e[ok])),
                "Observed - expectation": mean - float(np.mean(e[ok])),
                "Diff cluster CI low": cc.pct(bd)[0], "Diff cluster CI high": cc.pct(bd)[1],
                "Bootstrap p (observed vs expectation, cluster)": cc.p_boot(bd),
                "Account-perm null mean": centre,
                "Account-perm null 2.5 %": np.nan if np.isnan(centre) else float(np.nanpercentile(null_means, 2.5)),
                "Account-perm null 97.5 %": np.nan if np.isnan(centre) else float(np.nanpercentile(null_means, 97.5)),
                "Account-perm p": np.nan if np.isnan(centre) else float(
                    (1 + np.sum(np.abs(null_means - centre) >= abs(mean - centre))) / (N_PERM + 1)),
            })
    df = pd.DataFrame(rows)
    df["Bootstrap p (Holm, cluster)"] = cc.holm(df["Bootstrap p (observed vs expectation, cluster)"])
    df[f"Few accounts (< {FEW_ACCOUNTS})"] = df["Accounts"] < FEW_ACCOUNTS
    df = df.merge(old[["Author party", "Statistic", "Observed", "CI low", "CI high", "Wilcoxon p (Holm)",
                       "Permutation null 2.5 %", "Permutation null 97.5 %", "Permutation p"]]
                  .rename(columns={"Observed": "Observed_old", "CI low": "Post CI low", "CI high": "Post CI high",
                                   "Wilcoxon p (Holm)": "Old Wilcoxon p (Holm)",
                                   "Permutation null 2.5 %": "Old perm null 2.5 %",
                                   "Permutation null 97.5 %": "Old perm null 97.5 %",
                                   "Permutation p": "Old permutation p"}),
                  on=["Author party", "Statistic"], how="left")
    if df["Observed_old"].isna().any() or (df["Observed"] - df["Observed_old"]).abs().max() > 1e-3:
        raise SystemExit("composition point estimates differ from retrieval_chance_level.csv")
    df["Old sig (Wilcoxon Holm)"] = df["Old Wilcoxon p (Holm)"] < 0.05
    df["New sig (cluster boot Holm)"] = df["Bootstrap p (Holm, cluster)"] < 0.05
    df["Changed (test vs expectation)"] = df["Old sig (Wilcoxon Holm)"] != df["New sig (cluster boot Holm)"]
    na = df["Account-perm p"].isna()
    df["Old sig (perm)"] = [None if n else bool(p < 0.05) for n, p in zip(na, df["Old permutation p"])]
    df["New sig (account perm)"] = [None if n else bool(p < 0.05) for n, p in zip(na, df["Account-perm p"])]
    df = df.drop(columns="Observed_old")

    # rank 1 vs rank 5
    same = np.where(arr.G >= 0, (arr.G == arr.a[:, None]).astype(float), np.nan)
    oldr = pd.read_csv(OUT4 / "retrieval_rank_position.csv").set_index("Author party")
    rr = []
    for scope, mask in w.author_scopes(arr):
        s = same[mask]
        both = ~np.isnan(s[:, 0]) & ~np.isnan(s[:, 4])
        d = s[both, 0] - s[both, 4]
        bs = D.boot_mean(codes[mask][both], d)
        lo, hi = cc.pct(bs)
        rr.append({"Author party": scope, "Posts": int(both.sum()), "Accounts": D.n_accounts(arr.texts[mask][both]),
                   "Rank 1 - rank 5": float(d.mean()), "Post CI low": oldr.loc[scope, "CI low"],
                   "Post CI high": oldr.loc[scope, "CI high"], "Old Wilcoxon p (Holm)": oldr.loc[scope, "Wilcoxon p (Holm)"],
                   "Cluster CI low": lo, "Cluster CI high": hi, "Bootstrap p (cluster)": cc.p_boot(bs)})
    rank = pd.DataFrame(rr)
    rank["Bootstrap p (Holm, cluster)"] = cc.holm(rank["Bootstrap p (cluster)"])
    rank["Changed"] = (rank["Old Wilcoxon p (Holm)"] < 0.05) != (rank["Bootstrap p (Holm, cluster)"] < 0.05)

    # random-chunk reference: e5/simple minus random5, paired per post
    rarr, rname = random5_arrays(groups)
    pos_r = {t: i for i, t in enumerate(rarr.texts)}
    ir = np.array([pos_r[t] for t in arr.texts])
    robs = {k: v[ir] for k, v in rarr.per_text().items()}
    rows = []
    for scope, mask in w.author_scopes(arr):
        for stat, label in w.STAT_LABELS.items():
            o, r = obs[stat][mask], robs[stat][mask]
            ok = ~np.isnan(o) & ~np.isnan(r)
            if not ok.any():
                continue
            bs = D.boot_mean(codes[mask][ok], o[ok] - r[ok])
            lo, hi = cc.pct(bs)
            br = D.boot_mean(codes[mask], r)
            rows.append({"Author party": scope, "Statistic": label, "Posts": int(ok.sum()),
                         "e5/simple": float(np.nanmean(o)), "random5": float(np.nanmean(r)),
                         "random5 cluster CI low": cc.pct(br)[0], "random5 cluster CI high": cc.pct(br)[1],
                         "e5/simple - random5": float((o[ok] - r[ok]).mean()),
                         "Cluster CI low": lo, "Cluster CI high": hi, "Bootstrap p (cluster)": cc.p_boot(bs)})
    rnd = pd.DataFrame(rows)
    rnd["Bootstrap p (Holm, cluster)"] = cc.holm(rnd["Bootstrap p (cluster)"])
    rnd.attrs["source"] = rname

    df.to_csv(OUT4 / "retrieval_chance_level_cluster.csv", index=False)
    rank.to_csv(OUT4 / "retrieval_rank_position_cluster.csv", index=False)
    rnd.to_csv(OUT4 / "retrieval_random5_vs_e5_cluster.csv", index=False)

    md = ["# Item 3: retrieval composition (e5/simple) with account-level inference", "",
          f"775 posts and 72 accounts. e5/simple retrieves the same chunks for every model, so each statistic is "
          f"computed once per post. CIs come from the account cluster bootstrap ({cc.B} draws, seed {cc.SEED}, the "
          "same draws as for 5.1/5.2). The test against the knowledge-base expectation is the bootstrap of "
          "mean(observed − expected) over accounts, two-sided, Holm over all 35 rows. This replaces the per-post "
          "Wilcoxon test. With 2000 draws the smallest two-sided bootstrap p is 0.001, so the smallest attainable "
          "Holm p over 35 rows is 0.035: a Holm p of 0.035 means 'at the resolution limit'. Permutation null: the "
          "old version shuffled chunk lists across posts. The new version shuffles the author party across ACCOUNTS "
          f"({N_PERM} permutations, seed {cc.SEED}). Every account keeps its posts and their chunk lists and takes "
          "the party (bloc, position) of a randomly assigned account; the party scopes are defined by the permuted "
          "label. Rows based on fewer than 10 accounts (DIE LINKE 7, SPD 5, FDP 4) are flagged: with so few clusters "
          "neither the bootstrap nor the permutation is reliable, and changes in those rows are not counted.",
          "", "## Observed shares with old (post) and new (account) CIs", ""]
    tab = df.assign(**{
        "Observed [post CI]": [f"{o:.3f} [{a:.3f}, {b:.3f}]" for o, a, b in zip(df.Observed, df["Post CI low"], df["Post CI high"])],
        "Observed [account CI]": [f"{o:.3f} [{a:.3f}, {b:.3f}]" for o, a, b in zip(df.Observed, df["Cluster CI low"], df["Cluster CI high"])],
        "Old perm null 95 %": [("" if pd.isna(a) else f"[{a:.3f}, {b:.3f}]") for a, b in zip(df["Old perm null 2.5 %"], df["Old perm null 97.5 %"])],
        "Account perm null 95 %": [("" if pd.isna(a) else f"[{a:.3f}, {b:.3f}]") for a, b in zip(df["Account-perm null 2.5 %"], df["Account-perm null 97.5 %"])],
    })
    md.append(cc.md_table(tab[["Author party", "Statistic", "Posts", "Accounts", "Observed [post CI]",
                               "Observed [account CI]", "Knowledge-base expectation", "Old Wilcoxon p (Holm)",
                               "Bootstrap p (Holm, cluster)", "Old perm null 95 %", "Account perm null 95 %",
                               "Old permutation p", "Account-perm p"]]))
    perm_ch = (df["Old sig (perm)"] != df["New sig (account perm)"]) & df["Account-perm p"].notna()
    df["Changed (permutation)"] = perm_ch
    few = df[f"Few accounts (< {FEW_ACCOUNTS})"]
    ch = df[(df["Changed (test vs expectation)"] | perm_ch) & ~few]
    md += ["", f"Rows with at least 10 accounts whose significance changes (test against expectation or "
           f"permutation): {len(ch)}. A further {int(((df['Changed (test vs expectation)'] | perm_ch) & few).sum())} "
           "rows change but rest on fewer than 10 accounts.", ""]
    if len(ch):
        md.append(cc.md_table(ch[["Author party", "Statistic", "Accounts", "Observed - expectation", "Diff cluster CI low",
                                  "Diff cluster CI high", "Old Wilcoxon p (Holm)", "Bootstrap p (Holm, cluster)",
                                  "Old permutation p", "Account-perm p"]]))
    w_ = (df["Cluster CI high"] - df["Cluster CI low"]) / (df["Post CI high"] - df["Post CI low"])
    md += ["", f"CI width ratio account/post: median {w_.median():.2f} (range {w_.min():.2f}-{w_.max():.2f}).", "",
           "## Rank 1 vs rank 5 (same-party share)", "", cc.md_table(rank), "",
           f"## Comparison with the random-chunk reference (random5 run, {rname})", "",
           "Paired per post: the e5/simple chunk list minus the five random chunks shown to the same post. The random "
           "lists are identical for Llama-3.1-8B and Ministral-3-8B. Holm over all rows.", "",
           cc.md_table(rnd[rnd["Author party"].isin(["All posts", "DIE LINKE", "AfD"])]), "",
           "All author parties: `retrieval_random5_vs_e5_cluster.csv`.", ""]
    (OUT4 / "cluster_item3_composition.md").write_text("\n".join(md), encoding="utf-8")
    print("  item 3 ->", OUT4 / "cluster_item3_composition.md")


# ===========================================================================
# item 5: evidence use
# ===========================================================================

def boot_coefs(X, y, W, wt=None):
    """(B, p) weighted LS coefficients; NaN where a column has no weight in a draw."""
    Wt = W if wt is None else W * wt[None, :]
    n, p = X.shape
    XtWX = (Wt @ (X[:, :, None] * X[:, None, :]).reshape(n, p * p)).reshape(-1, p, p)
    XtWy = Wt @ (X * y[:, None])
    beta = np.einsum("bij,bj->bi", np.linalg.pinv(XtWX), XtWy)
    dead = (Wt @ X ** 2) == 0
    beta[dead] = np.nan
    return beta


def fit_both(formula, data, D, wt=None, boot=True, perm=None):
    """OLS/WLS fit with SEs clustered by text (old) and by account (new) + account bootstrap."""
    import statsmodels.formula.api as smf
    data = data.reset_index(drop=True)
    mod = smf.wls(formula, data, weights=data[wt]) if wt else smf.ols(formula, data)
    t_codes = pd.factorize(data["text_index"])[0]
    a_codes = D.codes(data["text_index"])
    f_old = mod.fit(cov_type="cluster", cov_kwds={"groups": t_codes})
    f_new = mod.fit(cov_type="cluster", cov_kwds={"groups": a_codes})
    G = len(np.unique(a_codes))
    res = {"old": f_old, "new": f_new, "G": G, "names": list(mod.exog_names), "n": int(f_new.nobs)}
    if boot:
        X, y = np.asarray(mod.exog), np.asarray(mod.endog)
        res["boot"] = boot_coefs(X, y, D.weights(a_codes), data[wt].to_numpy(float) if wt else None)
    return res


def coef_cols(res, term):
    j = res["names"].index(term)
    fo, fn = res["old"], res["new"]
    se = float(fn.bse[term])
    tcrit = stats.t.ppf(0.975, res["G"] - 1)
    out = {"Estimate": float(fn.params[term]),
           "Old CI low (text-clustered)": float(fo.conf_int().loc[term, 0]),
           "Old CI high (text-clustered)": float(fo.conf_int().loc[term, 1]),
           "Old p": float(fo.pvalues[term]),
           "Accounts": res["G"],
           "Account-clustered SE": se,
           "New CI low (account-clustered)": float(fn.conf_int().loc[term, 0]),
           "New CI high (account-clustered)": float(fn.conf_int().loc[term, 1]),
           "New p (account-clustered, normal)": float(fn.pvalues[term]),
           "New p (account-clustered, t G-1)": float(2 * stats.t.sf(abs(fn.params[term] / se), res["G"] - 1)),
           "New CI low (t G-1)": float(fn.params[term] - tcrit * se),
           "New CI high (t G-1)": float(fn.params[term] + tcrit * se)}
    if "boot" in res:
        b = res["boot"][:, j]
        out["Boot CI low (account)"], out["Boot CI high (account)"] = cc.pct(b)
        out["Boot p (account)"] = cc.p_boot(b)
        out["Boot NaN draws"] = int(np.isnan(b).sum())
    return out


def perm_E(data, D, fe: bool, controls: bool, rng, n=N_PERM):
    """Null of the E coefficient under the account-block permutation of E within model."""
    data = data.dropna(subset=["R", "B", "E"]).reset_index(drop=True)
    parts = [np.ones(len(data)), data["B"].to_numpy()]
    dummies = pd.get_dummies(data["model"], drop_first=True, dtype=float) if fe else pd.DataFrame(index=data.index)
    if controls:
        dummies = pd.concat([dummies, pd.get_dummies(data["input_party"], drop_first=True, dtype=float)], axis=1)
    Xf = np.column_stack(parts + ([dummies.to_numpy()] if dummies.shape[1] else []))
    y, E = data["R"].to_numpy(), data["E"].to_numpy()
    groups = [np.flatnonzero(data["model"].to_numpy() == m) for m in data["model"].unique()]
    codes = D.codes(data["text_index"])
    out = np.empty(n)
    for i in range(n):
        Ep = E.copy()
        for g in groups:
            Ep[g] = E[g][block_perm(codes[g], rng)]
        out[i] = np.linalg.lstsq(np.column_stack([Xf, Ep]), y, rcond=None)[0][-1]
    return out


def evidence(D, tag=""):
    pop = D.keep(aci.load_population())
    rng = np.random.default_rng(cc.SEED)
    # (a) chunk_influence evidence_use.csv: pooled, pooled + input party, per model
    rows = []
    specs = [("pooled", "R ~ B + E + C(model)", pop, True, False),
             ("pooled + input party", "R ~ B + E + C(model) + C(input_party)", pop, True, True)]
    specs += [(m, "R ~ B + E", g, False, False) for m, g in pop.groupby("model")]
    for name, f, data, fe, ctrl in specs:
        data = data.dropna(subset=["R", "B", "E"])
        res = fit_both(f, data, D)
        nul = perm_E(data, D, fe, ctrl, rng)
        r = {"model": name, "N": res["n"], **coef_cols(res, "E")}
        r["Account-perm null mean"] = float(nul.mean())
        r["Account-perm null 95 %"] = float(np.percentile(nul, 95))
        r["Account-perm p (one-sided)"] = float((np.sum(nul >= r["Estimate"]) + 1) / (N_PERM + 1))
        rows.append(r)
    ev = pd.DataFrame(rows)
    # (b) subsets and interactions (extremes_evidence_use)
    d, _ = w.extremes_data()
    d = D.keep(d)
    rows, inter = [], []
    for folder, *_ in w.MODELS:
        for scope in w.scopes(folder):
            g = w.in_scope(d[d["model"] == folder], scope).dropna(subset=["E"])
            lab = w.model_label(folder, scope)
            for s in ("all posts",) + w.SUBSETS:
                sub = g if s == "all posts" else g[g["subset"] == s]
                rows.append({"Model": lab, "Posts": scope, "Subset": s, "N posts": len(sub),
                             **coef_cols(fit_both("R ~ B + E", sub, D), "E")})
            inter += contrasts(lab, scope, g, D, fe=False)
    pooled = d.dropna(subset=["E"])
    for s in ("all posts",) + w.SUBSETS:
        sub = pooled if s == "all posts" else pooled[pooled["subset"] == s]
        rows.append({"Model": "All models (model fixed effects)", "Posts": w.SCOPE_ALL, "Subset": s,
                     "N posts": len(sub), **coef_cols(fit_both("R ~ B + E + C(model)", sub, D), "E")})
    inter += contrasts("All models (model fixed effects)", w.SCOPE_ALL, pooled, D, fe=True)
    use, con = pd.DataFrame(rows), pd.DataFrame(inter)
    # (c) cited / uncited, relevant / other (weighted quadrant sample)
    ch = aci.load_sample(aci.DEFAULT_RUN, "final", aci.load_population())
    t = D.keep(aci.text_table(ch))
    lab_rows = []
    for name, f, terms in (("cited vs. uncited chunks", "R ~ B + E_cited + E_uncited + C(model)", ["E_cited", "E_uncited"]),
                           ("relevant (R_top ≥ 4) vs. other chunks", "R ~ B + E_relevant + E_irrelevant + C(model)",
                            ["E_relevant", "E_irrelevant"])):
        data = t.dropna(subset=["R", "B", *terms]).copy()
        res = fit_both(f, data, D, wt="weight")
        for term in terms:
            lab_rows.append({"comparison": name, "term": term, "n_texts": res["n"], **coef_cols(res, term)})
        # difference of the two coefficients
        L = np.zeros(len(res["names"]))
        L[res["names"].index(terms[0])], L[res["names"].index(terms[1])] = 1, -1
        to, tn = res["old"].t_test(L), res["new"].t_test(L)
        bd = res["boot"][:, res["names"].index(terms[0])] - res["boot"][:, res["names"].index(terms[1])]
        lab_rows.append({"comparison": name, "term": f"{terms[0]} − {terms[1]}", "n_texts": res["n"],
                         "Estimate": float(tn.effect[0]), "Old p": float(to.pvalue),
                         "Accounts": res["G"], "New p (account-clustered, normal)": float(tn.pvalue),
                         "Boot CI low (account)": cc.pct(bd)[0], "Boot CI high (account)": cc.pct(bd)[1],
                         "Boot p (account)": cc.p_boot(bd)})
    lab = pd.DataFrame(lab_rows).drop(columns="_", errors="ignore")
    return ev, use, con, lab


def contrasts(label, scope, data, D, fe):
    f = "R ~ (B + E) * C(subset, Treatment('other parties'))" + (" + C(model)" if fe else "")
    res = fit_both(f, data, D)
    blk = []
    for grp in ("DIE LINKE", "AfD"):
        term = f"E:C(subset, Treatment('other parties'))[T.{grp}]"
        blk.append({"Model": label, "Posts": scope, "Contrast": f"E coefficient, {grp} − other parties",
                    **coef_cols(res, term)})
    for key, col in (("Old p", "Old p (Holm, 2)"), ("New p (account-clustered, normal)", "New p (Holm, 2)"),
                     ("New p (account-clustered, t G-1)", "New p t (Holm, 2)"), ("Boot p (account)", "Boot p (Holm, 2)")):
        for b, p in zip(blk, cc.holm([x[key] for x in blk])):
            b[col] = p
    return blk


def sig(lo, hi):
    return bool(lo > 0 or hi < 0)


def flag_use(use):
    """Subset fits: old / CR1 / bootstrap significance and the robust rule
    (CR1 with t(G-1) AND account bootstrap CI exclude 0)."""
    use["Old sig"] = [sig(a, b) for a, b in zip(use["Old CI low (text-clustered)"], use["Old CI high (text-clustered)"])]
    use["New sig (CR1)"] = [sig(a, b) for a, b in zip(use["New CI low (account-clustered)"], use["New CI high (account-clustered)"])]
    use["New sig (t G-1)"] = [sig(a, b) for a, b in zip(use["New CI low (t G-1)"], use["New CI high (t G-1)"])]
    use["Boot sig"] = [sig(a, b) for a, b in zip(use["Boot CI low (account)"], use["Boot CI high (account)"])]
    use["New sig (robust)"] = use["New sig (t G-1)"] & use["Boot sig"]
    use["Changed"] = use["Old sig"] != use["New sig (robust)"]
    return use


def flag_con(con):
    con["Old sig"] = con["Old p (Holm, 2)"] < 0.05
    con["New sig (CR1)"] = con["New p (Holm, 2)"] < 0.05
    con["New sig (t G-1)"] = con["New p t (Holm, 2)"] < 0.05
    con["Boot sig"] = con["Boot p (Holm, 2)"] < 0.05
    con["New sig (robust)"] = con["New sig (t G-1)"] & con["Boot sig"]
    con["Changed"] = con["Old sig"] != con["New sig (robust)"]
    return con


def evidence_report(full, sens, Dfull, Dsens):
    ev, use, con, lab = full
    old_ev = pd.read_csv(OUT_CI / "evidence_use.csv").set_index("model")
    old_use = pd.read_csv(OUT4 / "extremes_evidence_use.csv")
    old_con = pd.read_csv(OUT4 / "extremes_evidence_use_contrasts.csv")
    old_lab = pd.read_csv(OUT_CI / "evidence_by_label.csv")
    # consistency checks
    d1 = (ev.set_index("model")["Estimate"] - old_ev["coef"]).abs().max()
    m = use.merge(old_use, on=["Model", "Posts", "Subset"], suffixes=("", "_o"))
    d2 = (m["Estimate"] - m["E coefficient"]).abs().max()
    mc = con.merge(old_con, on=["Model", "Posts", "Contrast"])
    d3 = (mc["Estimate"] - mc["Difference"]).abs().max()
    if max(d1, d2, d3) > 1e-3:
        raise SystemExit(f"evidence point estimates differ from stored CSVs ({d1}, {d2}, {d3})")
    ev["Old perm p"] = ev["model"].map(old_ev["perm_p"])
    ev["Old sig"] = [sig(a, b) for a, b in zip(ev["Old CI low (text-clustered)"], ev["Old CI high (text-clustered)"])]
    ev["New sig"] = [sig(a, b) for a, b in zip(ev["New CI low (account-clustered)"], ev["New CI high (account-clustered)"])]
    flag_use(use)
    flag_con(con)
    lab["Old sig"] = lab["Old p"] < 0.05
    lab["New sig"] = lab["New p (account-clustered, normal)"] < 0.05
    for df in (ev, lab):
        df["Changed"] = df["Old sig"] != df["New sig"]
    ev.to_csv(OUT_CI / "evidence_use_cluster.csv", index=False)
    lab.to_csv(OUT_CI / "evidence_by_label_cluster.csv", index=False)
    use.to_csv(OUT4 / "extremes_evidence_use_cluster.csv", index=False)
    con.to_csv(OUT4 / "extremes_evidence_use_contrasts_cluster.csv", index=False)

    def ci_s(df, a, b):
        return [fmt_ci(e, l, h) for e, l, h in zip(df["Estimate"], df[a], df[b])]

    md = ["# Item 5: evidence use R ~ B + E with account-level inference", "",
          "OLS as before (pooled fits with model fixed effects). OLD = SEs clustered by text (one row per text and "
          "model, so per model this is the heteroskedasticity-robust SE). NEW = SEs clustered by author account "
          "(CR1, normal reference as before; the t(G−1) column uses the number of accounts minus 1 as df). Account "
          "bootstrap = pairs cluster bootstrap with the shared draws (weighted LS per draw). Permutation: E is permuted "
          "within each model by account blocks (see the script docstring), 1000 permutations, seed 42.", "",
          "## Population fits (chunk_influence_all/evidence_use.csv)", ""]
    t = ev.assign(**{"E [old CI]": ci_s(ev, "Old CI low (text-clustered)", "Old CI high (text-clustered)"),
                     "E [account-clustered CI]": ci_s(ev, "New CI low (account-clustered)", "New CI high (account-clustered)"),
                     "E [account bootstrap CI]": ci_s(ev, "Boot CI low (account)", "Boot CI high (account)")})
    md.append(cc.md_table(t[["model", "N", "Accounts", "E [old CI]", "E [account-clustered CI]",
                             "E [account bootstrap CI]", "Old perm p", "Account-perm null 95 %",
                             "Account-perm p (one-sided)", "Changed"]]))
    md += ["", "## By subset (extremes_evidence_use.csv)", ""]
    t = use.assign(**{"E [old CI]": ci_s(use, "Old CI low (text-clustered)", "Old CI high (text-clustered)"),
                      "E [account-clustered CI]": ci_s(use, "New CI low (account-clustered)", "New CI high (account-clustered)"),
                      "E [account bootstrap CI]": ci_s(use, "Boot CI low (account)", "Boot CI high (account)")})
    t = t[t["Subset"] != "all posts"]
    md.append(cc.md_table(t[["Model", "Subset", "N posts", "Accounts", "E [old CI]", "E [account-clustered CI]",
                             "E [account bootstrap CI]", "Old sig", "New sig (CR1)", "New sig (t G-1)", "Boot sig",
                             "New sig (robust)", "Changed"]]))
    md += ["", "DIE LINKE posts come from 7 accounts and AfD posts from 12, and one account (SWagenknecht, 121 of "
           "179 posts) dominates DIE LINKE. With so few clusters the CR1 SE is biased downward: it comes out "
           "narrower than the old text-clustered SE in several DIE LINKE rows. The percentile cluster bootstrap "
           "is unstable as well, because a replicate can contain zero or seven copies of the dominant account. For "
           "the subsets and the contrasts, a result therefore counts as significant ('robust') only if the CR1 "
           "t(G−1) CI AND the account bootstrap CI both exclude 0. 'Changed' compares this robust rule with the "
           "old result.", "",
           "## Interaction contrasts (extremes_evidence_use_contrasts.csv)", ""]
    t = con.assign(**{"Diff [old CI]": ci_s(con, "Old CI low (text-clustered)", "Old CI high (text-clustered)"),
                      "Diff [account-clustered CI]": ci_s(con, "New CI low (account-clustered)", "New CI high (account-clustered)"),
                      "Diff [account bootstrap CI]": ci_s(con, "Boot CI low (account)", "Boot CI high (account)")})
    md.append(cc.md_table(t[["Model", "Contrast", "Diff [old CI]", "Old p (Holm, 2)", "Diff [account-clustered CI]",
                             "New p (Holm, 2)", "New p t (Holm, 2)", "Diff [account bootstrap CI]",
                             "Boot p (Holm, 2)", "New sig (robust)", "Changed"]]))
    md += ["", "## Cited/uncited and relevant/other (weighted quadrant sample, chunk_influence_all/evidence_by_label.csv)", ""]
    md.append(cc.md_table(lab[["comparison", "term", "n_texts", "Accounts", "Estimate", "Old p",
                               "New CI low (account-clustered)", "New CI high (account-clustered)",
                               "New p (account-clustered, normal)", "Boot CI low (account)", "Boot CI high (account)",
                               "Boot p (account)", "Changed"]]))
    # sensitivity
    if sens is not None:
        sev, suse, scon, slab = sens
        flag_use(suse)
        flag_con(scon)
        md += ["", f"## Sensitivity: {Dsens.label}", "",
               "Population fits: account-clustered CI. Subsets and contrasts: the robust rule.", ""]
        rows = []
        lo, hi = "New CI low (account-clustered)", "New CI high (account-clustered)"
        j = ev.merge(sev, on=["model"], suffixes=("", "_s"))
        for _, r in j.iterrows():
            if sig(r[lo], r[hi]) != sig(r[lo + "_s"], r[hi + "_s"]):
                rows.append({"Table": "population", "Row": r["model"], "Full": fmt_ci(r["Estimate"], r[lo], r[hi]),
                             "Without largest account": fmt_ci(r["Estimate_s"], r[lo + "_s"], r[hi + "_s"])})
        j = use.merge(suse, on=["Model", "Subset"], suffixes=("", "_s"))
        for _, r in j.iterrows():
            if r["New sig (robust)"] != r["New sig (robust)_s"]:
                rows.append({"Table": "subset", "Row": f"{r['Model']} / {r['Subset']}",
                             "Full": fmt_ci(r["Estimate"], r[lo], r[hi]) + f" robust={r['New sig (robust)']}",
                             "Without largest account": fmt_ci(r["Estimate_s"], r[lo + "_s"], r[hi + "_s"])
                             + f" robust={r['New sig (robust)_s']}"})
        j = con.merge(scon, on=["Model", "Contrast"], suffixes=("", "_s"))
        for _, r in j.iterrows():
            if r["New sig (robust)"] != r["New sig (robust)_s"]:
                rows.append({"Table": "contrast", "Row": f"{r['Model']} / {r['Contrast']}",
                             "Full": f"{r['Estimate']:+.2f}, p_Holm CR1 {r['New p (Holm, 2)']:.3f} / boot {r['Boot p (Holm, 2)']:.3f}",
                             "Without largest account": f"{r['Estimate_s']:+.2f}, p_Holm CR1 {r['New p (Holm, 2)_s']:.3f}"
                                                        f" / boot {r['Boot p (Holm, 2)_s']:.3f}"})
        j = lab.merge(slab, on=["comparison", "term"], suffixes=("", "_s"))
        for _, r in j.iterrows():
            if (r["New p (account-clustered, normal)"] < 0.05) != (r["New p (account-clustered, normal)_s"] < 0.05):
                rows.append({"Table": "by label", "Row": f"{r['comparison']} / {r['term']}",
                             "Full": f"{r['Estimate']:+.2f}, p {r['New p (account-clustered, normal)']:.3f}",
                             "Without largest account": f"{r['Estimate_s']:+.2f}, p {r['New p (account-clustered, normal)_s']:.3f}"})
        md += [f"Conclusions (account-clustered) that change without the largest account: {len(rows)}", ""]
        if rows:
            md.append(cc.md_table(pd.DataFrame(rows)))
        for df, nm in ((sev, "evidence_use_cluster_sens.csv"), (slab, "evidence_by_label_cluster_sens.csv")):
            df.to_csv(OUT_CI / nm, index=False)
        suse.to_csv(OUT4 / "extremes_evidence_use_cluster_sens.csv", index=False)
        scon.to_csv(OUT4 / "extremes_evidence_use_contrasts_cluster_sens.csv", index=False)
    (OUT4 / "cluster_item5_evidence.md").write_text("\n".join(md), encoding="utf-8")
    print("  item 5 ->", OUT4 / "cluster_item5_evidence.md")


# ===========================================================================
# item 6: direction
# ===========================================================================

OVERALL = ["Mean absolute error", "Mean signed error", "Mean prediction", "SD of predictions", "Calibration slope",
           "Calibration intercept", "Spread AfD - DIE LINKE", "Share in centre band [3.4, 4.6)"]


def dir_metrics(T, P, L, A, W):
    centre = ((P >= LO_C) & (P < HI_C)).astype(float)
    slope = cc.weighted_slope(T, P, W)
    return {"Mean absolute error": cc.wmean(np.abs(P - T), W), "Mean signed error": cc.wmean(P - T, W),
            "Mean prediction": cc.wmean(P, W), "SD of predictions": cc.weighted_sd(P, W),
            "Calibration slope": slope, "Calibration intercept": cc.wmean(P, W) - slope * cc.wmean(T, W),
            "Spread AfD - DIE LINKE": cc.wmean(P, W, A) - cc.wmean(P, W, L),
            "Share in centre band [3.4, 4.6)": cc.wmean(centre, W)}


def direction(D):
    d = D.keep(w.direction_data())
    ov, mv, bp = [], [], []
    for folder, *_ in w.MODELS:
        for scope in w.scopes(folder):
            g = w.in_scope(d[d["model"] == folder], scope)
            lab = w.model_label(folder, scope)
            T, Bp, R = (g[c].to_numpy(float) for c in ("target", "B", "R"))
            L = (g["party"] == "DIE LINKE").to_numpy(float)
            A = (g["party"] == "AfD").to_numpy(float)
            codes = D.codes(g["text_index"])
            W1 = np.ones((1, len(g)))
            Wb = D.weights(codes)
            mb, mr = dir_metrics(T, Bp, L, A, W1), dir_metrics(T, R, L, A, W1)
            bb, br = dir_metrics(T, Bp, L, A, Wb), dir_metrics(T, R, L, A, Wb)
            for st in OVERALL:
                diff = br[st] - bb[st]
                lo, hi = cc.pct(diff)
                ov.append({"Model": lab, "Posts": scope, "N": len(g), "Accounts": D.n_accounts(g["text_index"]),
                           "Statistic": st, "Baseline": float(mb[st][0]), "RAG": float(mr[st][0]),
                           "Difference (RAG - baseline)": float(mr[st][0] - mb[st][0]),
                           "Cluster CI low": lo, "Cluster CI high": hi, "Bootstrap p (cluster)": cc.p_boot(diff)})
            move = R - Bp
            for party in ["All posts", "DIE LINKE", "AfD"]:
                msk = np.ones(len(g), bool) if party == "All posts" else (g["party"] == party).to_numpy()
                c = codes[msk]
                left, right = (move[msk] <= -w.MOVE_THRESHOLD).astype(float), (move[msk] >= w.MOVE_THRESHOLD).astype(float)
                row = {"Model": lab, "Posts": scope, "Author party": party, "N": int(msk.sum()),
                       "Accounts": D.n_accounts(g["text_index"][msk])}
                for nm, x in (("moved left", left), ("moved right", right), ("unchanged", 1 - left - right),
                              ("moved (|shift| >= 0.5)", left + right), ("right - left", right - left)):
                    bs = D.boot_mean(c, x)
                    row[f"Share {nm}"] = float(x.mean())
                    row[f"Share {nm} CI low"], row[f"Share {nm} CI high"] = cc.pct(bs)
                row["Bootstrap p (right - left)"] = cc.p_boot(D.boot_mean(c, right - left))
                mv.append(row)
            blk = []
            for party in w.AUTHOR_GROUPS:
                msk = (g["party"] == party).to_numpy()
                s = g[msk]
                dae = (s["ae_R"] - s["ae_B"]).to_numpy(float)
                dse = (s["se_R"] - s["se_B"]).to_numpy(float)
                b1, b2 = D.boot_mean(codes[msk], dae), D.boot_mean(codes[msk], dse)
                t1, t2 = cc.cr1_t(codes[msk], dae), cc.cr1_t(codes[msk], dse)
                blk.append({"Model": lab, "Posts": scope, "Author party": w.GROUP_DISPLAY[party], "N": len(s),
                            "Accounts": D.n_accounts(s["text_index"]),
                            "dMAE": float(dae.mean()), "dMAE CI low": cc.pct(b1)[0], "dMAE CI high": cc.pct(b1)[1],
                            "dMAE p (cluster)": cc.p_boot(b1), "d signed error": float(dse.mean()),
                            "d signed error CI low": cc.pct(b2)[0], "d signed error CI high": cc.pct(b2)[1],
                            "d signed error p (cluster)": cc.p_boot(b2), "NaN draws": int(np.isnan(b1).sum()),
                            "dMAE CR1-t low": t1[0], "dMAE CR1-t high": t1[1], "dMAE CR1-t p": t1[2],
                            "d signed error CR1-t low": t2[0], "d signed error CR1-t high": t2[1],
                            "d signed error CR1-t p": t2[2]})
            blk = pd.DataFrame(blk)
            blk["dMAE p (Holm, across parties)"] = cc.holm(blk["dMAE p (cluster)"])
            blk["d signed error p (Holm, across parties)"] = cc.holm(blk["d signed error p (cluster)"])
            bp.append(blk)
    ov = pd.DataFrame(ov)
    ov["p (Holm, within model)"] = np.nan
    for _, g in ov.groupby("Model"):
        fam = g[g["Statistic"] != "Mean prediction"]
        ov.loc[fam.index, "p (Holm, within model)"] = cc.holm(fam["Bootstrap p (cluster)"])
    bp = pd.concat(bp, ignore_index=True)
    bp["dMAE CR1-t p (Holm, across parties)"] = np.nan
    for _, ix in bp.groupby("Model").groups.items():
        bp.loc[ix, "dMAE CR1-t p (Holm, across parties)"] = cc.holm(bp.loc[ix, "dMAE CR1-t p"])
    comp = w.direction_compression(bp)
    comp_t = w.direction_compression(bp.assign(**{"d signed error CI low": bp["d signed error CR1-t low"],
                                                  "d signed error CI high": bp["d signed error CR1-t high"]}))
    comp["Classification (CR1-t)"] = comp_t["Classification"].values
    comp["Detail (CR1-t)"] = comp_t["Detail"].values
    return ov, pd.DataFrame(mv), bp, comp


def direction_report(full, sens, Dsens):
    ov, mv, bp, comp = full
    o_ov = pd.read_csv(OUT4 / "direction_overall.csv")
    o_bp = pd.read_csv(OUT4 / "direction_by_party.csv")
    o_bp["Author party"] = o_bp["Author party"].replace({"Grüne": "Die Grünen"})  # label changed since the old run
    o_cp = pd.read_csv(OUT4 / "direction_compression.csv")
    o_mv = pd.read_csv(OUT4 / "direction_movement.csv")
    ov = ov.merge(o_ov[["Model", "Statistic", "Difference (RAG - baseline)", "CI low", "CI high", "Test",
                        "p (Holm, within model)"]].rename(columns={
        "Difference (RAG - baseline)": "_d", "CI low": "Post CI low", "CI high": "Post CI high", "Test": "Old test",
        "p (Holm, within model)": "Old p (Holm)"}), on=["Model", "Statistic"], how="left")
    bp = bp.merge(o_bp[["Model", "Author party", "dMAE", "dMAE CI low", "dMAE CI high", "dMAE p (Holm, across parties)",
                        "d signed error CI low", "d signed error CI high", "d signed error p (Holm, across parties)"]]
                  .rename(columns=lambda c: c if c in ("Model", "Author party") else "Old " + c),
                  on=["Model", "Author party"], how="left")
    if ov["_d"].isna().any() or bp["Old dMAE"].isna().any():
        raise SystemExit("direction: some rows have no stored counterpart (label mismatch?)")
    if max((ov["Difference (RAG - baseline)"] - ov["_d"]).abs().max(), (bp["dMAE"] - bp["Old dMAE"]).abs().max()) > 1e-3:
        raise SystemExit("direction point estimates differ from stored CSVs")
    ov = ov.drop(columns="_d")
    bp = bp.drop(columns="Old dMAE")
    ov["Old CI direction"] = [cc.direction(l, h, "down", "up") for l, h in zip(ov["Post CI low"], ov["Post CI high"])]
    ov["New CI direction"] = [cc.direction(l, h, "down", "up") for l, h in zip(ov["Cluster CI low"], ov["Cluster CI high"])]
    mp_ = ov.Statistic == "Mean prediction"
    ov["Old sig (Holm)"] = [None if m else bool(p < 0.05) for m, p in zip(mp_, ov["Old p (Holm)"])]
    ov["New sig (Holm)"] = [None if m else bool(p < 0.05) for m, p in zip(mp_, ov["p (Holm, within model)"])]
    ov["Changed (CI)"] = ov["Old CI direction"] != ov["New CI direction"]
    for k in ("dMAE", "d signed error"):
        bp[f"Old {k} CI dir"] = [cc.direction(l, h, "down", "up") for l, h in zip(bp[f"Old {k} CI low"], bp[f"Old {k} CI high"])]
        bp[f"New {k} CI dir"] = [cc.direction(l, h, "down", "up") for l, h in zip(bp[f"{k} CI low"], bp[f"{k} CI high"])]
        bp[f"Changed {k} (CI)"] = bp[f"Old {k} CI dir"] != bp[f"New {k} CI dir"]
        bp[f"Changed {k} (Holm)"] = (bp[f"Old {k} p (Holm, across parties)"] < 0.05) != (bp[f"{k} p (Holm, across parties)"] < 0.05)
    comp = comp.merge(o_cp[["Model", "Classification", "DIE LINKE CI low", "DIE LINKE CI high", "AfD CI low", "AfD CI high"]]
                      .rename(columns=lambda c: c if c == "Model" else "Old " + c), on="Model", how="left")
    comp["Class changed"] = comp["Classification"] != comp["Old Classification"]
    mv = mv.merge(o_mv[["Model", "Author party", "Share moved left CI low", "Share moved left CI high",
                        "Share moved right CI low", "Share moved right CI high"]]
                  .rename(columns=lambda c: c if c in ("Model", "Author party") else "Old " + c),
                  on=["Model", "Author party"], how="left")
    ov.to_csv(OUT4 / "direction_overall_cluster.csv", index=False)
    mv.to_csv(OUT4 / "direction_movement_cluster.csv", index=False)
    bp.to_csv(OUT4 / "direction_by_party_cluster.csv", index=False)
    comp.to_csv(OUT4 / "direction_compression_cluster.csv", index=False)

    md = ["# Item 6: direction of the RAG shift with account-level inference", "",
          "Paired e5/simple vs no-RAG posts per model, the same posts and point estimates as 04_why_rag_direction.md. "
          "NEW = paired account cluster bootstrap with the shared draws. Every statistic is recomputed on the resampled "
          "accounts (weights = account multiplicities), with a two-sided bootstrap p and Holm within model, as before. "
          "OLD = post bootstrap CI, with Wilcoxon or post bootstrap p.", "",
          "## Overall statistics (difference RAG − baseline)", ""]
    keep = ["Mean signed error", "SD of predictions", "Calibration slope", "Spread AfD - DIE LINKE",
            "Share in centre band [3.4, 4.6)", "Mean absolute error"]
    t = ov[ov.Statistic.isin(keep)].copy()
    t["Old [post CI]"] = [fmt_ci(e, l, h) for e, l, h in zip(t["Difference (RAG - baseline)"], t["Post CI low"], t["Post CI high"])]
    t["New [account CI]"] = [fmt_ci(e, l, h) for e, l, h in zip(t["Difference (RAG - baseline)"], t["Cluster CI low"], t["Cluster CI high"])]
    md.append(cc.md_table(t[["Model", "Statistic", "Old [post CI]", "Old p (Holm)", "New [account CI]",
                             "p (Holm, within model)", "Old CI direction", "New CI direction", "Changed (CI)"]]))
    md += ["", "## Share of posts moving ≥ 0.5 (all posts)", ""]
    t = mv[mv["Author party"] == "All posts"]
    md.append(cc.md_table(t.assign(
        Moved=[fmt_ci(e, l, h) for e, l, h in zip(t["Share moved (|shift| >= 0.5)"], t["Share moved (|shift| >= 0.5) CI low"], t["Share moved (|shift| >= 0.5) CI high"])],
        **{"Right − left": [fmt_ci(e, l, h) for e, l, h in zip(t["Share right - left"], t["Share right - left CI low"], t["Share right - left CI high"])]}
    )[["Model", "N", "Moved", "Right − left", "Bootstrap p (right - left)"]]))
    md += ["", "DIE LINKE and AfD movement shares are in `direction_movement_cluster.csv`.", "",
           "## Per-party changes, DIE LINKE and AfD", ""]
    t = bp[bp["Author party"].isin(["DIE LINKE", "AfD"])].copy()
    t["dMAE old"] = [fmt_ci(e, l, h) for e, l, h in zip(t["dMAE"], t["Old dMAE CI low"], t["Old dMAE CI high"])]
    t["dMAE new"] = [fmt_ci(e, l, h) for e, l, h in zip(t["dMAE"], t["dMAE CI low"], t["dMAE CI high"])]
    t["dSE old"] = [fmt_ci(e, l, h) for e, l, h in zip(t["d signed error"], t["Old d signed error CI low"], t["Old d signed error CI high"])]
    t["dSE new"] = [fmt_ci(e, l, h) for e, l, h in zip(t["d signed error"], t["d signed error CI low"], t["d signed error CI high"])]
    t["dSE CR1-t"] = [fmt_ci(e, l, h) for e, l, h in zip(t["d signed error"], t["d signed error CR1-t low"], t["d signed error CR1-t high"])]
    md.append(cc.md_table(t[["Model", "Author party", "N", "Accounts", "dMAE old", "Old dMAE p (Holm, across parties)",
                             "dMAE new", "dMAE p (Holm, across parties)", "dSE old", "dSE new", "dSE CR1-t",
                             "Changed dMAE (CI)", "Changed d signed error (CI)"]]))
    many = bp["Accounts"] >= FEW_ACCOUNTS
    md += ["", "'new' = account percentile bootstrap. 'CR1-t' = mean with a cluster-robust SE and a t(G−1) interval. "
           "DIE LINKE rests on 7 accounts, and SWagenknecht holds 121 of its 179 posts. In a bootstrap replicate "
           "that account appears 0 to 4 times, so the percentile bounds are erratic (for example, an upper bound of "
           "+3.17 for Llama 3B). With an effective sample of 1-2 accounts, neither interval is reliable for DIE "
           "LINKE. SPD (5 accounts, 193 of 217 posts from Karl_Lauterbach) and FDP (4 accounts) are worse still.", "",
           f"Parties with ≥ {FEW_ACCOUNTS} accounts (Die Grünen, CDU/CSU, AfD): "
           f"{int(bp.loc[many, 'Changed dMAE (CI)'].sum())} of {int(many.sum())} dMAE CI directions change, "
           f"{int(bp.loc[many, 'Changed d signed error (CI)'].sum())} signed-error CI directions change and "
           f"{int(bp.loc[many, 'Changed dMAE (Holm)'].sum())} dMAE Holm decisions change. Parties with fewer accounts "
           f"(DIE LINKE, SPD, FDP): {int(bp.loc[~many, 'Changed dMAE (CI)'].sum())} of {int((~many).sum())} dMAE CI directions, "
           f"{int(bp.loc[~many, 'Changed d signed error (CI)'].sum())} signed-error CI directions and "
           f"{int(bp.loc[~many, 'Changed dMAE (Holm)'].sum())} Holm decisions change "
           "(all rows: `direction_by_party_cluster.csv`).", "",
           "## Compress / expand / shift classification", "",
           "The rule is unchanged (`why_rag_analysis.COMPRESSION_RULE`). It is applied once to the account bootstrap "
           "CIs and once to the CR1-t CIs of the DIE LINKE and AfD signed-error changes.", ""]
    t = comp.copy()
    t["DIE LINKE d [new CI]"] = [fmt_ci(e, l, h) for e, l, h in zip(t["DIE LINKE d signed error"], t["DIE LINKE CI low"], t["DIE LINKE CI high"])]
    t["AfD d [new CI]"] = [fmt_ci(e, l, h) for e, l, h in zip(t["AfD d signed error"], t["AfD CI low"], t["AfD CI high"])]
    tt = bp.set_index(["Model", "Author party"])
    t["DIE LINKE d [CR1-t]"] = [fmt_ci(tt.loc[(m, "DIE LINKE"), "d signed error"], tt.loc[(m, "DIE LINKE"), "d signed error CR1-t low"],
                                       tt.loc[(m, "DIE LINKE"), "d signed error CR1-t high"]) for m in t.Model]
    t["AfD d [CR1-t]"] = [fmt_ci(tt.loc[(m, "AfD"), "d signed error"], tt.loc[(m, "AfD"), "d signed error CR1-t low"],
                                 tt.loc[(m, "AfD"), "d signed error CR1-t high"]) for m in t.Model]
    t["Class changed (CR1-t)"] = t["Classification (CR1-t)"] != t["Old Classification"]
    md.append(cc.md_table(t[["Model", "Old Classification", "DIE LINKE d [new CI]", "AfD d [new CI]", "Classification",
                             "DIE LINKE d [CR1-t]", "AfD d [CR1-t]", "Classification (CR1-t)", "Class changed",
                             "Class changed (CR1-t)"]]))
    cnt = comp[comp["Posts"] == w.SCOPE_ALL]  # as in the old report: Llama 3B on all paired posts only
    vc = lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().items())  # noqa: E731
    md += ["", "Class counts (Llama 3B counted once, all posts, as in the old report). OLD: "
           + vc(cnt["Old Classification"]) + ". NEW (bootstrap): " + vc(cnt["Classification"])
           + ". NEW (CR1-t): " + vc(cnt["Classification (CR1-t)"]) + ".", ""]
    if sens is not None:
        sov, smv, sbp, scomp = sens
        sov.to_csv(OUT4 / "direction_overall_cluster_sens.csv", index=False)
        smv.to_csv(OUT4 / "direction_movement_cluster_sens.csv", index=False)
        sbp.to_csv(OUT4 / "direction_by_party_cluster_sens.csv", index=False)
        scomp.to_csv(OUT4 / "direction_compression_cluster_sens.csv", index=False)
        md += [f"## Sensitivity: {Dsens.label}", ""]
        j = ov.merge(sov, on=["Model", "Statistic"], suffixes=("", "_s"))
        j["dir_s"] = [cc.direction(l, h, "down", "up") for l, h in zip(j["Cluster CI low_s"], j["Cluster CI high_s"])]
        ch = j[(j["New CI direction"] != j["dir_s"]) & j.Statistic.isin(keep)]
        md += [f"Overall statistics whose account-CI direction changes: {len(ch)}", ""]
        if len(ch):
            md.append(cc.md_table(ch.assign(
                Full=[fmt_ci(e, l, h) for e, l, h in zip(ch["Difference (RAG - baseline)"], ch["Cluster CI low"], ch["Cluster CI high"])],
                Sens=[fmt_ci(e, l, h) for e, l, h in zip(ch["Difference (RAG - baseline)_s"], ch["Cluster CI low_s"], ch["Cluster CI high_s"])]
            )[["Model", "Statistic", "Full", "Sens"]]))
        j = comp.merge(scomp[["Model", "Classification", "Classification (CR1-t)", "DIE LINKE d signed error",
                              "AfD d signed error"]], on="Model", suffixes=("", "_s"))
        ch = j[(j["Classification"] != j["Classification_s"]) |
               (j["Classification (CR1-t)"] != j["Classification (CR1-t)_s"])]
        md += ["", f"Classes (bootstrap or CR1-t) that change without the largest account: {len(ch)}", ""]
        if len(ch):
            md.append(cc.md_table(ch[["Model", "Classification", "Classification_s", "Classification (CR1-t)",
                                      "Classification (CR1-t)_s", "DIE LINKE d signed error_s", "AfD d signed error_s"]]))
        j = bp.merge(sbp, on=["Model", "Author party"], suffixes=("", "_s"))
        j["dir_s"] = [cc.direction(l, h, "down", "up") for l, h in zip(j["dMAE CI low_s"], j["dMAE CI high_s"])]
        ch = j[j["New dMAE CI dir"] != j["dir_s"]]
        md += ["", f"Per-party dMAE whose account-CI direction changes: {len(ch)}", ""]
        if len(ch):
            md.append(cc.md_table(ch.assign(
                Full=[fmt_ci(e, l, h) for e, l, h in zip(ch["dMAE"], ch["dMAE CI low"], ch["dMAE CI high"])],
                Sens=[fmt_ci(e, l, h) for e, l, h in zip(ch["dMAE_s"], ch["dMAE CI low_s"], ch["dMAE CI high_s"])]
            )[["Model", "Author party", "N_s", "Full", "Sens"]]))
        md.append("")
    (OUT4 / "cluster_item6_direction.md").write_text("\n".join(md), encoding="utf-8")
    print("  item 6 ->", OUT4 / "cluster_item6_direction.md")


# ===========================================================================
# item 7: chunk ablation
# ===========================================================================

def wcorr(T, P, W):
    sw = W.sum(1, keepdims=True)
    mt, mp = (W * T).sum(1, keepdims=True) / sw, (W * P).sum(1, keepdims=True) / sw
    return (W * (T - mt) * (P - mp)).sum(1) / np.sqrt((W * (T - mt) ** 2).sum(1) * (W * (P - mp) ** 2).sum(1))


def paired_delta(df, model, cond, ref, posts, D):
    a = df[(df.model == model) & (df.condition == cond)][["text_index", "true", "pred"]]
    b = df[(df.model == model) & (df.condition == ref)][["text_index", "pred"]]
    m = a.merge(b, on="text_index", suffixes=("_a", "_b"))
    if posts is not None:
        m = m[m.text_index.isin(posts)]
    m = D.keep(m)
    T, Pa, Pb = (m[c].to_numpy(float) for c in ("true", "pred_a", "pred_b"))
    codes = D.codes(m.text_index)
    Wb = D.weights(codes)
    one = np.ones((1, len(m)))
    dae = np.abs(Pa - T) - np.abs(Pb - T)
    bs = D.boot_mean(codes, dae)
    dbias = lambda W: np.abs(cc.wmean(Pa - T, W)) - np.abs(cc.wmean(Pb - T, W))  # noqa: E731
    dr = lambda W: wcorr(T, Pa, W) - wcorr(T, Pb, W)  # noqa: E731
    out = {"Model": model, "Condition": cond, "Reference": ref, "N_paired": len(m),
           "Accounts": D.n_accounts(m.text_index), "dMAE": float(dae.mean()),
           "Cluster CI low": cc.pct(bs)[0], "Cluster CI high": cc.pct(bs)[1], "p_boot_cluster": cc.p_boot(bs),
           "d|Bias|": float(dbias(one)[0]), "d|Bias| CI low": cc.pct(dbias(Wb))[0], "d|Bias| CI high": cc.pct(dbias(Wb))[1],
           "dPearson_r": float(dr(one)[0]), "dPearson_r CI low": cc.pct(dr(Wb))[0],
           "dPearson_r CI high": cc.pct(dr(Wb))[1]}
    return out


def ablation(D):
    import chunk_ablation_analysis as cal
    df = cal.load()
    rtop = df[df.condition.str.contains("rtop")]
    df_main = df[~df.condition.str.contains("rtop")]
    rows = []
    for m in sorted(df_main.model.unique()):
        for ref in ("no_rag", "e5/simple"):
            rows.append({"Table": "all posts", **paired_delta(df_main, m, "e5pool/random5", ref, None, D)})
    for m in sorted(rtop.model.unique()):
        posts = set(rtop[rtop.model == m].text_index)
        both = pd.concat([df_main[df_main.text_index.isin(posts)], rtop])
        for cond, ref in (("e5/rtop4plus", "no_rag"), ("e5/rtop4plus", "e5/simple"), ("e5/rtop4plus", "e5pool/random5"),
                          ("e5pool/random5", "no_rag"), ("e5pool/random5", "e5/simple")):
            rows.append({"Table": "R_top posts", **paired_delta(both, m, cond, ref, posts, D)})
    out = pd.DataFrame(rows)
    out["p_holm_cluster"] = np.nan
    for _, ix in out.groupby(["Table", "Model"]).groups.items():
        out.loc[ix, "p_holm_cluster"] = cc.holm(out.loc[ix, "p_boot_cluster"])
    return out


def ablation_report(full, sens, Dsens):
    old = pd.concat([pd.read_csv(OUT5 / "chunk_ablation_paired.csv").assign(Table="all posts"),
                     pd.read_csv(OUT5 / "chunk_ablation_rtop_paired.csv").assign(Table="R_top posts")])
    old = old.rename(columns={"ΔMAE (RAG-Base)": "dMAE_old", "ΔMAE (RAG-Base)_lo": "Post CI low",
                              "ΔMAE (RAG-Base)_hi": "Post CI high", "ΔMAE_p_holm": "Old p_holm",
                              "N_paired": "N_old"})
    f = full.merge(old[["Table", "Model", "Condition", "Reference", "N_old", "dMAE_old", "Post CI low", "Post CI high",
                        "Old p_holm"]], on=["Table", "Model", "Condition", "Reference"], how="left")
    if ((f.N_paired != f.N_old) | ((f.dMAE - f.dMAE_old).abs() > 1e-3)).any():
        print(f[["Table", "Model", "Condition", "Reference", "N_paired", "N_old", "dMAE", "dMAE_old"]])
        raise SystemExit("ablation point estimates differ from stored CSVs")
    f = f.drop(columns=["N_old", "dMAE_old"])
    # old Holm per (model, condition) family of one comparison -> raw bootstrap p; new: Holm within table x model
    f["Old sig"] = f["Old p_holm"] < 0.05
    f["New sig"] = f["p_holm_cluster"] < 0.05
    f["Changed"] = f["Old sig"] != f["New sig"]
    f.to_csv(OUT5 / "chunk_ablation_paired_cluster.csv", index=False)
    md = ["# Item 7: chunk ablations with account-level inference", "",
          "Paired ΔMAE (condition − reference) on the posts shared with the reference, as in chunk_ablation.md and "
          "chunk_ablation_rtop.md. NEW = account cluster bootstrap with the shared draws, two-sided bootstrap p, Holm "
          "within model over the comparisons of a table (2 on all posts, 5 on the R_top posts). OLD = post bootstrap. "
          "Its Holm family had one comparison per call, so the old 'p_holm' is the unadjusted bootstrap p.", ""]
    t = f.assign(Model=f.Model.map(cc.short),
                 **{"dMAE [post CI]": [fmt_ci(e, l, h, 3) for e, l, h in zip(f.dMAE, f["Post CI low"], f["Post CI high"])],
                    "dMAE [account CI]": [fmt_ci(e, l, h, 3) for e, l, h in zip(f.dMAE, f["Cluster CI low"], f["Cluster CI high"])],
                    "d|Bias| [account CI]": [fmt_ci(e, l, h) for e, l, h in zip(f["d|Bias|"], f["d|Bias| CI low"], f["d|Bias| CI high"])],
                    "dr [account CI]": [fmt_ci(e, l, h) for e, l, h in zip(f["dPearson_r"], f["dPearson_r CI low"], f["dPearson_r CI high"])]})
    md.append(cc.md_table(t[["Table", "Model", "Condition", "Reference", "N_paired", "Accounts", "dMAE [post CI]",
                             "Old p_holm", "dMAE [account CI]", "p_holm_cluster", "d|Bias| [account CI]",
                             "dr [account CI]", "Changed"]]))
    acc = f[f.Table == "R_top posts"].drop_duplicates("Model")[["Model", "N_paired", "Accounts"]]
    md += ["", "Accounts in the relevant-only (R_top ≥ 4) subsets: " +
           ", ".join(f"{cc.short(m)} {n} posts from {a} accounts" for m, n, a in acc.itertuples(index=False)) + ".", ""]
    if sens is not None:
        sens.to_csv(OUT5 / "chunk_ablation_paired_cluster_sens.csv", index=False)
        j = f.merge(sens, on=["Table", "Model", "Condition", "Reference"], suffixes=("", "_s"))
        ch = j[(j["p_holm_cluster"] < 0.05) != (j["p_holm_cluster_s"] < 0.05)]
        md += [f"## Sensitivity: {Dsens.label}", "",
               f"Comparisons whose Holm decision changes: {len(ch)}", ""]
        if len(ch):
            md.append(cc.md_table(ch.assign(Model=ch.Model.map(cc.short),
                Full=[fmt_ci(e, l, h, 3) for e, l, h in zip(ch.dMAE, ch["Cluster CI low"], ch["Cluster CI high"])],
                Sens=[fmt_ci(e, l, h, 3) for e, l, h in zip(ch.dMAE_s, ch["Cluster CI low_s"], ch["Cluster CI high_s"])]
            )[["Table", "Model", "Condition", "Reference", "Full", "p_holm_cluster", "Sens", "p_holm_cluster_s"]]))
        md.append("")
        md.append(cc.md_table(sens.assign(Model=sens.Model.map(cc.short))[
            ["Table", "Model", "Condition", "Reference", "N_paired", "Accounts", "dMAE", "Cluster CI low",
             "Cluster CI high", "p_holm_cluster"]]))
    (OUT5 / "cluster_item7_ablation.md").write_text("\n".join(md), encoding="utf-8")
    print("  item 7 ->", OUT5 / "cluster_item7_ablation.md")


# ===========================================================================
# item 9: weights
# ===========================================================================

def weights(D):
    pop = aci.load_population()
    ch = aci.load_sample(aci.DEFAULT_RUN, "final", pop)
    t = ch.drop_duplicates(["model", "text_index"])
    calc = (t.groupby(["model", "quadrant"]).agg(Sampled=("text_index", "size"), Weight=("weight", "first"),
                                                 Accounts=("text_index", lambda s: D.n_accounts(s)))
            .reset_index())
    popc = pop.groupby(["model", "quadrant"]).size().rename("Population").reset_index()
    calc = calc.merge(popc, on=["model", "quadrant"])
    calc["Pop/Sampled"] = calc["Population"] / calc["Sampled"]
    calc["Weight matches"] = (calc["Pop/Sampled"] - calc["Weight"]).abs() < 1e-9
    stored = pd.read_csv(OUT4 / "judge_sampling_weights.csv")
    stored["model"] = stored["Model"].map({v: k for k, v in w.DISPLAY.items()} | {"Llama 3B (all)": "llama_3b"})
    j = calc.merge(stored.rename(columns={"Quadrant": "quadrant"}), on=["model", "quadrant"], how="left")
    j["Stored weight matches"] = (j["Weight_x"] - j["Weight_y"]).abs() < 1e-3
    j = j.rename(columns={"Weight_x": "Weight (analysis)", "Weight_y": "Weight (judge_sampling_weights.csv)"})
    j.to_csv(OUT4 / "judge_sampling_weights_check.csv", index=False)
    per_model = t.groupby("model")["text_index"].agg(lambda s: D.n_accounts(s)).rename("Accounts in sample")
    md = ["# Item 9: quadrant sampling weights (check, no rerun)", "",
          "Weight = population texts of the model's quadrant / sampled texts (inverse sampling probability). It is "
          "recomputed from `aci.load_population` and `final_annotations.jsonl` and compared with "
          "`judge_sampling_weights.csv`. Llama 3B uses the 'all paired posts' row. The weights correct only for the "
          "quadrant design. They do not account for the clustering of texts within accounts.", "",
          f"All weights equal Population / Sampled: {bool(j['Weight matches'].all())}; all equal the stored CSV: "
          f"{bool(j['Stored weight matches'].all())}.", "",
          cc.md_table(j[["model", "quadrant", "Population", "Sampled", "Accounts", "Weight (analysis)",
                         "Weight (judge_sampling_weights.csv)", "Stored weight matches"]]), "",
          "Accounts covered by the sampled texts per model: " +
          ", ".join(f"{w.DISPLAY[m]} {n}" for m, n in per_model.items()) + ".", ""]
    (OUT4 / "cluster_item9_weights.md").write_text("\n".join(md), encoding="utf-8")
    print("  item 9 ->", OUT4 / "cluster_item9_weights.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["composition", "evidence", "direction", "ablation", "weights", "all"])
    ap.add_argument("--no-sens", action="store_true")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    D = cc.Design()
    Ds = None if a.no_sens else cc.Design(drop=[cc.largest_account()[0]])
    todo = ["composition", "evidence", "direction", "ablation", "weights"] if a.what == "all" else [a.what]
    if "composition" in todo:
        composition(D)
    if "evidence" in todo:
        evidence_report(evidence(D), evidence(Ds) if Ds else None, D, Ds)
    if "direction" in todo:
        direction_report(direction(D), direction(Ds) if Ds else None, Ds)
    if "ablation" in todo:
        ablation_report(ablation(D), ablation(Ds) if Ds else None, Ds)
    if "weights" in todo:
        weights(D)


if __name__ == "__main__":
    main()
