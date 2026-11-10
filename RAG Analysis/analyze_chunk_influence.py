"""How retrieved chunks shape the RAG predictions (offline).

Two data sources:

* **Population** - every paired text of a model (baseline and RAG batch logs
  listed in ``results/qualitative/<family>/<size>/qualitative_e5_simple_diagnostics.json``,
  about 775 texts per model). Not selected on the outcome. The retrieved chunks'
  parties are in the RAG log, so the evidence position works for all texts.
* **Sample** - the 25-per-quadrant texts with chunk labels (R_top, R_ideo,
  A_caus) from ``final_annotations.jsonl`` (``src/merge_judge_review.py``).
  Selected on the outcome; pooled numbers are weighted by the quadrant base
  rates, and the 'buffer' texts (error between the pass and fail thresholds)
  are not in it at all.

Party positions: CHES 2024 ``lrgen`` on the project 1-7 scale
(``1 + lrgen / 10 * 6``), i.e. the same definition as the ground truth
``label_ideology`` (AfD 6.6, CSU 5.5, CDU 4.9, FDP 4.6, BSW 3.2, SPD 3.1,
Greens 2.8, Linke 1.9). The Bundestag faction 'CDU/CSU' gets the CDU value, as
in the ground truth.

Analyses:

1. **Base rates and party composition** of the quadrants in the population, and
   which parties a constant prediction would count as 'correct'.
2. **Evidence use** (population): ``R ~ B + E + model`` with R = RAG prediction,
   B = baseline prediction, E = mean party position of the retrieved chunks; the
   coefficient on E is the evidence use, with standard errors clustered by text
   (the same texts are judged by every model). A within-model permutation of E
   gives the null; it also shows how much of the naive ``corr(R − B, E − B)``
   ('pull') arises without any evidence use (both terms contain −B).
3. **Evidence use by chunk label** (sample, descriptive): R on the positions of
   the cited vs. uncited and relevant vs. irrelevant chunks, weighted.
4. **Quadrant descriptions** (sample): chunk labels and citations per quadrant.
   Quadrants are selected on the outcome, so these describe the error cases;
   they are not separate mechanisms. Q1 vs. Q2 differences are tested with the
   input party and model as controls.
5. **Noise floor** (``--baseline-rerun``): how many texts would land in Q1 / Q2
   if the 'RAG' run were just a second baseline run.
6. **Helped or hurt** (population): per model the share of texts whose error RAG
   reduced / increased by at least ``CHANGE_THRESHOLD``, overall and split by
   whether the retrieved parties pointed toward the ground truth; one overview
   figure and one figure per model (``per_model/``).

    python "RAG Analysis/analyze_chunk_influence.py"
    python "RAG Analysis/analyze_chunk_influence.py" --labels judge --out "RAG Analysis/chunk_influence_judge"
    python "RAG Analysis/analyze_chunk_influence.py" --baseline-rerun llama_8b=<rerun no_rag log> ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RUN = HERE / "LLM judge results" / "families"
DEFAULT_OUT = HERE / "chunk_influence"
DIAGNOSTICS_GLOB = "results/qualitative/*/*/qualitative_e5_simple_diagnostics.json"
CHES_PATH = ROOT / "src/datasets/ground_truth/ches_germany_1999-2024.csv"
CHES_WAVE = 2024
REVIEWED_MODELS = ("llama_70b", "mistral_14b", "qwen_3b")
QUADRANTS = ("Q1", "Q2", "Q3", "Q4")
QUADRANT_NAMES = {
    "Q1": "RAG win",
    "Q2": "RAG distraction",
    "Q3": "joint failure",
    "Q4": "baseline sufficient",
    "buffer": "between thresholds",
}
DIMENSIONS = ("R_top", "R_ideo", "A_caus")
N_PERMUTATIONS = 1000
CHANGE_THRESHOLD = 0.5  # |error change| below this counts as 'unchanged' (= the pass threshold)
SEED = 0

# Casefolded, whitespace-normalised spellings -> canonical party (as in
# src/evaluate_metrics.py). Fraktionslos / unknown stay unmapped.
PARTY_ALIASES = {
    "die linke": "DIE LINKE",
    "linke": "DIE LINKE",
    "bündnis 90/die grünen": "BÜNDNIS 90/DIE GRÜNEN",
    "die grünen": "BÜNDNIS 90/DIE GRÜNEN",
    "grüne": "BÜNDNIS 90/DIE GRÜNEN",
    "spd": "SPD",
    "fdp": "FDP",
    "afd": "AfD",
    "cdu": "CDU",
    "csu": "CSU",
    "cdu/csu": "CDU",  # Bundestag faction: CDU value, as in the ground truth
    "bsw": "BSW",
}


def _load_strata_module():
    """extract_qualitative_strata.py: the quadrant rule and log parsing it used."""
    spec = importlib.util.spec_from_file_location("strata", HERE / "extract_qualitative_strata.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strata"] = module  # dataclasses look their module up while executing
    spec.loader.exec_module(module)
    return module


strata = _load_strata_module()


def norm_party(name) -> str | None:
    if not isinstance(name, str):
        return None
    return PARTY_ALIASES.get(" ".join(name.replace("\xa0", " ").split()).casefold())


def load_positions(path: Path = CHES_PATH, wave: int = CHES_WAVE) -> dict:
    """Canonical party -> lrgen on the 1-7 scale (latest wave <= ``wave``)."""
    ches = pd.read_csv(path)
    ches = ches[ches["year"] <= wave].sort_values("year").groupby("std_party").tail(1)
    out = {}
    for _, row in ches.iterrows():
        party = norm_party(row["std_party"]) or row["std_party"]
        out[party] = 1 + row["lrgen"] / 10 * 6
    return out


POSITIONS = load_positions()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def quadrant_short(q: str | None) -> str:
    return q.split("_")[0] if q else "buffer"


def load_population() -> pd.DataFrame:
    """Every paired text per model with predictions, quadrant and evidence position."""
    rows = []
    for diag_path in sorted(ROOT.glob(DIAGNOSTICS_GLOB)):
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        model = f"{diag['family']}_{diag['size']}"
        thr = diag.get("thresholds") or {"pass": 0.5, "fail": 1.0}
        base = strata._merge_logs([ROOT / p for p in diag["baseline_paths"]])
        rag = strata._merge_logs([ROOT / p for p in diag["rag_paths"]])
        for text_index in set(base.records) & set(rag.records):
            b_rec, r_rec = base.records[text_index], rag.records[text_index]
            b, gt, _ = strata.extract_scalar_prediction(b_rec, "label_ideology")
            r, gt_r, _ = strata.extract_scalar_prediction(r_rec, "label_ideology")
            if None in (b, gt, r, gt_r) or abs(gt - gt_r) > 1e-6:
                continue
            chunks = (r_rec.get("inputs") or {}).get("retrieved_chunks") or []
            input_party = norm_party((r_rec.get("input_metadata") or {}).get("party"))
            parties = [norm_party(c.get("party")) for c in chunks]
            pos = [POSITIONS[p] for p in parties if p in POSITIONS]
            rows.append(
                {
                    "model": model,
                    "family": diag["family"],
                    "text_index": str(text_index),
                    "input_party": input_party or "other",
                    "ground_truth": gt,
                    "B": b,
                    "R": r,
                    "quadrant": quadrant_short(
                        strata.classify_quadrant(abs(b - gt), abs(r - gt), thr["pass"], thr["fail"])
                    ),
                    "E": np.mean(pos) if pos else np.nan,
                    "n_positioned": len(pos),
                    "same_party_share": np.mean([p == input_party for p in parties]) if parties else np.nan,
                }
            )
    pop = pd.DataFrame(rows)
    if pop.empty:
        raise SystemExit(f"No diagnostics found under {ROOT / DIAGNOSTICS_GLOB}")
    pop["shift"] = pop["R"] - pop["B"]
    pop["pull"] = pop["E"] - pop["B"]
    return pop


def load_sample(run: Path, labels: str, pop: pd.DataFrame) -> pd.DataFrame:
    """Labelled chunks of the sampled texts (one row per chunk), with weights."""
    rows = []
    for path in sorted(run.glob("*/final_annotations.jsonl")):
        model = path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            sc = rec.get("source_context") or {}
            for c in rec.get("chunk_annotations", []):
                lab = c.get("judge_labels") if labels == "judge" else c
                party = norm_party((c.get("chunk_metadata") or {}).get("party"))
                rows.append(
                    {
                        "model": model,
                        "tier": "reviewed" if model in REVIEWED_MODELS else "judge",
                        "text_index": str(rec["text_index"]),
                        "quadrant": (sc.get("quadrant") or "")[:2],
                        "chunk_index": c["chunk_index"],
                        "chunk_party": party or "other",
                        "chunk_position": POSITIONS.get(party, np.nan),
                        **{d: lab.get(d) for d in DIMENSIONS},
                    }
                )
    ch = pd.DataFrame(rows)
    if ch.empty:
        raise SystemExit(f"No final_annotations.jsonl under {run}")
    keys = ["model", "text_index"]
    ch = ch.merge(pop[keys + ["input_party", "ground_truth", "B", "R", "shift"]], on=keys, how="left")
    ch["same_party"] = ch["chunk_party"] == ch["input_party"]
    # Inverse sampling probability: population texts of the quadrant / sampled texts.
    pop_counts = pop.groupby(["model", "quadrant"]).size()
    samp_counts = ch.drop_duplicates(keys).groupby(["model", "quadrant"]).size()
    weight = (pop_counts / samp_counts).rename("weight").reset_index()
    return ch.merge(weight, on=["model", "quadrant"], how="left")


def text_table(ch: pd.DataFrame) -> pd.DataFrame:
    """Per sampled text: chunk features, overall and for cited / relevant chunks."""

    def mean_pos(g):
        v = g["chunk_position"].dropna()
        return v.mean() if len(v) else np.nan

    out = []
    for (model, text_index), g in ch.groupby(["model", "text_index"], sort=False):
        cited, uncited = g[g["A_caus"] == 1], g[g["A_caus"] != 1]
        out.append(
            {
                "model": model,
                "tier": g["tier"].iloc[0],
                "text_index": text_index,
                **{k: g[k].iloc[0] for k in ("quadrant", "input_party", "ground_truth", "B", "R", "shift", "weight")},
                "mean_R_top": g["R_top"].mean(),
                "share_relevant": (g["R_top"] >= 4).mean(),
                "mean_R_ideo": g["R_ideo"].mean(),
                "share_ideological": (g["R_ideo"] >= 4).mean(),
                "n_cited": len(cited),
                "cited_R_top": cited["R_top"].mean() if len(cited) else np.nan,
                "cited_R_ideo": cited["R_ideo"].mean() if len(cited) else np.nan,
                "share_same_party": g["same_party"].mean(),
                "cited_same_party": cited["same_party"].mean() if len(cited) else np.nan,
                "E_all": mean_pos(g),
                "E_cited": mean_pos(cited),
                "E_uncited": mean_pos(uncited),
                "E_relevant": mean_pos(g[g["R_top"] >= 4]),
                "E_irrelevant": mean_pos(g[g["R_top"] < 4]),
            }
        )
    t = pd.DataFrame(out)
    t["cited_gt_distance"] = (t["E_cited"] - t["ground_truth"]).abs()
    return t


# ---------------------------------------------------------------------------
# 1. Base rates and party composition (population)
# ---------------------------------------------------------------------------

ORDER = (*QUADRANTS, "buffer")


def base_rates(pop: pd.DataFrame) -> pd.DataFrame:
    counts = pop.groupby(["model", "quadrant"]).size().unstack(fill_value=0).reindex(columns=ORDER, fill_value=0)
    counts.loc["pooled"] = counts.sum()
    shares = counts.div(counts.sum(axis=1), axis=0)
    out = shares.add_suffix(" share").join(counts.add_suffix(" n"))
    out["texts"] = counts.sum(axis=1)
    return out.reset_index()


def party_composition(pop: pd.DataFrame) -> pd.DataFrame:
    """Share of each input party per quadrant (pooled population), and per party
    how often the baseline is correct."""
    comp = pd.crosstab(pop["input_party"], pop["quadrant"], normalize="columns").reindex(columns=ORDER)
    comp["all texts"] = pop["input_party"].value_counts(normalize=True)
    by_party = pop.assign(base_correct=(pop["B"] - pop["ground_truth"]).abs() <= 0.5).groupby("input_party")
    comp["baseline correct (|err| ≤ 0.5)"] = by_party["base_correct"].mean()
    comp["position"] = [POSITIONS.get(p, np.nan) for p in comp.index]
    return comp.sort_values("position").reset_index()


def constant_prediction_hits(threshold: float = 0.5) -> pd.DataFrame:
    """Which parties a constant prediction c counts as correct (|c − pos| ≤ threshold)."""
    rows = []
    for c in np.arange(1.0, 7.01, 0.5):
        hits = [p for p, v in sorted(POSITIONS.items(), key=lambda x: x[1]) if abs(c - v) <= threshold]
        rows.append({"constant prediction": c, "counted correct for": ", ".join(hits) or "—"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Evidence use (population)
# ---------------------------------------------------------------------------


def _ols_cluster(formula: str, data: pd.DataFrame, weights=None):
    data = data.dropna(subset=["R", "B", "E"]).copy()
    model = smf.wls(formula, data, weights=weights.loc[data.index] if weights is not None else None) \
        if weights is not None else smf.ols(formula, data)
    codes = pd.factorize(data["text_index"])[0]
    return model.fit(cov_type="cluster", cov_kwds={"groups": codes}), data


def _coef_row(name: str, fit, term: str, n_texts: int) -> dict:
    ci = fit.conf_int().loc[term]
    return {
        "model": name,
        "term": term,
        "coef": fit.params[term],
        "ci_low": ci[0],
        "ci_high": ci[1],
        "p": fit.pvalues[term],
        "n": int(fit.nobs),
        "n_texts": n_texts,
        "r2": fit.rsquared,
    }


def _permuted_e_coef(data: pd.DataFrame, rng, n: int, controls: bool) -> tuple[np.ndarray, np.ndarray]:
    """Null distributions of the E coefficient and of corr(shift, pull) when E is
    shuffled across texts within each model."""
    data = data.dropna(subset=["R", "B", "E"]).reset_index(drop=True)
    dummies = pd.get_dummies(data["model"], drop_first=True, dtype=float)
    parts = [np.ones(len(data)), data["B"].to_numpy()]
    if controls:
        parties = pd.get_dummies(data["input_party"], drop_first=True, dtype=float)
        dummies = pd.concat([dummies, parties], axis=1)
    X_fixed = np.column_stack(parts + ([dummies.to_numpy()] if dummies.shape[1] else []))
    y = data["R"].to_numpy()
    groups = [np.flatnonzero(data["model"].to_numpy() == m) for m in data["model"].unique()]
    E = data["E"].to_numpy()
    coefs, corrs = np.empty(n), np.empty(n)
    for i in range(n):
        Ep = E.copy()
        for idx in groups:
            Ep[idx] = E[rng.permutation(idx)]
        X = np.column_stack([X_fixed, Ep])
        coefs[i] = np.linalg.lstsq(X, y, rcond=None)[0][-1]
        corrs[i] = np.corrcoef(y - data["B"], Ep - data["B"])[0, 1]
    return coefs, corrs


def evidence_use(pop: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(SEED)
    rows, null = [], {}
    n_texts = pop["text_index"].nunique()
    specs = {
        "pooled": ("R ~ B + E + C(model)", False),
        "pooled + input party": ("R ~ B + E + C(model) + C(input_party)", True),
    }
    for name, (formula, controls) in specs.items():
        fit, data = _ols_cluster(formula, pop)
        row = _coef_row(name, fit, "E", n_texts)
        coefs, corrs = _permuted_e_coef(pop, rng, N_PERMUTATIONS, controls)
        observed_corr = np.corrcoef(data["shift"], data["pull"])[0, 1]
        row.update(
            perm_null_mean=coefs.mean(),
            perm_null_95=np.percentile(coefs, 95),
            perm_p=(np.sum(coefs >= row["coef"]) + 1) / (N_PERMUTATIONS + 1),
            naive_pull_corr=observed_corr,
            naive_pull_corr_null_mean=corrs.mean(),
            B_coef=fit.params["B"],
        )
        rows.append(row)
        null[name] = coefs
    for model, g in pop.groupby("model"):
        fit, data = _ols_cluster("R ~ B + E", g)
        row = _coef_row(model, fit, "E", g["text_index"].nunique())
        coefs, corrs = _permuted_e_coef(g, rng, N_PERMUTATIONS, False)
        row.update(
            perm_null_mean=coefs.mean(),
            perm_null_95=np.percentile(coefs, 95),
            perm_p=(np.sum(coefs >= row["coef"]) + 1) / (N_PERMUTATIONS + 1),
            naive_pull_corr=np.corrcoef(data["shift"], data["pull"])[0, 1],
            naive_pull_corr_null_mean=corrs.mean(),
            B_coef=fit.params["B"],
        )
        rows.append(row)
        null[model] = coefs
    return pd.DataFrame(rows), null


def evidence_by_label(t: pd.DataFrame) -> pd.DataFrame:
    """Sample, weighted: which chunks' positions go with the RAG prediction."""
    rows = []
    specs = {
        "cited vs. uncited chunks": ("R ~ B + E_cited + E_uncited + C(model)", ["E_cited", "E_uncited"]),
        "relevant (R_top ≥ 4) vs. other chunks": ("R ~ B + E_relevant + E_irrelevant + C(model)",
                                                  ["E_relevant", "E_irrelevant"]),
    }
    for name, (formula, terms) in specs.items():
        data = t.dropna(subset=["R", "B", *terms]).copy()
        fit = smf.wls(formula, data, weights=data["weight"]).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(data["text_index"])[0]}
        )
        for term in terms:
            ci = fit.conf_int().loc[term]
            rows.append({"comparison": name, "term": term, "coef": fit.params[term], "ci_low": ci[0],
                         "ci_high": ci[1], "p": fit.pvalues[term], "n_texts": int(fit.nobs)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Quadrant descriptions (sample)
# ---------------------------------------------------------------------------

PROFILE_COLS = {
    "mean_R_top": "mean R_top",
    "share_relevant": "share chunks R_top ≥ 4",
    "mean_R_ideo": "mean R_ideo",
    "share_ideological": "share chunks R_ideo ≥ 4",
    "n_cited": "chunks cited (of 5)",
    "cited_R_top": "R_top of cited chunks",
    "cited_R_ideo": "R_ideo of cited chunks",
    "share_same_party": "share chunks from input's party",
    "cited_same_party": "share cited chunks from input's party",
    "cited_gt_distance": "|cited chunks' position − ground truth|",
}


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    ok = values.notna()
    return np.average(values[ok], weights=weights[ok]) if ok.any() else np.nan


def _q1_vs_q2(t: pd.DataFrame, col: str) -> tuple[float, float]:
    """Logit Q1 (vs. Q2) on the feature, controlling input party and model."""
    data = t[t["quadrant"].isin(["Q1", "Q2"])].dropna(subset=[col]).copy()
    data["is_q1"] = (data["quadrant"] == "Q1").astype(int)
    data["x"] = (data[col] - data[col].mean()) / (data[col].std() or 1)
    try:
        fit = smf.logit("is_q1 ~ x + C(input_party) + C(model)", data).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(data["text_index"])[0]}
        )
        return fit.params["x"], fit.pvalues["x"]
    except Exception:  # noqa: BLE001  (separation in small groups)
        return np.nan, np.nan


def quadrant_profiles(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in PROFILE_COLS.items():
        row = {"feature": label}
        for q in QUADRANTS:
            row[q] = t.loc[t["quadrant"] == q, col].mean()
        row["all (weighted, no buffer)"] = _weighted_mean(t[col], t["weight"])
        row["Q1 vs Q2: log-odds per SD"], row["p (party + model controlled)"] = _q1_vs_q2(t, col)
        rows.append(row)
    return pd.DataFrame(rows)


def party_stratified(t: pd.DataFrame, col: str = "cited_same_party") -> pd.DataFrame:
    out = t[t["quadrant"].isin(["Q1", "Q2"])].pivot_table(
        index="input_party", columns="quadrant", values=col, aggfunc=["mean", "count"]
    )
    out.columns = [f"{q} {stat}" for stat, q in out.columns]
    out["position"] = [POSITIONS.get(p, np.nan) for p in out.index]
    return out.sort_values("position").reset_index()


def utilisation(ch: pd.DataFrame) -> pd.DataFrame:
    """Weighted share of chunks cited, by label level."""
    rows = []
    for dim in ("R_top", "R_ideo"):
        for level in range(1, 6):
            sub = ch[ch[dim] == level]
            if sub.empty:
                continue
            row = {"dimension": dim, "level": level, "n chunks": len(sub),
                   "cited (weighted)": _weighted_mean(sub["A_caus"], sub["weight"])}
            for q in QUADRANTS:
                row[f"cited {q}"] = sub.loc[sub["quadrant"] == q, "A_caus"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Noise floor
# ---------------------------------------------------------------------------


def noise_floor(pop: pd.DataFrame, reruns: dict[str, list[Path]]) -> pd.DataFrame | None:
    """Quadrants with a second baseline run in place of RAG: pseudo-Q1 / Q2 rates."""
    if not reruns:
        return None
    rows = []
    for model, paths in reruns.items():
        log = strata._merge_logs(paths)
        g = pop[pop["model"] == model].set_index("text_index")
        counts = {q: 0 for q in ORDER}
        changed = 0
        for text_index, rec in log.records.items():
            if str(text_index) not in g.index:
                continue
            b2, gt, _ = strata.extract_scalar_prediction(rec, "label_ideology")
            if b2 is None:
                continue
            b1 = g.loc[str(text_index), "B"]
            changed += int(b2 != b1)
            counts[quadrant_short(strata.classify_quadrant(abs(b1 - gt), abs(b2 - gt)))] += 1
        n = sum(counts.values())
        real = pop[pop["model"] == model]["quadrant"].value_counts(normalize=True)
        rows.append({
            "model": model, "texts": n, "prediction changed": changed / n if n else np.nan,
            **{f"pseudo {q}": counts[q] / n for q in ("Q1", "Q2")},
            **{f"RAG {q}": real.get(q, 0.0) for q in ("Q1", "Q2")},
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Helped or hurt (population)
# ---------------------------------------------------------------------------

OUTCOMES = ("hurt", "unchanged", "helped")
OUTCOME_COLORS = {"hurt": "#e76f51", "unchanged": "#d9d9d9", "helped": "#2a9d8f"}


def add_outcome(pop: pd.DataFrame) -> pd.DataFrame:
    pop = pop.copy()
    pop["base_error"] = (pop["B"] - pop["ground_truth"]).abs()
    pop["rag_error"] = (pop["R"] - pop["ground_truth"]).abs()
    pop["error_change"] = pop["rag_error"] - pop["base_error"]  # < 0 = RAG better
    pop["outcome"] = np.select(
        [pop["error_change"] <= -CHANGE_THRESHOLD, pop["error_change"] >= CHANGE_THRESHOLD],
        ["helped", "hurt"], "unchanged",
    )
    # Did the retrieved parties point the right way? Their mean position is closer
    # to the ground truth than the baseline prediction was.
    pop["evidence_side"] = np.where(
        pop["E"].isna(), "no position",
        np.where((pop["E"] - pop["ground_truth"]).abs() < pop["base_error"], "toward truth", "away from truth"),
    )
    return pop


def helped_hurt(pop: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in pop.groupby("model"):
        shares = g["outcome"].value_counts(normalize=True)
        row = {
            "model": model,
            "texts": len(g),
            **{o: shares.get(o, 0.0) for o in OUTCOMES},
            "MAE baseline": g["base_error"].mean(),
            "MAE RAG": g["rag_error"].mean(),
            "evidence toward truth": (g["evidence_side"] == "toward truth").mean(),
        }
        for side in ("toward truth", "away from truth"):
            s = g[g["evidence_side"] == side]["outcome"].value_counts(normalize=True)
            row[f"helped | {side}"] = s.get("helped", np.nan)
            row[f"hurt | {side}"] = s.get("hurt", np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def helped_hurt_figures(pop: pd.DataFrame, hh: pd.DataFrame, out: Path) -> list[str]:
    files = []
    # Overview: one stacked bar per model.
    order = hh.sort_values("helped")["model"].tolist()
    h = hh.set_index("model").loc[order]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(h) + 1.6))
    left = np.zeros(len(h))
    for o in OUTCOMES:
        ax.barh(h.index, h[o], left=left, color=OUTCOME_COLORS[o], label=o)
        for y, (l, v) in enumerate(zip(left, h[o])):
            if v >= 0.06:
                ax.text(l + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8)
        left += h[o].to_numpy()
    for y, m in enumerate(h.index):
        ax.text(1.01, y, f"MAE {h.loc[m, 'MAE baseline']:.2f} → {h.loc[m, 'MAE RAG']:.2f}",
                va="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"share of texts (error change of at least {CHANGE_THRESHOLD} counts as helped / hurt)")
    ax.set_title("Did RAG help or hurt? All texts per model")
    ax.legend(ncol=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    files.append("helped_hurt_overview.png")
    fig.savefig(out / files[-1], dpi=130, bbox_inches="tight")
    plt.close(fig)

    per_model = out / "per_model"
    per_model.mkdir(exist_ok=True)
    side_colors = {"toward truth": "#2a9d8f", "away from truth": "#e76f51", "no position": "#999999"}
    for model, g in pop.groupby("model"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        ax = axes[0]
        for side, gs in g.groupby("evidence_side"):
            ax.scatter(gs["base_error"], gs["rag_error"], s=10, alpha=0.5, color=side_colors[side],
                       label=f"retrieved parties {side}" if side != "no position" else side)
        lim = max(g["base_error"].max(), g["rag_error"].max()) + 0.2
        ax.plot([0, lim], [0, lim], color="grey", lw=0.8)
        ax.fill_between([0, lim], [0, lim], [lim, lim], color=OUTCOME_COLORS["hurt"], alpha=0.06)
        ax.fill_between([0, lim], [0, 0], [0, lim], color=OUTCOME_COLORS["helped"], alpha=0.06)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("baseline error |B − truth|")
        ax.set_ylabel("RAG error |R − truth|")
        ax.set_title("per text (below the line = RAG better)")
        ax.legend(fontsize=7)

        ax = axes[1]
        bins = np.arange(-6, 6.25, 0.25)
        for o in OUTCOMES:
            ax.hist(g.loc[g["outcome"] == o, "error_change"], bins=bins, color=OUTCOME_COLORS[o], label=o)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_xlim(-5, 5)
        ax.set_xlabel("error change: RAG error − baseline error")
        ax.set_ylabel("texts")
        ax.set_title("how much the error changed")
        ax.legend(fontsize=7)

        ax = axes[2]
        groups = ["all texts", "toward truth", "away from truth"]
        for i, grp in enumerate(groups):
            sub = g if grp == "all texts" else g[g["evidence_side"] == grp]
            shares = sub["outcome"].value_counts(normalize=True)
            left = 0.0
            for o in OUTCOMES:
                v = shares.get(o, 0.0)
                ax.barh(i, v, left=left, color=OUTCOME_COLORS[o])
                if v >= 0.07:
                    ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8)
                left += v
        ax.set_yticks(range(len(groups)), [
            f"{grp}\n(n={len(g) if grp == 'all texts' else int((g['evidence_side'] == grp).sum())})"
            for grp in groups])
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of texts")
        ax.set_title("by direction of the retrieved parties")

        r = hh.set_index("model").loc[model]
        fig.suptitle(f"{model}: helped {r['helped']:.0%}, hurt {r['hurt']:.0%}, "
                     f"MAE {r['MAE baseline']:.2f} → {r['MAE RAG']:.2f}", fontsize=12)
        fig.tight_layout()
        fig.savefig(per_model / f"{model}.png", dpi=120)
        plt.close(fig)
        files.append(f"per_model/{model}.png")
    return files


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------

COLORS = {"Q1": "#2a9d8f", "Q2": "#e76f51", "Q3": "#8d99ae", "Q4": "#e9c46a", "buffer": "#d9d9d9"}


def figures(pop, rates, ev, null, ch, out: Path) -> list[str]:
    files = []
    r = rates[rates["model"] != "pooled"].set_index("model")
    fig, ax = plt.subplots(figsize=(9, 4))
    left = np.zeros(len(r))
    for q in ORDER:
        ax.barh(r.index, r[f"{q} share"], left=left, color=COLORS[q], label=f"{q} {QUADRANT_NAMES[q]}")
        left += r[f"{q} share"].to_numpy()
    ax.set_xlabel("share of all texts")
    ax.set_title("Quadrant base rates in the full population")
    ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    files.append("base_rates.png")
    fig.savefig(out / files[-1], dpi=130)
    plt.close(fig)

    per_model = ev[~ev["model"].str.startswith("pooled")]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(per_model))
    ax.errorbar(per_model["coef"], y, xerr=[per_model["coef"] - per_model["ci_low"],
                per_model["ci_high"] - per_model["coef"]], fmt="o", color="#264653", label="coefficient on E (95 % CI)")
    for i, m in enumerate(per_model["model"]):
        lo, hi = np.percentile(null[m], [2.5, 97.5])
        ax.plot([lo, hi], [i, i], color="#e76f51", lw=6, alpha=0.35, label="permutation null 95 %" if i == 0 else None)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(y, per_model["model"])
    ax.set_xlabel("evidence use: coefficient on E in R ~ B + E")
    ax.set_title("Does the RAG prediction depend on the retrieved parties?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    files.append("evidence_use.png")
    fig.savefig(out / files[-1], dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, dim in zip(axes, DIMENSIONS):
        levels = [0, 1] if dim == "A_caus" else [1, 2, 3, 4, 5]
        width = 0.8 / len(QUADRANTS)
        for i, q in enumerate(QUADRANTS):
            sub = ch[ch["quadrant"] == q][dim].dropna()
            ax.bar(np.arange(len(levels)) + i * width, [(sub == lv).mean() if len(sub) else 0 for lv in levels],
                   width, color=COLORS[q], label=f"{q} {QUADRANT_NAMES[q]}")
        ax.set_xticks(np.arange(len(levels)) + 0.4 - width / 2, levels)
        ax.set_title(f"{dim} (sampled texts, descriptive)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    files.append("labels_by_quadrant.png")
    fig.savefig(out / files[-1], dpi=130)
    plt.close(fig)
    return files


def _fmt(v, col: str = "") -> str:
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "—"
        if col in ("p", "perm_p") or col.startswith("p ("):
            return "<0.001" if v < 0.001 else f"{v:.3f}"
        return f"{v:.2f}"
    return str(v)


def md_table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c], str(c)) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", default=str(DEFAULT_RUN), help="folder with <model>/final_annotations.jsonl")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--labels", choices=("final", "judge"), default="final",
                        help="final (reviewed / transferred / judge) or the raw judge labels")
    parser.add_argument("--tier", choices=("all", "reviewed"), default="all",
                        help="sample analyses on every model or only the human-reviewed ones")
    parser.add_argument("--baseline-rerun", nargs="*", default=[], metavar="MODEL=LOG",
                        help="second no-RAG run per model (e.g. llama_8b=logs/.../no_rag/...jsonl) for the noise floor")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pop = add_outcome(load_population())
    ch = load_sample(Path(args.run), args.labels, pop)
    if args.tier == "reviewed":
        ch = ch[ch["tier"] == "reviewed"]
    t = text_table(ch)
    reruns: dict[str, list[Path]] = {}
    for spec in args.baseline_rerun:
        model, _, path = spec.partition("=")
        reruns.setdefault(model, []).append(Path(path))

    rates = base_rates(pop)
    comp = party_composition(pop)
    hits = constant_prediction_hits()
    ev, null = evidence_use(pop)
    by_label = evidence_by_label(t)
    profiles = quadrant_profiles(t)
    strat = party_stratified(t)
    util = utilisation(ch)
    noise = noise_floor(pop, reruns)
    hh = helped_hurt(pop)

    pop.to_csv(out / "population_texts.csv", index=False)
    ch.to_csv(out / "sample_chunks.csv", index=False)
    t.to_csv(out / "sample_texts.csv", index=False)
    for name, frame in (("base_rates", rates), ("party_composition", comp), ("evidence_use", ev),
                        ("evidence_by_label", by_label), ("quadrant_profiles", profiles),
                        ("party_stratified_cited_same_party", strat), ("utilisation", util),
                        ("helped_hurt", hh)):
        frame.to_csv(out / f"{name}.csv", index=False)
    if noise is not None:
        noise.to_csv(out / "noise_floor.csv", index=False)
    figs = figures(pop, rates, ev, null, ch, out)
    hh_figs = helped_hurt_figures(pop, hh, out)

    pooled = ev[ev["model"] == "pooled"].iloc[0]
    md = [
        "# Chunk influence on the RAG predictions",
        "",
        f"Labels: **{args.labels}**, sample tier: **{args.tier}**. Quadrants: correct = |error| ≤ 0.5, "
        "wrong = |error| ≥ 1.0, texts in between = buffer (not sampled).",
        "",
        "Party positions (CHES 2024 lrgen, 1-7 scale; = ground truth definition): "
        + ", ".join(f"{p} {v:.2f}" for p, v in sorted(POSITIONS.items(), key=lambda x: x[1])),
        "",
        "## 1. Base rates (full population)",
        "",
        "The sample has 25 texts per quadrant, so it over-represents rare quadrants (above all Q2). "
        "Pooled sample numbers below are weighted by these base rates; buffer texts are missing from the "
        "sample entirely.",
        "",
        md_table(rates[["model", "texts", *[f"{q} share" for q in ORDER]]]),
        "",
        "### Party composition per quadrant (share of the quadrant's texts)",
        "",
        "Parties close to the scale centre are 'correct' for a constant centrist prediction, so quadrant "
        "membership partly reflects the party mix:",
        "",
        md_table(comp),
        "",
        md_table(hits),
        "",
        "## 2. Evidence use (full population, not selected on outcome)",
        "",
        "R ~ B + E (+ model / + input party), E = mean position of the retrieved chunks' parties; SE "
        "clustered by text. The permutation null shuffles E across texts within each model "
        f"({N_PERMUTATIONS}×). `naive_pull_corr` is the earlier corr(R − B, E − B); its null mean shows how "
        "much of it appears without any evidence use.",
        "",
        md_table(ev[["model", "coef", "ci_low", "ci_high", "p", "perm_null_mean", "perm_null_95", "perm_p",
                     "B_coef", "naive_pull_corr", "naive_pull_corr_null_mean", "n", "r2"]]),
        "",
        f"Reading: holding the baseline prediction fixed, a one-point shift of the retrieved parties' mean "
        f"position moves the RAG prediction by {pooled['coef']:.2f} points (95 % CI {pooled['ci_low']:.2f} to "
        f"{pooled['ci_high']:.2f}); the permutation null is centred at {pooled['perm_null_mean']:.2f}. "
        f"The naive pull correlation is {pooled['naive_pull_corr']:.2f}, of which "
        f"{pooled['naive_pull_corr_null_mean']:.2f} arises with shuffled evidence.",
        "",
        "## 3. Which chunks carry the effect (sample, weighted, descriptive)",
        "",
        "Same regression on the sampled texts with the positions of cited vs. uncited and relevant vs. other "
        "chunks. The sample is selected on the outcome and lacks the buffer, so read this as descriptive.",
        "",
        md_table(by_label),
        "",
        "## 4. Quadrant descriptions (sample)",
        "",
        "Quadrants are defined by the errors, so they describe the error cases, not different mechanisms "
        "(Q1 / Q2 are selected for large moves, Q4 for none). The last columns test Q1 vs. Q2 with input "
        "party and model as controls.",
        "",
        md_table(profiles),
        "",
        "### Share of cited chunks from the input's own party, by input party (Q1 vs. Q2)",
        "",
        md_table(strat),
        "",
        "### Utilisation: share of chunks cited by label level",
        "",
        md_table(util),
        "",
        "## 5. Noise floor",
        "",
        md_table(noise) if noise is not None else
        "Not available: every model has one baseline run (greedy decoding, temperature 0). Re-run the no-RAG "
        "baseline once per model and pass it with `--baseline-rerun <model>=<log>`; the share of texts that "
        "land in 'Q1' / 'Q2' between two baseline runs is the part of the RAG quadrants explained by "
        "run-to-run variation (batching / FP8 non-determinism).",
        "",
        "## 6. Did RAG help or hurt? (full population)",
        "",
        f"helped / hurt = RAG reduced / increased the error by at least {CHANGE_THRESHOLD}. 'toward truth' = the "
        "retrieved parties' mean position is closer to the ground truth than the baseline prediction.",
        "",
        md_table(hh),
        "",
        *[f"![{f}]({f})" for f in hh_figs],
        "",
        *[f"![{f}]({f})" for f in figs],
        "",
    ]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(md_table(ev[["model", "coef", "ci_low", "ci_high", "perm_null_mean", "perm_p", "naive_pull_corr",
                       "naive_pull_corr_null_mean"]]))
    print(f"\nReport: {out / 'summary.md'}")


if __name__ == "__main__":
    main()
