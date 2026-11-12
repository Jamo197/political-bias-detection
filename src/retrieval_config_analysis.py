"""retrieval_config_analysis.py

Data for the thesis section 'Retrieval configurations' (first part of RQ1b).
Same data basis as chapter 02 (``evaluate_metrics.py --rag-vs-norag``):
party_label_* logs, k = 5, bge/hyde_hybrid excluded (HyDERetrieval ignores the
hybrid flag, so it is the same pipeline as bge/hyde), target = exact CHES lrgen.

Writes CSV + Markdown to results/thesis/03_retrieval_configs/:

  noise_floor_pairs.csv / noise_floor_summary.csv
        run-to-run differences of cells that were run more than once
        (fixed context = identical chunks; HyDE = regenerated hypothetical texts)
  config_spread.csv / config_variance.csv
        MAE matrix model x condition and its two-way decomposition
  config_effects.csv
        per-model embedder / strategy / hybrid effects (additive fit on the
        paired chapter-02 CSV, deviations from the model mean)
  pairwise_marginal.csv / pairwise_cells.csv
        post-level paired comparisons of embedders, strategies and hybrid
        within each model (bootstrap CI, Wilcoxon, Holm within model x factor)
  hyde_vs_size.csv
        ΔMAE(HyDE − simple) per model and embedder against model size
  chunk_overlap.csv / chunk_determinism.csv
        Jaccard overlap of the retrieved chunk sets between embedders and
        between strategies / hybrid
  hybrid_lexical.csv / hybrid_lexical_summary.csv
        lexical overlap with the post of chunks only found by hybrid vs only
        found by dense bge retrieval
  party_agreement.csv / party_agreement_error.csv
        same-party / same-bloc share of the chunks per condition, and
        absolute error by number of same-party chunks
  m3_coefficients.csv / m3_summary.csv
        mixed model M3 (RAG rows) + likelihood-ratio test of the design factors
  k_ablation.csv / k_pairwise.csv
        MAE against k for Llama-8B and Ministral-14B (e5/simple)
  retrieval_configs_ches_lrgen.md
        all tables in one report

Usage:  python src/retrieval_config_analysis.py [--cache PATH] [--n-boot 2000] [--skip-lmm]
"""

import argparse
import hashlib
import itertools
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_metrics as em  # noqa: E402

OUT_DIR = em.THESIS_ROOT / "03_retrieval_configs"
TARGET = "ches_lrgen"
PAIRED_CSV = em.THESIS_ROOT / "02_rag_vs_no_rag" / f"rag_vs_no_rag_{TARGET}.csv"
BASELINE_CSV = em.THESIS_ROOT / "01_baseline_no_rag" / f"point_metrics_no_rag_{TARGET}.csv"
EMBEDDERS = ["bge", "e5", "jina", "qwen3"]
STRATEGIES = ["simple", "hyde", "twostage"]
# Embedders whose vectors are Matryoshka-truncated to 1024 dims at ingestion.
MRL_TRUNCATED = {"jina": 2048, "qwen3": 4096}
NATIVE_DIM = {"bge": 1024, "e5": 1024, "jina": 2048, "qwen3": 4096}

K_MODELS = ["Llama-3.1-8B-Instruct", "Ministral-3-14B-Instruct-2512"]
K_CONDITION = "e5/simple"

# Runs superseded by a pipeline fix (not replicates of the final cell).
SUPERSEDED_RUNS = {
    ("Ministral-3-8B-Instruct-2512", "2026-08-08_eval_matrix_20260808_125211"),
}
MIN_REPLICATE_N = 700  # minimum shared posts for a replicate pair
FIXED_CONTEXT_MIN = 0.95  # share of posts with identical chunk lists

GERMAN_STOP = set("""
aber alle allem allen aller alles als also am an ander andere anderem anderen
anderer anderes anderm andern anders auch auf aus bei bin bis bist da damit dann
das dass dasselbe dazu dein deine deinem deinen deiner dem demselben den denn
denselben der derer derselbe derselben des desselben dessen dich die dies diese
dieselbe dieselben diesem diesen dieser dieses dir doch dort du durch ein eine
einem einen einer eines einig einige einigem einigen einiger einiges einmal er es
etwas euch euer eure eurem euren eurer eures für gegen gewesen hab habe haben hat
hatte hatten hier hin hinter ich ihm ihn ihnen ihr ihre ihrem ihren ihrer ihres im
in indem ins ist jede jedem jeden jeder jedes jene jenem jenen jener jenes jetzt
kann kein keine keinem keinen keiner keines können könnte machen man manche
manchem manchen mancher manches mein meine meinem meinen meiner meines mich mir
mit muss musste nach nicht nichts noch nun nur ob oder ohne sehr sein seine seinem
seinen seiner seines selbst sich sie sind so solche solchem solchen solcher
solches soll sollte sondern sonst über um und uns unsere unserem unseren unser
unseres unter viel vom von vor während war waren warst was weg weil weiter welche
welchem welchen welcher welches wenn werde werden wie wieder will wir wird wirst
wo wollen wollte würde würden zu zum zur zwar zwischen the and for http https amp
""".split())


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _ckey(chunk: dict) -> str:
    """Chunk identity: speech id + hash of the chunk text."""
    text = chunk.get("text", "") or ""
    return f"{chunk.get('speech_id', '')}#{hashlib.md5(text.encode()).hexdigest()[:12]}"


def read_logs(base_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    """All party_label_* rows (no dedup) with chunk keys / parties.

    Returns (rows, chunk_text[key], post_text[text_index])."""
    rows, chunk_text, post_text = [], {}, {}
    files = sorted(base_dir.rglob("party_label_*.jsonl"))
    for order, fp in enumerate(files):
        run = fp.relative_to(base_dir).parts[0]
        is_k_file = "k-ablation" in fp.name
        with open(fp, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                try:
                    log = json.loads(line)
                except json.JSONDecodeError:
                    continue
                p = log.get("parameters", {})
                emb = p.get("embedding_model", "none")
                mode = p.get("retrieval_mode", "unknown")
                is_rag = bool(p.get("is_rag", True)) and emb not in (None, "none", "") and mode != "no_rag"
                cond = f"{emb}/{mode}" if is_rag else "no_rag"
                meta = log.get("input_metadata", {})
                tid = str(meta.get("text_index", ""))
                chunks = log.get("inputs", {}).get("retrieved_chunks", []) or []
                keys = []
                for c in chunks:
                    k = _ckey(c)
                    keys.append(k)
                    chunk_text.setdefault(k, c.get("text", "") or "")
                if tid not in post_text:
                    post_text[tid] = log.get("inputs", {}).get("text", "") or ""
                rows.append({
                    "file_order": order,
                    "batch_run": run,
                    "k_file": is_k_file,
                    "model": em._clean_model_name(p.get("llm", "unknown")),
                    "llm_full": p.get("llm", "unknown"),
                    "condition": cond,
                    "embedding_model": emb if is_rag else "none",
                    "retrieval_mode": mode if is_rag else "no_rag",
                    "strategy": mode.replace("_hybrid", "") if is_rag else "no_rag",
                    "hybrid": int(bool(p.get("hybrid", False)) or mode.endswith("_hybrid")) if is_rag else 0,
                    "is_rag": is_rag,
                    "k_chunks": int(p.get("k_chunks", 0)) if is_rag else 0,
                    "text_index": tid,
                    "party": meta.get("party", ""),
                    "speaker": meta.get("speaker", ""),
                    "source": meta.get("source", ""),
                    "predicted_bias": (log.get("output") or {}).get("bias"),
                    "label_ideology": (log.get("ground_truth") or {}).get("label_ideology"),
                    "label_economic": (log.get("ground_truth") or {}).get("label_economic"),
                    "label_galtan": (log.get("ground_truth") or {}).get("label_galtan"),
                    "chunk_keys": tuple(keys),
                    "chunk_parties": tuple(c.get("party") for c in chunks),
                    "chunk_scores": tuple(c.get("score") for c in chunks),
                })
    df = pd.DataFrame(rows)
    print(f"Read {len(df)} rows from {len(files)} files, {len(chunk_text)} distinct chunks.")
    return df, chunk_text, post_text


def load_all(base_dir: Path, cache: Optional[Path]):
    if cache and cache.exists():
        print(f"Loading cache {cache}")
        obj = pd.read_pickle(cache)
        return obj["rows"], obj["chunk_text"], obj["post_text"]
    rows, chunk_text, post_text = read_logs(base_dir)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle({"rows": rows, "chunk_text": chunk_text, "post_text": post_text}, cache)
    return rows, chunk_text, post_text


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Attach CHES lrgen and the derived columns of evaluate_metrics."""
    df = em.attach_ches_lrgen(df.copy())
    return em.prepare_target_frame(df, TARGET)


def main_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Same selection as evaluate_metrics.load_main_matrix: no k-ablation files,
    no_rag + RAG with k = 5, hyde_hybrid excluded, dedup keep last (file order)."""
    d = rows[~rows["k_file"]]
    d = d[(~d["is_rag"]) | (d["k_chunks"] == em.DEFAULT_K)]
    d = d[~d["retrieval_mode"].isin(em.EXCLUDED_RETRIEVAL_MODES)]
    d = d.sort_values("file_order", kind="stable")
    d = d.drop_duplicates(["model", "condition", "text_index"], keep="last")
    return prepare(d)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def paired_boot(d: np.ndarray, n_boot: int, rng) -> Tuple[float, float]:
    if n_boot <= 0 or len(d) < 3:
        return np.nan, np.nan
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def wilcoxon_p(d: np.ndarray) -> float:
    nz = d[d != 0]
    return float(stats.wilcoxon(nz).pvalue) if len(nz) > 0 else 1.0


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else np.nan


def short(m: str) -> str:
    meta = em._meta_for(m)
    fam = {"Mistral": "Ministral"}.get(meta["family"], meta["family"])
    return f"{fam} {int(meta['params_b'])}B"


def tokens(text: str) -> set:
    return {w for w in re.findall(r"\w+", (text or "").lower())
            if len(w) >= 3 and w not in GERMAN_STOP and not w.isdigit()}


def r4(x):
    return em._round(x)


# ---------------------------------------------------------------------------
# 1. Noise floor
# ---------------------------------------------------------------------------

def noise_floor(rows: pd.DataFrame, n_boot: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    d = rows[(~rows["k_file"]) & ((~rows["is_rag"]) | (rows["k_chunks"] == em.DEFAULT_K))]
    d = d[~d.apply(lambda r: (r["model"], r["batch_run"]) in SUPERSEDED_RUNS, axis=1)]
    d = prepare(d)
    out = []

    def add_pair(model, cond_label, kind, a, b, run_a, run_b):
        m = a.merge(b, on="text_index", suffixes=("_a", "_b"))
        if len(m) < MIN_REPLICATE_N:
            return
        y = m["true_a"].to_numpy(float)
        ea, eb = np.abs(m["pred_a"] - y).to_numpy(), np.abs(m["pred_b"] - y).to_numpy()
        diff = ea - eb
        lo, hi = paired_boot(diff, n_boot, rng)
        same = np.mean([ka == kb for ka, kb in zip(m["chunk_keys_a"], m["chunk_keys_b"])])
        jac = np.nanmean([jaccard(ka, kb) for ka, kb in zip(m["chunk_keys_a"], m["chunk_keys_b"])])
        pred_same = np.mean(m["pred_a"].to_numpy() == m["pred_b"].to_numpy())
        if pred_same >= 0.999:
            # e.g. Qwen-32B bge/hyde_hybrid (2026-10-10) carries the HyDE texts and
            # predictions of bge/hyde (2026-10-09): a copied file, not a rerun.
            kind = "identical copy (excluded)"
        elif kind is None:
            kind = "fixed context" if same >= FIXED_CONTEXT_MIN else (
                "HyDE rerun" if "hyde" in cond_label else "retrieval varies")
        out.append({
            "Model": model, "Condition": cond_label, "Kind": kind,
            "Run_a": run_a, "Run_b": run_b, "N": len(m),
            "MAE_a": r4(ea.mean()), "MAE_b": r4(eb.mean()),
            "dMAE": r4(diff.mean()), "dMAE_lo": r4(lo), "dMAE_hi": r4(hi),
            "abs_dMAE": r4(abs(diff.mean())),
            "Pred_identical": r4(np.mean(m["pred_a"].to_numpy() == m["pred_b"].to_numpy())),
            "Chunks_identical": r4(same), "Chunk_jaccard": r4(jac),
        })

    cols = ["text_index", "pred", "true", "chunk_keys"]
    for (model, cond), g in d.groupby(["model", "condition"]):
        runs = [(r, gg.drop_duplicates("text_index", keep="last")[cols])
                for r, gg in g.groupby("batch_run")]
        for (ra, a), (rb, b) in itertools.combinations(runs, 2):
            add_pair(model, cond, None, a, b, ra, rb)

    # bge/hyde_hybrid is the bge/hyde pipeline (hybrid flag ignored by HyDE):
    # the final runs of both conditions are an extra HyDE replicate per model.
    fin = d.sort_values("file_order", kind="stable").drop_duplicates(
        ["model", "condition", "text_index"], keep="last")
    for model, g in fin.groupby("model"):
        a = g[g["condition"] == "bge/hyde"]
        b = g[g["condition"] == "bge/hyde_hybrid"]
        if len(a) and len(b):
            add_pair(model, "bge/hyde vs bge/hyde_hybrid", "HyDE rerun (hyde_hybrid)",
                     a[cols], b[cols], a["batch_run"].iloc[-1], b["batch_run"].iloc[-1])

    all_pairs = pd.DataFrame(out).sort_values(["Kind", "Model", "Condition"]).reset_index(drop=True)
    pairs = all_pairs[all_pairs["Kind"] != "identical copy (excluded)"]
    summ = (pairs.groupby("Kind")
            .agg(N_pairs=("dMAE", "size"),
                 Mean_abs_dMAE=("abs_dMAE", "mean"),
                 Max_abs_dMAE=("abs_dMAE", "max"),
                 RMS_dMAE=("dMAE", lambda s: float(np.sqrt(np.mean(np.square(s))))),
                 Mean_pred_identical=("Pred_identical", "mean"),
                 Mean_chunks_identical=("Chunks_identical", "mean"))
            .reset_index())
    # Run-to-run SD of a cell's MAE: Var(MAE_a − MAE_b) = 2 σ² → σ = RMS / √2.
    summ["SD_run_to_run"] = summ["RMS_dMAE"] / np.sqrt(2)
    def pooled(kind, sub):
        return {
            "Kind": kind, "N_pairs": len(sub), "Mean_abs_dMAE": sub["abs_dMAE"].mean(),
            "Max_abs_dMAE": sub["abs_dMAE"].max(),
            "RMS_dMAE": float(np.sqrt(np.mean(np.square(sub["dMAE"])))),
            "Mean_pred_identical": sub["Pred_identical"].mean(),
            "Mean_chunks_identical": sub["Chunks_identical"].mean(),
        }

    allp = pd.DataFrame([
        pooled("HyDE (pooled)", pairs[pairs["Kind"].str.startswith("HyDE")]),
        pooled("all", pairs),
    ])
    allp["SD_run_to_run"] = allp["RMS_dMAE"] / np.sqrt(2)
    summ = pd.concat([summ, allp], ignore_index=True).round(4)
    return all_pairs, summ


# ---------------------------------------------------------------------------
# 2. Spread of the conditions (paired chapter-02 CSV)
# ---------------------------------------------------------------------------

def config_spread(paired: pd.DataFrame, sd_noise: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    mat = paired.pivot(index="Model", columns="Condition", values="MAE_rag")
    # Full no-RAG baseline (all posts, tab:baseline-metrics), not the paired
    # MAE_base of one condition.
    base = pd.read_csv(BASELINE_CSV).set_index("Model")["MAE"]
    spread = pd.DataFrame({
        "Model": mat.index,
        "Params_B": [em._meta_for(m)["params_b"] for m in mat.index],
        "MAE_no_rag": base.reindex(mat.index).values,
        "N_conditions": mat.notna().sum(1).values,
        "MAE_rag_mean": mat.mean(1).values,
        "MAE_rag_min": mat.min(1).values,
        "MAE_rag_max": mat.max(1).values,
        "Range_conditions": (mat.max(1) - mat.min(1)).values,
        "SD_conditions": mat.std(1, ddof=1).values,
        "Best_condition": mat.idxmin(1).values,
        "Worst_condition": mat.idxmax(1).values,
    })
    spread["Range_over_noise_SD"] = spread["Range_conditions"] / sd_noise
    spread = spread.sort_values("Params_B").round(4)

    long = paired[["Model", "Condition", "MAE_rag"]].dropna()
    fit = smf.ols("MAE_rag ~ C(Model) + C(Condition)", data=long).fit()
    an = anova_lm(fit, typ=2)
    ss_total = float(((long["MAE_rag"] - long["MAE_rag"].mean()) ** 2).sum())
    var = pd.DataFrame({
        "Source": ["Model", "Condition", "Residual (model x condition)"],
        "df": an["df"].values,
        "SS": an["sum_sq"].values,
        "Share_of_SS": an["sum_sq"].values / ss_total,
        "F": an["F"].values,
        "p": an["PR(>F)"].values,
    })
    model_means = mat.mean(1)
    cond_means = mat.mean(0)
    extra = pd.DataFrame([
        {"Source": "SD of model means", "SS": float(model_means.std(ddof=1))},
        {"Source": "SD of condition means", "SS": float(cond_means.std(ddof=1))},
        {"Source": "Mean within-model SD across conditions", "SS": float(mat.std(1, ddof=1).mean())},
        {"Source": "Run-to-run SD (noise floor)", "SS": sd_noise},
    ])
    var = pd.concat([var, extra], ignore_index=True)
    return spread, var


# ---------------------------------------------------------------------------
# 3. Effects per embedder / strategy / hybrid (paired CSV)
# ---------------------------------------------------------------------------

def config_effects(paired: pd.DataFrame) -> pd.DataFrame:
    """Per model, additive fit MAE ~ embedder + strategy on the 12 non-hybrid
    cells; effect of a level = its balanced marginal mean minus the mean of all
    12 predicted cells (handles the missing Qwen-72B jina/hyde cell). Hybrid =
    MAE(x_hybrid) − MAE(x) for bge simple and twostage."""
    import statsmodels.formula.api as smf

    rows = []
    for model, g in paired.groupby("Model"):
        meta = em._meta_for(model)
        base = dict(Model=model, Family=meta["family"], Params_B=meta["params_b"])
        core = g[g["Strategy"].isin(STRATEGIES)]
        fit = smf.ols("MAE_rag ~ C(Embedding) + C(Strategy)", data=core).fit()
        grid = pd.DataFrame(list(itertools.product(EMBEDDERS, STRATEGIES)),
                            columns=["Embedding", "Strategy"])
        grid["pred"] = fit.predict(grid)
        gm = grid["pred"].mean()
        for e in EMBEDDERS:
            rows.append({**base, "Factor": "embedder", "Level": e,
                         "Effect": grid.loc[grid["Embedding"] == e, "pred"].mean() - gm})
        for s in STRATEGIES:
            rows.append({**base, "Factor": "strategy", "Level": s,
                         "Effect": grid.loc[grid["Strategy"] == s, "pred"].mean() - gm})
        cell = g.set_index("Condition")["MAE_rag"]
        for s in ("simple", "twostage"):
            a, b = f"bge/{s}_hybrid", f"bge/{s}"
            if a in cell and b in cell:
                rows.append({**base, "Factor": "hybrid", "Level": f"{s}_hybrid − {s}",
                             "Effect": cell[a] - cell[b]})
        rows.append({**base, "Factor": "residual_sd", "Level": "additive fit",
                     "Effect": float(np.sqrt(fit.mse_resid))})
    out = pd.DataFrame(rows)
    out["Effect"] = out["Effect"].round(4)
    return out


def embedder_ranking(effects: pd.DataFrame) -> pd.DataFrame:
    """Mean embedder effect and rank over models, with the MRL truncation flag."""
    e = effects[effects["Factor"] == "embedder"].copy()
    e["Rank_in_model"] = e.groupby("Model")["Effect"].rank()
    out = (e.groupby("Level")
           .agg(Mean_effect=("Effect", "mean"), SD_effect=("Effect", "std"),
                Mean_rank=("Rank_in_model", "mean"),
                N_best=("Rank_in_model", lambda s: int((s == 1).sum())),
                N_worst=("Rank_in_model", lambda s: int((s == 4).sum())))
           .reset_index().rename(columns={"Level": "Embedder"}))
    out["Native_dim"] = out["Embedder"].map(NATIVE_DIM)
    out["MRL_truncated_to_1024"] = out["Embedder"].isin(MRL_TRUNCATED)
    return out.sort_values("Mean_effect").round(4)


# ---------------------------------------------------------------------------
# 4. Pairwise post-level tests
# ---------------------------------------------------------------------------

def _pair_row(model, factor, scope, a, b, ea, eb, n_boot, rng):
    d = ea - eb
    lo, hi = paired_boot(d, n_boot, rng)
    return {"Model": model, "Factor": factor, "Scope": scope, "A": a, "B": b,
            "N": len(d), "MAE_A": r4(ea.mean()), "MAE_B": r4(eb.mean()),
            "dMAE": r4(d.mean()), "dMAE_lo": r4(lo), "dMAE_hi": r4(hi),
            "p_wilcoxon": wilcoxon_p(d)}


def pairwise(valid: pd.DataFrame, n_boot: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """dMAE = MAE(A) − MAE(B) on shared posts (negative = A better).

    marginal: per post, the abs error averaged over the levels of the other
    factor (embedder: over the 3 strategies; strategy: over the 4 embedders),
    posts with all cells only. cells: one comparison per fixed other factor.
    Holm within model x factor x table."""
    rng = np.random.default_rng(seed)
    rag = valid[valid["is_rag"]]
    marg, cells = [], []
    for model, g in rag.groupby("model"):
        err = g.pivot_table(index="text_index", columns="condition", values="abs_error")
        # marginal embedder
        cols_e = {e: [f"{e}/{s}" for s in STRATEGIES if f"{e}/{s}" in err] for e in EMBEDDERS}
        full_e = all(len(v) == 3 for v in cols_e.values())
        scope_e = "mean over 3 strategies" if full_e else "mean over shared strategies"
        if not full_e:  # e.g. Qwen-72B without jina/hyde -> use strategies all embedders have
            shared = [s for s in STRATEGIES if all(f"{e}/{s}" in err for e in EMBEDDERS)]
            cols_e = {e: [f"{e}/{s}" for s in shared] for e in EMBEDDERS}
            scope_e += f" ({', '.join(shared)})"
        sub = err[sum(cols_e.values(), [])].dropna()
        me = {e: sub[c].mean(1).to_numpy() for e, c in cols_e.items()}
        for a, b in itertools.combinations(EMBEDDERS, 2):
            marg.append(_pair_row(model, "embedder", scope_e, a, b, me[a], me[b], n_boot, rng))
        # marginal strategy
        cols_s = {s: [f"{e}/{s}" for e in EMBEDDERS if f"{e}/{s}" in err] for s in STRATEGIES}
        n_e = min(len(v) for v in cols_s.values())
        shared_e = [e for e in EMBEDDERS if all(f"{e}/{s}" in err for s in STRATEGIES)]
        cols_s = {s: [f"{e}/{s}" for e in shared_e] for s in STRATEGIES}
        scope_s = ("mean over 4 embedders" if len(shared_e) == 4
                   else f"mean over shared embedders ({', '.join(shared_e)})")
        sub = err[sum(cols_s.values(), [])].dropna()
        ms = {s: sub[c].mean(1).to_numpy() for s, c in cols_s.items()}
        for a, b in itertools.combinations(STRATEGIES, 2):
            marg.append(_pair_row(model, "strategy", scope_s, a, b, ms[a], ms[b], n_boot, rng))
        # hybrid (bge only)
        for s in ("simple", "twostage"):
            a, b = f"bge/{s}_hybrid", f"bge/{s}"
            if a in err and b in err:
                sub = err[[a, b]].dropna()
                marg.append(_pair_row(model, "hybrid", f"bge/{s}", "hybrid", "dense",
                                      sub[a].to_numpy(), sub[b].to_numpy(), n_boot, rng))
        # cells
        for s in STRATEGIES:
            for a, b in itertools.combinations(EMBEDDERS, 2):
                ca, cb = f"{a}/{s}", f"{b}/{s}"
                if ca in err and cb in err:
                    sub = err[[ca, cb]].dropna()
                    cells.append(_pair_row(model, "embedder", f"strategy={s}", a, b,
                                           sub[ca].to_numpy(), sub[cb].to_numpy(), n_boot, rng))
        for e in EMBEDDERS:
            for a, b in itertools.combinations(STRATEGIES, 2):
                ca, cb = f"{e}/{a}", f"{e}/{b}"
                if ca in err and cb in err:
                    sub = err[[ca, cb]].dropna()
                    cells.append(_pair_row(model, "strategy", f"embedder={e}", a, b,
                                           sub[ca].to_numpy(), sub[cb].to_numpy(), n_boot, rng))

    def finish(rows):
        df = pd.DataFrame(rows)
        df["p_holm"] = np.nan
        for _, idx in df.groupby(["Model", "Factor"]).groups.items():
            df.loc[idx, "p_holm"] = em._holm(df.loc[idx, "p_wilcoxon"].tolist())
        df["significant"] = df["p_holm"] < 0.05
        df["Model_short"] = df["Model"].map(short)
        df["Params_B"] = df["Model"].map(lambda m: em._meta_for(m)["params_b"])
        df["Family"] = df["Model"].map(lambda m: em._meta_for(m)["family"])
        return df.sort_values(["Factor", "Family", "Params_B", "Scope", "A", "B"]).reset_index(drop=True)

    return finish(marg), finish(cells)


# ---------------------------------------------------------------------------
# 5. HyDE vs size
# ---------------------------------------------------------------------------

def hyde_vs_size(paired: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cell = paired.set_index(["Model", "Condition"])["MAE_rag"]
    rows = []
    for model in paired["Model"].unique():
        meta = em._meta_for(model)
        for e in EMBEDDERS:
            h, s, t = (cell.get((model, f"{e}/{x}"), np.nan) for x in ("hyde", "simple", "twostage"))
            rows.append({"Model": model, "Family": meta["family"], "Params_B": meta["params_b"],
                         "Embedding": e, "dMAE_hyde_simple": h - s, "dMAE_twostage_simple": t - s})
    df = pd.DataFrame(rows)
    per_model = (df.groupby(["Model", "Family", "Params_B"])
                 [["dMAE_hyde_simple", "dMAE_twostage_simple"]].mean().reset_index())
    per_model["log2_params"] = np.log2(per_model["Params_B"])
    tests = []
    for col in ("dMAE_hyde_simple", "dMAE_twostage_simple"):
        rho, p = stats.spearmanr(per_model["Params_B"], per_model[col])
        sl = stats.linregress(per_model["log2_params"], per_model[col])
        tests.append({"Contrast": col, "N_models": len(per_model),
                      "Spearman_rho_vs_size": rho, "p_spearman": p,
                      "Slope_per_doubling": sl.slope, "Slope_lo": sl.slope - 1.96 * sl.stderr,
                      "Slope_hi": sl.slope + 1.96 * sl.stderr, "p_slope": sl.pvalue})
    out = per_model.merge(
        df.pivot(index="Model", columns="Embedding", values="dMAE_hyde_simple")
        .add_prefix("dHyDE_").reset_index(), on="Model")
    return out.sort_values(["Family", "Params_B"]).round(4), pd.DataFrame(tests).round(4)


# ---------------------------------------------------------------------------
# 6. Retrieved-chunk analyses
# ---------------------------------------------------------------------------

def chunk_determinism(valid: pd.DataFrame) -> pd.DataFrame:
    """Is the chunk list identical across the ten models (per condition)?"""
    rows = []
    for cond, g in valid[valid["is_rag"]].groupby("condition"):
        per_post = g.groupby("text_index")["chunk_keys"].agg(lambda s: len(set(s)))
        sets = g.groupby("text_index")["chunk_keys"].agg(lambda s: len({frozenset(x) for x in s}))
        rows.append({"Condition": cond, "N_posts": len(per_post),
                     "Share_identical_list_across_models": r4((per_post == 1).mean()),
                     "Share_identical_set_across_models": r4((sets == 1).mean())})
    return pd.DataFrame(rows)


def chunk_overlap(valid: pd.DataFrame) -> pd.DataFrame:
    """Mean per-post Jaccard of chunk sets (and speech sets) between two
    conditions of the same model, averaged over models."""
    rag = valid[valid["is_rag"]]
    keys = rag.set_index(["model", "condition", "text_index"])["chunk_keys"].sort_index()
    speech = keys.map(lambda ks: tuple(k.split("#")[0] for k in ks))
    comps = []
    for s in STRATEGIES:
        for a, b in itertools.combinations(EMBEDDERS, 2):
            comps.append(("embedder", s, f"{a}/{s}", f"{b}/{s}"))
    for e in EMBEDDERS:
        for a, b in itertools.combinations(STRATEGIES, 2):
            comps.append(("strategy", e, f"{e}/{a}", f"{e}/{b}"))
    for s in ("simple", "twostage"):
        comps.append(("hybrid", "bge", f"bge/{s}", f"bge/{s}_hybrid"))
    comps.append(("hybrid", "bge", "bge/simple_hybrid", "bge/twostage_hybrid"))

    rows = []
    models = rag["model"].unique()
    for factor, fixed, ca, cb in comps:
        jac, jac_sp, inter, top1 = [], [], [], []
        for m in models:
            try:
                ka, kb = keys.loc[(m, ca)], keys.loc[(m, cb)]
                sa, sb = speech.loc[(m, ca)], speech.loc[(m, cb)]
            except KeyError:
                continue
            common = ka.index.intersection(kb.index)
            for t in common:
                x, y = ka[t], kb[t]
                jac.append(jaccard(x, y))
                jac_sp.append(jaccard(sa[t], sb[t]))
                inter.append(len(set(x) & set(y)))
                top1.append(bool(x) and bool(y) and x[0] == y[0])
        if not jac:
            continue
        inter = np.asarray(inter)
        rows.append({"Factor": factor, "Fixed": fixed, "A": ca, "B": cb, "N_post_model": len(jac),
                     "Jaccard_chunks": r4(np.nanmean(jac)), "Jaccard_speeches": r4(np.nanmean(jac_sp)),
                     "Mean_shared_of_5": r4(inter.mean()),
                     "Share_disjoint": r4((inter == 0).mean()),
                     "Share_identical_set": r4((inter == 5).mean()),
                     "Share_same_top1": r4(np.mean(top1))})
    return pd.DataFrame(rows)


def hybrid_lexical(valid: pd.DataFrame, chunk_text: Dict[str, str], post_text: Dict[str, str],
                   seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """bge dense vs hybrid: for chunks found only by the hybrid run vs only by
    the dense run (same post, same model), share of the post's content words
    that occur in the chunk, and same-party share."""
    rag = valid[valid["is_rag"]]
    keys = rag.set_index(["model", "condition", "text_index"])
    tok_cache: Dict[str, set] = {}

    def tk(key):
        if key not in tok_cache:
            tok_cache[key] = tokens(chunk_text.get(key, ""))
        return tok_cache[key]

    rows = []
    for s in ("simple", "twostage"):
        dense, hyb = f"bge/{s}", f"bge/{s}_hybrid"
        # retrieval is deterministic across models for these conditions -> use one row per post
        d = keys.xs(dense, level="condition").reset_index().drop_duplicates("text_index")
        h = keys.xs(hyb, level="condition").reset_index().drop_duplicates("text_index")
        m = d.merge(h, on="text_index", suffixes=("_d", "_h"))
        for _, r in m.iterrows():
            q = tokens(post_text.get(r["text_index"], ""))
            if not q:
                continue
            author = em._norm_party(em.PARTY_ALIASES.get(r["party_d"], r["party_d"]))
            kd, kh = list(r["chunk_keys_d"]), list(r["chunk_keys_h"])
            pd_ = dict(zip(kd, r["chunk_parties_d"]))
            ph_ = dict(zip(kh, r["chunk_parties_h"]))
            only_d = [k for k in kd if k not in set(kh)]
            only_h = [k for k in kh if k not in set(kd)]
            shared = [k for k in kd if k in set(kh)]

            def cov(ks):
                return float(np.mean([len(q & tk(k)) / len(q) for k in ks])) if ks else np.nan

            def same(ks, pmap):
                v = [em._norm_party(pmap[k]) == author for k in ks if em._norm_party(pmap[k])]
                return float(np.mean(v)) if v and author else np.nan

            rows.append({"Strategy": s, "text_index": r["text_index"],
                         "N_only_dense": len(only_d), "N_only_hybrid": len(only_h),
                         "Coverage_only_dense": cov(only_d), "Coverage_only_hybrid": cov(only_h),
                         "Coverage_shared": cov(shared),
                         "SameParty_only_dense": same(only_d, pd_),
                         "SameParty_only_hybrid": same(only_h, ph_)})
    per_post = pd.DataFrame(rows)
    summ = []
    for s, g in per_post.groupby("Strategy"):
        changed = g[g["N_only_hybrid"] > 0]
        dcov = (changed["Coverage_only_hybrid"] - changed["Coverage_only_dense"]).dropna()
        dpar = (changed["SameParty_only_hybrid"] - changed["SameParty_only_dense"]).dropna()
        summ.append({
            "Strategy": s, "N_posts": len(g), "Share_posts_changed": r4(len(changed) / len(g)),
            "Mean_chunks_replaced": r4(changed["N_only_hybrid"].mean()),
            "Coverage_shared": r4(g["Coverage_shared"].mean()),
            "Coverage_only_dense": r4(changed["Coverage_only_dense"].mean()),
            "Coverage_only_hybrid": r4(changed["Coverage_only_hybrid"].mean()),
            "dCoverage_hybrid_minus_dense": r4(dcov.mean()),
            "p_wilcoxon_coverage": wilcoxon_p(dcov.to_numpy()),
            "SameParty_only_dense": r4(changed["SameParty_only_dense"].mean()),
            "SameParty_only_hybrid": r4(changed["SameParty_only_hybrid"].mean()),
            "dSameParty_hybrid_minus_dense": r4(dpar.mean()),
            "p_wilcoxon_sameparty": wilcoxon_p(dpar.to_numpy()),
        })
    return per_post, pd.DataFrame(summ)


def party_agreement(valid: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per post: share of mapped chunks from the author's party and bloc.
    Aggregated per model x condition; plus abs error by # same-party chunks."""
    positions = em.load_party_positions()
    rag = valid[valid["is_rag"]].copy()

    def per_row(party, cps):
        a = em._norm_party(party)
        ap = positions.get(a)
        mapped = [em._norm_party(c) for c in cps]
        mapped = [c for c in mapped if c]
        if not a or not mapped:
            return np.nan, np.nan, np.nan, 0
        same = sum(c == a for c in mapped)
        bloc = (np.mean([em._bloc(positions[c]) == em._bloc(ap) for c in mapped if c in positions])
                if ap is not None else np.nan)
        dist = (np.mean([abs(positions[c] - ap) for c in mapped if c in positions])
                if ap is not None else np.nan)
        return same / len(mapped), bloc, dist, same

    vals = [per_row(p, c) for p, c in zip(rag["party"], rag["chunk_parties"])]
    rag["same_party_share"] = [v[0] for v in vals]
    rag["same_bloc_share"] = [v[1] for v in vals]
    rag["party_distance"] = [v[2] for v in vals]
    rag["n_same_party"] = [v[3] for v in vals]

    per_cell = (rag.groupby(["model", "condition"])
                .agg(N=("text_index", "size"), Same_party=("same_party_share", "mean"),
                     Same_bloc=("same_bloc_share", "mean"), Party_distance=("party_distance", "mean"),
                     MAE=("abs_error", "mean"))
                .reset_index().rename(columns={"model": "Model", "condition": "Condition"}))
    per_cond = (per_cell.groupby("Condition")
                .agg(Same_party=("Same_party", "mean"), Same_party_sd_models=("Same_party", "std"),
                     Same_bloc=("Same_bloc", "mean"), Party_distance=("Party_distance", "mean"),
                     MAE_mean_models=("MAE", "mean"))
                .reset_index())
    # within-model correlation (across conditions) of party agreement and MAE
    corr = []
    for m, g in per_cell.groupby("Model"):
        rho, p = stats.spearmanr(g["Same_party"], g["MAE"])
        corr.append({"Model": m, "N_conditions": len(g), "Spearman_sameparty_MAE": rho, "p": p})
    per_cond = per_cond.round(4)
    per_cell = per_cell.round(4)
    by_n = (rag.groupby(["model", "n_same_party"])
            .agg(N=("abs_error", "size"), MAE=("abs_error", "mean"),
                 Mean_signed_error=("signed_error", "mean"))
            .reset_index().rename(columns={"model": "Model", "n_same_party": "N_same_party_chunks"}))
    return per_cell, per_cond, by_n.round(4), pd.DataFrame(corr).round(4)


# ---------------------------------------------------------------------------
# 7. Mixed model M3
# ---------------------------------------------------------------------------

def m3(valid: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """M3 of evaluate_metrics on RAG rows (ches_lrgen, hyde_hybrid excluded),
    plus the same model without design factors for an LR test."""
    df = valid[valid["is_rag"]].copy()
    df["size_c"] = df["log2_size"] - df["log2_size"].mean()
    fam = em._cat("family", "Llama", df)
    party = em._cat("party_group", "SPD", df)
    design = ["C(strategy, Treatment('simple'))", "hybrid", "C(embedding_model, Treatment('bge'))"]
    full = " + ".join([fam, "size_c", party, *design])
    reduced = " + ".join([fam, "size_c", party])
    coefs, summ = [], []
    for outcome in ("abs_error", "signed_error"):
        fits = {}
        for spec, rhs in (("M3_rag_design", full), ("M3_no_design", reduced)):
            print(f"  [LMM] {outcome} ~ {spec} (n={len(df)}) …")
            res, conv, err = em._fit_lmm(f"{outcome} ~ {rhs}", df)
            if res is None:
                print("   failed:", err)
                continue
            fits[spec] = res
            ci = res.conf_int().loc[res.fe_params.index]
            for term in res.fe_params.index:
                coefs.append({"Outcome": outcome, "Spec": spec, "Term": term,
                              "Estimate": r4(res.fe_params[term]), "SE": r4(res.bse_fe[term]),
                              "CI_lo": r4(ci.loc[term, 0]), "CI_hi": r4(ci.loc[term, 1]),
                              "p": em._round(res.pvalues[term], 6)})
            vt = float(res.cov_re.iloc[0, 0])
            summ.append({"Outcome": outcome, "Spec": spec, "N": len(df),
                         "N_texts": df["text_index"].nunique(), "Var_text": r4(vt),
                         "Var_resid": r4(res.scale), "ICC": r4(vt / (vt + res.scale)),
                         "LogLik": round(res.llf, 2), "AIC": round(res.aic, 2), "BIC": round(res.bic, 2),
                         "Converged": conv, "Formula": f"{outcome} ~ {rhs}"})
        if len(fits) == 2:
            lr = 2 * (fits["M3_rag_design"].llf - fits["M3_no_design"].llf)
            ddf = len(fits["M3_rag_design"].fe_params) - len(fits["M3_no_design"].fe_params)
            summ.append({"Outcome": outcome, "Spec": "LR test design factors",
                         "LogLik": round(lr, 2), "AIC": ddf,
                         "BIC": float(stats.chi2.sf(lr, ddf)),
                         "Formula": "LR = 2ΔlogLik (LogLik col), df (AIC col), p (BIC col)"})
    return pd.DataFrame(coefs), pd.DataFrame(summ)


# ---------------------------------------------------------------------------
# 8. k ablation
# ---------------------------------------------------------------------------

def k_ablation(rows: pd.DataFrame, valid: pd.DataFrame, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    kf = rows[rows["k_file"] & rows["model"].isin(K_MODELS) & (rows["condition"] == K_CONDITION)]
    kf = kf.sort_values("file_order", kind="stable").drop_duplicates(
        ["model", "k_chunks", "text_index"], keep="last")
    kf = prepare(kf)
    k5 = valid[valid["model"].isin(K_MODELS) & (valid["condition"] == K_CONDITION)].copy()
    k0 = valid[valid["model"].isin(K_MODELS) & (valid["condition"] == "no_rag")].copy()
    allk = pd.concat([kf, k5, k0], ignore_index=True)
    cols = ["model", "k_chunks", "text_index", "abs_error", "signed_error", "true", "pred",
            "party", "chunk_parties", "batch_run"]
    allk = allk[cols]
    positions = em.load_party_positions()

    def same_party(party, cps):
        a = em._norm_party(party)
        mp = [em._norm_party(c) for c in cps]
        mp = [c for c in mp if c]
        return np.mean([c == a for c in mp]) if a and mp else np.nan

    allk["same_party"] = [same_party(p, c) for p, c in zip(allk["party"], allk["chunk_parties"])]

    metrics, pairs = [], []
    for model, g in allk.groupby("model"):
        ks = sorted(g["k_chunks"].unique())
        common = set.intersection(*[set(g.loc[g["k_chunks"] == k, "text_index"]) for k in ks])
        for subset in ("all_available", "common_texts"):
            for k in ks:
                s = g[g["k_chunks"] == k]
                if subset == "common_texts":
                    s = s[s["text_index"].isin(common)]
                e = s["abs_error"].to_numpy()
                lo, hi = paired_boot(e, n_boot, rng)
                y, p = s["true"].to_numpy(float), s["pred"].to_numpy(float)
                slope = np.polyfit(y, p, 1)[0] if len(s) > 2 else np.nan
                metrics.append({"Subset": subset, "Model": model, "K": k,
                                "Source": ";".join(sorted(s["batch_run"].unique())),
                                "N": len(s), "MAE": r4(e.mean()), "MAE_lo": r4(lo), "MAE_hi": r4(hi),
                                "Mean_signed_error": r4(s["signed_error"].mean()),
                                "Calibration_slope": r4(slope),
                                "Same_party_share": r4(s["same_party"].mean())})
        piv = g.pivot_table(index="text_index", columns="k_chunks", values="abs_error")
        model_pairs = []
        for ref in (0, 5):
            for k in ks:
                if k == ref:
                    continue
                sub = piv[[k, ref]].dropna()
                row = _pair_row(model, "k", f"vs k={ref}", f"k={k}", f"k={ref}",
                                sub[k].to_numpy(), sub[ref].to_numpy(), n_boot, rng)
                model_pairs.append(row)
        for r, a in zip(model_pairs, em._holm([r["p_wilcoxon"] for r in model_pairs])):
            r["p_holm"] = a
            r["significant"] = a < 0.05
        pairs.extend(model_pairs)
    return pd.DataFrame(metrics), pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(t: Dict[str, pd.DataFrame], path: Path, n_boot: int, seed: int):
    def tab(df, cols=None, nd=4):
        d = df[cols] if cols else df
        return em._md_table(d.round(nd))

    nf = t["noise_summary"]
    md = [
        f"# Retrieval configurations (RQ1b, part 1) — target `{TARGET}`",
        "",
        f"Data: party_label_* logs, k = {em.DEFAULT_K}, bge/hyde_hybrid excluded (same pipeline "
        "as bge/hyde: HyDERetrieval ignores the hybrid flag). Paired bootstrap CIs "
        f"({n_boot} resamples, seed {seed}), Wilcoxon signed-rank, Holm within model × factor. "
        "ΔMAE = MAE(A) − MAE(B); negative = A better.",
        "",
        "## 1. Noise floor (run-to-run)",
        "",
        "Cells that were run more than once (≥ "
        f"{MIN_REPLICATE_N} shared posts). *fixed context*: ≥ {FIXED_CONTEXT_MIN:.0%} of posts "
        "got the identical chunk list, so only LLM sampling differs. *HyDE rerun*: hypothetical "
        "texts regenerated. The superseded 2026-08-08 Ministral-8B bge/twostage runs are excluded. "
        "SD_run_to_run = RMS(ΔMAE)/√2.",
        "",
        tab(nf),
        "",
        tab(t["noise_pairs"], ["Kind", "Model", "Condition", "N", "MAE_a", "MAE_b", "dMAE",
                               "dMAE_lo", "dMAE_hi", "Pred_identical", "Chunks_identical"]),
        "",
        "## 2. Spread of the conditions",
        "",
        "Two-way decomposition of the MAE matrix (10 models × 14 conditions, paired CSV of chapter 02).",
        "",
        tab(t["variance"]),
        "",
        tab(t["spread"], ["Model", "MAE_no_rag", "MAE_rag_mean", "MAE_rag_min", "MAE_rag_max",
                          "Range_conditions", "SD_conditions", "Range_over_noise_SD",
                          "Best_condition", "Worst_condition"]),
        "",
        "## 3. Embedder",
        "",
        "Effect = balanced marginal mean of the embedder minus the model mean (additive fit "
        "per model on the 12 non-hybrid cells). jina (2048 → 1024) and qwen3 (4096 → 1024) are "
        "Matryoshka-truncated; bge and e5 are native 1024.",
        "",
        tab(t["ranking"]),
        "",
        "Post-level, marginal over strategies:",
        "",
        tab(t["pw_marg"][t["pw_marg"]["Factor"] == "embedder"],
            ["Model_short", "A", "B", "N", "dMAE", "dMAE_lo", "dMAE_hi", "p_holm", "significant"]),
        "",
        "Chunk overlap between embedders (mean per-post Jaccard, averaged over models):",
        "",
        tab(t["overlap"][t["overlap"]["Factor"] == "embedder"]),
        "",
        "## 4. Retrieval strategy",
        "",
        "Post-level, marginal over embedders:",
        "",
        tab(t["pw_marg"][t["pw_marg"]["Factor"] == "strategy"],
            ["Model_short", "A", "B", "N", "dMAE", "dMAE_lo", "dMAE_hi", "p_holm", "significant"]),
        "",
        "HyDE − simple against model size (cell MAEs, mean over embedders):",
        "",
        tab(t["hyde"]),
        "",
        tab(t["hyde_tests"]),
        "",
        "Does the reranker / HyDE change the chunks? (same embedder):",
        "",
        tab(t["overlap"][t["overlap"]["Factor"] == "strategy"]),
        "",
        "Determinism of retrieval across the ten models:",
        "",
        tab(t["determinism"]),
        "",
        "## 5. Hybrid search (bge only)",
        "",
        tab(t["pw_marg"][t["pw_marg"]["Factor"] == "hybrid"],
            ["Model_short", "Scope", "N", "dMAE", "dMAE_lo", "dMAE_hi", "p_holm", "significant"]),
        "",
        tab(t["overlap"][t["overlap"]["Factor"] == "hybrid"]),
        "",
        "Chunks only in the hybrid list vs only in the dense list (same post). Coverage = share "
        "of the post's content words (≥ 3 letters, no stop words) that occur in the chunk.",
        "",
        tab(t["hyb_summary"]),
        "",
        "## 6. Party agreement of the chunks",
        "",
        tab(t["party_cond"]),
        "",
        "Within-model Spearman correlation across conditions: same-party share vs MAE.",
        "",
        tab(t["party_corr"]),
        "",
        "## 7. What explains the variation — mixed model M3",
        "",
        "Random intercept per post, RAG rows only. size_c = centred log2(parameters): its "
        "coefficient is the change per doubling of model size. References: Llama, SPD, simple, "
        "dense, bge.",
        "",
        tab(t["m3_coef"]) if not t["m3_coef"].empty else "(skipped)",
        "",
        tab(t["m3_summ"]) if not t["m3_summ"].empty else "",
        "",
        f"## 8. Number of chunks k ({K_CONDITION})",
        "",
        "k = 0 is the no-RAG run, k = 5 comes from the main matrix, the other k from the "
        "k-ablation runs (2026-10-14, 2026-10-15).",
        "",
        tab(t["k"]),
        "",
        tab(t["k_pairs"], ["Model", "Scope", "A", "B", "N", "dMAE", "dMAE_lo", "dMAE_hi",
                           "p_holm", "significant"]),
        "",
    ]
    path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", type=Path, default=em.BASE_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--cache", type=Path, default=None, help="Pickle cache of the parsed logs.")
    ap.add_argument("--n-boot", type=int, default=em.DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=em.DEFAULT_SEED)
    ap.add_argument("--skip-lmm", action="store_true", help="Skip the mixed model M3.")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    rows, chunk_text, post_text = load_all(args.base_dir, args.cache)
    valid = main_matrix(rows)
    paired = pd.read_csv(PAIRED_CSV)
    t: Dict[str, pd.DataFrame] = {}

    print("1. noise floor")
    t["noise_pairs"], t["noise_summary"] = noise_floor(rows, args.n_boot, args.seed)
    fixed = t["noise_summary"].set_index("Kind")
    sd_noise = float(fixed.loc["fixed context", "SD_run_to_run"]) if "fixed context" in fixed.index \
        else float(fixed.loc["all", "SD_run_to_run"])
    print("2. spread")
    t["spread"], t["variance"] = config_spread(paired, sd_noise)
    print("3. effects")
    t["effects"] = config_effects(paired)
    t["ranking"] = embedder_ranking(t["effects"])
    print("4. pairwise")
    t["pw_marg"], t["pw_cells"] = pairwise(valid, args.n_boot, args.seed)
    print("5. HyDE vs size")
    t["hyde"], t["hyde_tests"] = hyde_vs_size(paired)
    print("6. chunks")
    t["determinism"] = chunk_determinism(valid)
    t["overlap"] = chunk_overlap(valid)
    t["hyb_posts"], t["hyb_summary"] = hybrid_lexical(valid, chunk_text, post_text, args.seed)
    t["party_cell"], t["party_cond"], t["party_by_n"], t["party_corr"] = party_agreement(valid)
    print("7. M3")
    if args.skip_lmm:
        t["m3_coef"], t["m3_summ"] = pd.DataFrame(), pd.DataFrame()
    else:
        t["m3_coef"], t["m3_summ"] = m3(valid)
    print("8. k ablation")
    t["k"], t["k_pairs"] = k_ablation(rows, valid, args.n_boot, args.seed)

    files = {
        "noise_floor_pairs.csv": "noise_pairs", "noise_floor_summary.csv": "noise_summary",
        "config_spread.csv": "spread", "config_variance.csv": "variance",
        "config_effects.csv": "effects", "embedder_ranking.csv": "ranking",
        "pairwise_marginal.csv": "pw_marg", "pairwise_cells.csv": "pw_cells",
        "hyde_vs_size.csv": "hyde", "hyde_vs_size_tests.csv": "hyde_tests",
        "chunk_determinism.csv": "determinism", "chunk_overlap.csv": "overlap",
        "hybrid_lexical.csv": "hyb_posts", "hybrid_lexical_summary.csv": "hyb_summary",
        "party_agreement.csv": "party_cell", "party_agreement_by_condition.csv": "party_cond",
        "party_agreement_error.csv": "party_by_n", "party_agreement_corr.csv": "party_corr",
        "m3_coefficients.csv": "m3_coef", "m3_summary.csv": "m3_summ",
        "k_ablation.csv": "k", "k_pairwise.csv": "k_pairs",
    }
    for fn, key in files.items():
        if key in t and not t[key].empty:
            t[key].to_csv(out / fn, index=False)
    write_report(t, out / f"retrieval_configs_{TARGET}.md", args.n_boot, args.seed)
    print(f"  → {out}")


if __name__ == "__main__":
    main()
