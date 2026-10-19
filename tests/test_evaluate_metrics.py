"""Unit tests for the metric primitives, bootstrap and mixed models in
src/evaluate_metrics.py (synthetic data only)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import evaluate_metrics as em  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_ccc_perfect_and_shifted(rng):
    t = rng.uniform(1, 7, 200)
    assert em._vec_metrics(t, t)["CCC"] == pytest.approx(1.0)
    shifted = em._vec_metrics(t, t + 1.0)
    assert shifted["Pearson_r"] == pytest.approx(1.0)
    assert shifted["CCC"] < shifted["Pearson_r"]


def test_calibration_slope_recovered(rng):
    t = rng.uniform(1, 7, 500)
    m = em._vec_metrics(t, 0.5 * t + 2.0)
    assert m["Calibration_slope"] == pytest.approx(0.5)
    assert m["Calibration_intercept"] == pytest.approx(2.0)


def test_vec_metrics_match_scipy(rng):
    t = rng.integers(1, 8, 300).astype(float)
    p = t + rng.normal(0, 1, 300)
    m = em._vec_metrics(t, p)
    assert m["Pearson_r"] == pytest.approx(pearsonr(t, p)[0])
    assert m["Spearman_rho"] == pytest.approx(spearmanr(t, p)[0])
    assert m["MAE"] == pytest.approx(np.abs(p - t).mean())
    assert m["Mean_signed_error"] == pytest.approx((p - t).mean())


def test_zero_variance_gives_nan():
    t = np.full(10, 4.0)
    m = em._vec_metrics(t, np.arange(10.0))
    assert np.isnan(m["Pearson_r"]) and np.isnan(m["Calibration_slope"])


def test_moment_metrics_agree_with_vec(rng):
    t = rng.uniform(1, 7, 400)
    p = t * 0.7 + rng.normal(0, 0.5, 400)
    a = em._vec_metrics(t, p)
    b = em._metrics_from_moments(em._row_moments(t, p).sum(0, keepdims=True))
    for k in b:
        assert b[k][0] == pytest.approx(a[k]), k


def test_bootstrap_ci_contains_point_and_shrinks(rng):
    def mae(t, p):
        return np.abs(p - t).mean(-1)

    widths = []
    for n in (50, 800):
        t = rng.uniform(1, 7, n)
        p = t + rng.normal(0, 1, n)
        lo, hi, _ = em.bootstrap_ci(mae, [t, p], 1000, np.random.default_rng(1))
        assert lo <= np.abs(p - t).mean() <= hi
        widths.append(hi - lo)
    assert widths[1] < widths[0]


MODELS = (
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-32B-Instruct",
    "Ministral-3-8B-Instruct-2512",
    "Ministral-3-14B-Instruct-2512",
)


def _frame(n_texts=120, models=MODELS, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.uniform(2, 6, n_texts)
    text_effect = rng.normal(0, 0.6, n_texts)  # shared across models -> ICC > 0
    rows = []
    for model in models:
        for cond in ("no_rag", "e5/simple"):
            is_rag = cond != "no_rag"
            bias = 0.0 if model.startswith("Qwen") else 0.8  # injected family effect
            pred = true + bias + text_effect + rng.normal(0, 0.5, n_texts)
            for i in range(n_texts):
                rows.append(
                    dict(
                        model=model, condition=cond, is_rag=is_rag,
                        text_index=str(i), party="SPD" if i % 2 else "AfD",
                        predicted_bias=pred[i], label_ideology=true[i],
                        strategy="simple" if is_rag else "no_rag",
                        hybrid=0, embedding_model="e5" if is_rag else "none",
                    )
                )
    return pd.DataFrame(rows)


def test_paired_delta_zero_when_identical():
    df = _frame(models=("Qwen2.5-7B-Instruct",))
    base = df[df.condition == "no_rag"].copy()
    rag = base.copy()
    rag["condition"], rag["is_rag"] = "e5/simple", True
    valid = em.prepare_target_frame(pd.concat([base, rag]), "label_ideology")
    out = em.compute_pairwise_rag_delta(valid, "label_ideology", n_boot=200)
    assert out["ΔMAE (RAG-Base)"].iloc[0] == 0
    assert out["ΔMAE (RAG-Base)_lo"].iloc[0] == 0 and out["ΔMAE (RAG-Base)_hi"].iloc[0] == 0


def test_group_comparison_detects_family_bias():
    valid = em.prepare_target_frame(_frame(), "label_ideology")
    grp = em.compute_group_comparison(valid, "label_ideology", n_boot=300)
    fam = grp[(grp.Dimension == "family") & (grp.Scope == "all")].set_index("Level")
    assert fam.loc["Mistral", "Mean_signed_error"] == pytest.approx(0.8, abs=0.15)
    assert fam.loc["Qwen", "Mean_signed_error"] == pytest.approx(0.0, abs=0.15)
    assert fam.loc["Mistral", "Mean_signed_error_lo"] > fam.loc["Qwen", "Mean_signed_error_hi"]


def test_mixed_model_recovers_family_effect():
    pytest.importorskip("statsmodels")
    valid = em.prepare_target_frame(_frame(), "label_ideology")
    coef, summary = em.compute_factor_analysis(valid, "label_ideology")
    assert summary["Converged"].all()
    row = coef[
        (coef.Outcome == "signed_error")
        & (coef.Spec == "M1_family_size")
        & coef.Term.str.contains(r"\[T\.Qwen\]$")
        & ~coef.Term.str.contains("is_rag")
    ]
    # Llama is absent, so the reference level is Mistral; Qwen is 0.8 below it
    assert len(row) == 1
    assert abs(row["Estimate"].iloc[0]) == pytest.approx(0.8, abs=0.2)


def test_holm_monotone_and_bounded():
    adj = em._holm([0.01, 0.04, 0.03, np.nan])
    assert adj[0] == pytest.approx(0.03)
    assert np.isnan(adj[3])
    assert all(a <= 1 for a in adj if not np.isnan(a))


def _k_frame(rng, ks=(0, 1, 5), n=60, drop_last_for=None):
    true = rng.uniform(1, 7, n)
    rows = []
    for k in ks:
        noise = 1.5 / (1 + k)  # larger k -> less noise
        m = n - 20 if k == drop_last_for else n
        for i in range(m):
            rows.append({"model": "m", "text_index": str(i), "k_chunks": k, "true": true[i],
                         "pred": true[i] + rng.normal(0, noise)})
    return pd.DataFrame(rows)


def test_k_ablation_pairwise_direction_and_common_subset(rng):
    df = _k_frame(rng, drop_last_for=5)
    pair = em.compute_k_pairwise(df, n_boot=200)
    assert len(pair) == 3
    row = pair[(pair["K_low"] == 0) & (pair["K_high"] == 5)].iloc[0]
    assert row["N_paired"] == 40 and row["ΔMAE"] < 0  # higher k is better

    df["family"] = "x"
    df["condition"] = df["k_chunks"].map(em._k_label)
    metrics = em.compute_k_ablation_metrics(df, n_boot=0)
    common = metrics[metrics["Subset"] == "common_texts"]
    assert set(common["N_samples"]) == {40} and list(common["K"]) == [0, 1, 5]


def test_norm_party_messy_spellings():
    assert em._norm_party("BÜNDNIS\xa090/DIE\n                    GRÜNEN") == "BÜNDNIS 90/DIE GRÜNEN"
    assert em._norm_party("Die Linke") == "DIE LINKE"
    assert em._norm_party("CSU") == "CDU/CSU"
    assert em._norm_party("fraktionslos") is None and em._norm_party(None) is None


POS = {"DIE LINKE": 1.8, "SPD": 3.5, "FDP": 5.4, "AfD": 8.5}


def _rq_frame(chunks_per_text):
    return pd.DataFrame(
        [{"model": "m", "condition": "e5/simple", "is_rag": True, "text_index": str(i),
          "party": party, "chunk_parties": chunks}
         for i, (party, chunks) in enumerate(chunks_per_text)]
    )


def test_retrieval_quality_perfect():
    df = _rq_frame([("DIE LINKE", ["Die Linke"] * 3), ("FDP", ["FDP"] * 3), ("AfD", ["AfD"] * 3)])
    row = em.compute_retrieval_quality(df, n_boot=0, positions=POS).iloc[0]
    assert row["Macro_F1"] == 1 and row["Bloc_precision_at_k"] == 1 and row["Mean_party_distance"] == 0


def test_retrieval_quality_known_mix():
    df = _rq_frame([("DIE LINKE", ["SPD", "AfD", "fraktionslos"])])
    row = em.compute_retrieval_quality(df, n_boot=0, positions=POS).iloc[0]
    assert row["Bloc_precision_at_k"] == pytest.approx(0.5)
    assert row["Mean_party_distance"] == pytest.approx(((3.5 - 1.8) + (8.5 - 1.8)) / 2)
    assert row["N_chunks_unmapped"] == 1
