"""Calibration report for a Jev judge run (offline, no API calls).

Reads ``RAG Analysis/<run_id>/llm_judge_annotations.jsonl`` and compares the
judge with the adjudicated consensus (``consensus_reference_annotations.jsonl``),
which also carries each annotator's own ratings. Reported per dimension and for
all / dev / holdout texts (the split is by text, never by chunk, because chunks
of one text share an input and a justification; every third text is held out):

* ordinal Krippendorff's alpha (nominal for A_caus) judge vs consensus, with a
  95 % bootstrap CI that resamples texts;
* alpha with the judge as a 4th rater next to the three annotators, against the
  human-only alpha, and judge vs each annotator;
* QWK / kappa, exact, within-1, MAE, bias, per-class recall, balanced accuracy,
  label distributions (class imbalance);
* strata by human uncertainty (annotators unanimous vs spread) and by quadrant;
* the judge's expected level (from the gate probabilities) against the mean of
  the three annotator ratings (distribution alignment);
* per-gate AUC and a risk-coverage table (agreement when only the most
  confident share of chunks is kept and the rest is escalated to a human).

    python "RAG Analysis/judge_calibration/calibrate.py" 2026-10-20_154122
    python "RAG Analysis/judge_calibration/calibrate.py" runA runB --subset holdout
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "RAG Analysis"
CONSENSUS = RUNS / "human_annotations/revised/consensus_reference_annotations.jsonl"
sys.path.insert(0, str(RUNS / "human_annotations"))

from compare_human_annotations import krippendorff_alpha  # noqa: E402

ANNOTATORS = ("Awsam", "Jannes", "Neda")
DIMENSIONS = ("R_top", "R_ideo", "A_caus")
LEVELS = {"R_top": [1, 2, 3, 4, 5], "R_ideo": [1, 2, 3, 4, 5], "A_caus": [0, 1]}
KIND = {"R_top": "ordinal", "R_ideo": "ordinal", "A_caus": "nominal"}
# Gates behind each ordinal dimension (their order is the ladder order).
R_TOP_GATES = ("T_domain", "T_subarea", "T_item", "T_central")
R_IDEO_GATES = ("I_evaluative", "I_stance", "I_party")
# Gate -> (dimension, minimum human level that counts as "yes") for the AUC table.
GATE_TARGETS = {
    "T_domain": ("R_top", 2),
    "T_subarea": ("R_top", 3),
    "T_item": ("R_top", 4),
    "T_central": ("R_top", 5),
    "I_evaluative": ("R_ideo", 3),
    "I_stance": ("R_ideo", 4),
    "I_party": ("R_ideo", 5),
}
KAPPA_TARGET = 0.70
N_BOOT = 1000
COVERAGES = (1.0, 0.75, 0.5, 0.25)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_consensus() -> dict:
    """(text_index, chunk_index) -> {quadrant, human_ratings, unanimous}."""
    out = {}
    for line in CONSENSUS.open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        quadrant = (record.get("source_context") or {}).get("quadrant", "")
        for chunk in record["chunk_annotations"]:
            out[(record["text_index"], chunk["chunk_index"])] = {
                "quadrant": quadrant.split("_")[0],
                "human_ratings": chunk["human_ratings"],
                "unanimous": chunk.get("unanimous"),
            }
    return out


def split_texts() -> tuple[set, set]:
    ids = sorted({t for t, _ in load_consensus()}, key=int)
    holdout = set(ids[2::3])
    return set(ids) - holdout, holdout


def load_run(run: str) -> list[dict]:
    path = Path(run)
    if not path.suffix:
        path = RUNS / run / "llm_judge_annotations.jsonl"
    consensus = load_consensus()
    rows = []
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if "error" in row or "derived_labels" not in row:
            continue
        row.update(consensus[(row["text_index"], row["chunk_index"])])
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _alpha(matrix: np.ndarray, dim: str) -> float | None:
    return krippendorff_alpha(matrix, KIND[dim], LEVELS[dim])


def _agreement(cons: np.ndarray, judge: np.ndarray, dim: str) -> float | None:
    """Judge-vs-consensus agreement used for the headline number (alpha)."""
    if len(cons) < 2:
        return None
    return _alpha(np.column_stack([cons, judge]).astype(float), dim)


def _vectors(rows: list[dict], dim: str):
    cons = np.array([r["human"][dim] for r in rows], dtype=int)
    judge = np.array([r["derived_labels"][dim] for r in rows], dtype=int)
    humans = np.array(
        [[r["human_ratings"][dim][a] for a in ANNOTATORS] for r in rows], dtype=float
    )
    return cons, judge, humans


def bootstrap_ci(rows: list[dict], dim: str, seed: int = 0) -> tuple[float, float] | None:
    """95 % CI of alpha(judge, consensus), resampling whole texts."""
    by_text: dict[str, list[dict]] = {}
    for r in rows:
        by_text.setdefault(r["text_index"], []).append(r)
    texts = list(by_text)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(N_BOOT):
        sample = [r for t in rng.choice(texts, len(texts)) for r in by_text[t]]
        cons, judge, _ = _vectors(sample, dim)
        a = _agreement(cons, judge, dim)
        if a is not None:
            values.append(a)
    if len(values) < N_BOOT // 2:
        return None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def dimension_metrics(rows: list[dict], dim: str, with_ci: bool = True) -> dict | None:
    if len(rows) < 2:
        return None
    levels = LEVELS[dim]
    cons, judge, humans = _vectors(rows, dim)
    ordinal = KIND[dim] == "ordinal"
    qwk = cohen_kappa_score(
        cons, judge, labels=levels, weights="quadratic" if ordinal else None
    )
    recalls = {
        int(level): float(np.mean(judge[cons == level] == level))
        for level in levels
        if np.any(cons == level)
    }
    human_alpha = _alpha(humans, dim)
    with_judge = _alpha(np.column_stack([humans, judge]), dim)
    out = {
        "n": len(rows),
        "alpha_vs_consensus": _agreement(cons, judge, dim),
        "alpha_ci95": bootstrap_ci(rows, dim) if with_ci else None,
        "kappa_type": "qwk" if ordinal else "cohen",
        "kappa": float(qwk),
        "exact": float(np.mean(cons == judge)),
        "within_1": float(np.mean(np.abs(cons - judge) <= 1)),
        "mae": float(np.mean(np.abs(cons - judge))),
        "bias": float(np.mean(judge - cons)),
        "balanced_accuracy": float(np.mean(list(recalls.values()))),
        "recall_by_class": recalls,
        "distribution_consensus": {int(v): int(np.sum(cons == v)) for v in levels},
        "distribution_judge": {int(v): int(np.sum(judge == v)) for v in levels},
        "alpha_human_only": human_alpha,
        "alpha_with_judge_as_4th_rater": with_judge,
        "alpha_judge_vs_annotator": {
            a: _alpha(np.column_stack([humans[:, k], judge]), dim)
            for k, a in enumerate(ANNOTATORS)
        },
    }
    return out


def stratified_metrics(rows: list[dict], dim: str) -> dict:
    """Agreement by human uncertainty (annotator spread) and by quadrant."""
    cons, judge, humans = _vectors(rows, dim)
    spread = humans.max(axis=1) - humans.min(axis=1)
    out = {}

    def cell(mask: np.ndarray) -> dict | None:
        if mask.sum() < 2:
            return None
        return {
            "n": int(mask.sum()),
            "alpha": _agreement(cons[mask], judge[mask], dim),
            "exact": float(np.mean(cons[mask] == judge[mask])),
        }

    out["unanimous"] = cell(spread == 0)
    out["contested"] = cell(spread > 0)
    quadrants = np.array([r["quadrant"] for r in rows])
    for q in sorted(set(quadrants)):
        out[q] = cell(quadrants == q)
    return out


def expected_level_alignment(rows: list[dict], dim: str) -> dict | None:
    """Judge expected level vs the mean of the three annotator ratings."""
    if dim == "A_caus":
        # A_caus: the model probability vs the share of annotators saying 1.
        judge = np.array([r["raw_answers"]["A_caus"]["noul"] for r in rows])
    else:
        gates = R_TOP_GATES if dim == "R_top" else R_IDEO_GATES
        gates = [g for g in gates if g in rows[0]["raw_answers"]]
        base = 1 if dim == "R_top" else 2
        judge = base + np.sum(
            [[r["raw_answers"][g]["noul"] for g in gates] for r in rows], axis=1
        )
    _, _, humans = _vectors(rows, dim)
    mean_human = humans.mean(axis=1)
    from scipy.stats import spearmanr

    rho = spearmanr(judge, mean_human).correlation
    return {
        "spearman_vs_annotator_mean": float(rho),
        "mean_abs_gap": float(np.mean(np.abs(judge - mean_human))),
    }


def gate_auc(rows: list[dict]) -> dict:
    out = {}
    for gate, (dim, minimum) in GATE_TARGETS.items():
        if gate not in rows[0]["raw_answers"]:
            continue
        y = np.array([r["human"][dim] >= minimum for r in rows])
        p = np.array([r["raw_answers"][gate]["noul"] for r in rows])
        out[gate] = {
            "positives": int(y.sum()),
            "predicted_yes": int(np.sum(p >= 0.5)),
            "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        }
    return out


def risk_coverage(rows: list[dict], dim: str) -> list[dict]:
    """Agreement on the most confident share of chunks (rest -> human review)."""
    if dim == "A_caus":
        gates = ["A_caus"]
    else:
        gates = [g for g in (R_TOP_GATES if dim == "R_top" else R_IDEO_GATES)
                 if g in rows[0]["raw_answers"]]
    confidence = np.array(
        [min(abs(r["raw_answers"][g]["noul"] - 0.5) for g in gates) for r in rows]
    )
    cons, judge, _ = _vectors(rows, dim)
    order = np.argsort(-confidence, kind="stable")
    table = []
    for coverage in COVERAGES:
        keep = order[: max(2, int(round(coverage * len(rows))))]
        a = _agreement(cons[keep], judge[keep], dim)
        table.append(
            {
                "coverage": coverage,
                "n": int(len(keep)),
                "alpha": a,
                "exact": float(np.mean(cons[keep] == judge[keep])),
            }
        )
    return table


def review_metrics(rows: list[dict], dim: str) -> dict | None:
    """Agreement on the chunks the judge keeps (not flagged ``needs_review``)."""
    if "needs_review" not in rows[0]:
        return None
    kept = [r for r in rows if not r["needs_review"][dim]]
    if len(kept) < 2:
        return None
    cons, judge, _ = _vectors(kept, dim)
    return {
        "share_flagged": 1 - len(kept) / len(rows),
        "n_kept": len(kept),
        "alpha_kept": _agreement(cons, judge, dim),
        "exact_kept": float(np.mean(cons == judge)),
    }


def evaluate(rows: list[dict]) -> dict:
    report = {}
    for dim in DIMENSIONS:
        report[dim] = {
            "metrics": dimension_metrics(rows, dim),
            "strata": stratified_metrics(rows, dim),
            "expected_level": expected_level_alignment(rows, dim),
            "risk_coverage": risk_coverage(rows, dim),
            "review": review_metrics(rows, dim),
        }
    report["gate_auc"] = gate_auc(rows)
    return report


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _f(value, spec=".3f") -> str:
    return "  n/a" if value is None else format(value, spec)


def print_report(name: str, subset: str, rows: list[dict], report: dict) -> None:
    print(f"\n{'=' * 78}\nrun {name} | subset {subset} | chunks {len(rows)} | "
          f"texts {len({r['text_index'] for r in rows})}\n{'=' * 78}")
    print(f"{'dim':<8}{'kappa':>7}{'alpha':>7}  {'95% CI':<13}{'exact':>6}{'±1':>6}"
          f"{'bias':>7}{'balAcc':>7} | {'a(humans)':>9}{'a(+judge)':>10}")
    for dim in DIMENSIONS:
        m = report[dim]["metrics"]
        if m is None:
            continue
        ci = m["alpha_ci95"]
        ci_s = f"({ci[0]:.2f},{ci[1]:.2f})" if ci else ""
        flag = " <0.70" if m["kappa"] < KAPPA_TARGET else ""
        print(f"{dim:<8}{m['kappa']:>7.3f}{_f(m['alpha_vs_consensus'], '>7.3f')}  "
              f"{ci_s:<13}{m['exact']:>6.2f}{m['within_1']:>6.2f}{m['bias']:>+7.2f}"
              f"{m['balanced_accuracy']:>7.2f} | {_f(m['alpha_human_only'], '>9.3f')}"
              f"{_f(m['alpha_with_judge_as_4th_rater'], '>10.3f')}{flag}")
    for dim in DIMENSIONS:
        m = report[dim]["metrics"]
        if m is None:
            continue
        per = ", ".join(f"{a} {_f(v)}" for a, v in m["alpha_judge_vs_annotator"].items())
        print(f"\n{dim}: judge vs annotator alpha: {per}")
        print(f"  consensus dist {m['distribution_consensus']}  judge dist {m['distribution_judge']}")
        print(f"  recall by class {({k: round(v, 2) for k, v in m['recall_by_class'].items()})}")
        strata = report[dim]["strata"]
        cells = "  ".join(
            f"{k}: n={v['n']} a={_f(v['alpha'], '.2f')} ex={v['exact']:.2f}"
            for k, v in strata.items() if v
        )
        print(f"  strata  {cells}")
        e = report[dim]["expected_level"]
        print(f"  expected level vs annotator mean: spearman {_f(e['spearman_vs_annotator_mean'], '.3f')}, "
              f"mean |gap| {e['mean_abs_gap']:.2f}")
        rc = "  ".join(f"{int(c['coverage']*100)}%: a={_f(c['alpha'], '.2f')} ex={c['exact']:.2f}"
                       for c in report[dim]["risk_coverage"])
        print(f"  risk-coverage  {rc}")
        rv = report[dim]["review"]
        if rv:
            print(f"  needs_review flag: {rv['share_flagged']*100:.0f}% routed to a human; on the kept "
                  f"{rv['n_kept']} chunks alpha={_f(rv['alpha_kept'], '.3f')} exact={rv['exact_kept']:.2f}")
    print("\nGate AUC vs matching human level (yes = predicted prob >= 0.5):")
    for gate, g in report["gate_auc"].items():
        print(f"  {gate:<13} human-pos {g['positives']:>3}  pred-yes {g['predicted_yes']:>3}  AUC {_f(g['auc'])}")


# ---------------------------------------------------------------------------
# Post-hoc calibration candidates (nested leave-one-text-out CV)
# ---------------------------------------------------------------------------
#
# Every candidate maps the stored gate probabilities of one dimension to a
# level. Fitting happens on all texts but the held-out one, prediction on the
# held-out text, so the reported agreement is out-of-sample. With 25 texts the
# uncalibrated rule stays the default unless a candidate beats it clearly.


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 0.02, 0.98)
    return np.log(p / (1 - p))


def _gate_matrix(rows: list[dict], gates: list[str]) -> np.ndarray:
    return np.array([[r["raw_answers"][g]["noul"] for g in gates] for r in rows])


def _ladder_levels(dim: str, rows: list[dict]) -> tuple[list[str], int]:
    """Gates (ladder order) and the level a chunk gets when no gate fires."""
    if dim == "R_top":
        gates = [g for g in R_TOP_GATES if g in rows[0]["raw_answers"]]
        return gates, 1
    gates = [g for g in R_IDEO_GATES if g in rows[0]["raw_answers"]]
    return gates, 2


def _fit_predict_global_shift(Ptr, ytr, Pte, base, step):
    """One shared threshold for all gates, chosen on the training texts."""
    best, best_t = -9.0, 0.5
    for t in np.arange(0.3, 0.86, 0.05):
        pred = base + np.max((Ptr >= t) * step, axis=1)
        s = cohen_kappa_score(ytr, pred, labels=LEVELS_ALL, weights="quadratic")
        if s > best:
            best, best_t = s, t
    return base + np.max((Pte >= best_t) * step, axis=1)


def _fit_predict_platt(Ptr, ytr, Pte, base, step, minimums):
    """Per-gate Platt scaling towards 'human level >= gate level', then 0.5."""
    from sklearn.linear_model import LogisticRegression

    fired = np.zeros((len(Pte), Ptr.shape[1]), dtype=bool)
    for k, minimum in enumerate(minimums):
        target = ytr >= minimum
        if target.all() or (~target).all():
            fired[:, k] = Pte[:, k] >= 0.5
            continue
        model = LogisticRegression(C=1.0).fit(_logit(Ptr[:, [k]]), target)
        fired[:, k] = model.predict_proba(_logit(Pte[:, [k]]))[:, 1] >= 0.5
    return base + np.max(fired * step, axis=1)


def fit_ordinal(P: np.ndarray, y: np.ndarray, minimums: list[int], C: float = 0.5) -> list[dict]:
    """Frank-Hall ordinal logistic: one binary model per 'human level >= k'.

    Features are the gate logits. A threshold whose target is constant in the
    training data is stored as a constant instead of a model.
    """
    from sklearn.linear_model import LogisticRegression

    specs = []
    for minimum in minimums:
        target = y >= minimum
        if target.all() or not target.any():
            specs.append({"level": minimum, "constant": bool(target.all())})
            continue
        model = LogisticRegression(C=C, max_iter=500).fit(_logit(P), target)
        specs.append(
            {
                "level": minimum,
                "coef": [float(c) for c in model.coef_[0]],
                "intercept": float(model.intercept_[0]),
            }
        )
    return specs


def predict_ordinal(P: np.ndarray, specs: list[dict], base: int) -> np.ndarray:
    """Level = base + number of thresholds the model says the chunk reaches."""
    X = _logit(P)
    level = np.full(len(P), base)
    for spec in specs:
        if "constant" in spec:
            level = level + int(spec["constant"])
        else:
            z = X @ np.array(spec["coef"]) + spec["intercept"]
            level = level + (z >= 0)
    return level


def _fit_predict_ordinal(Ptr, ytr, Pte, minimums):
    return predict_ordinal(Pte, fit_ordinal(Ptr, ytr, minimums), minimums[0] - 1)


LEVELS_ALL = [1, 2, 3, 4, 5]


def posthoc_cv(rows: list[dict], dim: str) -> dict:
    gates, base = _ladder_levels(dim, rows)
    P = _gate_matrix(rows, gates)
    y = np.array([r["human"][dim] for r in rows])
    texts = np.array([r["text_index"] for r in rows])
    # gate k (1-based, ladder order) lifts the level to base + k, which the
    # annotators express as 'human level >= base + k'
    step = np.arange(1, len(gates) + 1)[None, :]
    minimums = list(range(base + 1, base + 1 + len(gates)))
    candidates = {
        "global_threshold": lambda tr, te: _fit_predict_global_shift(
            P[tr], y[tr], P[te], base, step
        ),
        "platt_per_gate": lambda tr, te: _fit_predict_platt(
            P[tr], y[tr], P[te], base, step, minimums
        ),
        "ordinal_logistic": lambda tr, te: _fit_predict_ordinal(
            P[tr], y[tr], P[te], minimums
        ),
    }
    stored = np.array([r["derived_labels"][dim] for r in rows])
    out = {
        "baseline": _score(y, stored, dim),
    }
    for name, fn in candidates.items():
        pred = np.zeros(len(rows), dtype=int)
        for t in np.unique(texts):
            te = texts == t
            pred[te] = fn(~te, te)
        pred = np.clip(pred, 1, 5)
        out[name] = _score(y, pred, dim)
    return out


def _score(y: np.ndarray, pred: np.ndarray, dim: str) -> dict:
    return {
        "alpha": _agreement(y, pred, dim),
        "qwk": float(cohen_kappa_score(y, pred, labels=LEVELS_ALL, weights="quadratic")),
        "exact": float(np.mean(y == pred)),
        "bias": float(np.mean(pred - y)),
    }


def export_posthoc(run: str, rows: list[dict], path: Path, dims: list[str], cv: dict) -> None:
    """Fit the ordinal mapping on every chunk of the run and write it as JSON.

    The out-of-sample quality of this mapping is the leave-one-text-out figure
    from ``posthoc_cv`` (stored alongside), not its fit on the training data.
    """
    spec = {}
    for dim in dims:
        gates, base = _ladder_levels(dim, rows)
        minimums = list(range(base + 1, base + 1 + len(gates)))
        y = np.array([r["human"][dim] for r in rows])
        spec[dim] = {
            "gates": gates,
            "base": base,
            "thresholds": fit_ordinal(_gate_matrix(rows, gates), y, minimums),
            "fitted_on": {
                "run": run,
                "n_chunks": len(rows),
                "n_texts": len({r["text_index"] for r in rows}),
                "questions_hash": rows[0].get("questions_hash"),
            },
            "leave_one_text_out_cv": {
                "uncalibrated": cv[dim]["baseline"],
                "ordinal_logistic": cv[dim]["ordinal_logistic"],
            },
        }
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote post-hoc model for {dims} -> {path}")


def posthoc_dev_to_holdout(rows: list[dict], dim: str) -> dict:
    """Fit the ordinal mapping on the dev texts only, score the untouched holdout."""
    dev, hold = split_texts()
    gates, base = _ladder_levels(dim, rows)
    minimums = list(range(base + 1, base + 1 + len(gates)))
    tr = [r for r in rows if r["text_index"] in dev]
    te = [r for r in rows if r["text_index"] in hold]
    specs = fit_ordinal(_gate_matrix(tr, gates), np.array([r["human"][dim] for r in tr]), minimums)
    y = np.array([r["human"][dim] for r in te])
    pred = np.clip(predict_ordinal(_gate_matrix(te, gates), specs, base), 1, 5)
    stored = np.array([r["derived_labels"][dim] for r in te])
    return {"uncalibrated": _score(y, stored, dim), "ordinal_logistic": _score(y, pred, dim)}


def print_posthoc(run: str, rows: list[dict]) -> dict:
    print(f"\n{'=' * 78}\nrun {run} | post-hoc calibration, leave-one-text-out CV "
          f"(all {len(rows)} chunks)\n{'=' * 78}")
    result = {}
    for dim in ("R_top", "R_ideo"):
        result[dim] = posthoc_cv(rows, dim)
        print(f"\n{dim}\n  {'rule':<20}{'alpha':>7}{'qwk':>7}{'exact':>7}{'bias':>7}")
        for name, s in result[dim].items():
            print(f"  {name:<20}{_f(s['alpha'], '>7.3f')}{s['qwk']:>7.3f}"
                  f"{s['exact']:>7.2f}{s['bias']:>+7.2f}")
        split = posthoc_dev_to_holdout(rows, dim)
        result[dim]["dev_to_holdout"] = split
        print("  fit on dev texts, scored on the untouched holdout texts:")
        for name, s in split.items():
            print(f"    {name:<18}{_f(s['alpha'], '>7.3f')}{s['qwk']:>7.3f}{s['exact']:>7.2f}{s['bias']:>+7.2f}")
    return result


def print_summary(runs: list[str]) -> None:
    """One line per run x subset: alpha (kappa) / exact / bias for each dimension."""
    dev, hold = split_texts()
    print(f"{'run':<22}{'subset':<8}" + "".join(f"{d:>26}" for d in DIMENSIONS))
    print(f"{'':<30}" + "".join(f"{'alpha (kappa)  exact  bias':>26}" for _ in DIMENSIONS))
    for run in runs:
        rows = load_run(run)
        for subset, texts in (("all", None), ("dev", dev), ("holdout", hold)):
            part = rows if texts is None else [r for r in rows if r["text_index"] in texts]
            cells = []
            for dim in DIMENSIONS:
                m = dimension_metrics(part, dim, with_ci=False)
                cells.append(
                    f"{_f(m['alpha_vs_consensus'], '>7.3f')} ({m['kappa']:.2f}) "
                    f"{m['exact']:>5.2f} {m['bias']:>+6.2f}"
                )
            print(f"{Path(run).stem[:21]:<22}{subset:<8}" + "".join(f"{c:>26}" for c in cells))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="run id(s) under RAG Analysis/ or a path to a flat JSONL")
    parser.add_argument("--subset", choices=("all", "dev", "holdout"), default=None,
                        help="print only one subset (default: all three)")
    parser.add_argument("--summary", action="store_true",
                        help="only print the compact alpha/kappa/exact/bias table per run and subset")
    parser.add_argument("--posthoc", action="store_true",
                        help="also compare post-hoc calibration rules with leave-one-text-out CV")
    parser.add_argument("--export-posthoc", metavar="DIM[,DIM]", default=None,
                        help="fit the ordinal post-hoc mapping on all chunks of the (single) run for "
                             "these dimensions (e.g. R_ideo) and write it to judge_calibration/posthoc_<run>.json")
    parser.add_argument("--json", action="store_true",
                        help="write calibration_report.json next to each run")
    args = parser.parse_args()

    if args.summary:
        print_summary(args.runs)
        return
    dev, hold = split_texts()
    for run in args.runs:
        rows = load_run(run)
        reports = {}
        for subset, texts in (("all", None), ("dev", dev), ("holdout", hold)):
            if args.subset and subset != args.subset:
                continue
            part = rows if texts is None else [r for r in rows if r["text_index"] in texts]
            reports[subset] = evaluate(part)
            print_report(run, subset, part, reports[subset])
        if args.posthoc or args.export_posthoc:
            reports["posthoc_cv"] = print_posthoc(run, rows)
        if args.export_posthoc:
            export_posthoc(run, rows, RUNS / "judge_calibration" / f"posthoc_{Path(run).stem}.json",
                           args.export_posthoc.split(","), reports["posthoc_cv"])
        if args.json:
            out =(RUNS / run if not Path(run).suffix else Path(run).parent) / "calibration_report.json"
            out.write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
