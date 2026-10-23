"""Compare two judge set-ups on the same chunks, with a paired bootstrap on alpha.

Each side is ``RUN[:DERIVATION]``. A derivation turns a run's stored raw answers
into a label for one dimension:

* ``gates``     - the labels stored in the run (ladder rules, the default)
* ``score``     - most probable level of the run's ``<dim>_score`` question
* ``scoreexp``  - expected level of that question, rounded
* ``combo``     - ordinal logistic on gate logits + expected score, evaluated
                  out-of-sample (leave-one-text-out over all chunks of the run)
* ``scoreord``  - ordinal logistic on the expected score only (leave-one-text-out)

A_caus always comes from the stored labels. Chunks are matched by (text, chunk);
``--exclude-texts`` drops whole texts from both sides (for example texts whose
excerpts were used as in-prompt examples). Delta alpha = side B - side A with a
95 % CI from resampling texts (paired), so a difference that does not clear the
noise is not read as an improvement.

    python "RAG Analysis/judge_calibration/compare_arms.py" final exp1_minimal
    python "RAG Analysis/judge_calibration/compare_arms.py" exp3_score:gates exp3_score:combo
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import calibrate as cal  # noqa: E402

N_BOOT = 2000


def _gate_names(dim: str) -> list[str]:
    return list(cal.R_TOP_GATES if dim == "R_top" else cal.R_IDEO_GATES)


def score_expectation(row: dict, dim: str) -> float | None:
    """Expected 1-based level of a ``<dim>_score`` answer, or None."""
    answer = row["raw_answers"].get(f"{dim}_score")
    probs = (answer or {}).get("probabilities")
    if not probs:
        return None
    total = sum(float(v) for v in probs.values()) or 1.0
    return 1 + sum(int(k) * float(v) for k, v in probs.items()) / total


def score_argmax(row: dict, dim: str) -> int | None:
    answer = row["raw_answers"].get(f"{dim}_score")
    probs = (answer or {}).get("probabilities")
    if not probs:
        return None
    return 1 + int(max(probs, key=lambda k: float(probs[k])))


def _loto_ordinal(rows: list[dict], dim: str, use_gates: bool) -> np.ndarray:
    """Out-of-sample ordinal logistic on gate logits and/or the expected score."""
    from sklearn.linear_model import LogisticRegression

    gates = [g for g in _gate_names(dim) if g in rows[0]["raw_answers"] and g != "I_procedural"]
    feats = []
    for r in rows:
        f = []
        if use_gates:
            p = np.clip([r["raw_answers"][g]["noul"] for g in gates], 0.02, 0.98)
            f += list(np.log(p / (1 - p)))
        f.append(score_expectation(r, dim) - 3.0)
        feats.append(f)
    X = np.array(feats)
    y = np.array([r["human"][dim] for r in rows])
    texts = np.array([r["text_index"] for r in rows])
    base = 1
    minimums = [2, 3, 4, 5]
    pred = np.zeros(len(rows), dtype=int)
    for t in np.unique(texts):
        te = texts == t
        level = np.full(te.sum(), base)
        for m in minimums:
            target = y[~te] >= m
            if target.all():
                level = level + 1
            elif target.any():
                model = LogisticRegression(C=0.5, max_iter=500).fit(X[~te], target)
                level = level + (model.predict_proba(X[te])[:, 1] >= 0.5)
        pred[te] = level
    return pred


def labels_for(rows: list[dict], derivation: str) -> dict[str, np.ndarray | None]:
    """Per dimension label vector under one derivation (None if not available)."""
    out: dict[str, np.ndarray | None] = {}
    for dim in cal.DIMENSIONS:
        stored = np.array([r["derived_labels"][dim] for r in rows])
        if derivation == "gates" or dim == "A_caus":
            out[dim] = stored
            continue
        if f"{dim}_score" not in rows[0]["raw_answers"]:
            out[dim] = stored
            continue
        if derivation == "score":
            out[dim] = np.array([score_argmax(r, dim) for r in rows])
        elif derivation == "scoreexp":
            out[dim] = np.clip(np.rint([score_expectation(r, dim) for r in rows]), 1, 5).astype(int)
        elif derivation == "combo":
            out[dim] = _loto_ordinal(rows, dim, use_gates=True)
        elif derivation == "scoreord":
            out[dim] = _loto_ordinal(rows, dim, use_gates=False)
        else:
            raise SystemExit(f"unknown derivation {derivation!r}")
    return out


def load_side(spec: str) -> tuple[dict, str]:
    run, _, derivation = spec.partition(":")
    rows = cal.load_run(run)
    labels = labels_for(rows, derivation or "gates")
    side = {}
    for i, r in enumerate(rows):
        side[(r["text_index"], r["chunk_index"])] = {
            "text": r["text_index"],
            "human": {d: r["human"][d] for d in cal.DIMENSIONS},
            "label": {d: int(labels[d][i]) for d in cal.DIMENSIONS},
        }
    return side, spec


def _alpha(cons: np.ndarray, judge: np.ndarray, dim: str) -> float | None:
    return cal._agreement(cons, judge, dim)


def vectors(side_a, side_b, keys, dim):
    c = np.array([side_a[k]["human"][dim] for k in keys])
    return c, np.array([side_a[k]["label"][dim] for k in keys]), np.array([side_b[k]["label"][dim] for k in keys])


def paired_delta(side_a, side_b, keys, dim, seed=0):
    texts = sorted({side_a[k]["text"] for k in keys})
    by_text = {t: [k for k in keys if side_a[k]["text"] == t] for t in texts}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(N_BOOT):
        sample = [k for t in rng.choice(texts, len(texts)) for k in by_text[t]]
        c, a, b = vectors(side_a, side_b, sample, dim)
        da, db = _alpha(c, a, dim), _alpha(c, b, dim)
        if da is not None and db is not None:
            deltas.append(db - da)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), float(np.mean(np.array(deltas) > 0))


def summarize(a_spec, b_spec, exclude, subsets, dims):
    side_a, _ = load_side(a_spec)
    side_b, _ = load_side(b_spec)
    common = sorted(set(side_a) & set(side_b), key=lambda k: (int(k[0]), k[1]))
    common = [k for k in common if k[0] not in exclude]
    dev, hold = cal.split_texts()
    print(f"A = {a_spec}   B = {b_spec}   matched chunks {len(common)} "
          f"({len({k[0] for k in common})} texts"
          f"{', excluded texts: ' + ','.join(sorted(exclude)) if exclude else ''})")
    result = {}
    for name in subsets:
        keys = common if name == "all" else [k for k in common if k[0] in (dev if name == "dev" else hold)]
        if len(keys) < 5:
            continue
        print(f"\n[{name}: {len(keys)} chunks, {len({k[0] for k in keys})} texts]")
        print(f"  {'dim':<8}{'alpha A':>9}{'alpha B':>9}{'delta':>8}  {'95% CI of delta':<18}{'P(B>A)':>7}"
              f"{'exact A':>9}{'exact B':>9}{'bias A':>8}{'bias B':>8}")
        for dim in dims:
            c, a, b = vectors(side_a, side_b, keys, dim)
            aa, ab = _alpha(c, a, dim), _alpha(c, b, dim)
            lo, hi, p = paired_delta(side_a, side_b, keys, dim)
            print(f"  {dim:<8}{aa:>9.3f}{ab:>9.3f}{ab - aa:>+8.3f}  ({lo:+.3f}, {hi:+.3f})   {p:>6.2f}"
                  f"{np.mean(c == a):>9.2f}{np.mean(c == b):>9.2f}{np.mean(a - c):>+8.2f}{np.mean(b - c):>+8.2f}")
            result.setdefault(name, {})[dim] = {
                "alpha_a": aa, "alpha_b": ab, "delta": ab - aa, "delta_ci95": [lo, hi], "p_b_better": p,
                "exact_a": float(np.mean(c == a)), "exact_b": float(np.mean(c == b)),
                "bias_a": float(np.mean(a - c)), "bias_b": float(np.mean(b - c)),
                "n": len(keys),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a", help="side A: RUN[:DERIVATION]")
    parser.add_argument("b", help="side B: RUN[:DERIVATION]")
    parser.add_argument("--exclude-texts", default="", help="comma-separated text_index values to drop from both sides")
    parser.add_argument("--dims", default="R_top,R_ideo,A_caus")
    parser.add_argument("--subsets", default="all,dev,holdout")
    parser.add_argument("--json", default=None, help="write the numbers to this path")
    args = parser.parse_args()
    exclude = {t for t in args.exclude_texts.split(",") if t}
    out = summarize(args.a, args.b, exclude, args.subsets.split(","), args.dims.split(","))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
