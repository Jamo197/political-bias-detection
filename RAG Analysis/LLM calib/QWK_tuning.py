"""QWK threshold tuning for the LLM judge (offline, no API calls).

The judge answers a ladder of yes/no gates with probabilities. Summing them gives a
continuous expected level (R_top: 1 + 4 gates, R_ideo: 2 + 3 gates), which is cut into
levels 1-5. The default cut is rounding (thresholds 1.5, 2.5, ...). Following the
"threshold tuning" trick for QWK, the four cut points are moved one at a time in small
steps to maximise quadratic weighted kappa against the consensus.

Because only 25 texts exist, tuning and scoring on the same chunks overstates the gain.
Besides the in-sample QWK this script reports leave-one-text-out QWK: thresholds are
tuned on all other texts and applied to the held-out text, then QWK is computed over the
pooled predictions. Only that number is a fair estimate.

    python "RAG Analysis/QWK_tuning.py" final_nometa
    python "RAG Analysis/QWK_tuning.py" final_nometa final --steps 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "judge_calibration"))

from calibrate import R_IDEO_GATES, R_TOP_GATES, load_run  # noqa: E402

LABELS = [1, 2, 3, 4, 5]
GATES = {"R_top": (R_TOP_GATES, 1), "R_ideo": (R_IDEO_GATES, 2)}
DEFAULT_THRESHOLDS = [1.5, 2.5, 3.5, 4.5]


def qwk(true: np.ndarray, pred: np.ndarray) -> float:
    return float(cohen_kappa_score(true, pred, labels=LABELS, weights="quadratic"))


def cut(score: np.ndarray, thresholds: list[float]) -> np.ndarray:
    """Continuous score -> level 1..5 (a level is the number of thresholds passed + 1)."""
    return np.searchsorted(np.sort(thresholds), score, side="right") + 1


def expected_level(rows: list[dict], dim: str) -> np.ndarray:
    gates, base = GATES[dim]
    gates = [g for g in gates if g in rows[0]["raw_answers"]]
    return base + np.array([[r["raw_answers"][g]["noul"] for g in gates] for r in rows]).sum(axis=1)


def find_thresholds(true, score, steps: int = 200, delta: float = 0.01):
    """Coordinate search over the four thresholds, starting at the rounding cut.

    Each threshold is moved in both directions in steps of ``delta`` until ``steps``
    consecutive moves fail to improve QWK. Sweeps repeat until nothing improves.
    """
    threshold = list(DEFAULT_THRESHOLDS)
    best = qwk(true, cut(score, threshold))
    improved = True
    while improved:
        improved = False
        for k in range(len(threshold)):
            for sign in (1, -1):
                v, stop, trial = threshold[k], 0, list(threshold)
                while stop < steps:
                    v += sign * delta
                    trial[k] = v
                    metric = qwk(true, cut(score, trial))
                    if metric > best + 1e-12:
                        best, threshold, stop, improved = metric, list(trial), 0, True
                    else:
                        stop += 1
    return best, [round(t, 3) for t in sorted(threshold)]


def leave_one_text_out(rows, true, score, **kw) -> np.ndarray:
    texts = np.array([r["text_index"] for r in rows])
    pred = np.zeros(len(rows), dtype=int)
    for t in np.unique(texts):
        te = texts == t
        _, thr = find_thresholds(true[~te], score[~te], **kw)
        pred[te] = cut(score[te], thr)
    return pred


def run_report(path: str, steps: int, delta: float) -> None:
    rows = load_run(path)
    print(f"\n=== {path}: {len(rows)} chunks, {len({r['text_index'] for r in rows})} texts ===")
    for dim in GATES:
        true = np.array([r["human"][dim] for r in rows])
        stored = np.array([r["derived_labels"][dim] for r in rows])
        score = expected_level(rows, dim)

        rounded = cut(score, DEFAULT_THRESHOLDS)
        best, thr = find_thresholds(true, score, steps, delta)
        cv_pred = leave_one_text_out(rows, true, score, steps=steps, delta=delta)

        print(f"\n{dim}")
        print(f"  stored derived label              QWK {qwk(true, stored):.3f}")
        print(f"  expected level, rounded           QWK {qwk(true, rounded):.3f}")
        print(f"  tuned thresholds {thr}  in-sample QWK {best:.3f}")
        print(f"  tuned, leave-one-text-out         QWK {qwk(true, cv_pred):.3f}   <- fair estimate")
        print(f"  exact agreement: stored {np.mean(stored == true):.3f}, "
              f"tuned (LOTO) {np.mean(cv_pred == true):.3f}")
        print(f"  mean bias (judge - human): stored {np.mean(stored - true):+.3f}, "
              f"tuned (LOTO) {np.mean(cv_pred - true):+.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run folder names under RAG Analysis, or LLM calib/<name>")
    ap.add_argument("--steps", type=int, default=200, help="early-stopping patience per threshold")
    ap.add_argument("--delta", type=float, default=0.01, help="threshold step size")
    args = ap.parse_args()
    for run in args.runs:
        path = HERE / run
        if not (path / "llm_judge_annotations.jsonl").exists():
            path = HERE / "LLM calib" / run
        run_report(str(path / "llm_judge_annotations.jsonl"), args.steps, args.delta)


if __name__ == "__main__":
    main()
