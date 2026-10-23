"""Alternative Annotator Test (Alt-Test) for the Jev judge (offline, no API calls).

Calderon, Reichart & Dror (2025): can the LLM replace one of the human annotators?
For every annotator h_j (leave-one-out) and every chunk x_i:

* S(f, x_i, H_-j) and S(h_j, x_i, H_-j): how well the judge f and the excluded
  annotator h_j align with the remaining annotators H_-j. Negative RMSE for the
  Likert dimensions (R_top, R_ideo), accuracy for the binary A_caus.
* W^f_i = 1[S_f >= S_h], W^h_i = 1[S_h >= S_f] (ties count for both);
  rho^f_j = mean W^f, rho^h_j = mean W^h (advantage probabilities).
* One-sided paired t-test on d_i = W^f_i - W^h_i,
  H0: rho^f_j <= rho^h_j - eps vs H1: rho^f_j > rho^h_j - eps
  (Wilcoxon signed-rank instead when fewer than 30 units are tested).
* Benjamini-Yekutieli over the annotators (dependent tests), q = 0.05;
  winning rate omega = share of rejected H0. The judge passes if omega >= 0.5.

Human labels are the revised per-annotator ratings (after the discussion
round, same table as ``human_agreement_report_revised.json``); judge labels are
``derived_labels`` of the run. Besides the main test on all 125 chunks it
reports an eps sweep, exact-match scoring for the Likert dimensions, the
calibration holdout texts only (the judge was tuned on these chunks), a
text-level variant (chunks of one text are not independent), and the original
pre-discussion ratings (the discussion pulls annotators towards each other).

    python "RAG Analysis/alt_test/alt_test.py"
    python "RAG Analysis/alt_test/alt_test.py" --run 2026-10-22_132304 --epsilon 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = Path(__file__).resolve().parent
RAG_DIR = SCRIPT_DIR.parent
HUMAN_DIR = RAG_DIR / "human_annotations"
sys.path.insert(0, str(HUMAN_DIR))

from compare_human_annotations import (  # noqa: E402
    DEFAULT_PATTERN,
    DEFAULT_REVISED_CSV,
    DIMENSIONS,
    build_table,
    discover_annotation_files,
)

DEFAULT_RUN = "2026-10-22_132304"
DEFAULT_EPSILON = 0.2  # paper's recommendation for expert annotators (0.1 for crowd-workers)
EPSILON_SWEEP = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
Q = 0.05
MIN_T_TEST_N = 30
# Primary alignment score per dimension (paper: -RMSE for continuous, ACC for discrete labels).
PRIMARY_SCORE = {"R_top": "neg_rmse", "R_ideo": "neg_rmse", "A_caus": "acc"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_judge(run: str) -> dict:
    path = RAG_DIR / run / "llm_judge_annotations.jsonl"
    judge = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                judge[(str(row["text_index"]), int(row["chunk_index"]))] = row
    return judge


def load_data(run: str, revised: bool = True):
    """Human matrix (n_items x n_annotators) and judge vector per dimension."""
    files = discover_annotation_files(HUMAN_DIR, DEFAULT_PATTERN)
    revised_csv = DEFAULT_REVISED_CSV if revised else None
    table, _ = build_table(files, list(DIMENSIONS), "shared", revised_csv)
    judge = load_judge(run)
    missing = set(table.keys) - set(judge)
    if missing:
        raise SystemExit(f"Judge run {run} lacks {len(missing)} chunks, e.g. {sorted(missing)[:3]}")
    humans, llm = {}, {}
    for dim in DIMENSIONS:
        humans[dim] = table.matrices[dim]
        if np.isnan(humans[dim]).any():
            raise SystemExit(f"{dim}: missing human ratings; the Alt-Test here needs complete rows")
        llm[dim] = np.array([judge[k]["derived_labels"][dim] for k in table.keys], dtype=float)
    return table.annotators, table.keys, humans, llm, judge


def holdout_texts(keys) -> set:
    """Calibration holdout: every third text by numeric id (as in judge_calibration/calibrate.py)."""
    ids = sorted({t for t, _ in keys}, key=int)
    return set(ids[2::3])


# ---------------------------------------------------------------------------
# Alt-Test
# ---------------------------------------------------------------------------


def alignment(pred: np.ndarray, others: np.ndarray, score: str) -> np.ndarray:
    """Per-item alignment of ``pred`` (n,) with the remaining annotators ``others`` (n, k)."""
    diff = others - pred[:, None]
    if score == "neg_rmse":
        return -np.sqrt(np.mean(diff**2, axis=1))
    if score == "acc":
        return np.mean(diff == 0, axis=1)
    raise ValueError(score)


def one_sided_p(d: np.ndarray, epsilon: float) -> tuple[float, str, float | None]:
    """p-value for H1: mean(d) > -eps; t-test for n >= 30, else Wilcoxon."""
    shifted = d + epsilon
    if len(d) < MIN_T_TEST_N:
        if np.allclose(shifted, 0):
            return 1.0, "wilcoxon", None
        res = stats.wilcoxon(shifted, alternative="greater", zero_method="wilcox")
        return float(res.pvalue), "wilcoxon", float(res.statistic)
    if np.std(d, ddof=1) == 0:  # degenerate: no variance, decide by the mean alone
        return (0.0 if shifted.mean() > 0 else 1.0), "t-test (zero variance)", None
    res = stats.ttest_1samp(d, popmean=-epsilon, alternative="greater")
    return float(res.pvalue), "t-test", float(res.statistic)


def alt_test(
    humans: np.ndarray,
    llm: np.ndarray,
    annotators: list[str],
    score: str,
    epsilon: float,
    groups: np.ndarray | None = None,
) -> dict:
    """Run the Alt-Test for one dimension.

    With ``groups`` the indicators d_i are averaged within each group (text) first,
    so the test unit is the text instead of the chunk.
    """
    per_annotator, pvals = {}, []
    for j, name in enumerate(annotators):
        others = np.delete(humans, j, axis=1)
        s_f = alignment(llm, others, score)
        s_h = alignment(humans[:, j], others, score)
        w_f = (s_f >= s_h).astype(float)
        w_h = (s_h >= s_f).astype(float)
        d = w_f - w_h
        if groups is not None:
            d = np.array([d[groups == g].mean() for g in np.unique(groups)])
        p, test, statistic = one_sided_p(d, epsilon)
        pvals.append(p)
        per_annotator[name] = {
            "n_units": int(len(d)),
            "rho_f": float(w_f.mean()),
            "rho_h": float(w_h.mean()),
            "llm_wins": int(((w_f == 1) & (w_h == 0)).sum()),
            "human_wins": int(((w_h == 1) & (w_f == 0)).sum()),
            "ties": int(((w_f == 1) & (w_h == 1)).sum()),
            "mean_d": float(d.mean()),
            "test": test,
            "statistic": statistic,
            "p": p,
        }
    reject, p_by, _, _ = multipletests(pvals, alpha=Q, method="fdr_by")
    for name, rej, padj in zip(annotators, reject, p_by):
        per_annotator[name]["p_by"] = float(padj)
        per_annotator[name]["rejected"] = bool(rej)
    omega = float(np.mean(reject))
    return {
        "score": score,
        "epsilon": epsilon,
        "q": Q,
        "winning_rate": omega,
        "passed": omega >= 0.5,
        "avg_advantage_probability": float(np.mean([a["rho_f"] for a in per_annotator.values()])),
        "per_annotator": per_annotator,
    }


def min_passing_epsilon(humans, llm, annotators, score, groups=None, step=0.01, upper=1.0):
    """Smallest eps (grid ``step``) at which omega >= 0.5; None if none up to ``upper``."""
    for eps in np.arange(0.0, upper + 1e-9, step):
        if alt_test(humans, llm, annotators, score, round(float(eps), 4), groups)["passed"]:
            return round(float(eps), 4)
    return None


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_result(title: str, res: dict) -> None:
    verdict = "PASS" if res["passed"] else "fail"
    print(
        f"  {title}: omega = {res['winning_rate']:.2f} ({verdict}), "
        f"avg rho_f = {res['avg_advantage_probability']:.3f}"
    )
    for name, a in res["per_annotator"].items():
        mark = "rejected" if a["rejected"] else "kept"
        print(
            f"    {name:<7} n={a['n_units']:<4} rho_f={a['rho_f']:.3f} rho_h={a['rho_h']:.3f} "
            f"(LLM {a['llm_wins']} / human {a['human_wins']} / tie {a['ties']})  "
            f"p={a['p']:.4f} p_BY={a['p_by']:.4f}  H0 {mark}"
        )


def write_item_csv(path: Path, annotators, keys, humans, llm) -> None:
    """Per chunk and annotator: labels, both alignment scores and the indicators."""
    fields = ["text_index", "chunk_index", "dimension", "score", "left_out", "left_out_label",
              "others_labels", "llm_label", "S_llm", "S_human", "W_f", "W_h"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dim in DIMENSIONS:
            score = PRIMARY_SCORE[dim]
            for j, name in enumerate(annotators):
                others = np.delete(humans[dim], j, axis=1)
                s_f = alignment(llm[dim], others, score)
                s_h = alignment(humans[dim][:, j], others, score)
                for i, (t, c) in enumerate(keys):
                    writer.writerow({
                        "text_index": t, "chunk_index": c, "dimension": dim, "score": score,
                        "left_out": name, "left_out_label": int(humans[dim][i, j]),
                        "others_labels": "|".join(str(int(v)) for v in others[i]),
                        "llm_label": int(llm[dim][i]),
                        "S_llm": round(float(s_f[i]), 4), "S_human": round(float(s_h[i]), 4),
                        "W_f": int(s_f[i] >= s_h[i]), "W_h": int(s_h[i] >= s_f[i]),
                    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=DEFAULT_RUN, help="judge run folder in 'RAG Analysis/'")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--output-dir", default=None, help="default: alt_test/<run>/")
    args = parser.parse_args()

    annotators, keys, humans, llm, judge = load_data(args.run)
    _, orig_keys, orig_humans, _, _ = load_data(args.run, revised=False)
    if orig_keys != keys:
        raise SystemExit("Original and revised tables cover different chunks")
    texts = np.array([t for t, _ in keys])
    hold = holdout_texts(keys)
    hold_mask = np.array([t in hold for t in texts])
    model = next(iter(judge.values())).get("model")
    print(f"Alt-Test: judge {model} (run {args.run}) vs {', '.join(annotators)}; "
          f"{len(keys)} chunks / {len(set(texts))} texts; eps = {args.epsilon}, q = {Q} (BY)\n")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run": args.run,
        "model": model,
        "annotators": annotators,
        "human_labels": "revised per-annotator ratings (" + DEFAULT_REVISED_CSV.name + ")",
        "n_items": len(keys),
        "n_texts": int(len(set(texts))),
        "holdout_texts": sorted(hold, key=int),
        "epsilon": args.epsilon,
        "q": Q,
        "per_dimension": {},
    }

    for dim in DIMENSIONS:
        score = PRIMARY_SCORE[dim]
        H, f = humans[dim], llm[dim]
        print(f"== {dim} (score: {score})")
        main_res = alt_test(H, f, annotators, score, args.epsilon)
        print_result("main, all chunks", main_res)
        sweep = {str(e): alt_test(H, f, annotators, score, e)["winning_rate"] for e in EPSILON_SWEEP}
        print("  eps sweep (omega): " + ", ".join(f"{e}: {w:.2f}" for e, w in sweep.items()))
        min_eps = min_passing_epsilon(H, f, annotators, score)
        print(f"  smallest passing eps: {min_eps}")

        robustness = {}
        holdout = alt_test(H[hold_mask], f[hold_mask], annotators, score, args.epsilon)
        print_result(f"calibration holdout ({hold_mask.sum()} chunks)", holdout)
        robustness["holdout"] = holdout
        by_text = alt_test(H, f, annotators, score, args.epsilon, groups=texts)
        print_result("text-level units (Wilcoxon, n=25)", by_text)
        robustness["text_level"] = by_text
        if score != "acc":
            exact = alt_test(H, f, annotators, "acc", args.epsilon)
            print_result("exact-match (ACC) scoring", exact)
            robustness["acc_scoring"] = exact
        original = alt_test(orig_humans[dim], f, annotators, score, args.epsilon)
        print_result("original (pre-discussion) ratings", original)
        original["min_passing_epsilon"] = min_passing_epsilon(orig_humans[dim], f, annotators, score)
        print(f"    smallest passing eps (original ratings): {original['min_passing_epsilon']}")
        robustness["original_ratings"] = original
        print()

        report["per_dimension"][dim] = {
            "main": main_res,
            "epsilon_sweep_winning_rate": sweep,
            "min_passing_epsilon": min_eps,
            "robustness": robustness,
        }

    summary = {d: {"winning_rate": r["main"]["winning_rate"], "passed": r["main"]["passed"],
                   "avg_advantage_probability": r["main"]["avg_advantage_probability"],
                   "min_passing_epsilon": r["min_passing_epsilon"]}
               for d, r in report["per_dimension"].items()}
    report["summary"] = summary
    print("Summary (main test):")
    for d, s in summary.items():
        print(f"  {d:<7} omega = {s['winning_rate']:.2f}  {'PASS' if s['passed'] else 'fail'}  "
              f"avg rho_f = {s['avg_advantage_probability']:.3f}  min eps = {s['min_passing_epsilon']}")

    out_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "alt_test_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    write_item_csv(out_dir / "alt_test_items.csv", annotators, keys, humans, llm)
    print(f"\nSaved {out_dir / 'alt_test_report.json'} and alt_test_items.csv")


if __name__ == "__main__":
    main()
