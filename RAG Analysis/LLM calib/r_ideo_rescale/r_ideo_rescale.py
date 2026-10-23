"""R_ideo scale collapse: does agreement improve if the rarely used levels are merged?

The annotators practically never used R_ideo 1 or 2 (revised: 0 / 8 of 375
ratings), so the 5-level scale behaves like a 3-level one. This script maps the
existing ratings onto collapsed scales (no re-annotation) and reports for each:

* human Krippendorff alpha (ordinal; nominal for two levels) with a 95 %
  bootstrap CI that resamples texts, plus mean pairwise kappa (quadratic weights);
* the same on the original (pre-discussion) ratings;
* the judge (``derived_labels`` of the run) against the consensus (median of
  the mapped ratings), and the Alt-Test winning rate at eps = 0.2.

Note: ordinal alpha only uses the rank order and the observed frequencies of
the levels, so merging an empty or nearly empty level barely changes it. Only
merging levels where annotators actually disagree (3/4, 4/5) can move it.

    python "RAG Analysis/r_ideo_rescale/r_ideo_rescale.py"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RAG_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(RAG_DIR / "human_annotations"))
sys.path.insert(0, str(RAG_DIR / "alt_test"))

from alt_test import DEFAULT_EPSILON, DEFAULT_RUN, alt_test, load_data  # noqa: E402
from compare_human_annotations import krippendorff_alpha, safe_kappa  # noqa: E402

DIM = "R_ideo"
N_BOOT = 2000
SEED = 42

# name -> (description, mapping of the original levels 1-5)
SCHEMES: dict[str, tuple[str, dict[int, int]]] = {
    "5-level (as annotated)": ("1 | 2 | 3 | 4 | 5", {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}),
    "4-level: 1+2": ("descriptive/balanced | implicit | stance | party", {1: 1, 2: 1, 3: 2, 4: 3, 5: 4}),
    "3-level: 1+2+3": ("at most implicit framing | clear stance | explicit party", {1: 1, 2: 1, 3: 1, 4: 2, 5: 3}),
    "binary: <=3 vs 4+5": ("no clear stance | clear stance or party", {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}),
    "binary: <=4 vs 5": ("no explicit party grounding | explicit party", {1: 0, 2: 0, 3: 0, 4: 0, 5: 1}),
    "3-level: 3+4 merged": ("descriptive/balanced | framing or stance | party", {1: 1, 2: 1, 3: 2, 4: 2, 5: 3}),
}


def apply_map(values: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.vectorize(lambda v: mapping[int(v)])(values).astype(float)


def levels_of(mapping: dict[int, int]) -> list[int]:
    return sorted(set(mapping.values()))


def human_alpha(matrix: np.ndarray, mapping: dict[int, int]) -> float | None:
    levels = levels_of(mapping)
    return krippendorff_alpha(matrix, "ordinal" if len(levels) > 2 else "nominal", levels)


def mean_pairwise_kappa(matrix: np.ndarray, mapping: dict[int, int]) -> float:
    levels = levels_of(mapping)
    weights = "quadratic" if len(levels) > 2 else None
    kappas = [
        safe_kappa(matrix[:, a].astype(int), matrix[:, b].astype(int), levels, weights)
        for a, b in combinations(range(matrix.shape[1]), 2)
    ]
    return float(np.mean([k for k in kappas if k is not None]))


def boot_ci(matrix: np.ndarray, texts: np.ndarray, fn, rng) -> tuple[float, float]:
    """95 % percentile CI of ``fn(matrix_subset)`` resampling texts with replacement."""
    ids = np.unique(texts)
    rows = {t: np.flatnonzero(texts == t) for t in ids}
    stats = []
    for _ in range(N_BOOT):
        pick = np.concatenate([rows[t] for t in rng.choice(ids, size=len(ids), replace=True)])
        value = fn(matrix[pick])
        if value is not None:
            stats.append(value)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=DEFAULT_RUN)
    args = parser.parse_args()

    annotators, keys, humans, llm, _ = load_data(args.run)
    _, _, orig_humans, _, _ = load_data(args.run, revised=False)
    texts = np.array([t for t, _ in keys])
    H, H_orig, f = humans[DIM], orig_humans[DIM], llm[DIM]

    counts = {
        "revised": {int(v): int((H == v).sum()) for v in range(1, 6)},
        "original": {int(v): int((H_orig == v).sum()) for v in range(1, 6)},
        "judge": {int(v): int((f == v).sum()) for v in range(1, 6)},
    }
    print(f"{DIM} rating counts (all annotators pooled, {len(keys)} chunks):")
    for name, c in counts.items():
        print(f"  {name:<8} " + "  ".join(f"{k}: {v}" for k, v in c.items()))
    print()

    rng = np.random.default_rng(SEED)
    results = {}
    for name, (description, mapping) in SCHEMES.items():
        Hm, Hm_orig, fm = apply_map(H, mapping), apply_map(H_orig, mapping), apply_map(f, mapping)
        alpha_fn = lambda m, mp=mapping: human_alpha(m, mp)  # noqa: E731
        consensus = np.median(Hm, axis=1)  # median of 3 commutes with a monotone mapping
        judge_pair = np.column_stack([consensus, fm])
        alt = alt_test(Hm, fm, annotators, "neg_rmse", DEFAULT_EPSILON)
        res = {
            "description": description,
            "mapping": mapping,
            "levels_used_by_humans": sorted({int(v) for v in Hm.ravel()}),
            "human_alpha": human_alpha(Hm, mapping),
            "human_alpha_ci": boot_ci(Hm, texts, alpha_fn, rng),
            "human_mean_pairwise_kappa": mean_pairwise_kappa(Hm, mapping),
            "human_alpha_original_ratings": human_alpha(Hm_orig, mapping),
            "judge_vs_consensus_alpha": human_alpha(judge_pair, mapping),
            "judge_vs_consensus_alpha_ci": boot_ci(judge_pair, texts, alpha_fn, rng),
            "judge_as_4th_rater_alpha": human_alpha(np.column_stack([Hm, fm]), mapping),
            "judge_exact_agreement": float(np.mean(consensus == fm)),
            "alt_test_winning_rate": alt["winning_rate"],
            "alt_test_rho": {a: (r["rho_f"], r["rho_h"]) for a, r in alt["per_annotator"].items()},
        }
        results[name] = res
        lo, hi = res["human_alpha_ci"]
        jlo, jhi = res["judge_vs_consensus_alpha_ci"]
        print(f"== {name}  [{description}]")
        print(f"  humans:  alpha {res['human_alpha']:.3f} ({lo:.2f}-{hi:.2f}), "
              f"mean pairwise kappa {res['human_mean_pairwise_kappa']:.3f}, "
              f"original ratings alpha {res['human_alpha_original_ratings']:.3f}")
        print(f"  judge:   vs consensus alpha {res['judge_vs_consensus_alpha']:.3f} ({jlo:.2f}-{jhi:.2f}), "
              f"exact {res['judge_exact_agreement']:.2f}, 4th-rater alpha {res['judge_as_4th_rater_alpha']:.3f}, "
              f"Alt-Test omega {res['alt_test_winning_rate']:.2f}")

    out_dir = SCRIPT_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run": args.run,
        "dimension": DIM,
        "n_items": len(keys),
        "n_boot": N_BOOT,
        "alt_test_epsilon": DEFAULT_EPSILON,
        "rating_counts": counts,
        "schemes": results,
    }
    with (out_dir / "r_ideo_rescale_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_dir / 'r_ideo_rescale_report.json'}")


if __name__ == "__main__":
    main()
