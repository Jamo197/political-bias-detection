"""Judge-vs-human agreement read against range restriction (offline, no API calls).

QWK shrinks when the labels sit in a narrow band (R_ideo is mostly 3-4), for the judge
and for the humans alike. This report therefore shows, per dimension:

1. the label distribution of every annotator, the consensus and the judge, next to
   the QWK (table, plus a histogram PNG);
2. judge-vs-annotator agreement on the same chunks next to human-human agreement, and
   the ratio of the two (the fair comparison: both suffer the same deflation); the ratio
   gets a 95 % bootstrap CI that resamples texts;
3. exact agreement, within-1 agreement, Spearman and ICC(2,1) for the judge's continuous
   expected level, which are less sensitive to range restriction than QWK.

Judge-vs-annotator numbers are the mean over the three annotators, human-human numbers
the mean over the three annotator pairs, so both are one-rater-against-one-rater.
The consensus is shown for reference only: it is built from the annotators' ratings, so
agreement with it is inflated.

    python "RAG Analysis/LLM calib/range_restriction_report.py"
    python "RAG Analysis/LLM calib/range_restriction_report.py" --run "RAG Analysis/LLM judge results/2026-10-22_132304"
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")

RUNS = Path(__file__).resolve().parents[1]
DEFAULT_RUN = RUNS / "LLM judge results" / "2026-10-22_132304"
CONSENSUS = RUNS / "human_annotations" / "revised" / "consensus_reference_annotations.jsonl"
ANNOTATORS = ("Awsam", "Jannes", "Neda")
LEVELS = {
    "R_top": [1, 2, 3, 4, 5],
    "R_ideo": [1, 2, 3, 4, 5],
    "R_ideo_3": [1, 2, 3],  # R_ideo collapsed to <=3 / 4 / 5 (annotators use 3-5 almost only)
    "A_caus": [0, 1],
}
# Gates behind the continuous expected level: (gates in ladder order, level when none fires).
GATES = {
    "R_top": (("T_domain", "T_subarea", "T_item", "T_central"), 1),
    "R_ideo": (("I_evaluative", "I_stance", "I_party"), 2),
}
N_BOOT = 1000


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load(run: Path, collapse_ideo: bool = False) -> dict:
    """Aligned arrays per dimension: annotators (n, 3), consensus, judge label, judge score."""
    ratings = {}
    for line in CONSENSUS.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            for ch in rec["chunk_annotations"]:
                ratings[(rec["text_index"], ch["chunk_index"])] = ch
    rows = [json.loads(line) for line in (run / "llm_judge_annotations.jsonl").open(encoding="utf-8") if line.strip()]
    rows = [r for r in rows if "error" not in r and r.get("derived_labels")]
    texts = np.array([r["text_index"] for r in rows])
    out = {"texts": texts}
    for dim in ("R_top", "R_ideo", "A_caus"):
        hr = [ratings[(r["text_index"], r["chunk_index"])]["human_ratings"][dim] for r in rows]
        if dim == "A_caus":
            score = np.array([r["raw_answers"]["A_caus"]["noul"] for r in rows])
        else:
            gates, base = GATES[dim]
            score = base + np.array([[r["raw_answers"][g]["noul"] for g in gates] for r in rows]).sum(axis=1)
        out[dim] = {
            "annotators": np.array([[h[a] for a in ANNOTATORS] for h in hr], dtype=float),
            "consensus": np.array([r["human"][dim] for r in rows]),
            "judge": np.array([r["derived_labels"][dim] for r in rows]),
            "score": score,
        }
    if collapse_ideo:
        out["R_ideo_3"] = collapse_ideo_levels(out["R_ideo"])
    return out


def collapse_ideo_levels(x: dict) -> dict:
    """<=3 -> 1, 4 -> 2, 5 -> 3; the expected level (base 2 + gates) is clipped onto the same 1-3 axis."""
    to3 = lambda v: np.clip(v, 3, 5) - 2  # noqa: E731
    return {
        "annotators": to3(x["annotators"]),
        "consensus": to3(x["consensus"]),
        "judge": to3(x["judge"]),
        "score": to3(x["score"]),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def agreement(a, b, dim: str) -> float:
    """QWK for ordinal dimensions, Cohen's kappa for the binary A_caus."""
    weights = "quadratic" if dim != "A_caus" else None
    return float(cohen_kappa_score(a.astype(int), b.astype(int), labels=LEVELS[dim], weights=weights))


def icc_2_1(x: np.ndarray) -> float:
    """ICC(2,1): two-way random, absolute agreement, single rater. x is (items, raters)."""
    n, k = x.shape
    grand = x.mean()
    ms_r = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_e = (resid**2).sum() / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def pair_metrics(a, b, dim: str) -> dict:
    m = {
        "qwk": agreement(a, b, dim),
        "exact": float(np.mean(a == b)),
        "spearman": float(spearmanr(a, b).correlation),
        "icc": icc_2_1(np.column_stack([a, b]).astype(float)),
    }
    if dim != "A_caus":
        m["within1"] = float(np.mean(np.abs(a - b) <= 1))
    return m


def judge_vs_annotators(x: dict, dim: str, idx=slice(None)) -> dict:
    """Mean over annotators of the judge (label for QWK/exact, expected level for rank/ICC)."""
    ann, judge, score = x["annotators"][idx], x["judge"][idx], x["score"][idx]
    per = []
    for j in range(ann.shape[1]):
        m = pair_metrics(judge, ann[:, j], dim)
        m["spearman_score"] = float(spearmanr(score, ann[:, j]).correlation)
        m["icc_score"] = icc_2_1(np.column_stack([score, ann[:, j]]))
        per.append(m)
    return {k: float(np.mean([p[k] for p in per])) for k in per[0]}


def human_vs_human(x: dict, dim: str, idx=slice(None)) -> dict:
    ann = x["annotators"][idx]
    per = [pair_metrics(ann[:, i], ann[:, j], dim) for i, j in itertools.combinations(range(ann.shape[1]), 2)]
    return {k: float(np.mean([p[k] for p in per])) for k in per[0]}


def bootstrap_ratio(d: dict, dim: str, texts: np.ndarray, seed: int = 0):
    """95 % CI of (judge-annotator QWK) / (human-human QWK), resampling texts."""
    rng = np.random.default_rng(seed)
    ids = np.unique(texts)
    members = {t: np.flatnonzero(texts == t) for t in ids}
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([members[t] for t in rng.choice(ids, len(ids))])
        try:
            vals.append(judge_vs_annotators(d[dim], dim, idx)["qwk"] / human_vs_human(d[dim], dim, idx)["qwk"])
        except Exception:
            continue
    return tuple(np.percentile(vals, [2.5, 97.5])) if vals else (None, None)


def distribution(values, levels) -> list[float]:
    values = np.asarray(values).astype(int)
    return [float(np.mean(values == lv)) for lv in levels]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def f(x, spec=".3f"):
    return "n/a" if x is None else format(x, spec)


def build(d: dict) -> dict:
    report = {}
    for dim, levels in LEVELS.items():
        if dim not in d:
            continue
        x = d[dim]
        jh, hh = judge_vs_annotators(d[dim], dim), human_vs_human(d[dim], dim)
        lo, hi = bootstrap_ratio(d, dim, d["texts"])
        dist = {a: distribution(x["annotators"][:, i], levels) for i, a in enumerate(ANNOTATORS)}
        dist["consensus"] = distribution(x["consensus"], levels)
        dist["judge"] = distribution(x["judge"], levels)
        qwk_per = {a: agreement(x["judge"], x["annotators"][:, i], dim) for i, a in enumerate(ANNOTATORS)}
        report[dim] = {
            "levels": levels,
            "distribution": dist,
            "levels_used_sd": {
                **{a: float(x["annotators"][:, i].std()) for i, a in enumerate(ANNOTATORS)},
                "consensus": float(x["consensus"].std()),
                "judge": float(x["judge"].std()),
            },
            "judge_vs_annotators_mean": jh,
            "judge_vs_each_annotator_qwk": qwk_per,
            "human_vs_human_mean": hh,
            "judge_vs_consensus_qwk": agreement(x["judge"], x["consensus"], dim),
            "ratio_judge_to_human": jh["qwk"] / hh["qwk"],
            "ratio_ci95": [lo, hi],
        }
    return report


def print_report(report: dict, name: str) -> None:
    print(f"\nJudge run: {name}")
    for dim, r in report.items():
        metric = "kappa" if dim == "A_caus" else "QWK"
        levels = r["levels"]
        print(f"\n{'=' * 78}\n{dim}  (agreement metric: {metric})\n{'=' * 78}")
        print("Label distribution (share of chunks), with SD of the labels")
        print(f"  {'':10s}" + "".join(f"{lv:>8d}" for lv in levels) + f"{'SD':>8s}")
        for who, shares in r["distribution"].items():
            print(f"  {who:10s}" + "".join(f"{s:8.2f}" for s in shares) + f"{r['levels_used_sd'][who]:8.2f}")
        jh, hh = r["judge_vs_annotators_mean"], r["human_vs_human_mean"]
        print(f"\n  {'':32s}{'judge-human':>13s}{'human-human':>13s}{'ratio':>8s}")
        rows = [("qwk", metric), ("exact", "exact agreement")]
        if dim != "A_caus":
            rows.append(("within1", "within-1 agreement"))
        rows += [("spearman", "Spearman (labels)"), ("icc", "ICC(2,1) (labels)")]
        for key, label in rows:
            ratio = jh[key] / hh[key] if hh[key] else float("nan")
            print(f"  {label:32s}{jh[key]:13.3f}{hh[key]:13.3f}{ratio:8.2f}")
        if dim != "A_caus":
            print(f"  {'Spearman (expected level)':32s}{jh['spearman_score']:13.3f}")
            print(f"  {'ICC(2,1) (expected level)':32s}{jh['icc_score']:13.3f}")
        else:
            print(f"  {'Spearman (P(A_caus=1))':32s}{jh['spearman_score']:13.3f}")
        lo, hi = r["ratio_ci95"]
        print(f"\n  {metric} judge vs each annotator: "
              + ", ".join(f"{a} {v:.3f}" for a, v in r["judge_vs_each_annotator_qwk"].items()))
        print(f"  {metric} judge vs consensus (inflated, reference only): {r['judge_vs_consensus_qwk']:.3f}")
        print(f"  Ratio judge-human / human-human {metric}: {r['ratio_judge_to_human']:.2f} "
              f"(95% CI {f(lo, '.2f')} to {f(hi, '.2f')}, text bootstrap)")


def plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(report), figsize=(5 * len(report), 3.6))
    colors = {"Awsam": "#9ecae1", "Jannes": "#6baed6", "Neda": "#3182bd", "judge": "#d95f02"}
    for ax, (dim, r) in zip(axes, report.items()):
        levels = r["levels"]
        series = [*ANNOTATORS, "judge"]
        width = 0.8 / len(series)
        for i, who in enumerate(series):
            ax.bar(np.arange(len(levels)) + (i - (len(series) - 1) / 2) * width, r["distribution"][who],
                   width, label=who, color=colors[who])
        ax.set_xticks(range(len(levels)), levels)
        ax.set_xlabel("level")
        ax.set_ylabel("share of chunks")
        metric = "kappa" if dim == "A_caus" else "QWK"
        ax.set_title(f"{dim}: judge-human {metric} {r['judge_vs_annotators_mean']['qwk']:.2f}, "
                     f"human-human {r['human_vs_human_mean']['qwk']:.2f}", fontsize=9)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", type=Path, default=DEFAULT_RUN, help="folder with llm_judge_annotations.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: next to this script)")
    ap.add_argument("--collapse-ideo", action="store_true",
                    help="also report R_ideo collapsed to <=3 / 4 / 5 (written to a *_collapsed_ideo file)")
    args = ap.parse_args()
    out = args.out or Path(__file__).resolve().parent
    data = load(args.run, args.collapse_ideo)
    report = build(data)
    print_report(report, args.run.name)
    stem = f"range_restriction_{args.run.name}" + ("_collapsed_ideo" if args.collapse_ideo else "")
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot(report, out / f"{stem}.png")
    print(f"\nWrote {out / (stem + '.json')} and {out / (stem + '.png')}")


if __name__ == "__main__":
    main()
