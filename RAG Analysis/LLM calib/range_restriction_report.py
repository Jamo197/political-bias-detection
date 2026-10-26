"""Judge-vs-human agreement read against range restriction (offline, no API calls).

QWK shrinks when the labels sit in a narrow band (R_ideo is mostly 3-4), for the judge
and for the humans alike. This report therefore shows, per dimension:

1. the label distribution of every annotator, the consensus and the judge, next to
   the QWK (table, plus a histogram PNG);
2. judge-vs-annotator agreement on the same chunks next to human-human agreement, and
   the ratio of the two (the fair comparison: both suffer the same deflation); the ratio
   gets a 95 % bootstrap CI that resamples texts;
3. exact agreement, within-1 agreement, macro-F1, Spearman and ICC(2,1) for the judge's
   continuous expected level, which are less sensitive to range restriction than QWK;
4. a compact summary table (printed and written as summary_table.md) with the average agreement
   (QWK / kappa, exact agreement, macro-F1) of the judge with the annotators next to
   human-human, for the thesis text.

Judge-vs-annotator numbers are the mean over the three annotators, human-human numbers
the mean over the three annotator pairs, so both are one-rater-against-one-rater.
Macro-F1 is the unweighted mean of the per-level F1 over the levels that occur in either
rating (F1 is symmetric in the two raters, so human-human is well defined); for A_caus
the F1 of the positive class (A_caus = 1) is reported as well.
The consensus is shown for reference only: it is built from the annotators' ratings, so
agreement with it is inflated.

Every run writes to its own folder, range_restriction_runs/<timestamp>_<judge run>/, with
report.json (incl. run_info: which judge run, model, settings, chunk counts), summary_table.md,
console_output.txt, label_distribution.png and a copy of the judge run's llm_judge_report.json.

    python "RAG Analysis/LLM calib/range_restriction_report.py"
    python "RAG Analysis/LLM calib/range_restriction_report.py" --run "RAG Analysis/LLM judge results/2026-10-26_143417"
"""

from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score

warnings.filterwarnings("ignore")

RUNS = Path(__file__).resolve().parents[1]
DEFAULT_RUN = RUNS / "LLM judge results" / "2026-10-22_132304"
OUT_ROOT = Path(__file__).resolve().parent / "range_restriction_runs"
CONSENSUS = (
    RUNS / "human_annotations" / "revised" / "consensus_reference_annotations.jsonl"
)
ANNOTATORS = ("Awsam", "Jannes", "Neda")
LEVELS = {
    "R_top": [1, 2, 3, 4, 5],
    "R_ideo": [1, 2, 3, 4, 5],
    "R_ideo_3": [
        1,
        2,
        3,
    ],  # R_ideo collapsed to <=3 / 4 / 5 (annotators use 3-5 almost only)
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
    rows = [
        json.loads(line)
        for line in (run / "llm_judge_annotations.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    n_rows = len(rows)
    rows = [r for r in rows if "error" not in r and r.get("derived_labels")]
    texts = np.array([r["text_index"] for r in rows])
    out = {"texts": texts, "n_rows": n_rows}
    for dim in ("R_top", "R_ideo", "A_caus"):
        hr = [
            ratings[(r["text_index"], r["chunk_index"])]["human_ratings"][dim]
            for r in rows
        ]
        if dim == "A_caus":
            score = np.array([r["raw_answers"]["A_caus"]["noul"] for r in rows])
        else:
            gates, base = GATES[dim]
            score = base + np.array(
                [[r["raw_answers"][g]["noul"] for g in gates] for r in rows]
            ).sum(axis=1)
        out[dim] = {
            "annotators": np.array(
                [[h[a] for a in ANNOTATORS] for h in hr], dtype=float
            ),
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
    return float(
        cohen_kappa_score(
            a.astype(int), b.astype(int), labels=LEVELS[dim], weights=weights
        )
    )


def icc_2_1(x: np.ndarray) -> float:
    """ICC(2,1): two-way random, absolute agreement, single rater. x is (items, raters)."""
    n, k = x.shape
    grand = x.mean()
    ms_r = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_e = (resid**2).sum() / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def macro_f1(a, b) -> float:
    """Unweighted mean of the per-level F1, over the levels present in either rating."""
    a, b = a.astype(int), b.astype(int)
    return float(
        f1_score(a, b, labels=sorted(set(a) | set(b)), average="macro", zero_division=0)
    )


def pair_metrics(a, b, dim: str) -> dict:
    m = {
        "qwk": agreement(a, b, dim),
        "exact": float(np.mean(a == b)),
        "f1_macro": macro_f1(a, b),
        "spearman": float(spearmanr(a, b).correlation),
        "icc": icc_2_1(np.column_stack([a, b]).astype(float)),
    }
    if dim != "A_caus":
        m["within1"] = float(np.mean(np.abs(a - b) <= 1))
    else:
        m["f1_pos"] = float(
            f1_score(a.astype(int), b.astype(int), pos_label=1, zero_division=0)
        )
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
    per = [
        pair_metrics(ann[:, i], ann[:, j], dim)
        for i, j in itertools.combinations(range(ann.shape[1]), 2)
    ]
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
            vals.append(
                judge_vs_annotators(d[dim], dim, idx)["qwk"]
                / human_vs_human(d[dim], dim, idx)["qwk"]
            )
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
        dist = {
            a: distribution(x["annotators"][:, i], levels)
            for i, a in enumerate(ANNOTATORS)
        }
        dist["consensus"] = distribution(x["consensus"], levels)
        dist["judge"] = distribution(x["judge"], levels)
        qwk_per = {
            a: agreement(x["judge"], x["annotators"][:, i], dim)
            for i, a in enumerate(ANNOTATORS)
        }
        f1_per = {
            a: macro_f1(x["judge"], x["annotators"][:, i])
            for i, a in enumerate(ANNOTATORS)
        }
        report[dim] = {
            "levels": levels,
            "distribution": dist,
            "levels_used_sd": {
                **{
                    a: float(x["annotators"][:, i].std())
                    for i, a in enumerate(ANNOTATORS)
                },
                "consensus": float(x["consensus"].std()),
                "judge": float(x["judge"].std()),
            },
            "judge_vs_annotators_mean": jh,
            "judge_vs_each_annotator_qwk": qwk_per,
            "judge_vs_each_annotator_f1_macro": f1_per,
            "human_vs_human_mean": hh,
            "judge_vs_consensus_qwk": agreement(x["judge"], x["consensus"], dim),
            "judge_vs_consensus_f1_macro": macro_f1(x["judge"], x["consensus"]),
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
            print(
                f"  {who:10s}"
                + "".join(f"{s:8.2f}" for s in shares)
                + f"{r['levels_used_sd'][who]:8.2f}"
            )
        jh, hh = r["judge_vs_annotators_mean"], r["human_vs_human_mean"]
        print(f"\n  {'':32s}{'judge-human':>13s}{'human-human':>13s}{'ratio':>8s}")
        rows = [("qwk", metric), ("exact", "exact agreement"), ("f1_macro", "macro-F1")]
        if dim != "A_caus":
            rows.append(("within1", "within-1 agreement"))
        else:
            rows.append(("f1_pos", "F1 (A_caus = 1)"))
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
        print(
            f"\n  {metric} judge vs each annotator: "
            + ", ".join(
                f"{a} {v:.3f}" for a, v in r["judge_vs_each_annotator_qwk"].items()
            )
        )
        print(
            "  macro-F1 judge vs each annotator: "
            + ", ".join(
                f"{a} {v:.3f}" for a, v in r["judge_vs_each_annotator_f1_macro"].items()
            )
        )
        print(
            f"  {metric} / macro-F1 judge vs consensus (inflated, reference only): "
            f"{r['judge_vs_consensus_qwk']:.3f} / {r['judge_vs_consensus_f1_macro']:.3f}"
        )
        print(
            f"  Ratio judge-human / human-human {metric}: {r['ratio_judge_to_human']:.2f} "
            f"(95% CI {f(lo, '.2f')} to {f(hi, '.2f')}, text bootstrap)"
        )


def summary_table(report: dict) -> str:
    """Markdown table for the thesis: average agreement with the annotators next to human-human."""
    head = (
        "| Dimension | Agreement (QWK; kappa for A_caus) judge-annot. | human-human "
        "| Exact agr. judge-annot. | human-human | Macro-F1 judge-annot. | human-human "
        "| Ratio judge/human (95% CI) |"
    )
    lines = [head, "|" + "---|" * 8]
    for dim, r in report.items():
        jh, hh = r["judge_vs_annotators_mean"], r["human_vs_human_mean"]
        lo, hi = r["ratio_ci95"]
        lines.append(
            f"| {dim} | {jh['qwk']:.2f} | {hh['qwk']:.2f} | {jh['exact']:.2f} | {hh['exact']:.2f} "
            f"| {jh['f1_macro']:.2f} | {hh['f1_macro']:.2f} "
            f"| {r['ratio_judge_to_human']:.2f} ({f(lo, '.2f')} to {f(hi, '.2f')}) |"
        )
    lines.append("")
    lines.append(
        "Judge-annot. = mean over the three annotators; human-human = mean over the three annotator pairs."
    )
    return "\n".join(lines)


def plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(report), figsize=(5 * len(report), 3.6))
    colors = {
        "Awsam": "#9ecae1",
        "Jannes": "#6baed6",
        "Neda": "#3182bd",
        "judge": "#d95f02",
    }
    for ax, (dim, r) in zip(axes, report.items()):
        levels = r["levels"]
        series = [*ANNOTATORS, "judge"]
        width = 0.8 / len(series)
        for i, who in enumerate(series):
            ax.bar(
                np.arange(len(levels)) + (i - (len(series) - 1) / 2) * width,
                r["distribution"][who],
                width,
                label=who,
                color=colors[who],
            )
        ax.set_xticks(range(len(levels)), levels)
        ax.set_xlabel("level")
        ax.set_ylabel("share of chunks")
        metric = "kappa" if dim == "A_caus" else "QWK"
        ax.set_title(
            f"{dim}: judge-human {metric} {r['judge_vs_annotators_mean']['qwk']:.2f}, "
            f"human-human {r['human_vs_human_mean']['qwk']:.2f}",
            fontsize=9,
        )
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_info(run: Path, data: dict, argv: list[str]) -> dict:
    """What went into this report: the judge run, its settings, and how many chunks were used."""
    info = {
        "judge_run": run.name,
        "judge_run_path": str(run),
        "judge_files": sorted(p.name for p in run.iterdir()),
        "consensus_file": str(CONSENSUS),
        "chunks_in_annotations": data["n_rows"],
        "chunks_used": int(len(data["texts"])),
        "texts_used": int(len(np.unique(data["texts"]))),
        "annotators": list(ANNOTATORS),
        "command": " ".join(argv),
    }
    report_file = run / "llm_judge_report.json"
    if report_file.exists():
        rep = json.loads(report_file.read_text(encoding="utf-8"))
        for key in (
            "model",
            "questions_hash",
            "a_caus_threshold",
            "timestamp",
            "total_chunks",
            "judged_chunks",
        ):
            info[f"judge_{key}"] = rep.get(key)
    return info


def info_lines(info: dict) -> str:
    keys = (
        "judge_run",
        "judge_model",
        "judge_questions_hash",
        "judge_a_caus_threshold",
        "judge_timestamp",
    )
    lines = [f"- {k}: {info[k]}" for k in keys if info.get(k) is not None]
    lines.append(
        f"- chunks used: {info['chunks_used']} of {info['chunks_in_annotations']} ({info['texts_used']} texts)"
    )
    lines.append(f"- judge run folder: {info['judge_run_path']}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--run",
        type=Path,
        default=DEFAULT_RUN,
        help="folder with llm_judge_annotations.jsonl",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder (default: a new timestamped folder in range_restriction_runs/)",
    )
    ap.add_argument(
        "--collapse-ideo",
        action="store_true",
        help="also report R_ideo collapsed to <=3 / 4 / 5 (written to a *_collapsed_ideo file)",
    )
    args = ap.parse_args()
    data = load(args.run, args.collapse_ideo)
    report = build(data)
    info = run_info(args.run, data, sys.argv)
    suffix = "_collapsed_ideo" if args.collapse_ideo else ""
    out = (
        args.out
        or OUT_ROOT / f"{datetime.now():%Y-%m-%d_%H%M%S}_{args.run.name}{suffix}"
    )
    out.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("Judge results used\n" + info_lines(info) + "\n")
        print_report(report, args.run.name)
        table = summary_table(report)
        print(f"\nSummary table\n\n{table}")
    print(buf.getvalue())

    (out / "report.json").write_text(
        json.dumps({"run_info": info, **report}, indent=2), encoding="utf-8"
    )
    (out / "summary_table.md").write_text(
        f"Judge results used:\n\n{info_lines(info)}\n\n{table}\n", encoding="utf-8"
    )
    (out / "console_output.txt").write_text(buf.getvalue(), encoding="utf-8")
    plot(report, out / "label_distribution.png")
    # keep a copy of the judge run's own summary so the folder shows what was used
    judge_report = args.run / "llm_judge_report.json"
    if judge_report.exists():
        shutil.copy(judge_report, out / "judge_run_report.json")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
