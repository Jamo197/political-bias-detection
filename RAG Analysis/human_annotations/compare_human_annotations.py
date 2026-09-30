#!/usr/bin/env python3
"""Compare human qualitative annotations against each other.

Loads the per-annotator ``*_qualitative_annotations.jsonl`` files (Awsam,
Jannes, Neda, ...), aligns them on ``(text_index, chunk_index)`` and reports
inter-annotator agreement for the rubric dimensions:

* ``R_top``  — ordinal 1-5 (topical relevance)
* ``R_ideo`` — ordinal 1-5 (ideological specificity)
* ``A_caus`` — nominal 0/1 (causal utilisation)

Two families of coefficients are computed, pairwise and across all raters:

* ``quadratic weighted kappa`` (Cohen) for the ordinal scales and unweighted
  Cohen's kappa for the binary scale — matching ``src/run_llm_judge.py``, so
  human-human numbers are directly comparable to the human-vs-Jev report.
* ``Krippendorff's alpha`` (nominal and ordinal), which generalises cleanly to
  more than two raters. Implemented here with numpy only (no extra dependency)
  following Krippendorff's coincidence-matrix formulation.

The report also ranks the chunks where raters disagree most, shows each
annotator's leniency/severity per dimension, and summarises which dimension and
which annotator pair show the weakest agreement — i.e. where the biggest error
lies.

Usage (from anywhere):

    python "RAG Analysis/human_annotations/compare_human_annotations.py"
    python "RAG Analysis/human_annotations/compare_human_annotations.py" \
        --show-disagreements 15
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PATTERN = "*_qualitative_annotations.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "human_agreement_report.json"
DEFAULT_SHOW_DISAGREEMENTS = 10

# Dimension metadata. ``levels`` is the full label domain (used for confusion
# matrices and kappa), ``kind`` selects the weighting/distance function and
# ``alpha`` the Krippendorff metric.
DIMENSIONS: dict[str, dict] = {
    "R_top": {"levels": [1, 2, 3, 4, 5], "kind": "ordinal", "alpha": "ordinal"},
    "R_ideo": {"levels": [1, 2, 3, 4, 5], "kind": "ordinal", "alpha": "ordinal"},
    "A_caus": {"levels": [0, 1], "kind": "nominal", "alpha": "nominal"},
}

# Extra fields we can read but that only some annotators provide. These are
# reported as "single rater / not comparable" when fewer than two annotators
# filled them.
OPTIONAL_DIMENSIONS: dict[str, dict] = {
    "N_info": {"levels": [1, 2, 3], "kind": "ordinal", "alpha": "ordinal"},
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def discover_annotation_files(directory: Path, pattern: str) -> list[Path]:
    """Return sorted annotation files matching ``pattern`` in ``directory``."""
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No annotation files matching {pattern!r} in {directory}"
        )
    return files


def annotator_name(path: Path) -> str:
    """Derive a short annotator label from the file name."""
    return path.name.split("_", 1)[0]


def _coerce_value(value: object) -> int | None:
    """Return an int label for a raw annotation value, or None if missing."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if float(value).is_integer():
            return int(value)
        return None
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def load_annotations(path: Path, dims: list[str]) -> dict[tuple[str, int], dict]:
    """Flatten one annotation file to ``(text_index, chunk_index) -> labels``.

    Later records for the same ``text_index`` do not silently win: every chunk
    of every record is collected, and duplicate keys keep the first occurrence
    (the files are single-pass exports, so this should not trigger).
    """
    items: dict[tuple[str, int], dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            text_index = str(record.get("text_index"))
            for chunk in record.get("chunk_annotations", []) or []:
                chunk_index = chunk.get("chunk_index")
                if chunk_index is None:
                    continue
                key = (text_index, int(chunk_index))
                labels = {dim: _coerce_value(chunk.get(dim)) for dim in dims}
                meta = {
                    "party": (chunk.get("chunk_metadata") or {}).get("party"),
                    "speaker": (chunk.get("chunk_metadata") or {}).get("speaker"),
                }
                if key in items:
                    continue
                items[key] = {"labels": labels, "meta": meta}
    return items


def build_alignment(
    files: list[Path], dims: list[str]
) -> tuple[dict[str, dict], dict[str, list[str]]]:
    """Align per-annotator items and return (data, diagnostics).

    ``data`` maps annotator name -> {(text_index, chunk_index): {"labels",
    "meta"}}. ``diagnostics`` records dropped/extra keys and missing values.
    """
    per_annotator: dict[str, dict] = {}
    for path in files:
        name = annotator_name(path)
        per_annotator[name] = load_annotations(path, dims)

    names = list(per_annotator)
    key_sets = {name: set(items) for name, items in per_annotator.items()}
    shared = set.intersection(*key_sets.values()) if key_sets else set()
    union = set.union(*key_sets.values()) if key_sets else set()

    diagnostics = {
        "annotators": names,
        "per_annotator_keys": {n: len(k) for n, k in key_sets.items()},
        "shared_keys": len(shared),
        "union_keys": len(union),
        # Keys an annotator has that are not present everywhere, and universal
        # keys an annotator is missing. Items are dropped from agreement when
        # either list is non-empty.
        "not_shared_by_all": {
            name: sorted(keys - shared) for name, keys in key_sets.items()
        },
        "missing_from": {
            name: sorted(shared - keys) for name, keys in key_sets.items()
        },
    }

    # Restrict every annotator to the shared key set so vectors align.
    data: dict[str, dict] = {}
    for name, items in per_annotator.items():
        data[name] = {key: items[key] for key in shared if key in items}

    # Missing-value counts per annotator/dimension over the shared keys.
    diagnostics["missing_values"] = {
        name: {
            dim: sum(1 for item in items.values() if item["labels"].get(dim) is None)
            for dim in dims
        }
        for name, items in data.items()
    }
    return data, diagnostics


# ---------------------------------------------------------------------------
# Pairwise Cohen's kappa
# ---------------------------------------------------------------------------


def _safe_kappa(y1, y2, labels, weights=None) -> float | None:
    """Cohen's kappa, handling the all-agree/NaN edge case like run_llm_judge."""
    k = cohen_kappa_score(y1, y2, labels=labels, weights=weights)
    if np.isnan(k):
        if len(y1) > 0 and np.all(np.asarray(y1) == np.asarray(y2)):
            return 1.0
        return None
    return float(k)


# ---------------------------------------------------------------------------
# Krippendorff's alpha (self-contained)
# ---------------------------------------------------------------------------


def krippendorff_alpha(
    data: np.ndarray,
    level: str = "ordinal",
    value_domain: list[int] | None = None,
) -> float | None:
    """Krippendorff's alpha for a units x raters reliability matrix.

    ``data`` may contain ``np.nan`` for missing ratings. ``level`` is either
    ``"nominal"`` or ``"ordinal"``. Returns ``None`` when alpha is undefined
    (fewer than two units with at least two ratings).

    Implements the coincidence-matrix formulation: ``alpha = 1 - Do / De`` with
    ``Do`` the observed and ``De`` the expected disagreement, both using the
    squared difference function ``delta`` (0/1 for nominal, the ordered
    marginal-rank metric for ordinal).
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a 2D units x raters matrix")

    if value_domain is None:
        observed = data[~np.isnan(data)]
        if observed.size == 0:
            return None
        domain = sorted(set(int(v) for v in observed))
    else:
        domain = list(value_domain)
    index = {value: i for i, value in enumerate(domain)}
    n_values = len(domain)
    if n_values < 2:
        # Only one distinct value ever observed -> no disagreement possible.
        return 1.0

    # Coincidence matrix over the coincidence of values within units.
    coincidence = np.zeros((n_values, n_values), dtype=float)
    usable_units = 0
    for row in data:
        values = [index[int(v)] for v in row if not np.isnan(v)]
        m = len(values)
        if m < 2:
            continue
        usable_units += 1
        counts = np.bincount(values, minlength=n_values).astype(float)
        for c in range(n_values):
            if counts[c] == 0:
                continue
            for k in range(n_values):
                if c == k:
                    coincidence[c, k] += counts[c] * (counts[c] - 1) / (m - 1)
                else:
                    if counts[k] == 0:
                        continue
                    coincidence[c, k] += counts[c] * counts[k] / (m - 1)

    if usable_units < 2:
        return None

    n_c = coincidence.sum(axis=1)
    n_total = n_c.sum()
    if n_total <= 1:
        return None

    # Squared difference function between values.
    delta = np.zeros((n_values, n_values), dtype=float)
    cumulative = np.cumsum(n_c)
    for c in range(n_values):
        for k in range(n_values):
            if level == "nominal":
                delta[c, k] = 0.0 if c == k else 1.0
            else:  # ordinal
                lo, hi = (c, k) if c <= k else (k, c)
                # Sum of marginals from lo..hi, minus half of the two endpoints.
                between = cumulative[hi] - (cumulative[lo - 1] if lo > 0 else 0.0)
                delta[c, k] = (between - (n_c[c] + n_c[k]) / 2.0) ** 2

    do = float((coincidence * delta).sum() / n_total)
    de = float(
        sum(
            n_c[c] * n_c[k] * delta[c, k]
            for c in range(n_values)
            for k in range(n_values)
        )
        / (n_total * (n_total - 1))
    )
    if np.isclose(de, 0.0):
        # No expected disagreement: perfect agreement by construction.
        return 1.0 if np.isclose(do, 0.0) else None
    return float(1.0 - do / de)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_pairwise_metrics(data: dict[str, dict], dim: str, cfg: dict) -> dict:
    """Agreement metrics for every annotator pair on one dimension."""
    annotators = sorted(data)
    levels = cfg["levels"]
    ordinal = cfg["kind"] == "ordinal"
    weights = "quadratic" if ordinal else None
    results: dict[str, dict] = {}

    for a, b in combinations(annotators, 2):
        vec_a, vec_b = [], []
        for key in sorted(data[a]):
            va = data[a][key]["labels"].get(dim)
            vb = data[b][key]["labels"].get(dim)
            if va is None or vb is None:
                continue
            vec_a.append(va)
            vec_b.append(vb)
        if not vec_a:
            results[f"{a}|{b}"] = None  # type: ignore
            continue
        y1 = np.array(vec_a)
        y2 = np.array(vec_b)
        diffs = y2 - y1

        cm = confusion_matrix(y1, y2, labels=levels)
        entry = {
            "rater_a": a,
            "rater_b": b,
            "n": int(len(y1)),
            "exact_agreement": float(np.mean(y1 == y2)),
            "kappa": _safe_kappa(y1, y2, labels=levels, weights=weights),
            "alpha": krippendorff_alpha(
                np.column_stack([y1, y2]),
                level=cfg["alpha"],
                value_domain=levels,
            ),
            "mean_diff_a_minus_b": float(np.mean(y1 - y2)),
            "mean_abs_diff": float(np.mean(np.abs(diffs))),
            "pct_within_1": float(np.mean(np.abs(diffs) <= 1)),
            "labels": levels,
            "confusion_matrix": cm.tolist(),
        }
        results[f"{a}|{b}"] = entry
    return results


def compute_multirater_metrics(data: dict[str, dict], dim: str, cfg: dict) -> dict:
    """Krippendorff's alpha across all annotators plus mean pairwise kappa."""
    annotators = sorted(data)
    keys = sorted(data[annotators[0]]) if annotators else []
    matrix = []
    for key in keys:
        row = []
        for name in annotators:
            value = data[name][key]["labels"].get(dim)
            row.append(np.nan if value is None else value)
        matrix.append(row)

    alpha = krippendorff_alpha(
        np.array(matrix, dtype=float),
        level=cfg["alpha"],
        value_domain=cfg["levels"],
    )
    pairwise = compute_pairwise_metrics(data, dim, cfg)
    kappas = [e["kappa"] for e in pairwise.values() if e and e["kappa"] is not None]
    return {
        "n_raters": len(annotators),
        "n_items": int(len(keys)),
        "alpha": alpha,
        "mean_pairwise_kappa": float(np.mean(kappas)) if kappas else None,
        "min_pairwise_kappa": float(np.min(kappas)) if kappas else None,
        "max_pairwise_kappa": float(np.max(kappas)) if kappas else None,
        "pairwise": pairwise,
    }


def compute_annotator_bias(data: dict[str, dict], dim: str) -> dict:
    """Mean level and mean deviation from the per-item mean for each annotator.

    A positive deviation means the annotator rates the item above the other
    raters (more lenient); negative means stricter. For ``A_caus`` the mean is
    the share of chunks marked as used.
    """
    annotators = sorted(data)
    keys = sorted(data[annotators[0]]) if annotators else []
    means: dict[str, list[float]] = {name: [] for name in annotators}
    deviations: dict[str, list[float]] = {name: [] for name in annotators}

    for key in keys:
        present = {}
        for name in annotators:
            value = data[name][key]["labels"].get(dim)
            if value is not None:
                present[name] = value
        if not present:
            continue
        item_mean = float(np.mean(list(present.values())))
        for name, value in present.items():
            means[name].append(value)
            deviations[name].append(value - item_mean)

    return {
        name: {
            "n": len(means[name]),
            "mean_level": float(np.mean(means[name])) if means[name] else None,
            "mean_deviation_from_item_mean": (
                float(np.mean(deviations[name])) if deviations[name] else None
            ),
        }
        for name in annotators
    }


def get_worst_disagreements(
    data: dict[str, dict], dim: str, cfg: dict, n: int = 10
) -> list[dict]:
    """Rank items by across-rater spread for one dimension.

    Ordinal dimensions are ranked by the value range (max - min); ties break on
    mean absolute deviation. ``A_caus`` splits (range 1) are simply listed.
    """
    annotators = sorted(data)
    rows = []
    for key in sorted(data[annotators[0]]):
        present = {}
        for name in annotators:
            value = data[name][key]["labels"].get(dim)
            if value is not None:
                present[name] = int(value)
        if len(present) < 2:
            continue
        values = list(present.values())
        spread = max(values) - min(values)
        if spread == 0:
            continue
        rows.append(
            {
                "text_index": key[0],
                "chunk_index": key[1],
                "dimension": dim,
                "spread": int(spread),
                "mean_abs_deviation": float(
                    np.mean(np.abs(np.array(values) - np.mean(values)))
                ),
                "ratings": present,
                "party": data[annotators[0]][key]["meta"].get("party"),
                "speaker": data[annotators[0]][key]["meta"].get("speaker"),
            }
        )
    rows.sort(key=lambda r: (-r["spread"], -r["mean_abs_deviation"], r["text_index"]))
    return rows[:n]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt(value: float | None, spec: str = ">6.3f") -> str:
    return f"{value:{spec}}" if value is not None else "   N/A"


def print_report(
    per_dim: dict[str, dict],
    bias: dict[str, dict],
    worst: dict[str, list[dict]],
    diagnostics: dict,
) -> None:
    print("=== Human Inter-Annotator Agreement Report ===")
    print(f"Annotators: {', '.join(diagnostics['annotators'])}")
    print(
        f"Items: {diagnostics['shared_keys']} shared "
        f"(union {diagnostics['union_keys']}, per annotator "
        f"{diagnostics['per_annotator_keys']})"
    )

    for dim in per_dim:
        print(f"\n--- {dim} ---")
        multi = per_dim[dim]["multirater"]
        print(
            f"All {multi['n_raters']} raters: "
            f"Krippendorff alpha={_fmt(multi['alpha'])}, "
            f"mean pairwise kappa={_fmt(multi['mean_pairwise_kappa'])}, "
            f"kappa range "
            f"{_fmt(multi['min_pairwise_kappa'])}..{_fmt(multi['max_pairwise_kappa'])}"
        )
        header = (
            f"  {'pair':<18} {'n':>5} {'agree%':>7} {'kappa':>7} "
            f"{'alpha':>7} {'meanDiff':>9} {'|diff|':>7} {'<=1':>6}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for pair, entry in multi["pairwise"].items():
            if entry is None:
                print(f"  {pair:<18} {'—':>5}")
                continue
            print(
                f"  {pair:<18} {entry['n']:>5} "
                f"{entry['exact_agreement']*100:>6.1f}% "
                f"{_fmt(entry['kappa'])} {_fmt(entry['alpha'])} "
                f"{entry['mean_diff_a_minus_b']:>+9.2f} "
                f"{entry['mean_abs_diff']:>7.2f} "
                f"{entry['pct_within_1']*100:>5.1f}%"
            )

        print("  annotator bias (mean level / mean deviation from item mean):")
        for name, stats in bias[dim].items():
            print(
                f"    {name:<10} n={stats['n']:>3} "
                f"level={_fmt(stats['mean_level'], '>5.2f')} "
                f"dev={_fmt(stats['mean_deviation_from_item_mean'], '+5.2f')}"
            )

    for dim, rows in worst.items():
        if not rows:
            continue
        print(f"\nTop {len(rows)} disagreements — {dim} (spread = max-min):")
        annotators = diagnostics["annotators"]
        for row in rows:
            ratings = ", ".join(
                f"{name}={row['ratings'].get(name)}" for name in annotators
            )
            print(
                f"  text {row['text_index']:>4} chunk {row['chunk_index']:>2} "
                f"spread={row['spread']} [{ratings}] "
                f"({row['party']}, {row['speaker']})"
            )


def print_summary(per_dim: dict[str, dict]) -> None:
    """State plainly which dimension and pair show the weakest agreement."""
    print("\n=== Summary: where agreement is weakest ===")
    weakest_dim = None
    weakest_val = None
    for dim, info in per_dim.items():
        val = info["multirater"]["mean_pairwise_kappa"]
        if val is None:
            continue
        print(
            f"  {dim:<8} mean pairwise kappa={val:>6.3f}  "
            f"alpha={_fmt(info['multirater']['alpha'])}"
        )
        if weakest_val is None or val < weakest_val:
            weakest_val = val
            weakest_dim = dim
    if weakest_dim is not None:
        print(f"  => weakest dimension: {weakest_dim} (kappa={weakest_val:.3f})")

    worst_pair = None
    worst_pair_val = None
    for dim, info in per_dim.items():
        for pair, entry in info["multirater"]["pairwise"].items():
            if entry is None or entry["kappa"] is None:
                continue
            if worst_pair_val is None or entry["kappa"] < worst_pair_val:
                worst_pair_val = entry["kappa"]
                worst_pair = f"{pair} on {dim}"
    if worst_pair is not None:
        print(
            f"  => most divergent pair: {worst_pair} " f"(kappa={worst_pair_val:.3f})"
        )


def build_report(
    data: dict[str, dict],
    diagnostics: dict,
    per_dim: dict[str, dict],
    bias: dict[str, dict],
    worst: dict[str, list[dict]],
    files: list[Path],
    dims: list[str],
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files": [str(p) for p in files],
        "dimensions": dims,
        "annotators": diagnostics["annotators"],
        "n_items": diagnostics["shared_keys"],
        "n_union_items": diagnostics["union_keys"],
        "n_dropped_items": diagnostics["union_keys"] - diagnostics["shared_keys"],
        "diagnostics": diagnostics,
        "per_dimension": per_dim,
        "annotator_bias": bias,
        "worst_disagreements": worst,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare human qualitative annotations (QWK + Krippendorff "
        "alpha) pairwise and across all raters."
    )
    parser.add_argument(
        "--dir",
        default=str(SCRIPT_DIR),
        help="Directory containing the annotation JSONL files.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern for annotation files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explicit annotation files (overrides --dir/--pattern).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the JSON report.",
    )
    parser.add_argument(
        "--show-disagreements",
        type=int,
        default=DEFAULT_SHOW_DISAGREEMENTS,
        help="Number of worst-disagreement items to list per dimension.",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        default=list(DIMENSIONS),
        help="Dimensions to compare (default: R_top R_ideo A_caus).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = discover_annotation_files(Path(args.dir), args.pattern)

    dims = [d for d in args.dims if d in DIMENSIONS]
    unknown = [d for d in args.dims if d not in DIMENSIONS]
    if unknown:
        print(f"Warning: ignoring unknown dimensions: {unknown}", file=sys.stderr)
    if len(dims) < 1:
        raise SystemExit("No valid dimensions selected.")

    all_dims = dims + [d for d in OPTIONAL_DIMENSIONS if d not in dims]
    data, diagnostics = build_alignment(files, all_dims)
    if len(data) < 2:
        raise SystemExit("Need at least two annotators to compare.")

    per_dim = {}
    bias = {}
    worst = {}
    for dim in dims:
        cfg = DIMENSIONS[dim]
        per_dim[dim] = {
            "config": cfg,
            "multirater": compute_multirater_metrics(data, dim, cfg),
        }
        bias[dim] = compute_annotator_bias(data, dim)
        worst[dim] = get_worst_disagreements(data, dim, cfg, n=args.show_disagreements)

    # Optional dimensions filled by a single annotator: report coverage only.
    single_rater = {}
    for dim in OPTIONAL_DIMENSIONS:
        if dim in dims:
            continue
        counts = {
            name: sum(
                1 for item in items.values() if item["labels"].get(dim) is not None
            )
            for name, items in data.items()
        }
        if sum(1 for c in counts.values() if c > 0) < 2:
            single_rater[dim] = counts

    print_report(per_dim, bias, worst, diagnostics)
    if single_rater:
        print("\nNot comparable (fewer than two annotators filled the field):")
        for dim, counts in single_rater.items():
            print(f"  {dim}: {counts}")
    print_summary(per_dim)

    report = build_report(data, diagnostics, per_dim, bias, worst, files, dims)
    if single_rater:
        report["single_rater_dimensions"] = single_rater
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
