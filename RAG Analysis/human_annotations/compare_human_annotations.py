#!/usr/bin/env python3
"""Compare human qualitative annotations against each other.

Loads the per-annotator ``*_qualitative_annotations.jsonl`` files (Awsam,
Jannes, Neda, ...), aligns them on ``(text_index, chunk_index)`` and reports
inter-annotator agreement for the rubric dimensions:

* ``R_top``  — ordinal 1-5 (topical relevance)
* ``R_ideo`` — ordinal 1-5 (ideological specificity)
* ``A_caus`` — nominal 0/1 (causal utilisation; JSON ``true``/``false`` is
  accepted and mapped to 1/0)

Two families of coefficients are computed, pairwise and across all raters:

* Cohen's kappa — quadratic weighted for the ordinal scales, unweighted for the
  binary scale — matching ``src/run_llm_judge.py``, so human-human numbers are
  directly comparable to the human-vs-LLM report.
* Krippendorff's alpha (ordinal / nominal), which handles more than two raters
  and missing ratings. Implemented with numpy only, following Krippendorff's
  coincidence-matrix formulation (Krippendorff 2011, "Computing
  Krippendorff's Alpha-Reliability"). Checked against the worked example in
  that paper (nominal 0.743, ordinal 0.815).

The report also ranks the chunks where raters disagree most, shows each
annotator's leniency/severity per dimension, and summarises which dimension and
which annotator pair (per dimension) show the weakest agreement.

Usage (from anywhere):

    python "RAG Analysis/human_annotations/compare_human_annotations.py"
    python "RAG Analysis/human_annotations/compare_human_annotations.py" \
        --show-disagreements 15 --alignment all
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
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
DEFAULT_SHOW_DISAGREEMENTS = 20

# ``levels`` is the full label domain (confusion matrices, kappa, alpha);
# ``kind`` selects kappa weighting ("ordinal" -> quadratic) and the
# Krippendorff metric.
DIMENSIONS: dict[str, dict] = {
    "R_top": {"levels": [1, 2, 3, 4, 5], "kind": "ordinal"},
    "R_ideo": {"levels": [1, 2, 3, 4, 5], "kind": "ordinal"},
    "A_caus": {"levels": [0, 1], "kind": "nominal"},
}

# Fields only some annotators fill. They are loaded and can be analysed with
# ``--dims``; otherwise their coverage is reported.
OPTIONAL_DIMENSIONS: dict[str, dict] = {
    "N_info": {"levels": [1, 2, 3], "kind": "ordinal"},
}

ALL_DIMENSIONS: dict[str, dict] = {**DIMENSIONS, **OPTIONAL_DIMENSIONS}

_TRUE_STRINGS = {"true", "yes", "y", "t"}
_FALSE_STRINGS = {"false", "no", "n", "f"}

Key = tuple[str, int]


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
    """Derive a short annotator label from the file name (``Name_...``)."""
    return path.name.split("_", 1)[0]


def coerce_label(value: object) -> int | None:
    """Map a raw annotation value to an int label.

    Returns ``None`` for a missing value (``None`` / empty string) and raises
    ``ValueError`` for anything that cannot be interpreted. Booleans map to
    1/0 so ``A_caus`` may be stored as ``true``/``false``.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # must precede int: bool is a subclass of int
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError(f"non-integer rating {value!r}")
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "":
            return None
        if text in _TRUE_STRINGS:
            return 1
        if text in _FALSE_STRINGS:
            return 0
        try:
            number = float(text)
        except ValueError:
            raise ValueError(f"unparseable rating {value!r}") from None
        if number.is_integer():
            return int(number)
        raise ValueError(f"non-integer rating {value!r}")
    raise ValueError(f"unsupported rating type {type(value).__name__}")


@dataclass
class Item:
    labels: dict[str, int | None]
    meta: dict[str, object]


def load_annotations(path: Path, dims: list[str]) -> dict[Key, Item]:
    """Flatten one annotation file to ``(text_index, chunk_index) -> Item``.

    Malformed lines, invalid labels and duplicate keys raise, so no annotation
    is silently dropped or replaced.
    """
    items: dict[Key, Item] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            where = f"{path.name}:{line_no}"
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON ({exc})") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{where}: record is not a JSON object")
            if record.get("text_index") is None:
                raise ValueError(f"{where}: record is missing text_index")
            text_index = str(record["text_index"]).strip()

            for chunk in record.get("chunk_annotations") or []:
                if not isinstance(chunk, dict):
                    raise ValueError(f"{where}: chunk annotation is not an object")
                try:
                    chunk_index = int(chunk["chunk_index"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{where}: missing/invalid chunk_index "
                        f"{chunk.get('chunk_index')!r}"
                    ) from exc
                key = (text_index, chunk_index)
                if key in items:
                    raise ValueError(f"{where}: duplicate annotation key {key!r}")

                labels: dict[str, int | None] = {}
                for dim in dims:
                    try:
                        value = coerce_label(chunk.get(dim))
                    except ValueError as exc:
                        raise ValueError(f"{where} {key}: {dim}: {exc}") from exc
                    levels = ALL_DIMENSIONS[dim]["levels"]
                    if value is not None and value not in levels:
                        raise ValueError(
                            f"{where} {key}: {dim} value {value} outside {levels}"
                        )
                    labels[dim] = value

                chunk_meta = chunk.get("chunk_metadata") or {}
                items[key] = Item(
                    labels=labels,
                    meta={
                        "party": chunk_meta.get("party"),
                        "speaker": chunk_meta.get("speaker"),
                    },
                )
    return items


def _natural_key(key: Key) -> tuple:
    """Sort text indices numerically when they are numbers ("2" < "10")."""
    text_index, chunk_index = key
    parts = re.split(r"(\d+)", text_index)
    return (
        tuple((0, int(p)) if p.isdigit() else (1, p) for p in parts if p),
        chunk_index,
    )


# ---------------------------------------------------------------------------
# Rating table: one units x raters matrix per dimension
# ---------------------------------------------------------------------------


@dataclass
class RatingTable:
    annotators: list[str]
    keys: list[Key]
    matrices: dict[str, np.ndarray]  # dim -> (n_keys, n_annotators), NaN = missing
    meta: dict[Key, dict] = field(default_factory=dict)

    def column(self, dim: str, name: str) -> np.ndarray:
        return self.matrices[dim][:, self.annotators.index(name)]


def build_table(
    files: list[Path], dims: list[str], alignment: str
) -> tuple[RatingTable, dict]:
    """Load all files and align them into a :class:`RatingTable`.

    ``alignment="shared"`` keeps only chunks every annotator rated (one common
    item set for all coefficients). ``alignment="all"`` keeps the union; alpha
    then uses every chunk with >= 2 ratings and each pairwise kappa uses that
    pair's overlap.
    """
    per_annotator: dict[str, dict[Key, Item]] = {}
    for path in files:
        name = annotator_name(path)
        if name in per_annotator:
            raise ValueError(
                f"Two files map to annotator {name!r}; rename one of them "
                f"(annotator = file name up to the first underscore)."
            )
        per_annotator[name] = load_annotations(path, dims)

    annotators = sorted(per_annotator)
    key_sets = {name: set(items) for name, items in per_annotator.items()}
    shared = set.intersection(*key_sets.values())
    union = set.union(*key_sets.values())
    keys = sorted(shared if alignment == "shared" else union, key=_natural_key)

    matrices = {}
    for dim in dims:
        matrix = np.full((len(keys), len(annotators)), np.nan)
        for i, key in enumerate(keys):
            for j, name in enumerate(annotators):
                item = per_annotator[name].get(key)
                if item is not None and item.labels.get(dim) is not None:
                    matrix[i, j] = item.labels[dim]
        matrices[dim] = matrix

    # First non-empty metadata per chunk, whichever annotator provides it.
    meta = {}
    for key in keys:
        found = {"party": None, "speaker": None}
        for name in annotators:
            item = per_annotator[name].get(key)
            if item is None:
                continue
            for field_name in found:
                if found[field_name] is None:
                    found[field_name] = item.meta.get(field_name)
        meta[key] = found

    diagnostics = {
        "alignment": alignment,
        "annotators": annotators,
        "per_annotator_keys": {n: len(key_sets[n]) for n in annotators},
        "shared_keys": len(shared),
        "union_keys": len(union),
        "analysed_keys": len(keys),
        # Chunks this annotator rated that someone else did not, and chunks
        # someone rated that this annotator did not.
        "not_shared_by_all": {
            n: sorted(key_sets[n] - shared, key=_natural_key) for n in annotators
        },
        "missing_from": {
            n: sorted(union - key_sets[n], key=_natural_key) for n in annotators
        },
        "missing_values": {
            n: {dim: int(np.isnan(matrices[dim][:, j]).sum()) for dim in dims}
            for j, n in enumerate(annotators)
        },
    }
    return RatingTable(annotators, keys, matrices, meta), diagnostics


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------


def safe_kappa(y1, y2, labels, weights=None) -> float | None:
    """Cohen's kappa; 1.0 when both raters give the same constant label."""
    with np.errstate(divide="ignore", invalid="ignore"):
        k = cohen_kappa_score(y1, y2, labels=labels, weights=weights)
    if np.isnan(k):
        if len(y1) > 0 and np.array_equal(np.asarray(y1), np.asarray(y2)):
            return 1.0
        return None
    return float(k)


def krippendorff_alpha(
    data: np.ndarray,
    level: str = "ordinal",
    value_domain: list[int] | None = None,
) -> float | None:
    """Krippendorff's alpha for a units x raters matrix (NaN = missing).

    ``alpha = 1 - D_o / D_e`` computed from the coincidence matrix. ``level``
    is ``"nominal"`` or ``"ordinal"``. Returns ``None`` when alpha is undefined
    (no pairable values, or no variation at all so D_e = 0).
    """
    if level not in {"nominal", "ordinal"}:
        raise ValueError(f"unsupported level {level!r}")
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a 2D units x raters matrix")

    if value_domain is None:
        observed = data[~np.isnan(data)]
        domain = sorted({int(v) for v in observed})
    else:
        domain = list(value_domain)
    if not domain:
        return None
    index = {value: i for i, value in enumerate(domain)}
    n_values = len(domain)

    # Coincidence matrix: each unit with m >= 2 values contributes
    # (outer(counts, counts) - diag(counts)) / (m - 1).
    coincidence = np.zeros((n_values, n_values))
    for row in data:
        values = [index[int(v)] for v in row[~np.isnan(row)]]
        m = len(values)
        if m < 2:
            continue
        counts = np.bincount(values, minlength=n_values).astype(float)
        coincidence += (np.outer(counts, counts) - np.diag(counts)) / (m - 1)

    n_c = coincidence.sum(axis=1)
    n_total = n_c.sum()
    if n_total <= 1:
        return None

    # Squared difference function delta^2(c, k).
    if level == "nominal":
        delta = 1.0 - np.eye(n_values)
    else:
        # Ordinal: (sum_{g=c..k} n_g - (n_c + n_k) / 2)^2
        cumulative = np.concatenate([[0.0], np.cumsum(n_c)])
        lo = np.minimum.outer(np.arange(n_values), np.arange(n_values))
        hi = np.maximum.outer(np.arange(n_values), np.arange(n_values))
        between = cumulative[hi + 1] - cumulative[lo]
        delta = (between - (n_c[:, None] + n_c[None, :]) / 2.0) ** 2

    d_observed = float((coincidence * delta).sum() / n_total)
    d_expected = float((np.outer(n_c, n_c) * delta).sum() / (n_total * (n_total - 1)))
    if np.isclose(d_expected, 0.0):
        # Every rating is the same value: alpha is mathematically undefined.
        return None
    return float(1.0 - d_observed / d_expected)


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _alpha_level(cfg: dict) -> str:
    return "ordinal" if cfg["kind"] == "ordinal" else "nominal"


def compute_pairwise_metrics(table: RatingTable, dim: str, cfg: dict) -> dict:
    """Agreement metrics for every annotator pair on one dimension."""
    levels = cfg["levels"]
    ordinal = cfg["kind"] == "ordinal"
    results: dict[str, dict | None] = {}

    for a, b in combinations(table.annotators, 2):
        col_a, col_b = table.column(dim, a), table.column(dim, b)
        both = ~np.isnan(col_a) & ~np.isnan(col_b)
        if not both.any():
            results[f"{a}|{b}"] = None
            continue
        y1 = col_a[both].astype(int)
        y2 = col_b[both].astype(int)
        diff = y1 - y2  # positive -> a rates higher than b
        results[f"{a}|{b}"] = {
            "rater_a": a,
            "rater_b": b,
            "n": int(both.sum()),
            "exact_agreement": float(np.mean(diff == 0)),
            "kappa": safe_kappa(
                y1, y2, labels=levels, weights="quadratic" if ordinal else None
            ),
            "kappa_type": "quadratic_weighted" if ordinal else "unweighted",
            "alpha": krippendorff_alpha(
                np.column_stack([y1, y2]), _alpha_level(cfg), levels
            ),
            "mean_diff_a_minus_b": float(np.mean(diff)),
            "mean_abs_diff": float(np.mean(np.abs(diff))),
            "pct_within_1": float(np.mean(np.abs(diff) <= 1)) if ordinal else None,
            "labels": levels,
            # rows = rater_a, columns = rater_b
            "confusion_matrix": confusion_matrix(y1, y2, labels=levels).tolist(),
        }
    return results


def compute_multirater_metrics(table: RatingTable, dim: str, cfg: dict) -> dict:
    """Krippendorff's alpha across all annotators plus pairwise summaries."""
    matrix = table.matrices[dim]
    alpha = krippendorff_alpha(matrix, _alpha_level(cfg), cfg["levels"])
    pairwise = compute_pairwise_metrics(table, dim, cfg)
    kappas = [e["kappa"] for e in pairwise.values() if e and e["kappa"] is not None]
    n_rated = (~np.isnan(matrix)).sum(axis=1)
    return {
        "n_raters": len(table.annotators),
        "n_items": int(len(table.keys)),
        "n_pairable_items": int((n_rated >= 2).sum()),
        "alpha": alpha,
        "alpha_level": _alpha_level(cfg),
        "mean_pairwise_kappa": float(np.mean(kappas)) if kappas else None,
        "min_pairwise_kappa": float(np.min(kappas)) if kappas else None,
        "max_pairwise_kappa": float(np.max(kappas)) if kappas else None,
        "primary_metric": alpha,
        "primary_metric_name": f"krippendorff_alpha_{_alpha_level(cfg)}",
        "pairwise": pairwise,
    }


def get_reference_pairwise(
    pairwise: dict[str, dict | None], reference_annotator: str
) -> dict[str, dict]:
    """Pairwise results involving the reference annotator."""
    return {
        pair: entry
        for pair, entry in pairwise.items()
        if entry is not None
        and reference_annotator in (entry["rater_a"], entry["rater_b"])
    }


def compute_annotator_bias(table: RatingTable, dim: str) -> dict:
    """Mean level and mean deviation from the per-item mean for each annotator.

    Positive deviation = rates above the other raters (lenient); negative =
    stricter. Only chunks with >= 2 ratings contribute to the deviation. For
    ``A_caus`` the mean level is the share of chunks marked as used.
    """
    matrix = table.matrices[dim]
    rated = ~np.isnan(matrix)
    pairable = rated.sum(axis=1) >= 2
    with np.errstate(invalid="ignore"):
        item_mean = np.nanmean(np.where(pairable[:, None], matrix, np.nan), axis=1)

    result = {}
    for j, name in enumerate(table.annotators):
        own = matrix[:, j]
        levels_seen = own[rated[:, j]]
        dev_mask = rated[:, j] & pairable
        deviations = own[dev_mask] - item_mean[dev_mask]
        result[name] = {
            "n": int(rated[:, j].sum()),
            "mean_level": float(levels_seen.mean()) if levels_seen.size else None,
            "mean_deviation_from_item_mean": (
                float(deviations.mean()) if deviations.size else None
            ),
        }
    return result


def get_worst_disagreements(table: RatingTable, dim: str, n: int = 10) -> list[dict]:
    """Rank chunks by across-rater spread (max - min) for one dimension.

    Ties break on mean pairwise absolute difference, then mean absolute
    deviation from the item mean. For binary ``A_caus`` every split has
    spread 1, so the ordering mainly reflects how even the split is.
    """
    matrix = table.matrices[dim]
    rows = []
    for i, key in enumerate(table.keys):
        present = {
            name: int(matrix[i, j])
            for j, name in enumerate(table.annotators)
            if not np.isnan(matrix[i, j])
        }
        if len(present) < 2:
            continue
        values = np.array(list(present.values()))
        spread = int(values.max() - values.min())
        if spread == 0:
            continue
        distances = [
            (abs(present[a] - present[b]), f"{a}|{b}")
            for a, b in combinations(sorted(present), 2)
        ]
        max_gap = max(d for d, _ in distances)
        rows.append(
            {
                "text_index": key[0],
                "chunk_index": key[1],
                "dimension": dim,
                "spread": spread,
                "mean_abs_deviation": float(np.mean(np.abs(values - values.mean()))),
                "mean_pairwise_abs_diff": float(np.mean([d for d, _ in distances])),
                "max_pairwise_diff": int(max_gap),
                "max_disagreement_pairs": [p for d, p in distances if d == max_gap],
                "ratings": present,
                "party": table.meta[key].get("party"),
                "speaker": table.meta[key].get("speaker"),
            }
        )
    rows.sort(
        key=lambda r: (
            -r["spread"],
            -r["mean_pairwise_abs_diff"],
            -r["mean_abs_deviation"],
            _natural_key((r["text_index"], r["chunk_index"])),
        )
    )
    return rows[:n]


def optional_dimension_coverage(files: list[Path], analysed: list[str]) -> dict:
    """How many chunks each annotator filled for optional, unanalysed dims."""
    extra = [d for d in OPTIONAL_DIMENSIONS if d not in analysed]
    if not extra:
        return {}
    coverage: dict[str, dict[str, int]] = {d: {} for d in extra}
    for path in files:
        items = load_annotations(path, extra)
        for dim in extra:
            coverage[dim][annotator_name(path)] = sum(
                1 for item in items.values() if item.labels[dim] is not None
            )
    return coverage


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
    reference_annotator: str | None,
) -> None:
    annotators = diagnostics["annotators"]
    print("=== Human Inter-Annotator Agreement Report ===")
    print(f"Annotators: {', '.join(annotators)}")
    print(
        f"Items analysed: {diagnostics['analysed_keys']} "
        f"(alignment={diagnostics['alignment']}; shared by all "
        f"{diagnostics['shared_keys']}, union {diagnostics['union_keys']}, "
        f"per annotator {diagnostics['per_annotator_keys']})"
    )
    for name, missing in diagnostics["missing_from"].items():
        if missing:
            preview = ", ".join(f"{t}/{c}" for t, c in missing[:5])
            more = " ..." if len(missing) > 5 else ""
            print(f"  ! {name} did not rate {len(missing)} chunk(s): {preview}{more}")

    for dim, info in per_dim.items():
        multi = info["multirater"]
        print(f"\n--- {dim} ({info['config']['kind']}) ---")
        print(
            f"All {multi['n_raters']} raters: "
            f"Krippendorff alpha ({multi['alpha_level']})={_fmt(multi['alpha'])}, "
            f"mean pairwise kappa={_fmt(multi['mean_pairwise_kappa'])}, "
            f"range {_fmt(multi['min_pairwise_kappa'])}.."
            f"{_fmt(multi['max_pairwise_kappa'])}  "
            f"[n pairable={multi['n_pairable_items']}]"
        )
        if reference_annotator is not None:
            print(f"  {reference_annotator} vs each other annotator:")
            for pair, entry in multi["reference_pairwise"].items():
                print(
                    f"    {pair}: kappa={_fmt(entry['kappa'])}, "
                    f"alpha={_fmt(entry['alpha'])}, n={entry['n']}"
                )
        header = (
            f"  {'pair':<18} {'n':>5} {'agree%':>7} {'kappa':>7} "
            f"{'alpha':>7} {'meanDiff':>9} {'|diff|':>7} {'<=1':>7}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for pair, entry in multi["pairwise"].items():
            if entry is None:
                print(f"  {pair:<18} {'—':>5}  (no overlapping ratings)")
                continue
            within = (
                f"{entry['pct_within_1'] * 100:>6.1f}%"
                if entry["pct_within_1"] is not None
                else "    N/A"
            )
            print(
                f"  {pair:<18} {entry['n']:>5} "
                f"{entry['exact_agreement'] * 100:>6.1f}% "
                f"{_fmt(entry['kappa'], '>7.3f')} {_fmt(entry['alpha'], '>7.3f')} "
                f"{entry['mean_diff_a_minus_b']:>+9.2f} "
                f"{entry['mean_abs_diff']:>7.2f} {within}"
            )

        print("  annotator bias (mean level / mean deviation from item mean):")
        for name, stats in bias[dim].items():
            print(
                f"    {name:<10} n={stats['n']:>4} "
                f"level={_fmt(stats['mean_level'], '>5.2f')} "
                f"dev={_fmt(stats['mean_deviation_from_item_mean'], '+6.2f')}"
            )

    for dim, rows in worst.items():
        if not rows:
            continue
        print(f"\nTop {len(rows)} disagreements — {dim} (spread = max-min):")
        for row in rows:
            ratings = ", ".join(f"{n}={row['ratings'].get(n, '-')}" for n in annotators)
            print(
                f"  text {row['text_index']:>4} chunk {row['chunk_index']:>2} "
                f"spread={row['spread']} [{ratings}] "
                f"max pair gap={row['max_pairwise_diff']} "
                f"({', '.join(row['max_disagreement_pairs'])}) "
                f"({row['party']}, {row['speaker']})"
            )


def print_summary(per_dim: dict[str, dict]) -> dict:
    """Print and return the weakest dimension and the weakest pair per dimension.

    Pairs are only compared within a dimension: quadratic weighted kappa (1-5
    scales) and unweighted kappa (binary) are not on a common footing.
    """
    print("\n=== Summary: where agreement is weakest ===")
    summary: dict[str, object] = {"weakest_pair_per_dimension": {}}
    weakest_dim, weakest_val = None, None
    for dim, info in per_dim.items():
        multi = info["multirater"]
        val = multi["primary_metric"]
        print(
            f"  {dim:<8} {multi['primary_metric_name']}={_fmt(val)}  "
            f"mean pairwise kappa={_fmt(multi['mean_pairwise_kappa'])}"
        )
        if val is not None and (weakest_val is None or val < weakest_val):
            weakest_dim, weakest_val = dim, val

        entries = [
            (pair, e["kappa"])
            for pair, e in multi["pairwise"].items()
            if e is not None and e["kappa"] is not None
        ]
        if entries:
            pair, kappa = min(entries, key=lambda x: x[1])
            summary["weakest_pair_per_dimension"][dim] = {"pair": pair, "kappa": kappa}
            print(f"           weakest pair: {pair} (kappa={kappa:.3f})")

    if weakest_dim is not None:
        print(
            f"  => lowest Krippendorff alpha: {weakest_dim} (alpha={weakest_val:.3f})"
        )
        summary["weakest_dimension"] = {"dimension": weakest_dim, "alpha": weakest_val}
    return summary


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare human qualitative annotations (Cohen's kappa + "
        "Krippendorff's alpha) pairwise and across all raters."
    )
    parser.add_argument(
        "--dir",
        default=str(SCRIPT_DIR),
        help="Directory containing the annotation JSONL files.",
    )
    parser.add_argument(
        "--pattern", default=DEFAULT_PATTERN, help="Glob pattern for annotation files."
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Explicit annotation files (overrides --dir/--pattern).",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT), help="Path for the JSON report."
    )
    parser.add_argument(
        "--show-disagreements",
        type=int,
        default=DEFAULT_SHOW_DISAGREEMENTS,
        help="Number of worst-disagreement chunks per dimension.",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        default=list(DIMENSIONS),
        help=f"Dimensions to compare (default: {' '.join(DIMENSIONS)};"
        f" also available: {' '.join(OPTIONAL_DIMENSIONS)}).",
    )
    parser.add_argument(
        "--alignment",
        choices=["shared", "all"],
        default="shared",
        help="'shared': only chunks rated by every annotator "
        "(default). 'all': every chunk; alpha uses chunks with >=2 "
        "ratings, kappa uses each pair's overlap.",
    )
    parser.add_argument(
        "--reference-annotator",
        default="Jannes",
        help="Annotator for explicit pairwise listings "
        "(default: Jannes; pass '' to disable).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    files = (
        [Path(f) for f in args.files]
        if args.files
        else discover_annotation_files(Path(args.dir), args.pattern)
    )

    unknown = [d for d in args.dims if d not in ALL_DIMENSIONS]
    if unknown:
        print(f"Warning: ignoring unknown dimensions: {unknown}", file=sys.stderr)
    dims = [d for d in dict.fromkeys(args.dims) if d in ALL_DIMENSIONS]
    if not dims:
        raise SystemExit("No valid dimensions selected.")

    table, diagnostics = build_table(files, dims, args.alignment)
    if len(table.annotators) < 2:
        raise SystemExit("Need at least two annotators to compare.")
    if not table.keys:
        raise SystemExit("No chunks to compare after alignment.")
    reference = args.reference_annotator or None
    if reference is not None and reference not in table.annotators:
        raise SystemExit(
            f"Reference annotator {reference!r} not found; "
            f"available: {', '.join(table.annotators)}"
        )

    per_dim, bias, worst = {}, {}, {}
    for dim in dims:
        cfg = ALL_DIMENSIONS[dim]
        multirater = compute_multirater_metrics(table, dim, cfg)
        if reference is not None:
            multirater["reference_pairwise"] = get_reference_pairwise(
                multirater["pairwise"], reference
            )
        per_dim[dim] = {"config": cfg, "multirater": multirater}
        bias[dim] = compute_annotator_bias(table, dim)
        worst[dim] = get_worst_disagreements(table, dim, n=args.show_disagreements)

    coverage = optional_dimension_coverage(files, dims)

    print_report(per_dim, bias, worst, diagnostics, reference)
    for dim, counts in coverage.items():
        filled = sum(1 for c in counts.values() if c > 0)
        note = (
            "fewer than two annotators filled it"
            if filled < 2
            else f"add '--dims ... {dim}' to compare"
        )
        print(f"\nOptional field {dim} not analysed ({note}): {counts}")
    summary = print_summary(per_dim)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "files": [str(p) for p in files],
        "dimensions": dims,
        "annotators": table.annotators,
        "alignment": args.alignment,
        "n_items": diagnostics["analysed_keys"],
        "n_shared_items": diagnostics["shared_keys"],
        "n_union_items": diagnostics["union_keys"],
        "n_dropped_items": diagnostics["union_keys"] - diagnostics["analysed_keys"],
        "diagnostics": diagnostics,
        "per_dimension": per_dim,
        "annotator_bias": bias,
        "worst_disagreements": worst,
        "summary": summary,
        "reference_annotator": reference,
        "optional_dimension_coverage": coverage,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=_json_default)
    print(f"\nReport saved to: {output}")


if __name__ == "__main__":
    main()
