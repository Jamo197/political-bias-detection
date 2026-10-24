#!/usr/bin/env python3
"""Extract reproducible 2x2 qualitative samples from batch evaluation logs.

Pairs no-RAG and RAG JSONL records on ``text_index`` and applies strict
pass/fail thresholds to ``label_ideology`` errors.

Results are written to ``<output-dir>/<family>/<size>/qualitative_<embedding>_<strategy>*``.

Examples:
    python "RAG Analysis/extract_qualitative_strata.py"                      # all models, e5/simple
    python "RAG Analysis/extract_qualitative_strata.py" --family qwen --size 7b 32b
    python "RAG Analysis/extract_qualitative_strata.py" --embedding e5 bge --strategy simple hyde
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

QUADRANTS = (
    "Q1_RAG_Win",
    "Q2_RAG_Distraction",
    "Q3_Joint_Failure",
    "Q4_Baseline_Sufficiency",
)

# family -> size -> exact ``parameters.llm`` value in the batch logs
MODELS: Dict[str, Dict[str, str]] = {
    "llama": {
        "3b": "meta-llama/Llama-3.2-3B-Instruct",
        "8b": "meta-llama/Llama-3.1-8B-Instruct",
        "70b": "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8",
    },
    "qwen": {
        "3b": "Qwen/Qwen2.5-3B-Instruct",
        "7b": "Qwen/Qwen2.5-7B-Instruct",
        "32b": "Qwen/Qwen2.5-32B-Instruct",
        "72b": "RedHatAI/Qwen2.5-72B-Instruct-FP8-dynamic",
    },
    "mistral": {
        "3b": "mistralai/Ministral-3-3B-Instruct-2512",
        "8b": "mistralai/Ministral-3-8B-Instruct-2512",
        "14b": "mistralai/Ministral-3-14B-Instruct-2512",
    },
}

EMBEDDINGS = ("e5", "qwen3", "bge", "jina")
STRATEGIES = (
    "simple",
    "hyde",
    "twostage",
    "simple_hybrid",
    "hyde_hybrid",
    "twostage_hybrid",
)


@dataclass
class ParsedLog:
    path: Path
    records: Dict[str, Dict[str, Any]]
    duplicate_ids: List[str] = field(default_factory=list)
    malformed_lines: int = 0
    missing_ids: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, Path] = field(default_factory=dict)


@dataclass
class LogInfo:
    path: Path
    metadata: Dict[str, Any]
    mtime: float


def _normalise_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None


def _record_id(record: Dict[str, Any]) -> Optional[str]:
    metadata = record.get("input_metadata") or {}
    return _normalise_id(metadata.get("text_index", record.get("text_index")))


def _record_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    return record.get("parameters") or {}


def _log_metadata(record: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
    params = _record_metadata(record)
    return {
        "llm": params.get("llm"),
        "embedding_model": params.get("embedding_model"),
        "retrieval_mode": params.get("retrieval_mode"),
        "is_rag": params.get("is_rag"),
        "party_label_file": file_path.name.startswith("party_label_"),
    }


def _is_rag(metadata: Dict[str, Any]) -> bool:
    return bool(metadata.get("is_rag")) and metadata.get("retrieval_mode") != "no_rag"


def parse_jsonl(file_path: Path) -> ParsedLog:
    """Parse a JSONL file, retaining the last duplicate for diagnostics."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Log file not found: {file_path}")

    parsed = ParsedLog(path=file_path, records={})
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                parsed.malformed_lines += 1
                print(f"[WARNING] {file_path}:{line_number}: {error}", file=sys.stderr)
                continue
            if not isinstance(record, dict):
                parsed.malformed_lines += 1
                continue

            text_index = _record_id(record)
            if text_index is None:
                parsed.missing_ids += 1
                continue
            if text_index in parsed.records:
                parsed.duplicate_ids.append(text_index)
            parsed.records[text_index] = record

            if not parsed.metadata:
                parsed.metadata = _log_metadata(record, file_path)
    return parsed


@lru_cache(maxsize=None)
def _cached_log(file_path: Path) -> ParsedLog:
    return parse_jsonl(file_path)


def _first_record_metadata(file_path: Path) -> Optional[Dict[str, Any]]:
    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                return _log_metadata(record, file_path)
    return None


def _log_files(batch_root: Path, include_original: bool) -> Iterable[Path]:
    for path in sorted(batch_root.glob("**/*.jsonl")):
        if path.name == "evaluation_logs.jsonl":
            continue
        # k-ablation runs sweep k_chunks over the same texts and are not comparable.
        if "k-ablation" in path.name:
            continue
        if not include_original and not path.name.startswith("party_label_"):
            continue
        yield path


def build_log_index(batch_root: Path, include_original: bool = False) -> List[LogInfo]:
    """Read only the first record of each log so discovery needs no full parse."""
    index: List[LogInfo] = []
    for path in _log_files(batch_root, include_original):
        metadata = _first_record_metadata(path)
        if metadata is not None:
            index.append(LogInfo(path, metadata, path.stat().st_mtime))
    return index


def _model_family(model: Any) -> str:
    value = str(model or "").lower()
    value = (
        value.replace("meta-llama/", "")
        .replace("redhatai/", "")
        .replace("mistralai/", "")
    )
    value = value.replace("instruct-fp8", "instruct")
    return re.sub(r"[^a-z0-9]+", "", value)


def discover_logs(
    index: Sequence[LogInfo],
    *,
    llm: str,
    embedding: str,
    strategy: str,
    rag: bool,
    run_hint: Optional[str] = None,
) -> List[Path]:
    """Select the largest matching log and merge in disjoint partial runs.

    Candidates are ordered by (record count, mtime) descending; a candidate is
    added only if none of its ``text_index`` values are already covered, so a
    complete run wins over its duplicates while split runs are combined.
    """
    kind = "RAG" if rag else "baseline"
    candidates: List[Tuple[int, float, Path, set]] = []
    for info in index:
        if run_hint and run_hint not in str(info.path):
            continue
        params = info.metadata
        if params.get("llm") != llm or _is_rag(params) != rag:
            continue
        if rag and (
            params.get("embedding_model") != embedding
            or params.get("retrieval_mode") != strategy
        ):
            continue
        if not rag and params.get("embedding_model") not in (None, "", "none"):
            continue
        ids = set(_cached_log(info.path).records)
        candidates.append((len(ids), info.mtime, info.path, ids))

    if not candidates:
        raise FileNotFoundError(
            f"No {kind} log found for llm={llm!r}"
            + (f", embedding={embedding!r}, strategy={strategy!r}" if rag else "")
        )
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected: List[Path] = []
    covered: set = set()
    skipped: List[Path] = []
    for count, _, path, ids in candidates:
        if ids & covered:
            skipped.append(path)
            continue
        selected.append(path)
        covered |= ids
        print(f"[*] Selected {kind} log ({count} records): {path}")
    if skipped:
        print(
            f"[WARNING] Ignored {len(skipped)} overlapping {kind} log(s):\n    "
            + "\n    ".join(str(path) for path in skipped),
            file=sys.stderr,
        )
    return selected


def _merge_logs(paths: Sequence[Path]) -> ParsedLog:
    """Combine logs of disjoint runs, remembering each record's source file."""
    logs = [_cached_log(path) for path in paths]
    llms = {log.metadata.get("llm") for log in logs}
    if len(llms) > 1:
        raise ValueError(f"Cannot merge logs of different models: {sorted(map(str, llms))}")

    merged = ParsedLog(path=logs[0].path, records={}, metadata=dict(logs[0].metadata))
    for log in logs:
        for text_index, record in log.records.items():
            if text_index in merged.records:
                merged.duplicate_ids.append(text_index)
            merged.records[text_index] = record
            merged.sources[text_index] = log.path
        merged.duplicate_ids.extend(log.duplicate_ids)
        merged.malformed_lines += log.malformed_lines
        merged.missing_ids += log.missing_ids
    return merged


def extract_scalar_prediction(
    record: Dict[str, Any], target_key: str = "label_ideology"
) -> Tuple[Optional[float], Optional[float], str]:
    output = record.get("output") or {}
    ground_truth = record.get("ground_truth") or {}
    raw_pred = (
        output.get("bias")
        if output.get("bias") is not None
        else output.get("prediction")
    )
    prediction = _number(raw_pred)
    target = _number(ground_truth.get(target_key))
    return prediction, target, str(output.get("justification") or "")


def classify_quadrant(
    e_base: float,
    e_rag: float,
    correct_thresh: float = 0.5,
    incorrect_thresh: float = 1.0,
) -> Optional[str]:
    """Return a strict quadrant; the intermediate buffer [correct_thresh, incorrect_thresh) returns None."""
    base_pass = e_base <= correct_thresh
    base_fail = e_base >= incorrect_thresh
    rag_pass = e_rag <= correct_thresh
    rag_fail = e_rag >= incorrect_thresh

    if base_fail and rag_pass:
        return "Q1_RAG_Win"
    if base_pass and rag_fail:
        return "Q2_RAG_Distraction"
    if base_fail and rag_fail:
        return "Q3_Joint_Failure"
    if base_pass and rag_pass:
        return "Q4_Baseline_Sufficiency"
    return None


def _sort_id(value: str) -> Tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _sample_row(
    text_index: str,
    baseline: Dict[str, Any],
    rag: Dict[str, Any],
    target: float,
    quadrant: str,
    baseline_path: Path,
    rag_path: Path,
    pass_threshold: float,
    fail_threshold: float,
) -> Dict[str, Any]:
    baseline_params = baseline.get("parameters") or {}
    rag_params = rag.get("parameters") or {}
    baseline_output = baseline.get("output") or {}
    rag_output = rag.get("output") or {}

    base_prediction = _number(
        baseline_output.get("bias")
        if baseline_output.get("bias") is not None
        else baseline_output.get("prediction")
    )
    rag_prediction = _number(
        rag_output.get("bias")
        if rag_output.get("bias") is not None
        else rag_output.get("prediction")
    )

    base_error = abs(base_prediction - target)
    rag_error = abs(rag_prediction - target)
    base_residual = base_prediction - target
    rag_residual = rag_prediction - target
    directional_shift = rag_prediction - base_prediction

    return {
        "text_index": text_index,
        "quadrant": quadrant,
        "thresholds": {"pass": pass_threshold, "fail": fail_threshold},
        "input_metadata": rag.get("input_metadata")
        or baseline.get("input_metadata")
        or {},
        "input_text": (rag.get("inputs") or {}).get(
            "text", (baseline.get("inputs") or {}).get("text", "")
        ),
        "ground_truth": {"label_ideology": target},
        "metrics": {
            "base_prediction": base_prediction,
            "rag_prediction": rag_prediction,
            "base_error": round(base_error, 6),
            "rag_error": round(rag_error, 6),
            "error_reduction": round(base_error - rag_error, 6),
            "base_residual": round(base_residual, 6),
            "rag_residual": round(rag_residual, 6),
            "directional_shift": round(directional_shift, 6),
        },
        "baseline": {
            "source_path": str(baseline_path),
            "run_id": baseline.get("run_id"),
            "parameters": baseline_params,
            "justification": baseline_output.get("justification") or "",
        },
        "rag": {
            "source_path": str(rag_path),
            "run_id": rag.get("run_id"),
            "parameters": rag_params,
            "justification": rag_output.get("justification") or "",
            "retrieved_chunks": (rag.get("inputs") or {}).get("retrieved_chunks", []),
            "hyde_docs": (rag.get("inputs") or {}).get("hyde_docs", []),
        },
        "qualitative_annotation": {
            "R_top": None,
            "R_ideo": None,
            "N_info": None,
            "A_caus": None,
            "error_typology": None,
            "annotator_notes": "",
        },
    }


def build_stratified_dataset(
    baseline_paths: Sequence[Path],
    rag_paths: Sequence[Path],
    samples_per_quadrant: int = 25,
    correct_thresh: float = 0.5,
    incorrect_thresh: float = 1.0,
    target_key: str = "label_ideology",
    seed: int = 42,
    diagnostics: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Inner-join two logs, classify strict quadrants, and sample reproducibly."""
    if samples_per_quadrant < 1:
        raise ValueError("samples_per_quadrant must be positive")
    if correct_thresh >= incorrect_thresh:
        raise ValueError("pass threshold must be strictly smaller than fail threshold")

    baseline_log = _merge_logs(baseline_paths)
    rag_log = _merge_logs(rag_paths)
    baseline_model = baseline_log.metadata.get("llm")
    rag_model = rag_log.metadata.get("llm")

    if _model_family(baseline_model) != _model_family(rag_model):
        raise ValueError(
            f"Baseline ({baseline_model!r}) and RAG ({rag_model!r}) logs use different base models."
        )

    baseline_ids = set(baseline_log.records)
    rag_ids = set(rag_log.records)
    common_ids = sorted(baseline_ids & rag_ids, key=_sort_id)

    stats: Dict[str, Any] = {
        "baseline_records": len(baseline_ids),
        "rag_records": len(rag_ids),
        "inner_join_records": len(common_ids),
        "baseline_only_count": len(baseline_ids - rag_ids),
        "rag_only_count": len(rag_ids - baseline_ids),
        "baseline_duplicates": len(baseline_log.duplicate_ids),
        "rag_duplicates": len(rag_log.duplicate_ids),
        "malformed_lines": baseline_log.malformed_lines + rag_log.malformed_lines,
        "missing_text_index_lines": baseline_log.missing_ids + rag_log.missing_ids,
        "ground_truth_mismatches": 0,
        "invalid_value_ids": 0,
        "buffer_count": 0,
    }
    buckets: Dict[str, List[Dict[str, Any]]] = {quadrant: [] for quadrant in QUADRANTS}

    for text_index in common_ids:
        baseline = baseline_log.records[text_index]
        rag = rag_log.records[text_index]
        base_prediction, base_target, _ = extract_scalar_prediction(
            baseline, target_key
        )
        rag_prediction, rag_target, _ = extract_scalar_prediction(rag, target_key)

        if None in (base_prediction, base_target, rag_prediction, rag_target):
            stats["invalid_value_ids"] += 1
            continue
        if abs(base_target - rag_target) > 1e-6:
            stats["ground_truth_mismatches"] += 1
            continue

        base_error = abs(base_prediction - base_target)
        rag_error = abs(rag_prediction - rag_target)
        quadrant = classify_quadrant(
            base_error, rag_error, correct_thresh, incorrect_thresh
        )

        if quadrant is None:
            stats["buffer_count"] += 1
            continue

        buckets[quadrant].append(
            _sample_row(
                text_index,
                baseline,
                rag,
                base_target,
                quadrant,
                baseline_log.sources[text_index],
                rag_log.sources[text_index],
                correct_thresh,
                incorrect_thresh,
            )
        )

    population_counts = {quadrant: len(items) for quadrant, items in buckets.items()}
    rng = random.Random(seed)
    sampled: List[Dict[str, Any]] = []
    selected_counts: Dict[str, int] = {}

    for quadrant in QUADRANTS:
        pop_size = len(buckets[quadrant])
        k = min(samples_per_quadrant, pop_size)
        if pop_size < samples_per_quadrant:
            print(
                f"[WARNING] Quadrant '{quadrant}' has only {pop_size} eligible instances (requested {samples_per_quadrant}). "
                f"Sampling all {pop_size} instances.",
                file=sys.stderr,
            )
        chosen = rng.sample(buckets[quadrant], k)
        selected_counts[quadrant] = len(chosen)
        sampled.extend(chosen)

    sampled.sort(
        key=lambda row: (QUADRANTS.index(row["quadrant"]), _sort_id(row["text_index"]))
    )
    stats["population_counts"] = population_counts
    stats["selected_counts"] = selected_counts
    stats["valid_paired_records"] = (
        sum(population_counts.values()) + stats["buffer_count"]
    )

    if diagnostics is not None:
        diagnostics.update(stats)
    return sampled, population_counts


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _write_summary(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = [
        "text_index",
        "quadrant",
        "ground_truth",
        "base_prediction",
        "rag_prediction",
        "base_error",
        "rag_error",
        "error_reduction",
        "directional_shift",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            m = row["metrics"]
            writer.writerow(
                {
                    "text_index": row["text_index"],
                    "quadrant": row["quadrant"],
                    "ground_truth": row["ground_truth"]["label_ideology"],
                    "base_prediction": m["base_prediction"],
                    "rag_prediction": m["rag_prediction"],
                    "base_error": m["base_error"],
                    "rag_error": m["rag_error"],
                    "error_reduction": m["error_reduction"],
                    "directional_shift": m["directional_shift"],
                }
            )


def _export(
    args: argparse.Namespace,
    baseline_paths: List[Path],
    rag_paths: List[Path],
    output_dir: Path,
    prefix: str,
    extra: Dict[str, Any],
) -> None:
    diagnostics: Dict[str, Any] = {
        **extra,
        "baseline_paths": [str(path) for path in baseline_paths],
        "rag_paths": [str(path) for path in rag_paths],
        "seed": args.seed,
        "samples_per_quadrant": args.k,
        "target": args.target,
        "thresholds": {"pass": args.pass_threshold, "fail": args.fail_threshold},
    }
    rows, _ = build_stratified_dataset(
        baseline_paths,
        rag_paths,
        args.k,
        args.pass_threshold,
        args.fail_threshold,
        args.target,
        args.seed,
        diagnostics,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / f"{prefix}.jsonl", rows)
    _write_summary(output_dir / f"{prefix}_summary.csv", rows)
    (output_dir / f"{prefix}_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[+] Exported {len(rows)} samples to {output_dir / (prefix + '.jsonl')}")
    for quadrant in QUADRANTS:
        print(
            f"    {quadrant}: {diagnostics['selected_counts'][quadrant]}/{diagnostics['population_counts'][quadrant]}"
        )


def _run_model(
    args: argparse.Namespace,
    index: Sequence[LogInfo],
    family: str,
    size: str,
    embedding: str,
    strategy: str,
) -> None:
    llm = MODELS[family][size]
    print(f"\n=== {family}/{size} ({llm}) | {embedding}/{strategy} ===")
    baseline_paths = discover_logs(
        index,
        llm=llm,
        embedding=embedding,
        strategy=strategy,
        rag=False,
        run_hint=args.baseline_run_hint,
    )
    rag_paths = discover_logs(
        index,
        llm=llm,
        embedding=embedding,
        strategy=strategy,
        rag=True,
        run_hint=args.rag_run_hint,
    )
    _export(
        args,
        baseline_paths,
        rag_paths,
        args.output_dir / family / size,
        f"qualitative_{embedding}_{strategy}",
        {
            "family": family,
            "size": size,
            "llm": llm,
            "embedding": embedding,
            "strategy": strategy,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--family",
        choices=[*MODELS, "all"],
        default="all",
        help="Model family to process (default: all).",
    )
    parser.add_argument(
        "--size",
        nargs="+",
        type=str.lower,
        help="Restrict to these sizes, e.g. --size 3b 8b (default: all sizes).",
    )
    parser.add_argument("--batch-root", type=Path, default=Path("logs/batch_runs"))
    parser.add_argument("--baseline", type=Path, help="Explicit baseline JSONL path.")
    parser.add_argument("--rag", type=Path, help="Explicit RAG JSONL path.")
    parser.add_argument(
        "--embedding", nargs="+", default=["e5"], choices=EMBEDDINGS
    )
    parser.add_argument("--strategy", nargs="+", default=["simple"], choices=STRATEGIES)
    parser.add_argument("--k", type=int, default=25, help="Samples per quadrant.")
    parser.add_argument("--pass-threshold", type=float, default=0.5)
    parser.add_argument("--fail-threshold", type=float, default=1.0)
    parser.add_argument("--target", default="label_ideology")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qualitative"))
    parser.add_argument(
        "--output-prefix",
        default="qualitative_explicit",
        help="Prefix for explicit-path output files.",
    )
    parser.add_argument(
        "--baseline-run-hint", help="Substring used to select a baseline run directory."
    )
    parser.add_argument(
        "--rag-run-hint", help="Substring used to select a RAG run directory."
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Also consider non-party_label JSONL files.",
    )
    args = parser.parse_args()

    if bool(args.baseline) != bool(args.rag):
        parser.error("--baseline and --rag must be supplied together")
    if args.baseline and args.rag:
        _export(
            args,
            [args.baseline],
            [args.rag],
            args.output_dir,
            args.output_prefix,
            {"mode": "explicit"},
        )
        return

    families = list(MODELS) if args.family == "all" else [args.family]
    if args.size:
        known = {size for family in families for size in MODELS[family]}
        unknown = sorted(set(args.size) - known)
        if unknown:
            parser.error(
                f"unknown size(s) {unknown} for {args.family}; available: {sorted(known)}"
            )
    jobs = [
        (family, size, embedding, strategy)
        for family in families
        for size in MODELS[family]
        if not args.size or size in args.size
        for embedding in args.embedding
        for strategy in args.strategy
    ]

    index = build_log_index(args.batch_root, args.include_original)
    print(f"[*] Indexed {len(index)} logs under {args.batch_root}")
    exported: List[str] = []
    skipped: List[str] = []
    for family, size, embedding, strategy in jobs:
        label = f"{family}/{size} {embedding}/{strategy}"
        try:
            _run_model(args, index, family, size, embedding, strategy)
        except FileNotFoundError as error:
            print(f"[WARNING] Skipping {label}: {error}", file=sys.stderr)
            skipped.append(f"{label} ({error})")
            continue
        exported.append(label)

    print(f"\n[*] Exported {len(exported)}/{len(jobs)} configurations")
    for label in exported:
        print(f"    + {label}")
    for label in skipped:
        print(f"    - {label}")


if __name__ == "__main__":
    main()
