#!/usr/bin/env python3
"""Update A_caus from citations in embedded RAG justifications.

By default this script performs a dry run. Use ``--apply`` to atomically
replace the annotation files.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PATTERN = "*_qualitative_annotations.jsonl"
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass
class FileStats:
    records: int = 0
    cited_chunks: int = 0
    changed_values: int = 0
    malformed_lines: int = 0
    missing_context: int = 0
    invalid_chunk_indices: int = 0
    ignored_citations: int = 0


def cited_chunk_indices(justification: str, chunk_count: int) -> tuple[set[int], int]:
    """Return cited zero-based chunk indices and the number of ignored refs."""
    cited: set[int] = set()
    ignored = 0
    for match in CITATION_PATTERN.finditer(justification or ""):
        reference = int(match.group(1))
        if 1 <= reference <= chunk_count:
            cited.add(reference - 1)
        else:
            ignored += 1
    return cited, ignored


def update_record(record: dict[str, Any], stats: FileStats) -> dict[str, Any]:
    """Promote cited chunks to A_caus=1 while preserving all other values."""
    source_context = record.get("source_context")
    rag = source_context.get("rag") if isinstance(source_context, dict) else None
    chunks = rag.get("retrieved_chunks") if isinstance(rag, dict) else None
    justification = rag.get("justification") if isinstance(rag, dict) else None
    annotations = record.get("chunk_annotations")

    if (
        not isinstance(rag, dict)
        or not isinstance(chunks, list)
        or not isinstance(justification, str)
        or not isinstance(annotations, list)
    ):
        stats.missing_context += 1
        return record

    cited, ignored = cited_chunk_indices(justification, len(chunks))
    stats.ignored_citations += ignored
    stats.cited_chunks += len(cited)

    for annotation in annotations:
        if not isinstance(annotation, dict):
            stats.invalid_chunk_indices += 1
            continue
        chunk_index = annotation.get("chunk_index")
        if isinstance(chunk_index, bool) or not isinstance(chunk_index, int):
            stats.invalid_chunk_indices += 1
            continue
        if chunk_index in cited and annotation.get("A_caus") != 1:
            annotation["A_caus"] = 1
            stats.changed_values += 1

    return record


def process_file(path: Path, apply: bool) -> FileStats:
    """Process one JSONL file, optionally replacing it atomically."""
    stats = FileStats()
    output_lines: list[str] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if not line.strip():
                output_lines.append(line)
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.malformed_lines += 1
                output_lines.append(line)
                continue
            if not isinstance(record, dict):
                stats.malformed_lines += 1
                output_lines.append(line)
                continue

            stats.records += 1
            changes_before = stats.changed_values
            update_record(record, stats)
            if stats.changed_values == changes_before:
                output_lines.append(line)
            else:
                updated = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                output_lines.append(updated + "\n")

    if apply and stats.changed_values:
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = handle.name
                handle.writelines(output_lines)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    return stats


def discover_files(directory: Path, pattern: str) -> list[Path]:
    """Return annotation files, excluding temporary and report files."""
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No annotation files matching {pattern!r} in {directory}"
        )
    return files


def print_summary(path: Path, stats: FileStats, apply: bool) -> None:
    mode = "updated" if apply else "dry run"
    print(
        f"{path.name}: {mode}; records={stats.records}, "
        f"cited_chunks={stats.cited_chunks}, changed={stats.changed_values}, "
        f"malformed={stats.malformed_lines}, missing_context={stats.missing_context}, "
        f"invalid_chunk_indices={stats.invalid_chunk_indices}, "
        f"ignored_citations={stats.ignored_citations}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Promote cited retrieved chunks to A_caus=1 in human annotations."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing annotation JSONL files.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob pattern for annotation files.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        help="Explicit files; overrides --dir and --pattern.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Atomically replace files. Without this flag, only report changes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    files = args.files or discover_files(args.dir, args.pattern)
    for path in files:
        print_summary(path, process_file(path, apply=args.apply), args.apply)


if __name__ == "__main__":
    main()
