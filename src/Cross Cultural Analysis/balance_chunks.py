"""Give every country the same chunk budget before embedding (RQ2 knowledge base).

Countries differ a lot in parliamentary output (Poland >> Latvia), and a bigger
corpus is a better retrieval target on its own. To keep the knowledge base size
equal across countries, this script samples WHOLE speeches per ``country_code``
(seeded shuffle) until the country reaches the budget, then writes the filtered
``chunks`` and ``full_speeches`` files for the ingest jobs.

Budget: the smallest country's chunk count, or ``--budget N`` (countries below
N keep everything and are reported).

Usage (from the repo root)::

    python "src/Cross Cultural Analysis/balance_chunks.py" \\
        --artifact-dir rag/ingest/artifacts/parlamint
    # -> chunks_balanced.jsonl + full_speeches_balanced.jsonl, then ingest with
    #    CHUNKS_FILE=.../chunks_balanced.jsonl SPEECHES_FILE=.../full_speeches_balanced.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def chunk_counts(chunks_path: Path, require_ches: bool) -> dict[str, dict[str, int]]:
    """``country_code -> {speech_id: number of chunks}`` (one streaming pass)."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in _read_jsonl(chunks_path):
        if require_ches and rec.get("ches_party_id") is None:
            continue
        counts[rec.get("country_code") or "unknown"][rec["speech_id"]] += 1
    return {code: dict(speeches) for code, speeches in counts.items()}


def select_speeches(
    counts: dict[str, dict[str, int]], budget: int | None, seed: int
) -> tuple[set[str], dict[str, int], int]:
    """Seeded sample of whole speeches per country up to the chunk budget."""
    totals = {code: sum(s.values()) for code, s in counts.items()}
    target = budget if budget is not None else min(totals.values())
    keep: set[str] = set()
    kept: dict[str, int] = {}
    for code in sorted(counts):
        speech_ids = sorted(counts[code])
        random.Random(f"{seed}:{code}").shuffle(speech_ids)
        n = 0
        for speech_id in speech_ids:
            if n >= target:
                break
            keep.add(speech_id)
            n += counts[code][speech_id]
        kept[code] = n
    return keep, kept, target


def write_filtered(src: Path, dst: Path, keep: set[str]) -> int:
    written = 0
    with dst.open("w", encoding="utf-8") as out:
        for rec in _read_jsonl(src):
            if rec.get("speech_id") in keep:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("rag/ingest/artifacts/parlamint"))
    parser.add_argument("--chunks", type=Path, default=None, help="Default: <artifact-dir>/chunks.jsonl")
    parser.add_argument("--speeches", type=Path, default=None, help="Default: <artifact-dir>/full_speeches.jsonl")
    parser.add_argument("--suffix", default="_balanced", help="Output name suffix.")
    parser.add_argument("--budget", type=int, default=None, help="Chunks per country (default: smallest country).")
    parser.add_argument("--require-ches", action="store_true", help="Drop chunks without a CHES party id first.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    chunks = args.chunks or args.artifact_dir / "chunks.jsonl"
    speeches = args.speeches or args.artifact_dir / "full_speeches.jsonl"
    counts = chunk_counts(chunks, args.require_ches)
    keep, kept, target = select_speeches(counts, args.budget, args.seed)

    print(f"Chunk budget per country: {target}")
    for code in sorted(counts):
        total = sum(counts[code].values())
        flag = "  (below budget: kept all)" if total < target else ""
        print(f"  {code}: {kept[code]:>8d} of {total:>8d} chunks{flag}")

    chunks_out = chunks.with_name(chunks.stem + args.suffix + chunks.suffix)
    speeches_out = speeches.with_name(speeches.stem + args.suffix + speeches.suffix)
    n_chunks = write_filtered(chunks, chunks_out, keep)
    n_speeches = write_filtered(speeches, speeches_out, keep) if speeches.exists() else 0
    print(f"Wrote {n_chunks} chunks -> {chunks_out}")
    print(f"Wrote {n_speeches} speeches -> {speeches_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
