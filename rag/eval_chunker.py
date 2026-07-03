"""Evaluate and compare chunking configurations.

Metrics:
1. Chunk size distribution -- descriptive stats + ASCII histogram
2. Fallback split rate -- % of chunks from RecursiveCharacterTextSplitter
3. Boundary coherence -- mean cosine drop between adjacent chunks,
   measured with a reference embedding model independent of the
   chunker's breakpoint model

Usage:
python -m rag.eval_chunker \
    --chunks A=tmp/config_A_chunks.jsonl \
    --chunks B=tmp/config_B_chunks.jsonl \
    --chunks C=tmp/config_C_chunks.jsonl \
    --chunks D=tmp/config_D_chunks.jsonl \
    --chunks F=tmp/config_F_chunks.jsonl \
    --batch-size 128 \
    --output tmp/report.json

If from_fallback is missing from chunks (older chunker runs), it is
reconstructed automatically from start_index overlap detection.

Add --no-coherence to skip Metric 3 (no embedding model needed).
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from collections import defaultdict
from pathlib import Path

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

_DEFAULT_REF = "sentence-transformers/LaBSE"


# ── loading ──────────────────────────────────────────────────────


def _load_chunks(path: Path) -> list[dict]:
    return [
        json.loads(l)
        for l in path.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]


# ── Metric 1: size distribution ──────────────────────────────────


def size_distribution(chunks: list[dict]) -> dict:
    sizes = [len(c["text"]) for c in chunks]
    if not sizes:
        return {"count": 0}
    return {
        "count": len(sizes),
        "min": min(sizes),
        "max": max(sizes),
        "mean": stats.mean(sizes),
        "median": stats.median(sizes),
        "stdev": stats.stdev(sizes) if len(sizes) > 1 else 0.0,
        "p25": float(np.percentile(sizes, 25)),
        "p75": float(np.percentile(sizes, 75)),
        "p90": float(np.percentile(sizes, 90)),
        "p95": float(np.percentile(sizes, 95)),
        "sizes": sizes,
    }


def _ascii_histogram(sizes: list[int], n_buckets: int = 20, width: int = 40) -> str:
    lo, hi = min(sizes), max(sizes)
    if hi == lo:
        return f"  {lo:>5} | {'#' * width} ({len(sizes)})"
    bw = (hi - lo) / n_buckets
    counts = [0] * n_buckets
    for s in sizes:
        idx = min(int((s - lo) / bw), n_buckets - 1)
        counts[idx] += 1
    mx = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        lo_b = int(lo + i * bw)
        hi_b = int(lo + (i + 1) * bw)
        bar = "#" * int(c / mx * width)
        lines.append(f"  {lo_b:>5}-{hi_b:>5} | {bar} ({c})")
    return "\n".join(lines)


# ── Metric 2: fallback split rate ────────────────────────────────


def _reconstruct_fallback(chunks: list[dict]) -> None:
    """Infer from_fallback from start_index overlap (in-place).

    RecursiveCharacterTextSplitter produces chunks that overlap by
    chunk_overlap chars, so start_index[i+1] < start_index[i] + len(text[i]).
    SemanticChunker never overlaps (start_index advances by exact chunk
    length). Any chunk involved in an overlap pair is a fallback chunk.
    """
    by_speech: dict[str, list[int]] = defaultdict(list)
    for i, c in enumerate(chunks):
        by_speech[c["speech_id"]].append(i)

    for indices in by_speech.values():
        for k in range(len(indices) - 1):
            ci, cj = chunks[indices[k]], chunks[indices[k + 1]]
            end_i = ci["start_index"] + len(ci["text"])
            if cj["start_index"] < end_i:
                ci["from_fallback"] = True
                cj["from_fallback"] = True
            else:
                ci.setdefault("from_fallback", False)
                cj.setdefault("from_fallback", False)
        # Last chunk if not already set
        chunks[indices[-1]].setdefault("from_fallback", False)


def fallback_rate(chunks: list[dict]) -> dict:
    total = len(chunks)
    had_field = any("from_fallback" in c for c in chunks)
    if not had_field:
        _reconstruct_fallback(chunks)
    fb = sum(1 for c in chunks if c.get("from_fallback", False))
    sem = total - fb
    sp_fb = len({c["speech_id"] for c in chunks if c.get("from_fallback", False)})
    sp_tot = len({c["speech_id"] for c in chunks})
    return {
        "available": True,
        "reconstructed": not had_field,
        "total_chunks": total,
        "fallback_chunks": fb,
        "fallback_pct": fb / total * 100 if total else 0.0,
        "semantic_chunks": sem,
        "semantic_pct": sem / total * 100 if total else 0.0,
        "speeches_with_fallback": sp_fb,
        "total_speeches": sp_tot,
    }


# ── Metric 3: boundary coherence ─────────────────────────────────


def boundary_coherence(
    chunks: list[dict], embedder: HuggingFaceEmbeddings, batch_size: int
) -> dict:
    # Build (speech_id, chunk_index) -> position index
    pos: dict[tuple[str, int], int] = {}
    for i, c in enumerate(chunks):
        pos[(c["speech_id"], c["chunk_index"])] = i

    # Group chunk indices by speech
    by_speech: dict[str, list[int]] = defaultdict(list)
    for c in chunks:
        by_speech[c["speech_id"]].append(c["chunk_index"])
    for sid in by_speech:
        by_speech[sid].sort()

    # Collect adjacent pairs as (pos_i, pos_j)
    pairs: list[tuple[int, int]] = []
    for sid, indices in by_speech.items():
        for k in range(len(indices) - 1):
            pairs.append((pos[(sid, indices[k])], pos[(sid, indices[k + 1])]))

    if not pairs:
        return {"n_pairs": 0, "note": "No adjacent chunk pairs found."}

    # Embed all chunk texts in order
    texts = [c["text"] for c in chunks]
    all_embs: list[list[float]] = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[start : start + batch_size]
        all_embs.extend(embedder.embed_documents(batch))

    emb_arr = np.asarray(all_embs, dtype=np.float32)

    # Cosine drop for each adjacent pair
    drops: list[float] = []
    for i, j in pairs:
        a, b = emb_arr[i], emb_arr[j]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        drops.append(1.0 - cos)

    return {
        "n_pairs": len(drops),
        "mean_drop": float(np.mean(drops)),
        "median_drop": float(np.median(drops)),
        "std_drop": float(np.std(drops)),
        "min_drop": float(np.min(drops)),
        "max_drop": float(np.max(drops)),
        "p25": float(np.percentile(drops, 25)),
        "p75": float(np.percentile(drops, 75)),
    }


# ── report formatting ────────────────────────────────────────────

_SIZE_FIELDS = [
    "count",
    "min",
    "max",
    "mean",
    "median",
    "stdev",
    "p25",
    "p75",
    "p90",
    "p95",
]

_FB_FIELDS = [
    "total_chunks",
    "fallback_chunks",
    "fallback_pct",
    "semantic_chunks",
    "semantic_pct",
    "speeches_with_fallback",
    "total_speeches",
]


def _print_fallback_note(results: dict, labels: list[str]) -> None:
    reconstructed = [l for l in labels if results[l]["fallback"].get("reconstructed")]
    if reconstructed:
        print(
            f"\n  NOTE: from_fallback reconstructed from start_index overlap for: {', '.join(reconstructed)}"
        )
        print(
            f"        (RecursiveCharacterTextSplitter uses 150-char overlap; semantic chunks don't)"
        )


_COH_FIELDS = [
    "n_pairs",
    "mean_drop",
    "median_drop",
    "std_drop",
    "min_drop",
    "max_drop",
    "p25",
    "p75",
]


def _print_table(
    title: str,
    labels: list[str],
    rows: list[tuple[str, dict]],
    pct_fields: set[str] | None = None,
) -> None:
    pct_fields = pct_fields or set()
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}\n")
    col_w = 12
    header = f"  {'metric':>20}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(header)
    for field_name, field_map in rows:
        row = f"  {field_name:>20}"
        for l in labels:
            v = field_map.get(l, "-")
            if v == "-":
                row += f"{'n/a':>{col_w}}"
            elif field_name in pct_fields and isinstance(v, (int, float)):
                row += f"{v:>{col_w}.1f}%"
            elif isinstance(v, float):
                row += f"{v:>{col_w}.2f}"
            else:
                row += f"{v:>{col_w}}"
        print(row)


def print_report(results: dict, reference_model: str) -> None:
    labels = list(results.keys())

    print("\n" + "=" * 60)
    print("  CHUNKING CONFIGURATION COMPARISON")
    print("=" * 60)

    # Metric 1
    size_maps: list[tuple[str, dict]] = []
    for f in _SIZE_FIELDS:
        m = {}
        for l in labels:
            v = results[l]["size"].get(f)
            if isinstance(v, float):
                m[l] = round(v, 1)
            else:
                m[l] = v
        size_maps.append((f, m))
    _print_table("Metric 1: Chunk Size Distribution (chars)", labels, size_maps)

    # Histograms
    for l in labels:
        print(f"\n  Histogram [{l}]:")
        print(results[l]["histogram"])

    # Metric 2
    fb_maps: list[tuple[str, dict]] = []
    any_fb = any(results[l]["fallback"].get("available") for l in labels)
    if not any_fb:
        print(f"\n{'─' * 60}")
        print("  Metric 2: Fallback Split Rate")
        print(f"{'─' * 60}\n")
        print("  (from_fallback field not found -- re-run chunker with updated code)")
    else:
        for f in _FB_FIELDS:
            m = {}
            for l in labels:
                fb = results[l]["fallback"]
                if not fb.get("available"):
                    m[l] = "-"
                else:
                    m[l] = fb.get(f)
            fb_maps.append((f, m))
        _print_table(
            "Metric 2: Fallback Split Rate",
            labels,
            fb_maps,
            pct_fields={"fallback_pct", "semantic_pct"},
        )
        _print_fallback_note(results, labels)

    # Metric 3
    coh_maps: list[tuple[str, dict]] = []
    any_coh = any(results[l].get("coherence", {}).get("n_pairs", 0) > 0 for l in labels)
    print(f"\n{'─' * 60}")
    print(f"  Metric 3: Boundary Coherence (cosine drop)")
    print(f"  reference model: {reference_model}")
    print(f"{'─' * 60}\n")
    if not any_coh:
        print("  (skipped or no adjacent pairs)")
    else:
        header = f"  {'metric':>20}" + "".join(f"{l:>12}" for l in labels)
        print(header)
        for f in _COH_FIELDS:
            row = f"  {f:>20}"
            for l in labels:
                coh = results[l].get("coherence", {})
                v = coh.get(f, "-")
                if isinstance(v, float):
                    row += f"{v:>12.4f}"
                else:
                    row += f"{v:>12}"
            print(row)
        print(f"\n  Higher mean_drop = clearer semantic transitions at boundaries.")
        print(f"  NOTE: If a config uses the SAME model as the reference for")
        print(f"        chunking, its coherence is biased upward (self-agreement).")


# ── main ─────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare chunking configurations across 3 metrics."
    )
    parser.add_argument(
        "--chunks",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="A chunked JSONL file. Repeatable: --chunks A=a.jsonl --chunks B=b.jsonl",
    )
    parser.add_argument(
        "--reference-model",
        default=_DEFAULT_REF,
        help=f"Embedding model for boundary coherence (default: {_DEFAULT_REF})",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Save JSON report to this path.",
    )
    parser.add_argument(
        "--no-coherence",
        action="store_true",
        help="Skip boundary coherence metric (no embedding needed).",
    )
    args = parser.parse_args()

    # Parse LABEL=PATH pairs
    configs: list[tuple[str, Path]] = []
    for spec in args.chunks:
        if "=" not in spec:
            parser.error(f"--chunks must be LABEL=PATH, got: {spec}")
        label, path_str = spec.split("=", 1)
        p = Path(path_str)
        if not p.exists():
            parser.error(f"File not found: {p}")
        configs.append((label.strip(), p))

    labels = [l for l, _ in configs]

    # Load all configs
    all_chunks: dict[str, list[dict]] = {}
    for label, path in configs:
        chunks = _load_chunks(path)
        all_chunks[label] = chunks
        print(f"Loaded {label}: {len(chunks)} chunks from {path.name}")

    # Compute metrics
    results: dict[str, dict] = {}
    for label in labels:
        chunks = all_chunks[label]
        sd = size_distribution(chunks)
        hist = _ascii_histogram(sd["sizes"]) if sd.get("sizes") else ""
        sd_clean = {k: v for k, v in sd.items() if k != "sizes"}
        results[label] = {
            "size": sd_clean,
            "histogram": hist,
            "fallback": fallback_rate(chunks),
        }

    # Boundary coherence (requires embedding)
    if not args.no_coherence:
        print(f"\nLoading reference model: {args.reference_model}")
        embedder = HuggingFaceEmbeddings(model_name=args.reference_model)
        for label in labels:
            results[label]["coherence"] = boundary_coherence(
                all_chunks[label], embedder, args.batch_size
            )

    # Print report
    print_report(results, args.reference_model)

    # Save JSON report
    if args.output:
        serializable = {}
        for label, metrics in results.items():
            serializable[label] = {k: v for k, v in metrics.items() if k != "histogram"}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(serializable, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
