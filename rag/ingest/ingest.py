"""Stage 2: embed the pre-computed chunks with one model and upsert to Qdrant.

Usage:
    python -m rag.ingest.ingest --model {e5|bge|jina|qwen3} \
        --chunks rag/ingest/artifacts/chunks.jsonl \
        --qdrant-url $QDRANT_URL [--reset] [--batch-size 64]

Chunk IDs are deterministic, so re-running is idempotent (upsert overwrites).
The parent-document collection is populated once from full_speeches.jsonl
(use --skip-parents to avoid re-writing it on subsequent model runs).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

from .config import (
    BGE_DENSE_VECTOR_NAME,
    BGE_SPARSE_VECTOR_NAME,
    PARENT_COLLECTION,
    get_model_config,
)
from .embedders import EmbedResult, build_embedder
from .schema import ensure_chunk_collection, ensure_parent_collection

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env.local")


# Payload keys that are NOT part of the stored metadata payload.
_NON_PAYLOAD_KEYS = {"chunk_id"}


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_points(
    cfg, batch: list[dict], result: EmbedResult
) -> list[models.PointStruct]:
    points: list[models.PointStruct] = []
    for i, record in enumerate(batch):
        payload = {k: v for k, v in record.items() if k not in _NON_PAYLOAD_KEYS}
        dense_vec = result.dense[i].tolist()

        if cfg.hybrid_sparse:
            sparse_map = (result.sparse or [{}])[i]
            vector = {
                BGE_DENSE_VECTOR_NAME: dense_vec,
                BGE_SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=list(sparse_map.keys()),
                    values=list(sparse_map.values()),
                ),
            }
        else:
            vector = dense_vec

        points.append(
            models.PointStruct(id=record["chunk_id"], vector=vector, payload=payload)
        )
    return points


def upload_parents(client: QdrantClient, speeches_path: Path, batch_size: int) -> int:
    if not speeches_path.exists():
        print(f"[parents] {speeches_path} not found; skipping parent upload.")
        return 0
    buffer: list[models.PointStruct] = []
    total = 0
    for rec in tqdm(_read_jsonl(speeches_path), desc="Parents"):
        buffer.append(models.PointStruct(id=rec["id"], vector={}, payload=rec))
        if len(buffer) >= batch_size:
            client.upsert(collection_name=PARENT_COLLECTION, points=buffer)
            total += len(buffer)
            buffer = []
    if buffer:
        client.upsert(collection_name=PARENT_COLLECTION, points=buffer)
        total += len(buffer)
    print(f"[parents] Upserted {total} full speeches.")
    return total


def run(args: argparse.Namespace) -> None:
    cfg = get_model_config(args.model)
    client = QdrantClient(url=args.qdrant_url, timeout=args.qdrant_timeout)

    print(f"=== Ingesting '{cfg.key}' -> collection '{cfg.collection}' ===")
    ensure_chunk_collection(client, cfg, reset=args.reset)
    ensure_parent_collection(client, reset=False)

    if not args.skip_parents:
        upload_parents(client, args.speeches, args.batch_size)

    embedder = build_embedder(cfg, device=args.device, vllm_base_url=args.vllm_base_url)

    batch: list[dict] = []
    total = 0

    def flush(batch: list[dict]) -> int:
        if not batch:
            return 0
        texts = [r["text"] for r in batch]
        result = embedder.embed_passages(texts)
        points = _build_points(cfg, batch, result)
        client.upsert(collection_name=cfg.collection, points=points)
        return len(points)

    for record in tqdm(_read_jsonl(args.chunks), desc=f"Embedding [{cfg.key}]"):
        batch.append(record)
        if len(batch) >= args.batch_size:
            total += flush(batch)
            batch = []
    total += flush(batch)

    print(f"Done. Upserted {total} chunks into '{cfg.collection}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: embed + upsert per model.")
    parser.add_argument(
        "--model", required=True, choices=["e5", "bge", "jina", "qwen3"]
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=_PROJECT_ROOT / "rag/ingest/artifacts/chunks.jsonl",
    )
    parser.add_argument(
        "--speeches",
        type=Path,
        default=_PROJECT_ROOT / "rag/ingest/artifacts/full_speeches.jsonl",
    )
    parser.add_argument(
        "--qdrant-url",
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
    parser.add_argument("--qdrant-timeout", type=float, default=120.0)
    parser.add_argument(
        "--vllm-base-url",
        default=os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1"),
        help="vLLM OpenAI endpoint (qwen3 only).",
    )
    parser.add_argument("--device", default=None, help="cuda / cpu (ST/Flag models).")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop + recreate ONLY this model's collection before ingesting.",
    )
    parser.add_argument(
        "--skip-parents",
        action="store_true",
        help="Do not (re)upload the parent full-speech collection.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
