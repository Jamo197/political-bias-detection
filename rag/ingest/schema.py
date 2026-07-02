"""Qdrant collection schemas, one per embedding model.

Dense-only models (e5, jina, qwen3) use the default unnamed 1024-dim vector.
bge-m3 uses a hybrid schema: a named dense vector + a sparse vector (IDF
modifier). ColBERT multivectors are intentionally NOT configured.

Every chunk collection carries the same payload indexes as the original
pipeline so ground-truth filtering + parent-document retrieval keep working.
"""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import (
    BGE_DENSE_VECTOR_NAME,
    BGE_SPARSE_VECTOR_NAME,
    PARENT_COLLECTION,
    PAYLOAD_INDEX_FIELDS,
    ModelConfig,
)


def _create_payload_indexes(client: QdrantClient, collection: str) -> None:
    for field_name, is_int in PAYLOAD_INDEX_FIELDS:
        schema = (
            models.PayloadSchemaType.INTEGER
            if is_int
            else models.PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=schema,
        )


def ensure_chunk_collection(
    client: QdrantClient, cfg: ModelConfig, reset: bool = False
) -> None:
    """Create (or reset) the chunk collection for a single model."""
    exists = client.collection_exists(cfg.collection)

    if exists and reset:
        client.delete_collection(cfg.collection)
        exists = False
        print(f"[reset] Dropped existing collection '{cfg.collection}'.")

    if exists:
        print(f"[skip] Collection '{cfg.collection}' already exists.")
        return

    if cfg.hybrid_sparse:
        # bge-m3 hybrid: named dense + sparse (IDF). No ColBERT.
        client.create_collection(
            collection_name=cfg.collection,
            vectors_config={
                BGE_DENSE_VECTOR_NAME: models.VectorParams(
                    size=cfg.target_dim, distance=models.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                BGE_SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF
                ),
            },
        )
    else:
        # Dense-only (default unnamed vector).
        client.create_collection(
            collection_name=cfg.collection,
            vectors_config=models.VectorParams(
                size=cfg.target_dim, distance=models.Distance.COSINE
            ),
        )

    _create_payload_indexes(client, cfg.collection)
    print(
        f"[create] Collection '{cfg.collection}' "
        f"({'hybrid dense+sparse' if cfg.hybrid_sparse else 'dense'}, "
        f"{cfg.target_dim}d) with payload indexes."
    )


def ensure_parent_collection(client: QdrantClient, reset: bool = False) -> None:
    """Vectorless collection storing full speeches (shared across models)."""
    exists = client.collection_exists(PARENT_COLLECTION)
    if exists and reset:
        client.delete_collection(PARENT_COLLECTION)
        exists = False
    if exists:
        return
    client.create_collection(
        collection_name=PARENT_COLLECTION,
        vectors_config={},
    )
    _create_payload_indexes(client, PARENT_COLLECTION)
    print(f"[create] Parent collection '{PARENT_COLLECTION}' (vectorless).")
