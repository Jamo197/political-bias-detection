"""Central configuration for the multi-model Qdrant ingestion pipeline.

Four embedding models are compared directly. Because bge-m3 and
multilingual-e5-large-instruct are fixed at 1024 dimensions, Qwen3 (native 4096)
and Jina v4 (native 2048) are truncated to 1024 via Matryoshka Representation
Learning (MRL) and re-normalized. Every collection therefore stores 1024-dim
dense vectors so cosine similarity is comparable across models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Unified target dimensionality for ALL models (fair comparison baseline).
TARGET_DIM: int = 1024

ModelKey = Literal["e5", "bge", "jina", "qwen3"]


@dataclass(frozen=True)
class ModelConfig:
    key: str
    hf_model_id: str
    native_dim: int
    target_dim: int
    collection: str
    # Backend used to produce embeddings.
    backend: Literal["sentence_transformers", "flag_embedding", "vllm_server"]
    # Whether bge-style sparse (lexical) vectors are produced/stored.
    hybrid_sparse: bool = False
    # Default English instruction used on the QUERY side for instruction-tuned
    # models. NOTE: only consumed by retrieval (follow-up task), not ingestion.
    # The Qwen team recommends English instructions even for German/French text.
    default_query_instruction: str = (
        "Given a political statement, retrieve relevant parliamentary speech "
        "excerpts that express a similar ideological position."
    )


# Name of the dense vector inside the bge hybrid collection. The pure-dense
# collections use the default (unnamed) vector for simplicity.
BGE_DENSE_VECTOR_NAME = "dense"
BGE_SPARSE_VECTOR_NAME = "sparse"


MODELS: dict[str, ModelConfig] = {
    "e5": ModelConfig(
        key="e5",
        hf_model_id="intfloat/multilingual-e5-large-instruct",
        native_dim=1024,
        target_dim=1024,
        collection="chunks_e5",
        backend="sentence_transformers",
    ),
    "bge": ModelConfig(
        key="bge",
        hf_model_id="BAAI/bge-m3",
        native_dim=1024,
        target_dim=1024,
        collection="chunks_bge",
        backend="flag_embedding",
        hybrid_sparse=True,
    ),
    "jina": ModelConfig(
        key="jina",
        hf_model_id="jinaai/jina-embeddings-v4",
        native_dim=2048,
        target_dim=1024,
        collection="chunks_jina",
        backend="sentence_transformers",
    ),
    "qwen3": ModelConfig(
        key="qwen3",
        hf_model_id="Qwen/Qwen3-Embedding-8B",
        native_dim=4096,
        target_dim=1024,
        collection="chunks_qwen3",
        backend="vllm_server",
    ),
}

# Parent-document (full speech) collection. Vectorless, shared across models.
PARENT_COLLECTION = "bundestag_speeches"

# Payload fields indexed for ground-truth filtering + parent-doc retrieval.
# (field_name, is_integer)
PAYLOAD_INDEX_FIELDS: list[tuple[str, bool]] = [
    ("party", False),
    ("speaker", False),
    ("year", True),
    ("legislative_period", True),
    ("speech_id", False),
]

# Model used ONLY to detect semantic breakpoints during chunking. It does not
# influence the stored comparison vectors, only where the text is split.
CHUNKER_BREAKPOINT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Hard cap enforced after semantic chunking (chars) before fallback splitting.
MAX_CHUNK_CHARS = 1000
# Minimum chars for a semantic chunk; smaller ones merge into the next chunk.
MIN_CHUNK_CHARS = 200
FALLBACK_CHUNK_SIZE = 1000
FALLBACK_CHUNK_OVERLAP = 150


def get_model_config(key: str) -> ModelConfig:
    if key not in MODELS:
        raise ValueError(
            f"Unknown model '{key}'. Valid options: {sorted(MODELS.keys())}"
        )
    return MODELS[key]
