"""Model-specific embedding backends with a unified ingestion interface.

Every embedder exposes ``embed_passages(texts) -> EmbedResult`` for ingestion.
Query-side encoding (instruction prompting, prompt_name="query", etc.) is a
FOLLOW-UP task handled in rag/retrieval.py; the hooks are marked with
``TODO(retrieval)`` below so the model-specific query logic is not forgotten.

Model quirks handled here:
  * e5     : passages prefixed with "passage: ", L2-normalized.
  * bge-m3 : FlagEmbedding produces dense + sparse (lexical) weights. No ColBERT
             (dropped by design). Sparse stored alongside dense for hybrid search.
  * jina   : Matryoshka truncate_dim=1024 (native, internally renormalized);
             task="retrieval", prompt_name="passage".
  * qwen3  : 8B decoder served by vLLM; raw 4096 vectors are sliced to [:1024]
             (MRL) and **re-normalized with L2** — required for correct cosine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import TARGET_DIM, ModelConfig


def _patch_jina_rope_compat() -> None:
    """Add 'default' rope init back for jina-embeddings-v4 compatibility.

    transformers 5.x removed the 'default' key from ROPE_INIT_FUNCTIONS, but
    jina-embeddings-v4's custom Qwen2.5-VL code (loaded via trust_remote_code)
    still references ``ROPE_INIT_FUNCTIONS['default']``.  We restore the mapping
    so model loading does not crash with ``KeyError: 'default'``.
    """
    try:
        import transformers.modeling_rope_utils as _rope
    except ImportError:
        return

    if "default" in _rope.ROPE_INIT_FUNCTIONS:
        return

    import torch

    if hasattr(_rope, "_compute_default_rope_parameters"):
        _rope.ROPE_INIT_FUNCTIONS["default"] = _rope._compute_default_rope_parameters
    else:

        def _default_rope_init(config, device, **kwargs):
            base = getattr(config, "rope_theta", 1_000_000.0)
            head_dim = getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            )
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
            )
            return inv_freq, 1.0

        _rope.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Critical after MRL slicing so cosine is valid."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / norm if norm > 0 else vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@dataclass
class EmbedResult:
    """Output of a passage embedding batch.

    dense:  (N, target_dim) float32, L2-normalized.
    sparse: optional list of {index: weight} dicts (bge hybrid only).
    """

    dense: np.ndarray
    sparse: Optional[list[dict[int, float]]] = None


class BaseEmbedder:
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    def embed_passages(self, texts: list[str]) -> EmbedResult:  # pragma: no cover
        raise NotImplementedError

    # TODO(retrieval): implement embed_query() per model with the instruction /
    # prompt_name logic. Ingestion only needs passages, so query encoding is
    # deferred to the retrieval follow-up (see rag/retrieval.py).
    def embed_query(self, text: str, instruction: Optional[str] = None) -> np.ndarray:
        raise NotImplementedError(
            "Query encoding is a retrieval-side follow-up; see TODO(retrieval)."
        )


class E5Embedder(BaseEmbedder):
    """intfloat/multilingual-e5-large-instruct via SentenceTransformers."""

    def __init__(self, cfg: ModelConfig, device: Optional[str] = None) -> None:
        super().__init__(cfg)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(cfg.hf_model_id, device=device)

    def embed_passages(self, texts: list[str]) -> EmbedResult:
        # e5 convention: documents are prefixed with "passage: ".
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return EmbedResult(dense=l2_normalize(vecs))

    # TODO(retrieval): e5 queries MUST be formatted as
    #   f"Instruct: {instruction}\nQuery: {query}"  (then normalize_embeddings=True)


class BGEEmbedder(BaseEmbedder):
    """BAAI/bge-m3 via the official FlagEmbedding library (dense + sparse).

    ColBERT vectors are intentionally NOT produced (return_colbert_vecs=False).
    """

    def __init__(
        self, cfg: ModelConfig, device: Optional[str] = None, use_fp16: bool = True
    ) -> None:
        super().__init__(cfg)
        from FlagEmbedding import BGEM3FlagModel

        kwargs: dict = {"use_fp16": use_fp16}
        if device:
            kwargs["devices"] = device
        self.model = BGEM3FlagModel(cfg.hf_model_id, **kwargs)

    def embed_passages(self, texts: list[str]) -> EmbedResult:
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = l2_normalize(np.asarray(out["dense_vecs"], dtype=np.float32))

        # lexical_weights is a list of {token_id(str): weight(float)} dicts.
        sparse: list[dict[int, float]] = []
        for lw in out["lexical_weights"]:
            sparse.append({int(k): float(v) for k, v in lw.items()})

        return EmbedResult(dense=dense, sparse=sparse)

    # TODO(retrieval): bge-m3 hybrid query = dense + sparse; fuse via Qdrant
    # prefetch + RRF/DBSF. No instruction prefix needed for bge-m3.


class JinaEmbedder(BaseEmbedder):
    """jinaai/jina-embeddings-v4 via SentenceTransformers, MRL-truncated to 1024.

    Jina v4 supports Matryoshka dims [128,256,512,1024,2048]; truncate_dim=1024
    truncates AND re-normalizes internally. We L2-normalize again defensively.
    """

    def __init__(self, cfg: ModelConfig, device: Optional[str] = None) -> None:
        super().__init__(cfg)
        _patch_jina_rope_compat()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            cfg.hf_model_id, trust_remote_code=True, device=device
        )

    def embed_passages(self, texts: list[str]) -> EmbedResult:
        vecs = self.model.encode(
            texts,
            task="retrieval",
            prompt_name="passage",
            truncate_dim=TARGET_DIM,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return EmbedResult(dense=l2_normalize(vecs))

    # TODO(retrieval): jina queries use prompt_name="query" (same task=
    # "retrieval", truncate_dim=1024).


class Qwen3VLLMEmbedder(BaseEmbedder):
    """Qwen/Qwen3-Embedding-8B served by a vLLM OpenAI-compatible server.

    The server is launched separately (slurm/02_vllm_qwen3.sbatch) with
    ``--task embed``. We hit POST {base_url}/embeddings, then apply MRL:
    slice 4096 -> 1024 and **re-normalize** (the crucial correctness step).
    """

    def __init__(
        self,
        cfg: ModelConfig,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ) -> None:
        super().__init__(cfg)
        from openai import OpenAI

        base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
        self.model_id = cfg.hf_model_id
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def embed_passages(self, texts: list[str]) -> EmbedResult:
        # Passages are embedded raw (no instruction prefix on the document side).
        resp = self.client.embeddings.create(model=self.model_id, input=texts)
        # Preserve request order.
        ordered = sorted(resp.data, key=lambda d: d.index)
        raw = np.asarray([d.embedding for d in ordered], dtype=np.float32)

        # MRL truncation: slice to target dim, THEN L2-renormalize.
        sliced = raw[:, : self.cfg.target_dim]
        return EmbedResult(dense=l2_normalize(sliced))

    # TODO(retrieval): Qwen3 queries prepend an ENGLISH instruction even for
    # German/French text (Qwen team recommendation):
    #   f"Instruct: {instruction}\nQuery: {query}"
    # Ensure padding_side='left' if you ever swap vLLM for raw PyTorch.


def build_embedder(
    cfg: ModelConfig,
    device: Optional[str] = None,
    vllm_base_url: Optional[str] = None,
) -> BaseEmbedder:
    if cfg.backend == "sentence_transformers" and cfg.key == "e5":
        return E5Embedder(cfg, device=device)
    if cfg.backend == "sentence_transformers" and cfg.key == "jina":
        return JinaEmbedder(cfg, device=device)
    if cfg.backend == "flag_embedding":
        return BGEEmbedder(cfg, device=device)
    if cfg.backend == "vllm_server":
        return Qwen3VLLMEmbedder(cfg, base_url=vllm_base_url)
    raise ValueError(f"No embedder wiring for model '{cfg.key}'")
