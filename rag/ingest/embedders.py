"""Model-specific embedding backends with a unified interface.

Every embedder exposes two methods:
  * ``embed_passages(texts) -> EmbedResult``  — for ingestion (document side).
  * ``embed_query(text, instruction) -> EmbedResult``  — for retrieval (query side).
  * ``embed_queries(texts, instruction) -> EmbedResult``  — batch query side (HyDE).

Query-side encoding mirrors the passage-side quirks so that ingestion and
retrieval stay consistent:

  * e5     : passages prefixed "passage: "; queries formatted as
             f"Instruct: {instruction}\\nQuery: {query}", normalize_embeddings=True.
  * bge-m3 : FlagEmbedding produces dense + sparse (lexical) weights for both
             passages and queries. No ColBERT (dropped by design). No instruction
             prefix needed.
  * jina   : Matryoshka truncate_dim=1024 (native, internally renormalized);
             task="retrieval", prompt_name="passage" / "query".
  * qwen3  : 8B decoder served by vLLM; raw 4096 vectors are sliced to [:1024]
             (MRL) and **re-normalized with L2** — required for correct cosine.
             Queries prepend an ENGLISH instruction (Qwen team recommendation).
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


def _resolve_device(device: Optional[str]) -> Optional[str]:
    """Return *device* unless CUDA is requested but unavailable.

    If the user passes ``--device cuda`` but the installed torch was compiled
    for a newer CUDA than the HPC driver, ``torch.cuda.is_available()`` returns
    False.  Rather than crashing inside SentenceTransformer / FlagEmbedding, we
    fall back to ``None`` (auto-detect -> CPU) and print a warning pointing at
    ``slurm/setup_torch_compat.sh``.
    """
    if device is None:
        return None
    if device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                print(
                    "[warn] CUDA requested (--device cuda) but "
                    "torch.cuda.is_available() is False. "
                    "Falling back to CPU. "
                    "Run: bash slurm/setup_venv.sh  (one-time, on a login node)"
                )
                return None
        except ImportError:
            pass
    return device


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

    def embed_query(self, text: str, instruction: Optional[str] = None) -> EmbedResult:
        raise NotImplementedError

    def embed_queries(
        self, texts: list[str], instruction: Optional[str] = None
    ) -> EmbedResult:
        raise NotImplementedError

    def _resolve_instruction(self, instruction: Optional[str]) -> str:
        return instruction or self.cfg.default_query_instruction


class E5Embedder(BaseEmbedder):
    """intfloat/multilingual-e5-large-instruct via SentenceTransformers."""

    def __init__(self, cfg: ModelConfig, device: Optional[str] = None) -> None:
        super().__init__(cfg)
        device = _resolve_device(device)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(cfg.hf_model_id, device=device)

    def embed_passages(self, texts: list[str]) -> EmbedResult:
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return EmbedResult(dense=l2_normalize(vecs))

    def embed_query(self, text: str, instruction: Optional[str] = None) -> EmbedResult:
        formatted = f"Instruct: {self._resolve_instruction(instruction)}\nQuery: {text}"
        vec = self.model.encode(
            [formatted],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return EmbedResult(dense=l2_normalize(vec[0]))

    def embed_queries(
        self, texts: list[str], instruction: Optional[str] = None
    ) -> EmbedResult:
        inst = self._resolve_instruction(instruction)
        formatted = [f"Instruct: {inst}\nQuery: {t}" for t in texts]
        vecs = self.model.encode(
            formatted,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return EmbedResult(dense=l2_normalize(vecs))


class BGEEmbedder(BaseEmbedder):
    """BAAI/bge-m3 via the official FlagEmbedding library (dense + sparse).

    ColBERT vectors are intentionally NOT produced (return_colbert_vecs=False).
    """

    def __init__(
        self, cfg: ModelConfig, device: Optional[str] = None, use_fp16: bool = True
    ) -> None:
        super().__init__(cfg)
        device = _resolve_device(device)
        from FlagEmbedding import BGEM3FlagModel

        kwargs: dict = {}
        if device:
            kwargs["devices"] = device
            kwargs["use_fp16"] = use_fp16
        else:
            kwargs["use_fp16"] = False
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

    def embed_query(self, text: str, instruction: Optional[str] = None) -> EmbedResult:
        out = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = l2_normalize(
            np.asarray(out["dense_vecs"][0], dtype=np.float32)
        )
        sparse = [{int(k): float(v) for k, v in out["lexical_weights"][0].items()}]
        return EmbedResult(dense=dense, sparse=sparse)

    def embed_queries(
        self, texts: list[str], instruction: Optional[str] = None
    ) -> EmbedResult:
        out = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = l2_normalize(np.asarray(out["dense_vecs"], dtype=np.float32))
        sparse = [
            {int(k): float(v) for k, v in lw.items()} for lw in out["lexical_weights"]
        ]
        return EmbedResult(dense=dense, sparse=sparse)


class JinaEmbedder(BaseEmbedder):
    """jinaai/jina-embeddings-v4 via SentenceTransformers, MRL-truncated to 1024.

    Jina v4 supports Matryoshka dims [128,256,512,1024,2048]; truncate_dim=1024
    truncates AND re-normalizes internally. We L2-normalize again defensively.
    """

    def __init__(self, cfg: ModelConfig, device: Optional[str] = None) -> None:
        super().__init__(cfg)
        device = _resolve_device(device)
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

    def embed_query(self, text: str, instruction: Optional[str] = None) -> EmbedResult:
        vec = self.model.encode(
            [text],
            task="retrieval",
            prompt_name="query",
            truncate_dim=TARGET_DIM,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return EmbedResult(dense=l2_normalize(vec[0]))

    def embed_queries(
        self, texts: list[str], instruction: Optional[str] = None
    ) -> EmbedResult:
        vecs = self.model.encode(
            texts,
            task="retrieval",
            prompt_name="query",
            truncate_dim=TARGET_DIM,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return EmbedResult(dense=l2_normalize(vecs))


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
        resp = self.client.embeddings.create(model=self.model_id, input=texts)
        ordered = sorted(resp.data, key=lambda d: d.index)
        raw = np.asarray([d.embedding for d in ordered], dtype=np.float32)

        sliced = raw[:, : self.cfg.target_dim]
        return EmbedResult(dense=l2_normalize(sliced))

    def embed_query(self, text: str, instruction: Optional[str] = None) -> EmbedResult:
        formatted = f"Instruct: {self._resolve_instruction(instruction)}\nQuery: {text}"
        resp = self.client.embeddings.create(model=self.model_id, input=[formatted])
        ordered = sorted(resp.data, key=lambda d: d.index)
        raw = np.asarray([d.embedding for d in ordered], dtype=np.float32)
        sliced = raw[0, : self.cfg.target_dim]
        return EmbedResult(dense=l2_normalize(sliced))

    def embed_queries(
        self, texts: list[str], instruction: Optional[str] = None
    ) -> EmbedResult:
        inst = self._resolve_instruction(instruction)
        formatted = [f"Instruct: {inst}\nQuery: {t}" for t in texts]
        resp = self.client.embeddings.create(model=self.model_id, input=formatted)
        ordered = sorted(resp.data, key=lambda d: d.index)
        raw = np.asarray([d.embedding for d in ordered], dtype=np.float32)
        sliced = raw[:, : self.cfg.target_dim]
        return EmbedResult(dense=l2_normalize(sliced))


class OpenRouterEmbedder(BaseEmbedder):
    """Query-only embedder that calls the OpenRouter embeddings API.

    Supports bge-m3 and qwen3-embedding-8b via OpenRouter, eliminating the need
    for a local GPU or vLLM server during evaluation. Sparse vectors are NOT
    available from OpenRouter — hybrid retrieval is automatically disabled.

    Dimension handling: we request native-dim vectors and apply the same MRL
    slicing + L2 renorm locally, guaranteeing consistency with ingestion.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
    ):
        super().__init__(cfg)
        import os

        from openai import OpenAI

        api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            timeout=timeout,
        )
        self.model_id = cfg.openrouter_model_id

    def embed_passages(self, texts: list[str]) -> EmbedResult:
        raise NotImplementedError("OpenRouterEmbedder is query-only.")

    def _embed_one(self, text: str) -> EmbedResult:
        resp = self.client.embeddings.create(model=self.model_id, input=[text])
        ordered = sorted(resp.data, key=lambda d: d.index)
        raw = np.asarray([d.embedding for d in ordered], dtype=np.float32)
        sliced = raw[0, : self.cfg.target_dim]
        return EmbedResult(dense=l2_normalize(sliced))

    def _embed_batch(self, texts: list[str]) -> EmbedResult:
        resp = self.client.embeddings.create(model=self.model_id, input=texts)
        ordered = sorted(resp.data, key=lambda d: d.index)
        raw = np.asarray([d.embedding for d in ordered], dtype=np.float32)
        sliced = raw[:, : self.cfg.target_dim]
        return EmbedResult(dense=l2_normalize(sliced))

    def embed_query(self, text: str, instruction: Optional[str] = None) -> EmbedResult:
        if self.cfg.key == "qwen3":
            formatted = f"Instruct: {self._resolve_instruction(instruction)}\nQuery: {text}"
        else:
            formatted = text
        return self._embed_one(formatted)

    def embed_queries(
        self, texts: list[str], instruction: Optional[str] = None
    ) -> EmbedResult:
        if self.cfg.key == "qwen3":
            inst = self._resolve_instruction(instruction)
            formatted = [f"Instruct: {inst}\nQuery: {t}" for t in texts]
        else:
            formatted = texts
        return self._embed_batch(formatted)


def build_embedder(
    cfg: ModelConfig,
    device: Optional[str] = None,
    vllm_base_url: Optional[str] = None,
    query_backend: str = "local",
    openrouter_api_key: Optional[str] = None,
) -> BaseEmbedder:
    if query_backend == "openrouter":
        if not cfg.openrouter_model_id:
            raise ValueError(
                f"Model '{cfg.key}' is not available on OpenRouter. "
                f"Use --query_backend local (requires GPU)."
            )
        return OpenRouterEmbedder(cfg, api_key=openrouter_api_key)
    if cfg.backend == "sentence_transformers" and cfg.key == "e5":
        return E5Embedder(cfg, device=device)
    if cfg.backend == "sentence_transformers" and cfg.key == "jina":
        return JinaEmbedder(cfg, device=device)
    if cfg.backend == "flag_embedding":
        return BGEEmbedder(cfg, device=device)
    if cfg.backend == "vllm_server":
        return Qwen3VLLMEmbedder(cfg, base_url=vllm_base_url)
    raise ValueError(f"No embedder wiring for model '{cfg.key}'")
