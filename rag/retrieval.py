"""Model-aware retrieval strategies for the multi-embedding RAG pipeline.

The retriever is now model-aware: query-side encoding uses the SAME embedding
model (and the SAME quirks) that produced the vectors at ingestion time. This
module reuses ``rag.ingest.embedders`` and ``rag.ingest.config`` so ingestion
and retrieval stay consistent.

Supported strategies:
  * Simple      — Bi-encoder dense vector retrieval (baseline).
  * SimpleHybrid — BGE-m3 only: dense + sparse RRF fusion via Qdrant prefetch.
  * HyDE        — Hypothetical Document Embeddings (generates fake parliamentary
                  speeches via a small LLM, averages embeddings with the query).
  * TwoStage    — Bi-encoder retrieval + Cross-Encoder reranking.

HyDE LLM backends:
  * OpenAIHyDELLM     — Any OpenAI-compatible endpoint (Ollama, vLLM server).
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag.ingest.config import (
    BGE_DENSE_VECTOR_NAME,
    BGE_SPARSE_VECTOR_NAME,
    ModelConfig,
    get_model_config,
)
from rag.ingest.embedders import BaseEmbedder, EmbedResult, build_embedder, l2_normalize

load_dotenv(Path(".env.local"))
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


# ---------------------------------------------------------------------------
# HyDE LLM
# ---------------------------------------------------------------------------


class OpenAIHyDELLM:
    """Any OpenAI-compatible endpoint (vLLM server or OpenRouter)."""

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        model_id: str = "qwen/qwen-2.5-7b-instruct",
        api_key: Optional[str] = None,
    ):
        from openai import OpenAI

        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY", "EMPTY")
        headers = {}
        if "openrouter.ai" in base_url:
            headers.update(
                {
                    "HTTP-Referer": "https://github.com/Jamo197/political-bias-detection",
                    "X-Title": "Political RAG Pipeline Engine",
                }
            )
        self.client = OpenAI(
            base_url=base_url, api_key=resolved_key, default_headers=headers
        )
        self.model_id = model_id

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Retrieval strategies
# ---------------------------------------------------------------------------


class RetrievalStrategy(ABC):
    """Abstract base class for all retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, limit: int = 3) -> List[models.PointStruct]:
        pass


class SimpleRetrieval(RetrievalStrategy):
    """Bi-encoder dense vector retrieval.

    For bge-m3 with hybrid=True, performs dense + sparse RRF fusion via
    Qdrant prefetch. For bge-m3 with hybrid=False, queries only the named
    dense vector. Other models use the default unnamed vector.
    """

    def __init__(
        self,
        client: QdrantClient,
        cfg: ModelConfig,
        embedder: BaseEmbedder,
        hybrid: bool = False,
    ):
        self.client = client
        self.cfg = cfg
        self.embedder = embedder
        self.hybrid = hybrid

    def retrieve(self, query: str, limit: int = 3) -> List[models.PointStruct]:
        result = self.embedder.embed_query(query)
        dense_vec = result.dense.tolist()

        if self.cfg.hybrid_sparse and self.hybrid and result.sparse:
            sparse_dict = result.sparse[0]
            sparse_vec = models.SparseVector(
                indices=list(sparse_dict.keys()),
                values=list(sparse_dict.values()),
            )
            prefetch_limit = max(limit * 5, 20)
            response = self.client.query_points(
                collection_name=self.cfg.collection,
                prefetch=[
                    models.Prefetch(
                        query=dense_vec,
                        using=BGE_DENSE_VECTOR_NAME,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=sparse_vec,
                        using=BGE_SPARSE_VECTOR_NAME,
                        limit=prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        elif self.cfg.hybrid_sparse:
            response = self.client.query_points(
                collection_name=self.cfg.collection,
                query=dense_vec,
                using=BGE_DENSE_VECTOR_NAME,
                limit=limit,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.cfg.collection,
                query=dense_vec,
                limit=limit,
                with_payload=True,
            )

        return response.points


class HyDERetrieval(RetrievalStrategy):
    """Hypothetical Document Embeddings (HyDE) contextual retrieval.

    Generates hypothetical parliamentary speech excerpts via a small LLM,
    embeds them alongside the original query, averages the dense vectors,
    and queries the collection. For bge-m3, only the dense vector is used
    (averaging sparse vectors across hypothetical docs is not meaningful).
    """

    def __init__(
        self,
        client: QdrantClient,
        cfg: ModelConfig,
        embedder: BaseEmbedder,
        hyde_llm: Optional[OpenAIHyDELLM] = None,
        country_context: str = "Germany",
    ):
        self.client = client
        self.cfg = cfg
        self.embedder = embedder
        self.hyde_llm = hyde_llm
        self.country_context = country_context

    def retrieve(
        self, query: str, limit: int = 3, num_hypothetical: int = 3
    ) -> List[models.PointStruct]:
        hypothetical_docs = self._generate_hypothetical_docs(query, num_hypothetical)
        all_texts = [query] + hypothetical_docs
        result = self.embedder.embed_queries(all_texts)
        avg_vector = np.mean(result.dense, axis=0)
        avg_vector = l2_normalize(avg_vector)
        avg_list = avg_vector.tolist()

        if self.cfg.hybrid_sparse:
            response = self.client.query_points(
                collection_name=self.cfg.collection,
                query=avg_list,
                using=BGE_DENSE_VECTOR_NAME,
                limit=limit,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.cfg.collection,
                query=avg_list,
                limit=limit,
                with_payload=True,
            )

        return response.points

    def _generate_hypothetical_docs(self, query: str, num_docs: int) -> List[str]:
        if not self.hyde_llm:
            return []

        prompt = f"""
            You are a political assistant in {self.country_context}. Generate {num_docs} short hypothetical
            parliamentary speech excerpts regarding: "{query}"

            Rules:
            - 1-3 sentences maximum.
            - Mirror the authentic rhetorical style and the original language of the parliament in {self.country_context}.

            IMPORTANT: Return ONLY the excerpts, one per line, no additional text or information, like "Here are the hypothetical documents: ...".
        """
        try:
            text = self.hyde_llm.generate(prompt)
            return [d.strip() for d in text.strip().split("\n") if len(d.strip()) > 10][
                :num_docs
            ]
        except Exception as e:
            logger.error(
                f"HyDE LLM Generation failed: {e}. Defaulting to standard vector path."
            )
            return []


class TwoStageRetrieval(RetrievalStrategy):
    """Bi-encoder retrieval coupled with Cross-Encoder reranking.

    The first stage uses SimpleRetrieval (including BGE hybrid if enabled)
    to over-fetch candidates. The second stage reranks with a cross-encoder.
    """

    def __init__(
        self,
        client: QdrantClient,
        cfg: ModelConfig,
        embedder: BaseEmbedder,
        cross_encoder,
        hybrid: bool = False,
    ):
        self.client = client
        self.cfg = cfg
        self.embedder = embedder
        self.cross_encoder = cross_encoder
        self.hybrid = hybrid

    def retrieve(
        self, query: str, limit: int = 3, rerank_top_k: int = 15
    ) -> List[models.PointStruct]:
        simple = SimpleRetrieval(self.client, self.cfg, self.embedder, self.hybrid)
        candidates = simple.retrieve(query, limit=rerank_top_k)

        if not candidates or not self.cross_encoder:
            return candidates[:limit]

        pairs = [[query, c.payload.get("text", "")] for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)

        return sorted(candidates, key=lambda x: x.score, reverse=True)[:limit]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class PoliticalRAGRetriever:
    """Model-aware RAG orchestrator.

    Query encoding uses the same embedding model that produced the vectors
    at ingestion time, ensuring cosine similarity is meaningful. Pre-built
    components (embedder, cross_encoder, hyde_llm) can be passed in to avoid
    reloading models across multiple strategy runs.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        model_key: str = "e5",
        retrieval_mode: str = "simple",
        country_context: str = "Germany",
        device: Optional[str] = None,
        vllm_base_url: Optional[str] = None,
        hybrid: bool = False,
        cross_encoder=None,
        hyde_llm: Optional[OpenAIHyDELLM] = None,
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.cfg = get_model_config(model_key)
        self.collection_name = self.cfg.collection
        self.country_context = country_context
        self.hybrid = hybrid

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = build_embedder(
                self.cfg, device=device, vllm_base_url=vllm_base_url
            )

        self.retrieval_strategy = self._init_retrieval_strategy(
            retrieval_mode, cross_encoder, hyde_llm, device
        )

    def _init_retrieval_strategy(
        self,
        mode: str,
        cross_encoder,
        hyde_llm: Optional[OpenAIHyDELLM],
        device: Optional[str],
    ) -> RetrievalStrategy:
        mode = mode.lower().replace(" ", "").replace("_", "")

        if mode == "hyde":
            return HyDERetrieval(
                self.client,
                self.cfg,
                self.embedder,
                hyde_llm,
                self.country_context,
            )

        elif mode == "twostage":
            if cross_encoder is None:
                try:
                    from sentence_transformers import CrossEncoder

                    logger.info(f"Loading Cross-Encoder: {CROSS_ENCODER_MODEL}")
                    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
                except ImportError:
                    logger.error(
                        "sentence-transformers dependency missing. Defaulting to Simple."
                    )
                    return SimpleRetrieval(
                        self.client, self.cfg, self.embedder, self.hybrid
                    )
            return TwoStageRetrieval(
                self.client,
                self.cfg,
                self.embedder,
                cross_encoder,
                self.hybrid,
            )

        return SimpleRetrieval(self.client, self.cfg, self.embedder, self.hybrid)

    def search(self, query: str, limit: int = 3) -> List[models.PointStruct]:
        return self.retrieval_strategy.retrieve(query=query, limit=limit)
