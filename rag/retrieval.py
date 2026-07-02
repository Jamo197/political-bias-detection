import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv(Path(".env.local"))
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RetrievalStrategy(ABC):
    """Abstract Base Class establishing structural interface for all retrieval strategies."""

    @abstractmethod
    def retrieve(self, query: str, limit: int = 3) -> List[models.PointStruct]:
        pass


class SimpleRetrieval(RetrievalStrategy):
    """Baseline Bi-Encoder dense vector retrieval."""

    def __init__(self, client: QdrantClient, collection_name: str, embeddings):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings

    def retrieve(self, query: str, limit: int = 3) -> List[models.PointStruct]:
        query_vector = self.embeddings.embed_query(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return response.points


class HyDERetrieval(RetrievalStrategy):
    """Hypothetical Document Embeddings (HyDE) contextual retrieval."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings,
        llm=None,
        country_context: str = "Germany",
    ):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.llm = llm
        self.country_context = country_context

    def retrieve(
        self, query: str, limit: int = 3, num_hypothetical: int = 3
    ) -> List[models.PointStruct]:
        hypothetical_docs = self._generate_hypothetical_docs(query, num_hypothetical)
        all_queries = [query] + hypothetical_docs
        query_vectors = self.embeddings.embed_documents(all_queries)
        avg_vector = np.mean(query_vectors, axis=0).tolist()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=avg_vector,
            limit=limit,
            with_payload=True,
        )
        return response.points

    def _generate_hypothetical_docs(self, query: str, num_docs: int) -> List[str]:
        if not self.llm:
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
            response = self.llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            return [d.strip() for d in text.strip().split("\n") if len(d.strip()) > 10][
                :num_docs
            ]
        except Exception as e:
            logger.error(
                f"HyDE LLM Generation failed: {e}. Defaulting to standard vector path."
            )
            return []


class TwoStageRetrieval(RetrievalStrategy):
    """Bi-Encoder Retrieval coupled with Cross-Encoder Reranking."""

    def __init__(
        self, client: QdrantClient, collection_name: str, bi_encoder, cross_encoder
    ):
        self.client = client
        self.collection_name = collection_name
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder

    def retrieve(
        self, query: str, limit: int = 3, rerank_top_k: int = 15
    ) -> List[models.PointStruct]:
        query_vector = self.bi_encoder.embed_query(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=rerank_top_k,
            with_payload=True,
        )

        candidates = response.points
        if not candidates or not self.cross_encoder:
            return candidates[:limit]

        pairs = [[query, c.payload.get("text", "")] for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)

        return sorted(candidates, key=lambda x: x.score, reverse=True)[:limit]


# TODO(retrieval): Multi-model query-side encoding follow-up.
# The HPC ingestion pipeline (rag/ingest/) now populates four separate Qdrant
# collections — chunks_e5, chunks_bge, chunks_jina, chunks_qwen3 — each with
# 1024-dim dense vectors (bge additionally stores a sparse IDF vector). This
# retriever must be made model-aware so queries are encoded with the SAME model
# (and the SAME quirks) used at ingestion. Specifically:
#   * e5    : query = f"Instruct: {instruction}\nQuery: {query}", normalize.
#   * qwen3 : prepend ENGLISH instruction (Qwen recommendation), slice [:1024],
#             then L2-renormalize (mirror rag/ingest/embedders.py).
#   * jina  : encode(task="retrieval", prompt_name="query", truncate_dim=1024).
#   * bge   : dense + sparse hybrid query (Qdrant prefetch + RRF/DBSF fusion).
# Reuse rag.ingest.embedders + rag.ingest.config (MODELS, default instruction)
# so ingestion and retrieval stay consistent. Until then, the legacy MiniLM
# baseline below remains for the old single-collection setup.
class PoliticalRAGRetriever:
    """Main RAG orchestrator optimized for CHES validation execution."""

    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        chunk_collection: str = "bundestag_speeches_chunks",
        retrieval_mode: str = "simple",
        country_context: str = "Germany",
        llm=None,
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = chunk_collection
        self.country_context = country_context
        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)
        self.retrieval_strategy = self._init_retrieval_strategy(retrieval_mode, llm)

    def _init_retrieval_strategy(self, mode: str, llm) -> RetrievalStrategy:
        mode = mode.lower().replace(" ", "").replace("_", "")
        if mode == "hyde":
            if not llm:
                try:
                    logger.info("Initializing local Ollama instance for HyDE...")
                    llm = OllamaLLM(model="gemma3", base_url="http://localhost:11434")
                except Exception as e:
                    logger.warning(
                        f"Ollama offline: {e}. Running HyDE in fallback mode."
                    )
            return HyDERetrieval(
                self.client,
                self.collection_name,
                self.embeddings,
                llm,
                self.country_context,
            )

        elif mode == "twostage":
            try:
                from sentence_transformers import CrossEncoder

                logger.info(
                    f"Loading Cross-Encoder sequence-transformer: {self.CROSS_ENCODER_MODEL}"
                )
                cross_encoder = CrossEncoder(self.CROSS_ENCODER_MODEL)
                return TwoStageRetrieval(
                    self.client, self.collection_name, self.embeddings, cross_encoder
                )
            except ImportError:
                logger.error(
                    "sentence-transformers dependency missing. Defaulting to Simple."
                )

        return SimpleRetrieval(self.client, self.collection_name, self.embeddings)

    def search(self, query: str, limit: int = 3) -> List[models.PointStruct]:
        return self.retrieval_strategy.retrieve(query=query, limit=limit)
