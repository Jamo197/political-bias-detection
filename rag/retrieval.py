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

# Environment Setup
load_dotenv(Path(".env.local"))
os.environ["HF_TOKEN"] = os.getenv("HT_TOKEN", "")

# Logging configuration prioritized for pipeline tracking
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RetrievalStrategy(ABC):
    """Abstract base class enforcing a standard interface for all retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        limit: int = 3,
        party_filter: Optional[str] = None,
    ) -> List[models.PointStruct]:
        pass

    @staticmethod
    def _build_party_filter(party_filter: Optional[str]) -> Optional[models.Filter]:
        """Centralized Qdrant filter construction for academic subsets."""
        if not party_filter:
            return None
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="party", match=models.MatchValue(value=party_filter)
                )
            ]
        )


class SimpleRetrieval(RetrievalStrategy):
    """Baseline Bi-Encoder retrieval"""

    def __init__(self, client: QdrantClient, collection_name: str, embeddings):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings

    def retrieve(
        self, query: str, limit: int = 3, party_filter: Optional[str] = None
    ) -> List[models.PointStruct]:
        query_vector = self.embeddings.embed_query(query)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._build_party_filter(party_filter),
            limit=limit,
            with_payload=True,
        )
        return response.points


class HyDERetrieval(RetrievalStrategy):
    """
    Hypothetical Document Embeddings (HyDE).
    Optimized for cross-cultural prompt adaptation (RQ2.1).
    """

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
        self,
        query: str,
        limit: int = 3,
        party_filter: Optional[str] = None,
        num_hypothetical: int = 3,
    ) -> List[models.PointStruct]:
        hypothetical_docs = self._generate_hypothetical_docs(query, num_hypothetical)

        all_queries = [query] + hypothetical_docs
        query_vectors = self.embeddings.embed_documents(all_queries)
        avg_vector = np.mean(query_vectors, axis=0).tolist()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=avg_vector,
            query_filter=self._build_party_filter(party_filter),
            limit=limit,
            with_payload=True,
        )
        return response.points

    def _generate_hypothetical_docs(self, query: str, num_docs: int) -> List[str]:
        if not self.llm:
            return []

        # Prompt parameterized for cross-cultural ideological variances
        prompt = f"""
            You are a political assistant in {self.country_context}. Generate {num_docs} short hypothetical 
            parliamentary speech excerpts regarding: "{query}"

            Rules:
            - 1-3 sentences maximum.
            - Mirror the authentic rhetorical style of the {self.country_context} parliament.
            - Return ONLY the excerpts, one per line.
        """
        try:
            response = self.llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)

            cleaned_docs = [
                d.strip() for d in text.strip().split("\n") if len(d.strip()) > 10
            ][:num_docs]
            return cleaned_docs
        except Exception as e:
            logger.error(
                f"HyDE LLM Generation failed: {e}. Defaulting to standard retrieval."
            )
            return []


class TwoStageRetrieval(RetrievalStrategy):
    """Bi-Encoder Retrieval + Cross-Encoder Reranking for high-precision alignment."""

    def __init__(
        self, client: QdrantClient, collection_name: str, bi_encoder, cross_encoder
    ):
        self.client = client
        self.collection_name = collection_name
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder

    def retrieve(
        self,
        query: str,
        limit: int = 3,
        party_filter: Optional[str] = None,
        rerank_top_k: int = 15,
    ) -> List[models.PointStruct]:
        # Stage 1: Fast Vector Search
        query_vector = self.bi_encoder.embed_query(query)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=self._build_party_filter(party_filter),
            limit=rerank_top_k,
            with_payload=True,
        )

        candidates = response.points
        if not candidates or not self.cross_encoder:
            return candidates[:limit]

        # Stage 2: Cross-Encoder Reranking
        pairs = [[query, c.payload.get("text", "")] for c in candidates]
        scores = self.cross_encoder.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)

        return sorted(candidates, key=lambda x: x.score, reverse=True)[:limit]


class PoliticalRAGRetriever:
    """
    Main RAG orchestrator optimized for CHES validation and bias classification.
    """

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

        logger.info(f"Loading Bi-Encoder: {self.EMBEDDING_MODEL}")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)

        self.retrieval_strategy = self._init_retrieval_strategy(retrieval_mode, llm)

    def _init_retrieval_strategy(self, mode: str, llm) -> RetrievalStrategy:
        mode = mode.lower()
        if mode == "hyde":
            if not llm:
                try:
                    logger.info("Initializing local Ollama LLM for HyDE...")
                    llm = OllamaLLM(model="gemma3", base_url="http://localhost:11434")
                except Exception as e:
                    logger.warning(
                        f"Ollama connection failed: {e}. HyDE will operate in fallback mode."
                    )
            return HyDERetrieval(
                self.client,
                self.collection_name,
                self.embeddings,
                llm,
                self.country_context,
            )

        elif mode in ("two_stage", "twostage"):
            try:
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading Cross-Encoder: {self.CROSS_ENCODER_MODEL}")
                cross_encoder = CrossEncoder(self.CROSS_ENCODER_MODEL)
                return TwoStageRetrieval(
                    self.client, self.collection_name, self.embeddings, cross_encoder
                )
            except ImportError:
                logger.error(
                    "sentence-transformers missing. Falling back to simple retrieval."
                )

        return SimpleRetrieval(self.client, self.collection_name, self.embeddings)

    @staticmethod
    def _extract_year(date_str: Optional[str]) -> Optional[int]:
        """
        Extracts a 4-digit year from a date string (e.g. '2021-11-03').
        Returns None if parsing fails or the string is empty/None.
        """
        if not date_str:
            return None
        try:
            return int(str(date_str)[:4])
        except (ValueError, TypeError):
            return None

    def search(
        self,
        query: str,
        limit: int = 3,
        party_filter: Optional[str] = None,
    ) -> List[models.PointStruct]:
        """
        Executes a standard Top-K search using the active retrieval strategy.
        Returns raw Qdrant PointStructs.
        """
        return self.retrieval_strategy.retrieve(
            query=query, limit=limit, party_filter=party_filter
        )
