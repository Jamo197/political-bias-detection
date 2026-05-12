import logging
import os
import uuid
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv(Path(".env.local"))
os.environ["HF_TOKEN"] = os.getenv("HT_TOKEN")

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        limit: int = 3,
        party_filter: Optional[str] = None,
    ) -> List[models.PointStruct]:
        """Retrieve relevant points based on query."""
        pass


class SimpleRetrieval(RetrievalStrategy):
    """Basic semantic similarity retrieval."""

    def __init__(self, client: QdrantClient, collection_name: str, embeddings):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings

    def retrieve(
        self,
        query: str,
        limit: int = 3,
        party_filter: Optional[str] = None,
    ) -> List[models.PointStruct]:
        """Retrieve using simple semantic similarity."""
        query_vector = self.embeddings.embed_query(query)
        query_filter = self._build_filter(party_filter)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return response.points

    @staticmethod
    def _build_filter(party_filter: Optional[str]) -> Optional[models.Filter]:
        """Build Qdrant filter for party if provided."""
        if not party_filter:
            return None

        return models.Filter(
            must=[
                models.FieldCondition(
                    key="party", match=models.MatchValue(value=party_filter)
                )
            ]
        )


class HyDERetrieval(RetrievalStrategy):
    """Hypothetical Document Embeddings (HyDE) retrieval.

    Generates hypothetical documents related to the query
    and uses their embeddings for retrieval, which can improve recall.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embeddings,
        llm=None,
    ):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.llm = llm

    def retrieve(
        self,
        query: str,
        limit: int = 3,
        party_filter: Optional[str] = None,
        num_hypothetical: int = 3,
    ) -> List[models.PointStruct]:
        """Retrieve using HyDE approach."""
        # Generate hypothetical documents
        hypothetical_docs = self._generate_hypothetical_docs(query, num_hypothetical)
        logger.debug(
            f"Generated {len(hypothetical_docs)} hypothetical documents for query"
        )

        # Embed all queries (original + hypothetical)
        all_queries = [query] + hypothetical_docs
        query_vectors = [self.embeddings.embed_query(q) for q in all_queries]

        # average the query_vectors
        avg_vector = np.mean(query_vectors, axis=0).tolist()

        # Retrieve using averaged embedding
        query_filter = SimpleRetrieval._build_filter(party_filter)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=avg_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return response.points

    def _generate_hypothetical_docs(self, query: str, num_docs: int = 3) -> List[str]:
        """Generate hypothetical documents related to the query.

        If LLM is available, uses it to generate realistic documents.
        Otherwise, uses simple heuristic-based approach.
        """
        if self.llm:
            prompt = f"""
                You are a German parliamentary assistant. Generate {num_docs} short hypothetical 
                parliamentary speech excerpts that might contain information about: "{query}"

                Rules:
                - Keep each excerpt concise (1-3 sentences)
                - Make them sound like real parliamentary speeches
                - Use German political terminology where appropriate

                Return ONLY the speech excerpts, one per line. No numbering, no extra text.
            """
            try:
                response = self.llm.invoke(prompt)
                # Handle different response types
                if hasattr(response, "content"):
                    text = response.content
                else:
                    text = str(response)

                docs = text.strip().split("\n")
                cleaned_docs = [
                    d.strip() for d in docs if d.strip() and len(d.strip()) > 10
                ][:num_docs]

                if cleaned_docs:
                    logger.debug(
                        f"Generated {len(cleaned_docs)} hypothetical docs from LLM"
                    )
                    return cleaned_docs
                else:
                    logger.warning(
                        "LLM generated no valid hypothetical docs, using fallback"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to generate hypothetical docs from LLM: {e}, using fallback"
                )

        # Fallback: simple heuristic approach
        fallback_docs = [
            f"speech addressing {query} in the Bundestag",
            f"parliamentary debate about {query} policies",
            f"discussion on {query} implementation and effects",
        ][:num_docs]
        logger.debug("Using fallback hypothetical documents")
        return fallback_docs


class TwoStageRetrieval(RetrievalStrategy):
    """Two-stage retrieval: Bi-Encoder (fast) + Cross-Encoder (reranking).

    First retrieves candidates using fast bi-encoder embeddings,
    then reranks them using a cross-encoder for better precision.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        bi_encoder,
        cross_encoder=None,
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
        rerank_top_k: int = 10,
    ) -> List[models.PointStruct]:
        """Retrieve using two-stage approach."""
        # Stage 1: Fast retrieval with bi-encoder
        candidates = self._bi_encoder_retrieval(
            query, limit=rerank_top_k, party_filter=party_filter
        )

        if not candidates:
            return []

        # Stage 2: Rerank with cross-encoder if available
        if self.cross_encoder:
            candidates = self._cross_encoder_rerank(query, candidates)

        return candidates[:limit]

    def _bi_encoder_retrieval(
        self,
        query: str,
        limit: int,
        party_filter: Optional[str] = None,
    ) -> List[models.PointStruct]:
        """Stage 1: Fast retrieval using bi-encoder embeddings."""
        query_vector = self.bi_encoder.embed_query(query)
        query_filter = SimpleRetrieval._build_filter(party_filter)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return response.points

    def _cross_encoder_rerank(
        self, query: str, candidates: List[models.PointStruct]
    ) -> List[models.PointStruct]:
        """Stage 2: Rerank candidates using cross-encoder."""
        if not candidates:
            return []

        # Prepare pairs: (query, document_text)
        pairs = [[query, candidate.payload.get("text", "")] for candidate in candidates]

        # Get cross-encoder scores
        scores = self.cross_encoder.predict(pairs)

        # Sort by cross-encoder scores
        scored_candidates = [
            (candidate, score) for candidate, score in zip(candidates, scores)
        ]
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Update scores in candidates
        reranked = []
        for candidate, score in scored_candidates:
            candidate.score = float(score)
            reranked.append(candidate)

        return reranked


class PoliticalRAGRetriever:
    """Main retriever for political bias detection using RAG.

    Supports multiple retrieval strategies:
    - Simple: Basic semantic similarity
    - HyDE: Hypothetical Document Embeddings
    - TwoStage: Bi-Encoder + Cross-Encoder reranking
    """

    EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
    CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        chunk_collection: str = "bundestag_speeches_chunks",
        retrieval_mode: str = "simple",
        llm=None,
    ):
        """Initialize Political RAG Retriever.

        Args:
            qdrant_url: URL to Qdrant instance
            chunk_collection: Name of collection with speech chunks
            retrieval_mode: 'simple', 'hyde', or 'two_stage'
            llm: Optional LLM for HyDE document generation
        """
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = chunk_collection
        self.embeddings = HuggingFaceEmbeddings(model_name=self.EMBEDDING_MODEL)

        # Initialize retrieval strategy
        self.retrieval_strategy = self._init_retrieval_strategy(retrieval_mode, llm)

    def _init_retrieval_strategy(
        self, retrieval_mode: str, llm=None
    ) -> RetrievalStrategy:
        """Initialize the appropriate retrieval strategy."""
        mode = retrieval_mode.lower()

        if mode == "hyde":
            # If no LLM provided, initialize Ollama Gemma3
            if llm is None:
                try:
                    logger.info("Initializing Ollama Gemma3 LLM for HyDE...")
                    llm = OllamaLLM(model="gemma3", base_url="http://localhost:11434")
                    # Test connection
                    _ = llm.invoke("test")
                    logger.info("Successfully connected to Ollama Gemma3")
                except Exception as e:
                    logger.warning(
                        f"Failed to connect to Ollama: {e}. "
                        "HyDE will use fallback mode."
                    )
                    llm = None

            return HyDERetrieval(
                self.client, self.collection_name, self.embeddings, llm
            )
        elif mode == "two_stage":
            try:
                from sentence_transformers import CrossEncoder

                cross_encoder = CrossEncoder(self.CROSS_ENCODER_MODEL)
                return TwoStageRetrieval(
                    self.client,
                    self.collection_name,
                    self.embeddings,
                    cross_encoder,
                )
            except ImportError:
                logger.warning(
                    "sentence-transformers not available. "
                    "Falling back to simple retrieval."
                )
                return SimpleRetrieval(
                    self.client, self.collection_name, self.embeddings
                )
        else:  # default to simple
            return SimpleRetrieval(self.client, self.collection_name, self.embeddings)

    def search(
        self,
        query: str,
        limit: int = 3,
        party_filter: Optional[str] = None,
    ) -> List[models.PointStruct]:
        """Execute search using configured retrieval strategy."""
        return self.retrieval_strategy.retrieve(
            query, limit=limit, party_filter=party_filter
        )

    @lru_cache(maxsize=128)
    def _get_cached_speech_uuid(self, raw_speech_id: str) -> str:
        """Cache UUID generation for performance."""
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_speech_id))

    def get_parent_speech(self, raw_speech_id: str) -> Optional[str]:
        """Retrieve full speech text by speech_id."""
        speech_uuid = self._get_cached_speech_uuid(raw_speech_id)

        try:
            result = self.client.retrieve(
                collection_name="bundestag_speeches",
                ids=[speech_uuid],
                with_payload=True,
            )
            return result[0].payload["full_text"] if result else None
        except Exception as e:
            print(f"Error retrieving speech {raw_speech_id}: {e}")
            return None

    def retrieve_for_bias_analysis(
        self,
        query: str,
        party_filter: Optional[str] = None,
        include_metadata: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve relevant context for bias analysis.

        Args:
            query: Query text
            party_filter: Optional filter for specific party
            include_metadata: Whether to include full metadata

        Returns:
            Dictionary with relevant chunk, full context, and metadata
        """
        chunks = self.search(query, limit=1, party_filter=party_filter)

        if not chunks:
            return None

        best_chunk = chunks[0]
        payload = best_chunk.payload

        speech_id = payload.get("speech_id")
        full_text = self.get_parent_speech(speech_id) if speech_id else None

        result = {
            "relevant_chunk": payload.get("text"),
            "full_context": full_text,
            "similarity_score": best_chunk.score,
        }

        if include_metadata:
            result["metadata"] = {
                "speech_id": speech_id,
                "speaker": payload.get("speaker"),
                "party": payload.get("party"),
                "date": payload.get("date"),
                "interjections": payload.get("interjections", []),
            }

        return result

    def batch_retrieve(
        self,
        queries: List[str],
        limit: int = 3,
        party_filter: Optional[str] = None,
    ) -> List[List[models.PointStruct]]:
        """Retrieve for multiple queries efficiently."""
        return [
            self.search(query, limit=limit, party_filter=party_filter)
            for query in queries
        ]

    def print_results(
        self, points: List[models.PointStruct], verbose: bool = False
    ) -> None:
        """Print formatted retrieval results."""
        print("-" * 70)
        if not points:
            print("No results found.")
            return

        for i, hit in enumerate(points, 1):
            payload = hit.payload
            self._print_hit(i, hit, verbose)

        print("-" * 70)

    @staticmethod
    def _print_hit(index: int, hit: models.PointStruct, verbose: bool = False) -> None:
        """Print a single hit result."""
        payload = hit.payload

        print(f"\nResult {index} | Similarity Score: {hit.score:.4f}")
        print(
            f"Speaker: {payload.get('speaker')} ({payload.get('party')}) | "
            f"Date: {payload.get('date')}"
        )
        print(f"Speech ID: {payload.get('speech_id')}")
        print(f"Text:\n{payload.get('text')[:500]}...")  # First 500 chars

        interjections = payload.get("interjections", [])
        if interjections:
            print(f"Interjections: {interjections}")

        if verbose:
            print(f"Full Payload: {payload}")
