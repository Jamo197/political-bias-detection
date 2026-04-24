import json
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv(Path(".env.local"))
os.environ["HF_TOKEN"] = ""


class PoliticalRAGIngestor:

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        chunk_collection: str = "bundestag_speeches_chunks",
        full_spech_collection: str = "bundestag_speeches",
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.chunk_collection = chunk_collection
        self.full_spech_collection = full_spech_collection

        self.embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
        )

        self.semantic_splitter = SemanticChunker(
            self.embeddings, breakpoint_threshold_type="percentile"
        )
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )

        self._setup_collections()

    def _setup_collections(self):
        """Erstellt die Vektor-Datenbank und die Payload-Indizes."""
        collections = self.client.get_collections().collections

        if not any(c.name == self.chunk_collection for c in collections):
            self.client.create_collection(
                collection_name=self.chunk_collection,
                vectors_config=models.VectorParams(
                    size=384, distance=models.Distance.COSINE
                ),
            )

            # Payload Indizes für Ground-Truth Filterung UND Parent Document Retrieval
            index_fields = [
                "party",
                "speaker",
                "year",
                "legislative_period",
                "speech_id",
            ]
            for field in index_fields:
                schema = (
                    models.PayloadSchemaType.INTEGER
                    if field in ["year", "legislative_period"]
                    else models.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    collection_name=self.chunk_collection,
                    field_name=field,
                    field_schema=schema,
                )
            print(
                f"Collection '{self.chunk_collection}' mit strukturierten Indizes erstellt."
            )

        if not any(c.name == self.full_spech_collection for c in collections):
            self.client.create_collection(
                collection_name=self.full_spech_collection,
                vectors_config={},
            )

    def process_and_upload(self, file_path: str, batch_size: int = 50):
        """Liest das JSON, führt das Hybrid-Chunking durch und lädt in Qdrant hoch."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        points = []
        for entry in data:
            text = entry.get("text", "")
            metadata = entry.get("metadata", {})

            raw_speech_id = metadata.get("speech_id")
            if not raw_speech_id:
                continue

            speech_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_speech_id))

            # 0. Speichere die komplette Rede EINMAL pro speech_id
            self.client.upsert(
                collection_name=self.full_spech_collection,
                points=[
                    models.PointStruct(
                        id=speech_uuid,
                        vector={},
                        payload={"full_text": text, **metadata},
                    )
                ],
            )

            # 1. Semantisches Chunking (Originaltext inkl. Zwischenrufe)
            semantic_chunks = self.semantic_splitter.create_documents([text])

            final_chunks = []
            # 2. Chunk Size Limits durchsetzen
            for chunk in semantic_chunks:
                if len(chunk.page_content) > 1000:
                    # Wenn der Chunk zu groß ist, greift der Charakter-Splitter
                    final_chunks.extend(
                        self.fallback_splitter.create_documents([chunk.page_content])
                    )
                else:
                    final_chunks.append(chunk)

            # 3. Vektorisierung & Upload
            for i, chunk in enumerate(final_chunks):
                chunk_id = str(uuid.uuid4())
                vector = self.embeddings.embed_query(chunk.page_content)

                payload = {
                    "chunk_index": i,
                    "text": chunk.page_content,
                    **metadata,
                }

                points.append(
                    models.PointStruct(id=chunk_id, vector=vector, payload=payload)
                )

                if len(points) >= batch_size:
                    self.client.upsert(
                        collection_name=self.chunk_collection, points=points
                    )
                    points = []

        if points:
            self.client.upsert(collection_name=self.chunk_collection, points=points)
        print("Upload abgeschlossen.")


if __name__ == "__main__":
    ingestor = PoliticalRAGIngestor()
    ingestor.process_and_upload(
        "extraction/datasets/bundestag_data/WP_21/2026-01/speeches/2026-01-14_5763_speeches.json"
    )
