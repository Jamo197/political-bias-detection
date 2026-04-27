import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv(Path(".env.local"))
# FIXME: setup token correctly
os.environ["HF_TOKEN"] = ""


class PoliticalRAGRetriever:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        chunk_collection: str = "bundestag_speeches_chunks",
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = chunk_collection

        # Das exakt selbe Embedding-Modell wie beim Ingest
        self.embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

    def search(self, query: str, limit: int = 3, party_filter: str = None):
        """
        Führt eine semantische Suche durch, optional mit hartem Filter auf eine Partei.
        Nutzt den modernen query_points Endpunkt von Qdrant.
        """
        # print(f"Suche nach: '{query}'")

        query_vector = self.embeddings.embed_query(query)

        query_filter = None
        if party_filter:
            # print(f"Aktiver Filter: Partei = {party_filter}")
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="party", match=models.MatchValue(value=party_filter)
                    )
                ]
            )

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        # self._print_results(response.points)
        return response.points

    def get_parent_speech(self, raw_speech_id: str):
        """Holt den Volltext basierend auf der speech_id."""
        speech_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_speech_id))

        result = self.client.retrieve(
            collection_name="bundestag_speeches",
            ids=[speech_uuid],
            with_payload=True,
        )

        return result[0].payload["full_text"] if result else None

    def retrieve_for_bias_analysis(self, query: str, party_filter: str = None):
        chunks = self.search(query, limit=1, party_filter=party_filter)

        if not chunks:
            return None

        best_chunk = chunks[0]
        speech_id = best_chunk.payload["speech_id"]

        full_text = self.get_parent_speech(speech_id)

        return {
            "relevant_chunk": best_chunk.payload["text"],
            "full_context": full_text,
            "metadata": best_chunk.payload,
        }

    def _print_results(self, points):
        print("-" * 50)
        if not points:
            print("Keine Ergebnisse gefunden.")
            return

        for i, hit in enumerate(points, 1):
            score = hit.score
            payload = hit.payload

            print(f"Treffer {i} | Score: {score:.4f}")
            print(
                f"Redner: {payload.get('speaker')} ({payload.get('party')}) | Datum: {payload.get('date')}"
            )
            print(f"Speech ID (für Parent Retrieval): {payload.get('speech_id')}")
            print(f"Text-Chunk:\n{payload.get('text')}")

            interjections = payload.get("interjections", [])
            if interjections:
                print(f"Zwischenrufe im Originaltext: {interjections}")
            print("-" * 50)


# --- Test-Ausführung ---
if __name__ == "__main__":
    retriever = PoliticalRAGRetriever()

    # Test 1: Baseline Retrieval (Ungesteuert)
    text = retriever.search(
        query="#Inflation trifft Geringverdiener besonders hart,  da diese einen viel größeren Anteil ihres Einkommens für #Miete,  Lebensmittel,  #Strom &amp; Heizung ausgeben. Daher: #Mindestlohn von 13 Euro,  #Mindestrente von 1200 Euro &amp; bundesweiten #Mietendeckel einführen! ",
        limit=2,
    )
    print(text)
