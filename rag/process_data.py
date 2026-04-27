import json
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

load_dotenv(Path(".env.local"))
# FIXME: setup token correctly
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

    def reset_collections(self):
        """Löscht die bestehenden Collections für einen sauberen Neustart."""
        try:
            self.client.delete_collection(self.chunk_collection)
            self.client.delete_collection(self.full_spech_collection)
            print("Erfolg: Alte Collections wurden gelöscht. Starte bei Null.")
        except Exception as e:
            print(
                f"Hinweis beim Löschen (kann ignoriert werden, wenn sie nicht existieren): {e}"
            )

        # Baut die Collections inkl. Indizes direkt wieder frisch auf
        self._setup_collections()

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


class PartyNormalizer:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        collection_name: str = "bundestag_speeches_chunks",
    ):
        self.client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name

    def clean_party_name(self, raw_name: str) -> str:
        """
        Zentrale Mapping-Funktion, die String-Artefakte bereinigt.
        """
        if not raw_name:
            return "UNKNOWN"

        # 1. Alle mehrfachen Leerzeichen und Zeilenumbrüche (\n) in ein einzelnes Leerzeichen umwandeln
        clean = re.sub(r"\s+", " ", raw_name).strip()
        clean_upper = clean.upper()

        # 2. Regelbasiertes Mapping
        if "GRÜNEN" in clean_upper:
            return "BÜNDNIS 90/DIE GRÜNEN"
        elif "LINKE" in clean_upper:
            return "DIE LINKE"
        elif "FRAKTIONSLOS" in clean_upper:
            return "Fraktionslos"
        elif clean_upper == "SPDSPD":
            return "SPD"
        elif clean_upper == "SPDCDU/CSU":
            # Parsing-Fehler, bei dem zwei Parteien verschmolzen sind.
            # Für eine saubere Ground-Truth-Evaluation setzen wir dies auf UNKNOWN.
            return "UNKNOWN"
        elif clean_upper == "CDU/CSU":
            return "CDU/CSU"
        elif clean_upper == "SPD":
            return "SPD"
        elif clean_upper == "AFD":
            return "AfD"
        elif clean_upper == "FDP":
            return "FDP"
        elif clean_upper == "BSW":
            return "BSW"

        # Fallback
        return clean

    def normalize_database(self):
        print(f"Starte Normalisierung für Collection: '{self.collection_name}'...\n")

        # 1. Alle aktuellen (schmutzigen) Facetten abrufen
        response = self.client.facet(
            collection_name=self.collection_name, key="party", limit=100, exact=True
        )

        raw_parties = [hit.value for hit in response.hits]

        # 2. Iteriere über alle vorhandenen Parteinamen
        for raw_party in raw_parties:
            clean_party = self.clean_party_name(raw_party)

            # Wenn der Name bereits sauber ist, überspringen wir ihn
            if raw_party == clean_party:
                continue

            print(f"Korrigiere: '{raw_party}'  --->  '{clean_party}'")

            # 3. Alle Point-IDs finden, die diesen schmutzigen Namen haben
            # Da einige Kategorien tausende Einträge haben, nutzen wir scroll()
            offset = None
            total_updated = 0

            while True:
                records, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="party", match=models.MatchValue(value=raw_party)
                            )
                        ]
                    ),
                    limit=2000,  # Batch-Größe
                    with_payload=False,  # Wir brauchen nur die IDs, keinen Text/Vektoren
                    with_vectors=False,
                )

                point_ids = [record.id for record in records]

                if point_ids:
                    # 4. In-Place Update: Wir überschreiben NUR den Key "party",
                    # alles andere (text, speaker, date, interjections) bleibt erhalten!
                    self.client.set_payload(
                        collection_name=self.collection_name,
                        payload={"party": clean_party},
                        points=point_ids,
                    )
                    total_updated += len(point_ids)

                if offset is None:
                    break

            print(f"  -> {total_updated} Chunks erfolgreich aktualisiert.")

        print("\nNormalisierung abgeschlossen!")


if __name__ == "__main__":
    normalizer = PartyNormalizer()
    normalizer.normalize_database()

# if __name__ == "__main__":
#     ingestor = PoliticalRAGIngestor()

#     # 2. Definiere den Basis-Pfad zu deinen Daten
#     # Passe den Pfad an, je nachdem von wo du das Skript ausführst
#     base_data_path = Path("extraction/datasets/bundestag_data")

#     # 3. Finde alle JSON-Dateien rekursiv (rglob)
#     # print("Schritt 2: Suche nach Dateien...")
#     # json_files = list(base_data_path.rglob("speeches/*.json"))

#     # if not json_files:
#     #     print(f"Fehler: Keine JSON-Dateien im Pfad {base_data_path} gefunden.")
#     # else:
#     #     print(f"{len(json_files)} Dateien gefunden. Starte Massen-Upload...")

#     #     # 4. Iteriere durch alle Dateien mit Fortschrittsbalken
#     #     for file_path in tqdm(json_files, desc="Verarbeite Legislaturperioden"):
#     #         try:
#     #             # wandle das Path-Objekt für die Methode in einen String um
#     #             ingestor.process_and_upload(str(file_path))
#     #         except Exception as e:
#     #             # Error Handling: Verhindert, dass das ganze Skript bei einer defekten JSON abstürzt
#     #             print(f"\nFehler beim Verarbeiten von {file_path.name}: {e}")

#     #     print(
#     #         "\nUpload komplett abgeschlossen! Das RAG-System ist bereit für die Bias-Klassifizierung."
#     #     )
#     #     )
