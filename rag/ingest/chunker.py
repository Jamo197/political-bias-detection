"""Stage 1 of the HPC ingestion pipeline: chunk the corpus EXACTLY ONCE.

The same chunk set is later embedded by all four models so the retrieval
comparison is not confounded by differing chunk boundaries. Chunk IDs are
deterministic (uuid5 of speech_id + chunk_index), so:
  * the identical chunk_id is reused across all four model collections, and
  * re-running ingestion is idempotent (upserts overwrite, never duplicate).

Outputs (JSON Lines):
  chunks.jsonl         one record per chunk (text + metadata + ids)
  full_speeches.jsonl  one record per speech_id (parent-document collection)

The SemanticChunker needs *an* embedder only to detect breakpoints; this does
NOT affect the stored comparison vectors. We keep the original MiniLM model used
on the local machine so chunk boundaries match the prior methodology.

CHUNKING TEST STRATEGY:
    Configuration A: MiniLM + MAX=1000
    Configuration B: e5-small + MAX=1000
    Configuration C: e5-small + MAX=1500
    Configuration D: e5-small + MAX=2000
    Configuration F: MiniLM + MAX=2000
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from ._semantic_chunker import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .config import (
    CHUNKER_BREAKPOINT_MODEL,
    FALLBACK_CHUNK_OVERLAP,
    FALLBACK_CHUNK_SIZE,
    MAX_CHUNK_CHARS,
    MIN_CHUNK_CHARS,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env.local")


def chunk_uuid(speech_id: str, chunk_index: int) -> str:
    """Deterministic, collision-free chunk id shared across all collections."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{speech_id}:{chunk_index}"))


def speech_uuid(speech_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, speech_id))


class CorpusChunker:
    def __init__(self) -> None:
        model_name = CHUNKER_BREAKPOINT_MODEL
        self.max_chunk_chars = MAX_CHUNK_CHARS
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.semantic_splitter = SemanticChunker(
            self.embeddings,
            breakpoint_threshold_type="percentile",
            min_chunk_size=MIN_CHUNK_CHARS,
            add_start_index=True,
        )
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_chars,
            chunk_overlap=FALLBACK_CHUNK_OVERLAP,
            add_start_index=True,
        )

    def _split_text(self, text: str) -> list[dict]:
        semantic_chunks = self.semantic_splitter.create_documents([text])
        final: list[dict] = []
        for chunk in semantic_chunks:
            content = chunk.page_content
            base_offset = chunk.metadata.get("start_index", 0)
            if len(content) > self.max_chunk_chars:
                for d in self.fallback_splitter.create_documents([content]):
                    sub_offset = d.metadata.get("start_index", 0)
                    final.append(
                        {
                            "text": d.page_content,
                            "start_index": base_offset + sub_offset,
                            "from_fallback": True,
                        }
                    )
            else:
                final.append(
                    {
                        "text": content,
                        "start_index": base_offset,
                        "from_fallback": False,
                    }
                )
        return final

    def iter_files(self, base_path: Path) -> list[Path]:
        files = sorted(base_path.rglob("speeches/*_cleaned.json"))
        if not files:
            # Fallback: any *_cleaned.json under the tree.
            files = sorted(base_path.rglob("*_cleaned.json"))
        return files

    def process(
        self, base_path: Path, chunks_out: Path, speeches_out: Path
    ) -> tuple[int, int]:
        files = self.iter_files(base_path)
        if not files:
            raise FileNotFoundError(f"No speech JSON files found under {base_path}")

        seen_speeches: set[str] = set()
        n_chunks = 0
        n_speeches = 0

        chunks_out.parent.mkdir(parents=True, exist_ok=True)
        with (
            chunks_out.open("w", encoding="utf-8") as cf,
            speeches_out.open("w", encoding="utf-8") as sf,
        ):
            for file_path in tqdm(files, desc="Chunking speech files"):
                try:
                    data = json.loads(file_path.read_text(encoding="utf-8"))
                except Exception as e:  # noqa: BLE001 - never abort the whole run
                    print(f"\n[WARN] Skipping unreadable {file_path.name}: {e}")
                    continue

                for entry in data:
                    text = entry.get("text", "") or ""
                    metadata = entry.get("metadata", {}) or {}
                    interjections = entry.get("interjections", []) or []
                    raw_speech_id = metadata.get("speech_id")
                    if not raw_speech_id or not text.strip():
                        continue

                    # Parent document: write once per speech_id.
                    if raw_speech_id not in seen_speeches:
                        seen_speeches.add(raw_speech_id)
                        sf.write(
                            json.dumps(
                                {
                                    "id": speech_uuid(raw_speech_id),
                                    "speech_id": raw_speech_id,
                                    "full_text": text,
                                    "interjections": interjections,
                                    **metadata,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        n_speeches += 1

                    for idx, chunk in enumerate(self._split_text(text)):
                        record = {
                            "chunk_id": chunk_uuid(raw_speech_id, idx),
                            "speech_id": raw_speech_id,
                            "chunk_index": idx,
                            "start_index": chunk["start_index"],
                            "from_fallback": chunk["from_fallback"],
                            "text": chunk["text"],
                            "interjections": interjections,
                            **metadata,
                        }
                        cf.write(json.dumps(record, ensure_ascii=False) + "\n")
                        n_chunks += 1

        return n_chunks, n_speeches


def read_chunks(chunks_path: Path) -> Iterator[dict]:
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: chunk corpus once.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_PROJECT_ROOT / "extraction/datasets/bundestag_data",
        help="Root directory containing speeches/*_cleaned.json files.",
    )
    parser.add_argument(
        "--chunks-out",
        type=Path,
        default=_PROJECT_ROOT / "rag/ingest/artifacts/chunks.jsonl",
    )
    parser.add_argument(
        "--speeches-out",
        type=Path,
        default=_PROJECT_ROOT / "rag/ingest/artifacts/full_speeches.jsonl",
    )
    args = parser.parse_args()

    chunker = CorpusChunker()
    n_chunks, n_speeches = chunker.process(
        args.data_dir, args.chunks_out, args.speeches_out
    )
    print(
        f"Done. {n_chunks} chunks from {n_speeches} speeches.\n"
        f"  chunks   -> {args.chunks_out}\n"
        f"  speeches -> {args.speeches_out}"
    )


if __name__ == "__main__":
    main()
