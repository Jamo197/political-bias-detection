"""Multi-model Qdrant ingestion pipeline for HPC.

Stage 1 (chunker.py):  chunk the corpus once -> chunks.jsonl
Stage 2 (ingest.py):   embed chunks with one of {e5, bge, jina, qwen3} -> Qdrant
"""
