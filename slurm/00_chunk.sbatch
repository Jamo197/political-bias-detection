#!/bin/bash
#SBATCH --job-name=chunk_corpus
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/chunk_%j.out
#SBATCH --error=logs/slurm/chunk_%j.err

# Stage 1: chunk the corpus ONCE. CPU-only (semantic breakpoint model is small).
# Runs Python INSIDE vllm.sif (it already contains torch + transformers +
# sentence-transformers + langchain). No host venv needed.
set -euo pipefail

module load singularity || module load apptainer || true

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm rag/ingest/artifacts

VLLM_SIF="${VLLM_SIF:-$PWD/vllm.sif}"

# --bind $PWD so the container sees the repo; --pwd sets the working dir inside.
singularity exec \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    "$VLLM_SIF" \
    python -m rag.ingest.chunker \
        --data-dir extraction/datasets/bundestag_data \
        --chunks-out rag/ingest/artifacts/chunks.jsonl \
        --speeches-out rag/ingest/artifacts/full_speeches.jsonl

echo "Chunking complete."
