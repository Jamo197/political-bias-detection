#!/bin/bash
#SBATCH --job-name=ingest_bge
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/ingest_bge_%j.out
#SBATCH --error=logs/slurm/ingest_bge_%j.err

# Embed chunks with bge-m3 via FlagEmbedding (dense + sparse hybrid, no ColBERT),
# using the HPC venv (.venv-hpc) with CUDA-12.8-compatible torch.
#
# Prerequisites:
#   1. Run: bash slurm/setup_venv.sh   (one-time, on a login node)
#   2. Qdrant server is auto-started if not already running (01_qdrant.sbatch).
set -euo pipefail

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VENV_PY="${VENV_PY:-$PWD/.venv-hpc/bin/python}"
if [[ ! -f "$VENV_PY" ]]; then
    echo "ERROR: venv not found at $VENV_PY"
    echo "  Run: bash slurm/setup_venv.sh"
    exit 1
fi

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"; mkdir -p "$HF_HOME"

# Ensure the Qdrant server is running (auto-starts 01_qdrant.sbatch if needed).
source slurm/qdrant_ensure.sh
ensure_qdrant
echo "Qdrant -> $QDRANT_URL"
echo "Python -> $VENV_PY ($($VENV_PY --version 2>&1))"

# --skip-parents: parents already uploaded by the first model job (idempotent).
"$VENV_PY" -m rag.ingest.ingest \
    --model bge \
    --chunks rag/ingest/artifacts/chunks.jsonl \
    --speeches rag/ingest/artifacts/full_speeches.jsonl \
    --qdrant-url "$QDRANT_URL" \
    --qdrant-timeout 300 \
    --device cuda \
    --batch-size 32 \
    --reset \
    --skip-parents
