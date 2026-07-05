#!/bin/bash
#SBATCH --job-name=ingest_e5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/ingest_e5_%j.out
#SBATCH --error=logs/slurm/ingest_e5_%j.err

# Embed chunks with multilingual-e5-large-instruct (SentenceTransformers),
# using the HPC venv (.venv-hpc) with CUDA-12.8-compatible torch.
#
# Prerequisites:
#   1. Run: bash slurm/setup_venv.sh   (one-time, on a login node)
#   2. Start: sbatch slurm/01_qdrant.sbatch   (Qdrant server must be running)
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

# Point at the Qdrant server job (auto-resolve from its host file, or override).
if [[ -z "${QDRANT_URL:-}" && -f logs/slurm/qdrant_host.txt ]]; then
    export QDRANT_URL="http://$(cat logs/slurm/qdrant_host.txt)"
fi
export QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
echo "Qdrant -> $QDRANT_URL"
echo "Python -> $VENV_PY ($($VENV_PY --version 2>&1))"

"$VENV_PY" -m rag.ingest.ingest \
    --model e5 \
    --chunks rag/ingest/artifacts/chunks.jsonl \
    --speeches rag/ingest/artifacts/full_speeches.jsonl \
    --qdrant-url "$QDRANT_URL" \
    --qdrant-timeout 300 \
    --device cuda \
    --batch-size 64 \
    --reset
