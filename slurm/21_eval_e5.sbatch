#!/bin/bash
#SBATCH --job-name=eval_e5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/eval_e5_%j.out
#SBATCH --error=logs/slurm/eval_e5_%j.err

# Evaluate e5 embeddings with all retrieval strategies (simple, hyde, twostage).
# Loads multilingual-e5-large-instruct (~2GB) + cross-encoder (~0.5GB) +
# HyDE LLM Qwen2.5-0.5B-Instruct (~1GB) on one GPU.
#
# Prerequisites:
#   1. Qdrant server running with chunks_e5 collection populated.
#   2. Run: bash slurm/setup_venv.sh   (one-time, on a login node)
#
# All eval jobs share the same RUN_ID and RUN_DIR so logs land together.
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

source slurm/qdrant_ensure.sh
ensure_qdrant
echo "Qdrant -> $QDRANT_URL"

RUN_ID="${RUN_ID:-eval_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-logs/batch_runs/$(date +%Y-%m-%d)_${RUN_ID}}"
export RUN_ID RUN_DIR

echo "Run ID: $RUN_ID  |  Run dir: $RUN_DIR"
echo "Python -> $VENV_PY ($($VENV_PY --version 2>&1))"

SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
K_CHUNKS="${K_CHUNKS:-5}"
RANDOM_SEED="${RANDOM_SEED:-33}"
HYDE_MODEL="${HYDE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"

"$VENV_PY" -m src.run_batch \
    --embedding_model e5 \
    --strategies simple,hyde,twostage \
    --device cuda \
    --qdrant_url "$QDRANT_URL" \
    --k_chunks "$K_CHUNKS" \
    --sample_size "$SAMPLE_SIZE" \
    --random_seed "$RANDOM_SEED" \
    --hyde_model "$HYDE_MODEL" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"
