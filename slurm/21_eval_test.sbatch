#!/bin/bash
#SBATCH --job-name=eval_test_all
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/eval_test_all_%j.out
#SBATCH --error=logs/slurm/eval_test_all_%j.err

# Quick smoke-test: all retrieval strategies + no-RAG baseline (10 samples).
# Loads multilingual-e5-large-instruct (~2GB) + cross-encoder (~0.5GB).
#
# Prerequisites:
#   1. Qdrant server running with chunks_e5 collection populated.
#   2. Run: bash slurm/setup_venv.sh   (one-time, on a login node)
#
# Can be deleted afterwards
set -euo pipefail

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm logs/tmp

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

VLLM_HOST="http://$(cat logs/slurm/vllm_ministral_host.txt 2>/dev/null || echo "")"
if [[ -z "$VLLM_HOST" ]]; then
    echo "ERROR: vLLM host not found in logs/slurm/vllm_ministral_host.txt"
    exit 1
fi
echo "vLLM server: $VLLM_HOST"

RUN_ID="${RUN_ID:-test_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-logs/tmp/${RUN_ID}}"
export RUN_ID RUN_DIR

echo "Run ID: $RUN_ID  |  Run dir: $RUN_DIR"
echo "Python -> $VENV_PY ($($VENV_PY --version 2>&1))"

SAMPLE_SIZE=10
K_CHUNKS=5
RANDOM_SEED=33

# 1) No-RAG baseline
echo "=== 1/2 No-RAG baseline ==="
"$VENV_PY" -m src.run_batch \
    --no_rag \
    --sample_size "$SAMPLE_SIZE" \
    --random_seed "$RANDOM_SEED" \
    --llm_base_url "${VLLM_HOST}/v1" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

# 2) All retrieval strategies
echo "=== 2/2 RAG strategies ==="
"$VENV_PY" -m src.run_batch \
    --embedding_model e5 \
    --strategies simple,hyde,twostage \
    --device cuda \
    --qdrant_url "$QDRANT_URL" \
    --k_chunks "$K_CHUNKS" \
    --sample_size "$SAMPLE_SIZE" \
    --random_seed "$RANDOM_SEED" \
    --llm_base_url "${VLLM_HOST}/v1" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo "Test run complete. Logs in: $RUN_DIR"
