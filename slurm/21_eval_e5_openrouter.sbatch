#!/bin/bash
#SBATCH --job-name=eval_openrouter
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/eval_openrouter_%j.out
#SBATCH --error=logs/slurm/eval_openrouter_%j.err

# no-RAG baseline + e5 RAG via OpenRouter
# Embeddings (e5) and LLM (mistral-large) both go through OpenRouter —
# no vLLM server needed. GPU is only used for the cross-encoder (twostage).
#
# Prerequisites:
#   1. Qdrant server running with chunks_e5 collection populated.
#   2. Run: bash slurm/setup_venv.sh   (one-time, on a login node)
#   3. OPENROUTER_API_KEY set in environment
#
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

RUN_ID="${RUN_ID:-eval_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-logs/batch_runs/$(date +%Y-%m-%d)_${RUN_ID}}"
export RUN_ID RUN_DIR

echo "Run ID: $RUN_ID  |  Run dir: $RUN_DIR"
echo "Python -> $VENV_PY ($($VENV_PY --version 2>&1))"

SAMPLE_SIZE=2
K_CHUNKS=5
RANDOM_SEED=33

echo "=== 2/2 e5 RAG strategies (OpenRouter) ==="
"$VENV_PY" -m src.run_batch \
    --embedding_model e5 \
    --strategies simple,hyde,twostage \
    --qdrant_url "$QDRANT_URL" \
    --k_chunks "$K_CHUNKS" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

echo "Test run complete. Logs in: $RUN_DIR"