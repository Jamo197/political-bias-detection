#!/bin/bash
#SBATCH --job-name=eval_norag
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/eval_norag_%j.out
#SBATCH --error=logs/slurm/eval_norag_%j.err

# No-RAG baseline evaluation. No Qdrant, no vLLM, no GPU needed.
# Runs BiasPredictor on the dataset without any retrieved context.
#
# All eval jobs share the same RUN_ID and RUN_DIR so logs land together.
# Set RUN_ID before submitting if you want a specific run identifier:
#   RUN_ID="my_eval" sbatch slurm/20_eval_norag.sbatch
set -euo pipefail

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VENV_PY="${VENV_PY:-$PWD/.venv-hpc/bin/python}"
if [[ ! -f "$VENV_PY" ]]; then
    echo "ERROR: venv not found at $VENV_PY"
    echo "  Run: bash slurm/setup_venv.sh"
    exit 1
fi

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

RUN_ID="${RUN_ID:-eval_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-logs/batch_runs/$(date +%Y-%m-%d)_${RUN_ID}}"
export RUN_ID RUN_DIR

echo "Run ID: $RUN_ID  |  Run dir: $RUN_DIR"
echo "Python -> $VENV_PY ($($VENV_PY --version 2>&1))"

SAMPLE_SIZE="${SAMPLE_SIZE:-50}"
K_CHUNKS="${K_CHUNKS:-5}"
RANDOM_SEED="${RANDOM_SEED:-33}"

VLLM_HOST=$(cat logs/slurm/vllm_ministral_host.txt 2>/dev/null || echo "")
if [[ -z "$VLLM_HOST" ]]; then
    echo "ERROR: vLLM host not found in logs/slurm/vllm_ministral_host.txt"
    exit 1
fi
echo "vLLM server: $VLLM_HOST"

"$VENV_PY" -m src.run_batch \
    --no_rag \
    --llm_base_url "http://${VLLM_HOST}/v1" \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR"

# NEXT STEPS:
# Run for different LLM