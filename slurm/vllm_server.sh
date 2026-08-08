#!/bin/bash
#SBATCH --job-name=vllm_server
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/vllm_server_%j.out
#SBATCH --error=logs/slurm/vllm_server_%j.err

set -euo pipefail

module load singularity || module load apptainer || true

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VLLM_SIF="${VLLM_SIF:-/home/users/j/$USER/hpc-images/vllm.sif}"
VLLM_PORT="${VLLM_PORT:-8000}"

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-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}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN must be set in the environment (required for gated models)."
    exit 1
fi

# Pass the model ID as a variable so you can reuse this script for any LLM
TARGET_LLM="${1:-mistralai/Ministral-3-8B-Instruct-2512}"

echo "$(hostname):${VLLM_PORT}" > logs/slurm/vllm_active_host.txt
echo "Starting vLLM for $TARGET_LLM on host -> $(cat logs/slurm/vllm_active_host.txt)"

exec singularity exec --nv \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    --env HF_HOME="$HF_HOME" \
    --env HF_TOKEN="$HF_TOKEN" \
    --env VLLM_USE_FLASHINFER_SAMPLER="0" \
    "$VLLM_SIF" \
    vllm serve "$TARGET_LLM" \
        --host 0.0.0.0 \
        --port "${VLLM_PORT}" \
        --dtype bfloat16 \
        --trust-remote-code \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.85