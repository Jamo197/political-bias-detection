#!/bin/bash
#SBATCH --job-name=vllm_ministral_instruct
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/vllm_ministral_%j.out
#SBATCH --error=logs/slurm/vllm_ministral_%j.err

# Launch the vLLM OpenAI-compatible chat server for Ministral-3-8B-Instruct,
# from vllm.sif. Keep this job RUNNING while evaluation scripts query it.
set -euo pipefail

module load singularity || module load apptainer || true

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VLLM_SIF="${VLLM_SIF:-/home/users/j/$USER/hpc-images/vllm.sif}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Cache HF weights on scratch so they are not re-downloaded each run.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
mkdir -p "$HF_HOME"

echo "$(hostname):${VLLM_PORT}" > logs/slurm/vllm_ministral_host.txt
echo "vLLM host -> $(cat logs/slurm/vllm_ministral_host.txt)"

# --nv exposes the GPU drivers to the container.
exec singularity exec --nv \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    --env HF_HOME="$HF_HOME" \
    "$VLLM_SIF" \
    vllm serve mistralai/Ministral-3-8B-Instruct-2512 \
        --host 0.0.0.0 \
        --port "${VLLM_PORT}" \
        --dtype bfloat16 \
        --trust-remote-code
