#!/bin/bash
#SBATCH --job-name=vllm_qwen3_embed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1              # Qwen3-Embedding-8B: bf16 ~16GB weights + KV cache; needs >=24GB VRAM -> avoid P100/V100-16GB, prefer A100-40G/V100-32GB/H100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/vllm_qwen3_%j.out
#SBATCH --error=logs/slurm/vllm_qwen3_%j.err

# Launch the vLLM OpenAI-compatible embedding server for Qwen3-Embedding-8B,
# from vllm.sif. Keep this job RUNNING while 13_ingest_qwen3.sbatch talks to it.
set -euo pipefail

module load singularity || module load apptainer || true

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VLLM_SIF="${VLLM_SIF:-/home/users/j/$USER/hpc-images/vllm.sif}"
VLLM_PORT="${VLLM_PORT:-8000}"

# Cache HF weights on scratch so they are not re-downloaded each run.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
mkdir -p "$HF_HOME"

echo "$(hostname):${VLLM_PORT}" > logs/slurm/vllm_qwen3_host.txt
echo "vLLM host -> $(cat logs/slurm/vllm_qwen3_host.txt)"

# --nv exposes the GPU drivers to the container.
# --task embed enables the embeddings endpoint; vLLM handles left-padding.
exec singularity exec --nv \
    --bind "$PWD:$PWD" \
    --pwd "$PWD" \
    --env HF_HOME="$HF_HOME" \
    "$VLLM_SIF" \
    vllm serve Qwen/Qwen3-Embedding-8B \
        --task embed \
        --host 0.0.0.0 \
        --port "${VLLM_PORT}" \
        --dtype bfloat16 \
        --trust-remote-code
