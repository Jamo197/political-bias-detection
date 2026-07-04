#!/bin/bash
#SBATCH --job-name=ingest_e5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/ingest_e5_%j.out
#SBATCH --error=logs/slurm/ingest_e5_%j.err

# Embed chunks with multilingual-e5-large-instruct (SentenceTransformers),
# running Python inside vllm.sif.
set -euo pipefail

module load singularity || module load apptainer || true
cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VLLM_SIF="${VLLM_SIF:-/home/users/j/$USER/hpc-images/vllm.sif}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"; mkdir -p "$HF_HOME"

# Point at the Qdrant server job (auto-resolve from its host file, or override).
if [[ -z "${QDRANT_URL:-}" && -f logs/slurm/qdrant_host.txt ]]; then
    export QDRANT_URL="http://$(cat logs/slurm/qdrant_host.txt)"
fi
export QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"
echo "Qdrant -> $QDRANT_URL"

# Use CUDA-compatible torch if installed (see slurm/setup_torch_compat.sh).
# The container's torch is compiled for CUDA >=12.9 but the HPC driver is 12.8.
PIP_COMPAT="${PIP_COMPAT:-$PWD/.pip_compat}"
SING_PYTHONPATH=""
if [[ -d "$PIP_COMPAT/torch" ]]; then
    export PYTHONPATH="$PIP_COMPAT:${PYTHONPATH:-}"
    SING_PYTHONPATH="--env PYTHONPATH=$PYTHONPATH"
    echo "Using CUDA-compatible torch from $PIP_COMPAT"
else
    echo "No CUDA-compatible torch at $PIP_COMPAT — will fall back to CPU."
    echo "  For GPU: srun --partition=gpu --gres=gpu:1 --time=30:00 --mem=16G bash slurm/setup_torch_compat.sh"
fi

singularity exec --nv \
    --bind "$PWD:$PWD" --pwd "$PWD" \
    --env HF_HOME="$HF_HOME" --env QDRANT_URL="$QDRANT_URL" \
    $SING_PYTHONPATH \
    "$VLLM_SIF" \
    python -m rag.ingest.ingest \
        --model e5 \
        --chunks rag/ingest/artifacts/chunks.jsonl \
        --speeches rag/ingest/artifacts/full_speeches.jsonl \
        --qdrant-url "$QDRANT_URL" \
        --qdrant-timeout 300 \
        --device cuda \
        --batch-size 64 \
        --reset
