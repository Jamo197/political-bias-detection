#!/bin/bash
#SBATCH --job-name=ingest_qwen3
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/ingest_qwen3_%j.out
#SBATCH --error=logs/slurm/ingest_qwen3_%j.err

# Embed chunks with Qwen3-Embedding-8B by calling the vLLM server (02_vllm_qwen3).
# Vectors are sliced 4096 -> 1024 (MRL) + L2-renormalized inside the embedder.
# Start 02_vllm_qwen3.sbatch FIRST and wait until it is healthy.
set -euo pipefail

module load singularity || module load apptainer || true
cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VLLM_SIF="${VLLM_SIF:-$PWD/vllm.sif}"

if [[ -z "${QDRANT_URL:-}" && -f logs/slurm/qdrant_host.txt ]]; then
    export QDRANT_URL="http://$(cat logs/slurm/qdrant_host.txt)"
fi
export QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

# Resolve the vLLM endpoint written by the server job (override with VLLM_BASE_URL).
if [[ -z "${VLLM_BASE_URL:-}" && -f logs/slurm/vllm_qwen3_host.txt ]]; then
    export VLLM_BASE_URL="http://$(cat logs/slurm/vllm_qwen3_host.txt)/v1"
fi
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
echo "Qdrant -> $QDRANT_URL | vLLM -> $VLLM_BASE_URL"

# Wait for the server to become reachable (max ~5 min).
for i in $(seq 1 60); do
    if curl -sf "${VLLM_BASE_URL%/v1}/health" >/dev/null 2>&1; then
        echo "vLLM server is healthy."; break
    fi
    echo "Waiting for vLLM server... ($i)"; sleep 5
done

# The client only needs qdrant-client + openai (both inside vllm.sif). No GPU.
singularity exec \
    --bind "$PWD:$PWD" --pwd "$PWD" \
    --env QDRANT_URL="$QDRANT_URL" --env VLLM_BASE_URL="$VLLM_BASE_URL" \
    "$VLLM_SIF" \
    python -m rag.ingest.ingest \
        --model qwen3 \
        --chunks rag/ingest/artifacts/chunks.jsonl \
        --speeches rag/ingest/artifacts/full_speeches.jsonl \
        --qdrant-url "$QDRANT_URL" \
        --vllm-base-url "$VLLM_BASE_URL" \
        --batch-size 64 \
        --reset \
        --skip-parents
