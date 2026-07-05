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
# The vLLM server is auto-started if not already running (02_vllm_qwen3.sbatch).
set -euo pipefail

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VENV_PY="${VENV_PY:-$PWD/.venv-hpc/bin/python}"
if [[ ! -f "$VENV_PY" ]]; then
    echo "ERROR: venv not found at $VENV_PY"
    echo "  Run: bash slurm/setup_venv.sh"
    exit 1
fi

# Ensure the Qdrant server is running (auto-starts 01_qdrant.sbatch if needed).
source slurm/qdrant_ensure.sh
ensure_qdrant

# Ensure the vLLM server is running (auto-starts 02_vllm_qwen3.sbatch if needed).
source slurm/vllm_ensure.sh
ensure_vllm
echo "Qdrant -> $QDRANT_URL | vLLM -> $VLLM_BASE_URL"

# Client only needs qdrant-client + openai (both in the venv). No GPU, no container.
"$VENV_PY" -m rag.ingest.ingest \
    --model qwen3 \
    --chunks rag/ingest/artifacts/chunks.jsonl \
    --speeches rag/ingest/artifacts/full_speeches.jsonl \
    --qdrant-url "$QDRANT_URL" \
    --qdrant-timeout 300 \
    --vllm-base-url "$VLLM_BASE_URL" \
    --batch-size 64 \
    --reset \
    --skip-parents
