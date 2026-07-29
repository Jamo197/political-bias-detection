#!/bin/bash
#SBATCH --job-name=eval_matrix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/eval_matrix_%j.out
#SBATCH --error=logs/slurm/eval_matrix_%j.err

set -euo pipefail

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VENV_PY="${VENV_PY:-$PWD/.venv-hpc/bin/python}"
if [[ ! -f "$VENV_PY" ]]; then
    echo "ERROR: venv not found at $VENV_PY"
    exit 1
fi

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"; mkdir -p "$HF_HOME"

source slurm/qdrant_ensure.sh
ensure_qdrant
echo "Qdrant -> $QDRANT_URL"

RUN_ID="${RUN_ID:-eval_matrix_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-logs/batch_runs/$(date +%Y-%m-%d)_${RUN_ID}}"
export RUN_ID RUN_DIR

VLLM_HOST_FILE="logs/slurm/vllm_active_host.txt"
VLLM_HOST=""
if [[ -s "$VLLM_HOST_FILE" ]]; then
    VLLM_HOST="http://$(cat "$VLLM_HOST_FILE")"
    echo "vLLM server target: $VLLM_HOST"
else
    echo "WARNING: vLLM host file missing or empty: $VLLM_HOST_FILE; using OpenRouter as LLM provider."
fi

# Pre-read qwen3 embedding server host (required for qwen3 model)
VLLM_EMBED_HOST_FILE="logs/slurm/vllm_qwen3_host.txt"
VLLM_EMBED_HOST=""
if [[ -s "$VLLM_EMBED_HOST_FILE" ]]; then
    VLLM_EMBED_HOST="http://$(cat "$VLLM_EMBED_HOST_FILE")"
fi

K_CHUNKS="${K_CHUNKS:-5}"

LLM_ARGS=()
if [[ -n "$VLLM_HOST" ]]; then
    LLM_ARGS+=(--llm_base_url "${VLLM_HOST}/v1")
fi

# --- Test 1: no-RAG baseline (no embedding model needed) ------------------
echo "--------------------------------------------------"
echo "Starting evaluation batch: no_rag (baseline)"
echo "--------------------------------------------------"

"$VENV_PY" -m src.run_batch \
    --no_rag \
    --run_id "${RUN_ID}_norag" \
    --run_dir "$RUN_DIR" \
    "${LLM_ARGS[@]}"

echo "Finished no_rag baseline"
echo ""

# --- Tests 2-5: RAG with each embedding model -------------------------------
EMBEDDING_MODELS=("e5" "bge" "jina" "qwen3")
STRATEGIES="simple,hyde,twostage"

for EMB in "${EMBEDDING_MODELS[@]}"; do
    if [[ "$EMB" == "bge" ]]; then
        STRATEGIES="simple,simple_hybrid,hyde,hyde_hybrid,twostage,twostage_hybrid"
    else
        STRATEGIES="simple,hyde,twostage"
    fi

    echo "--------------------------------------------------"
    echo "Starting evaluation batch: Embedding=$EMB"
    echo "--------------------------------------------------"

    EXTRA_ARGS=()
    if [[ "$EMB" == "qwen3" ]]; then
        if [[ -n "$VLLM_EMBED_HOST" ]]; then
            EXTRA_ARGS+=(--vllm_base_url "$VLLM_EMBED_HOST")
        else
            echo "WARNING: qwen3 vLLM embed server not found; falling back to OpenRouter."
            EXTRA_ARGS+=(--query_backend openrouter)
        fi
    fi

    "$VENV_PY" -m src.run_batch \
        --embedding_model "$EMB" \
        --strategies "$STRATEGIES" \
        --device cuda \
        --qdrant_url "$QDRANT_URL" \
        --k_chunks "$K_CHUNKS" \
        --run_id "${RUN_ID}_${EMB}" \
        --run_dir "$RUN_DIR" \
        "${EXTRA_ARGS[@]}" \
        "${LLM_ARGS[@]}"

    echo "Finished embedding model: $EMB"
    echo ""
done

echo "=== All 5 tests completed successfully! ==="
# TODO: add Openrouter fallback