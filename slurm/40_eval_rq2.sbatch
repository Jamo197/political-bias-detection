#!/bin/bash
#SBATCH --job-name=eval_rq2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/eval_rq2_%j.out
#SBATCH --error=logs/slurm/eval_rq2_%j.err

# RQ2 cross-cultural evaluation (conditions C0-C5, see src/run_rq2.py).
#
# Prerequisites:
#   1. Qdrant with chunks_<model>_parlamint (all countries incl. DE), see
#      HOWTO_CROSS_CULTURAL.md. C5 also needs chunks_<model>_parlamint_en.
#   2. src/datasets/EU_tweets_rq2.csv (label_tweets.py; tweet_text_en for C5).
#   3. A vLLM server for the generator (slurm/vllm_server.sh <model id>), which
#      writes its host to logs/slurm/vllm_active_host.txt.
#
#   COUNTRY=at CONDITIONS=C0,C1,C2,C3,C4 EMB=bge STRATEGIES=simple_hybrid,twostage \
#     LLM=qwen-32B sbatch slurm/40_eval_rq2.sbatch
#
# Re-submitting with the same RUN_ID/RUN_DIR resumes (logged tweets are skipped).
set -euo pipefail

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VENV_PY="${VENV_PY:-$PWD/.venv-hpc/bin/python}"
if [[ ! -f "$VENV_PY" ]]; then
    echo "ERROR: venv not found at $VENV_PY  (run: bash slurm/setup_venv.sh)"
    exit 1
fi

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"; mkdir -p "$HF_HOME"

source slurm/qdrant_ensure.sh
ensure_qdrant
echo "Qdrant -> $QDRANT_URL"

COUNTRY="${COUNTRY:-at}"
CONDITIONS="${CONDITIONS:-C0,C1,C2,C3,C4}"
EMB="${EMB:-bge}"
STRATEGIES="${STRATEGIES:-simple}"
LLM="${LLM:-qwen-32B}"
K_CHUNKS="${K_CHUNKS:-5}"

RUN_ID="${RUN_ID:-rq2_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-logs/rq2_runs/$(date +%Y-%m-%d)_${RUN_ID}}"

VLLM_HOST_FILE="${VLLM_HOST_FILE:-logs/slurm/vllm_active_host.txt}"
VLLM_HOST="$(cat "$VLLM_HOST_FILE" 2>/dev/null || echo "")"
if [[ -z "$VLLM_HOST" ]]; then
    echo "ERROR: vLLM host not found in $VLLM_HOST_FILE"
    exit 1
fi
echo "vLLM server: http://$VLLM_HOST | Run: $RUN_ID -> $RUN_DIR"

SAMPLE_ARGS=()
if [[ -n "${SAMPLE_SIZE:-}" ]]; then
    SAMPLE_ARGS=(--sample_size "$SAMPLE_SIZE")   # per country, smoke tests
fi

"$VENV_PY" -m src.run_rq2 \
    --country "$COUNTRY" \
    --conditions "$CONDITIONS" \
    --embedding_model "$EMB" \
    --strategies "$STRATEGIES" \
    --llm "$LLM" \
    --llm_base_url "http://${VLLM_HOST}/v1" \
    --qdrant_url "$QDRANT_URL" \
    --k_chunks "$K_CHUNKS" \
    --device cuda \
    --run_id "$RUN_ID" \
    --run_dir "$RUN_DIR" \
    ${SAMPLE_ARGS[@]+"${SAMPLE_ARGS[@]}"}
