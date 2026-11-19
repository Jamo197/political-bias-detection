#!/bin/bash
#SBATCH --job-name=chunk_ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/chunk_ablation_%j.out
#SBATCH --error=logs/slurm/chunk_ablation_%j.err

# Chunk-ablation runs on already-logged retrieval. NO Qdrant needed.
# One job = start vLLM for ONE model inside the job, run the ablation, stop vLLM.
#
#   Experiment 1 (5 random chunks, all 775 posts):
#     MODE=random MODEL_ID=meta-llama/Llama-3.1-8B-Instruct MODEL_KEY=llama-8B \
#         sbatch slurm/30_chunk_ablation.sbatch
#
#   Experiment 2 (only chunks with R_top>=4, the ~100 judged posts of a family):
#     MODE=rtop FAMILY=llama_8b   sbatch slurm/30_chunk_ablation.sbatch
#     MODE=rtop FAMILY=mistral_8b sbatch slurm/30_chunk_ablation.sbatch
#     # control with the low-R_top chunks of the same posts:
#     MODE=rtop FAMILY=llama_8b SELECT=low sbatch slurm/30_chunk_ablation.sbatch
#
# Big models (Llama-70B FP8, Qwen-72B FP8) need more GPU memory, override on the CLI:
#     sbatch --partition=h200_short --gres=gpu:1 ...
#
# Prerequisites: bash slurm/setup_venv.sh (client venv), vllm.sif, the repo with
# logs/batch_runs/2026-08-17_eval_matrix_20260817_123007 (e5/simple logs) and
# results/qualitative + "RAG Analysis/LLM judge results/families" present.
# Gated models (Llama): export HF_TOKEN before sbatch (not stored in this file).

# # Experiment 1: 5 random chunks, all 775 posts
# MODE=random MODEL_ID=meta-llama/Llama-3.1-8B-Instruct MODEL_KEY=llama-8B sbatch slurm/30_chunk_ablation.sbatch
# MODE=random MODEL_ID=mistralai/Ministral-3-8B-Instruct-2512 MODEL_KEY=mistral-8B sbatch slurm/30_chunk_ablation.sbatch

# # Experiment 2: only R_top>=4 chunks
# MODE=rtop FAMILY=llama_8b   sbatch slurm/30_chunk_ablation.sbatch
# MODE=rtop FAMILY=mistral_8b sbatch slurm/30_chunk_ablation.sbatch

set -euo pipefail

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm
module load singularity || module load apptainer || true

MODE="${MODE:?set MODE=random or MODE=rtop}"
VENV_PY="${VENV_PY:-$PWD/.venv-hpc/bin/python}"
VLLM_SIF="${VLLM_SIF:-/home/users/j/$USER/hpc-images/vllm.sif}"
VLLM_PORT="${VLLM_PORT:-$((8100 + SLURM_JOB_ID % 500))}"
CONCURRENCY="${CONCURRENCY:-16}"
SEED="${SEED:-33}"
SELECT="${SELECT:-high}"
MIN_RTOP="${MIN_RTOP:-4}"
NO_CHUNK_POLICY="${NO_CHUNK_POLICY:-skip}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

[[ -f "$VENV_PY" ]] || { echo "ERROR: venv not found at $VENV_PY (run slurm/setup_venv.sh)"; exit 1; }

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"; mkdir -p "$HF_HOME"
export HF_TOKEN="${HF_TOKEN:-}"
[[ -n "$HF_TOKEN" ]] || echo "WARNING: HF_TOKEN empty; fine for cached or ungated models, fails for gated ones."

# ---- resolve model + python args -------------------------------------------
if [[ "$MODE" == "rtop" ]]; then
    FAMILY="${FAMILY:?set FAMILY=llama_8b|mistral_8b|...}"
    MODEL_ID="${MODEL_ID:-$("$VENV_PY" -m src.run_chunk_ablation --mode rtop --family "$FAMILY" --print_model_id)}"
    PY_ARGS=(--mode rtop --family "$FAMILY" --select "$SELECT" --min_rtop "$MIN_RTOP"
             --no_chunk_policy "$NO_CHUNK_POLICY")
elif [[ "$MODE" == "random" ]]; then
    MODEL_ID="${MODEL_ID:?set MODEL_ID=<hf model id>}"
    MODEL_KEY="${MODEL_KEY:?set MODEL_KEY=e.g. llama-8B}"
    PY_ARGS=(--mode random --model_key "$MODEL_KEY" --seed "$SEED" --k_chunks "${K_CHUNKS:-5}")
else
    echo "ERROR: MODE must be random or rtop"; exit 1
fi
[[ -n "${RUN_ID:-}" ]] && PY_ARGS+=(--run_id "$RUN_ID")
echo "Mode=$MODE  Model=$MODEL_ID  Port=$VLLM_PORT"

# ---- start vLLM in the background ------------------------------------------
singularity exec --nv \
    --bind "$PWD:$PWD" --pwd "$PWD" \
    --env HF_HOME="$HF_HOME" --env HF_TOKEN="$HF_TOKEN" \
    --env VLLM_USE_FLASHINFER_SAMPLER="0" \
    "$VLLM_SIF" \
    vllm serve "$MODEL_ID" \
        --host 0.0.0.0 --port "$VLLM_PORT" \
        --dtype bfloat16 --max-model-len 32768 \
        --gpu-memory-utilization 0.92 --trust-remote-code $EXTRA_VLLM_ARGS \
    > "logs/slurm/chunk_ablation_vllm_${SLURM_JOB_ID}.log" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

echo "Waiting for vLLM health..."
for i in $(seq 1 360); do   # up to 30 min (large models load slowly)
    if curl -sf --max-time 5 "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM healthy after $((i*5))s"; break
    fi
    kill -0 $VLLM_PID 2>/dev/null || { echo "ERROR: vLLM died, see logs/slurm/chunk_ablation_vllm_${SLURM_JOB_ID}.log"; exit 1; }
    sleep 5
done
curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null || { echo "ERROR: vLLM never became healthy"; exit 1; }

# ---- run ---------------------------------------------------------------------
"$VENV_PY" -m src.run_chunk_ablation "${PY_ARGS[@]}" \
    --model_id "$MODEL_ID" \
    --llm_base_url "http://localhost:${VLLM_PORT}/v1" \
    --concurrency "$CONCURRENCY"
