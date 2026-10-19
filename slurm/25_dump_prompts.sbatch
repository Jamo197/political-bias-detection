#!/bin/bash
#SBATCH --job-name=dump_prompts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/dump_prompts_%j.out
#SBATCH --error=logs/slurm/dump_prompts_%j.err

# Imitates a prediction run for ONE dataset row and writes the exact prompts the
# LLM receives (No-RAG baseline and RAG with retrieved chunks) to disk.
# No LLM is called -- the request is captured and a stub answer is returned.
# Needs Qdrant (started automatically) and a GPU for the query embedder.
#
# Output: results/example_prompts/<RUN_ID>/{norag,rag}_prompt.txt (+ .json, chunks)
#
# Overrides (env vars):
#   EMBEDDING_MODEL=e5|bge|jina|qwen3   STRATEGY=simple|twostage
#   K_CHUNKS=5   RANDOM_SEED=33   ROW_INDEX=<n> (fixed row instead of seeded sample)
#   MODEL_ID=Qwen/Qwen2.5-32B-Instruct
#   e.g.  ROW_INDEX=12 EMBEDDING_MODEL=bge sbatch slurm/25_dump_prompts.sbatch
set -euo pipefail

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VENV_PY="${VENV_PY:-$PWD/.venv-hpc/bin/python}"
if [[ ! -f "$VENV_PY" ]]; then
    echo "ERROR: venv not found at $VENV_PY"
    echo "  Run: bash slurm/setup_venv.sh"
    exit 1
fi

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"; mkdir -p "$HF_HOME"
# Compute nodes have no internet: use the local cache instead of retrying the hub.
# Override with HF_HUB_OFFLINE=0 if the model is not cached yet.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

source slurm/qdrant_ensure.sh
ensure_qdrant
echo "Qdrant -> $QDRANT_URL"

RUN_ID="${RUN_ID:-prompts_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-results/example_prompts/$RUN_ID}"

ARGS=(
    --embedding_model "${EMBEDDING_MODEL:-e5}"
    --strategy "${STRATEGY:-simple}"
    --k_chunks "${K_CHUNKS:-5}"
    --random_seed "${RANDOM_SEED:-33}"
    --model_id "${MODEL_ID:-Qwen/Qwen2.5-32B-Instruct}"
    --device cuda
    --qdrant_url "$QDRANT_URL"
    --out_dir "$OUT_DIR"
)
[[ -n "${ROW_INDEX:-}" ]] && ARGS+=(--row_index "$ROW_INDEX")

"$VENV_PY" -u -m scripts.dump_example_prompts "${ARGS[@]}"

echo "=== No-RAG prompt ==="
cat "$OUT_DIR/norag_prompt.txt"
echo "=== RAG prompt ==="
cat "$OUT_DIR/rag_prompt.txt"
echo "Saved to $OUT_DIR"
