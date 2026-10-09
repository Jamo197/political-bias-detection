#!/bin/bash
#SBATCH --job-name=parlamint_export
#SBATCH --partition=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/parlamint_export_%j.out
#SBATCH --error=logs/slurm/parlamint_export_%j.err

# Download ParlaMint3 from HuggingFace and write Bundestag-style speech JSONs.
# Needs internet: if compute nodes are offline, run the same python command on a
# login node instead (see HOWTO_CROSS_CULTURAL.md).
#
#   COUNTRIES="pl gb" sbatch slurm/30_parlamint_export.sbatch
#   LIMIT=200 COUNTRIES=pl OUT=extraction/datasets/parlamint_test sbatch ...   (smoke test)
#   UNTIL=none / MAX_SPEECHES=20000 sbatch ...                                (window end / pre-cap)
#   MAPPING='src/Cross Cultural Analysis/party_mapping.csv' sbatch ...          (optional, adds CHES ids)
set -euo pipefail

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

VENV_PY="${VENV_PY:-$PWD/.venv-hpc/bin/python}"
if [[ ! -f "$VENV_PY" ]]; then
    echo "ERROR: venv not found at $VENV_PY  (run: bash slurm/setup_venv.sh)"
    exit 1
fi

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"; mkdir -p "$HF_HOME"

COUNTRIES="${COUNTRIES:-at be dk fr gb gr it lv nl pl pt se si}"
SINCE="${SINCE:-2020}"
UNTIL="${UNTIL:-2021-12-31}"   # RQ2 shared window (tweets end 2021)
OUT="${OUT:-extraction/datasets/parlamint_data}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS=(--limit "$LIMIT")   # first N rows per country, for smoke tests
fi

MAX_ARGS=()
if [[ -n "${MAX_SPEECHES:-}" ]]; then
    MAX_ARGS=(--max-speeches "$MAX_SPEECHES")   # seeded random pre-cap per country
fi

MAPPING_ARGS=()
if [[ -n "${MAPPING:-}" ]]; then
    MAPPING_ARGS=(--mapping "$MAPPING")
fi

# COUNTRIES is intentionally unquoted: one --config value per country.
"$VENV_PY" "src/Cross Cultural Analysis/parlamint_to_speeches.py" \
    --config $COUNTRIES \
    --since "$SINCE" \
    --until "$UNTIL" \
    --out "$OUT" \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"} \
    ${MAX_ARGS[@]+"${MAX_ARGS[@]}"} \
    ${MAPPING_ARGS[@]+"${MAPPING_ARGS[@]}"}

echo "Export complete -> $OUT"
