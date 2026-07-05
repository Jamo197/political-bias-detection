#!/bin/bash
# slurm/vllm_ensure.sh — source from ingest scripts to guarantee a running
# vLLM (Qwen3-Embedding-8B) server. If the server is not reachable,
# (re-)submits 02_vllm_qwen3.sbatch from inside the running job and blocks
# until vLLM's /health endpoint responds.
#
# A flock guard prevents several jobs that start simultaneously from each
# submitting their own vLLM server (which would contend for GPUs and corrupt
# the shared host file).
#
# Usage (after `cd "$PROJECT_ROOT"` and `mkdir -p logs/slurm`):
#     source slurm/vllm_ensure.sh
#     ensure_vllm             # exits the caller on failure (set -e)
#     echo "vLLM -> $VLLM_BASE_URL"
#
# Tunables (env vars):
#   VLLM_BASE_URL      — override the URL (skips host-file resolution)
#   VLLM_SBATCH        — path to the vllm sbatch (default: slurm/02_vllm_qwen3.sbatch)
#   VLLM_JOB_NAME      — SLURM job-name to look up in squeue (default: vllm_qwen3_embed)
#   VLLM_WAIT_MAX      — max seconds to wait for job scheduling + model load (default: 1800)
#   VLLM_POLL          — poll interval in seconds (default: 5)

# --- internal: HTTP health probe ------------------------------------------
# vLLM serves /health at the base URL (without the /v1 suffix).
_vllm_is_healthy() {
    local base="${VLLM_BASE_URL%/v1}"
    curl -sf --max-time 10 "${base}/health" >/dev/null 2>&1
}

# --- internal: actual logic (runs under flock) ----------------------------
_ensure_vllm_impl() {
    local root="${PROJECT_ROOT:-$PWD}"
    local vllm_sbatch="${VLLM_SBATCH:-$root/slurm/02_vllm_qwen3.sbatch}"
    local job_name="${VLLM_JOB_NAME:-vllm_qwen3_embed}"
    local host_file="$root/logs/slurm/vllm_qwen3_host.txt"
    local max_wait="${VLLM_WAIT_MAX:-1800}"
    local poll="${VLLM_POLL:-5}"
    local waited=0

    # Resolve URL from host file or env (same logic the ingest scripts used).
    if [[ -z "${VLLM_BASE_URL:-}" && -f "$host_file" ]]; then
        export VLLM_BASE_URL="http://$(cat "$host_file")/v1"
    fi
    export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"

    # Fast path — already healthy.
    if _vllm_is_healthy; then
        echo "vLLM already healthy at $VLLM_BASE_URL"
        return 0
    fi

    echo "vLLM not reachable at $VLLM_BASE_URL — starting it."

    if [[ ! -f "$vllm_sbatch" ]]; then
        echo "ERROR: vllm sbatch not found: $vllm_sbatch" >&2
        return 1
    fi

    # Remember old host-file content so we can detect the new job's overwrite.
    local old_host=""
    if [[ -f "$host_file" ]]; then
        old_host="$(cat "$host_file" 2>/dev/null || true)"
    fi

    # Check whether a vLLM job is already in the queue (avoids duplicates
    # even without the flock, e.g. submitted manually beforehand).
    local existing
    existing="$(squeue --name="$job_name" --noheader 2>/dev/null || true)"
    if [[ -n "$existing" ]]; then
        echo "A vLLM job is already queued/running ($(echo "$existing" | wc -l) match(es))."
    else
        local job_id
        job_id="$(sbatch --parsable "$vllm_sbatch" 2>&1)" || {
            echo "ERROR: sbatch failed for $vllm_sbatch:" >&2
            echo "$job_id" >&2
            return 1
        }
        echo "Submitted vLLM job $job_id."
    fi

    # Phase 1 — wait for a vLLM job to start and write the host file.
    local cur_host=""
    while (( waited < max_wait )); do
        if [[ -f "$host_file" ]]; then
            cur_host="$(cat "$host_file" 2>/dev/null || true)"
        fi
        if [[ -n "$cur_host" && "$cur_host" != "$old_host" ]]; then
            export VLLM_BASE_URL="http://$cur_host/v1"
            echo "vLLM job started; host -> ${VLLM_BASE_URL%/v1}"
            break
        fi
        echo "  waiting for vLLM job to be scheduled... (${waited}s)"
        sleep "$poll"
        waited=$((waited + poll))
    done

    if (( waited >= max_wait )); then
        echo "ERROR: timed out (${max_wait}s) waiting for vLLM job to start." >&2
        return 1
    fi

    # Phase 2 — wait for the vLLM HTTP server + model to come up.
    # vLLM loading an 8B model can take several minutes after scheduling.
    while (( waited < max_wait )); do
        if _vllm_is_healthy; then
            echo "vLLM healthy at $VLLM_BASE_URL (total wait ${waited}s)."
            return 0
        fi
        echo "  waiting for vLLM HTTP server / model load... (${waited}s)"
        sleep "$poll"
        waited=$((waited + poll))
    done

    echo "ERROR: vLLM did not become healthy within ${max_wait}s." >&2
    return 1
}

# --- public entry point (acquires flock, then runs _ensure_vllm_impl) ------
ensure_vllm() {
    local root="${PROJECT_ROOT:-$PWD}"
    local lock_file="$root/logs/slurm/vllm_ensure.lock"
    mkdir -p "$(dirname "$lock_file")"

    # fd 201 — distinct from qdrant_ensure.sh's fd 200 so both can be sourced.
    {
        flock -x 201
        _ensure_vllm_impl
    } 201>"$lock_file"
}
