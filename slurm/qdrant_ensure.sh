#!/bin/bash
# slurm/qdrant_ensure.sh — source from ingest scripts to guarantee a running
# Qdrant server. If the server is not reachable, (re-)submits 01_qdrant.sbatch
# from inside the running job and blocks until Qdrant responds.
#
# A flock guard prevents several ingest jobs that start simultaneously from
# each submitting their own Qdrant server (which would corrupt the shared
# storage directory).
#
# Usage (after `cd "$PROJECT_ROOT"` and `mkdir -p logs/slurm`):
#     source slurm/qdrant_ensure.sh
#     ensure_qdrant            # exits the caller on failure (set -e)
#     echo "Qdrant -> $QDRANT_URL"
#
# Tunables (env vars):
#   QDRANT_URL         — override the URL (skips host-file resolution)
#   QDRANT_SBATCH      — path to the qdrant sbatch (default: slurm/01_qdrant.sbatch)
#   QDRANT_WAIT_MAX    — max seconds to wait for job scheduling + server startup (default: 900)
#   QDRANT_POLL        — poll interval in seconds (default: 5)

# --- internal: HTTP health probe ------------------------------------------
_qdrant_is_healthy() {
    curl -sf --max-time 5 "${QDRANT_URL%/}/" >/dev/null 2>&1
}

# --- internal: actual logic (runs under flock) ----------------------------
_ensure_qdrant_impl() {
    local root="${PROJECT_ROOT:-$PWD}"
    local qdrant_sbatch="${QDRANT_SBATCH:-$root/slurm/01_qdrant.sbatch}"
    local host_file="$root/logs/slurm/qdrant_host.txt"
    local max_wait="${QDRANT_WAIT_MAX:-900}"
    local poll="${QDRANT_POLL:-5}"
    local waited=0

    # Resolve URL from host file or env (same logic the ingest scripts used).
    if [[ -z "${QDRANT_URL:-}" && -f "$host_file" ]]; then
        export QDRANT_URL="http://$(cat "$host_file")"
    fi
    export QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

    # Fast path — already healthy.
    if _qdrant_is_healthy; then
        echo "Qdrant already healthy at $QDRANT_URL"
        return 0
    fi

    echo "Qdrant not reachable at $QDRANT_URL — starting it."

    if [[ ! -f "$qdrant_sbatch" ]]; then
        echo "ERROR: qdrant sbatch not found: $qdrant_sbatch" >&2
        return 1
    fi

    # Remember old host-file content so we can detect the new job's overwrite.
    local old_host=""
    if [[ -f "$host_file" ]]; then
        old_host="$(cat "$host_file" 2>/dev/null || true)"
    fi

    # Check whether a qdrant job is already in the queue (avoids duplicates
    # even without the flock, e.g. submitted manually beforehand).
    local existing
    existing="$(squeue --name=qdrant --noheader 2>/dev/null || true)"
    if [[ -n "$existing" ]]; then
        echo "A Qdrant job is already queued/running ($(echo "$existing" | wc -l) match(es))."
    else
        local job_id
        job_id="$(sbatch --parsable "$qdrant_sbatch" 2>&1)" || {
            echo "ERROR: sbatch failed for $qdrant_sbatch:" >&2
            echo "$job_id" >&2
            return 1
        }
        echo "Submitted Qdrant job $job_id."
    fi

    # Phase 1 — wait for a qdrant job to start and write the host file.
    local cur_host=""
    while (( waited < max_wait )); do
        if [[ -f "$host_file" ]]; then
            cur_host="$(cat "$host_file" 2>/dev/null || true)"
        fi
        if [[ -n "$cur_host" && "$cur_host" != "$old_host" ]]; then
            export QDRANT_URL="http://$cur_host"
            echo "Qdrant job started; host -> $QDRANT_URL"
            break
        fi
        echo "  waiting for Qdrant job to be scheduled... (${waited}s)"
        sleep "$poll"
        waited=$((waited + poll))
    done

    if (( waited >= max_wait )); then
        echo "ERROR: timed out (${max_wait}s) waiting for Qdrant job to start." >&2
        return 1
    fi

    # Phase 2 — wait for the HTTP server inside the container to come up.
    while (( waited < max_wait )); do
        if _qdrant_is_healthy; then
            echo "Qdrant healthy at $QDRANT_URL (total wait ${waited}s)."
            return 0
        fi
        echo "  waiting for Qdrant HTTP server... (${waited}s)"
        sleep "$poll"
        waited=$((waited + poll))
    done

    echo "ERROR: Qdrant did not become healthy within ${max_wait}s." >&2
    return 1
}

# --- public entry point (acquires flock, then runs _ensure_qdrant_impl) ----
ensure_qdrant() {
    local root="${PROJECT_ROOT:-$PWD}"
    local lock_file="$root/logs/slurm/qdrant_ensure.lock"
    mkdir -p "$(dirname "$lock_file")"

    # fd 200 is scoped to the group — closing it releases the lock.
    {
        flock -x 200
        _ensure_qdrant_impl
    } 200>"$lock_file"
}
