#!/bin/bash
#SBATCH --job-name=qdrant
#SBATCH --partition=standard
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --signal=B:SIGTERM@1800
#SBATCH --output=logs/slurm/qdrant_%j.out
#SBATCH --error=logs/slurm/qdrant_%j.err

set -euo pipefail

module load singularity || module load apptainer || true

# --- Configuration ---
LOCAL_STORAGE="$TMPDIR/qdrant_storage"
BEEGFS_STORAGE="$PWD/qdrant_storage"
HOST_FILE="$PWD/qdrant_host.txt"

QDRANT_SIF="${QDRANT_SIF:-/home/users/j/$USER/hpc-images/qdrant.sif}"

export QDRANT__SERVICE__HOST="0.0.0.0"
export QDRANT__TELEMETRY_DISABLED="true"

mkdir -p "$LOCAL_STORAGE"

# 1. Copy persisted database from BeeGFS to Local NVMe
if [ -d "$BEEGFS_STORAGE" ]; then
    echo "Syncing database from BeeGFS to fast local NVMe ($TMPDIR)..."
    rsync -ah --delete "$BEEGFS_STORAGE/" "$LOCAL_STORAGE/"
fi

# 2. Expose the compute node's hostname for your other jobs
hostname -f > "$HOST_FILE"
echo "Qdrant will be available at: $(cat $HOST_FILE)"

# 3. Define cleanup function for graceful shutdown and data safety
cleanup() {
    echo "Signal received or job ending. Stopping Qdrant safely..."
    
    # 3a. Send SIGTERM to Qdrant and WAIT for it to exit
    # This is critical to prevent data loss: it forces Qdrant to flush memory/WAL to disk.
    if kill -0 "$QDRANT_PID" 2>/dev/null; then
        kill -TERM "$QDRANT_PID"
        wait "$QDRANT_PID" 2>/dev/null || true
    fi

    echo "Qdrant flushed and stopped. Syncing database back to BeeGFS..."
    # 3b. Sync the safe state back to shared storage
    rsync -ah --delete "$LOCAL_STORAGE/" "$BEEGFS_STORAGE/"
    
    # Clean up the host file so other jobs don't try to connect to a dead node
    rm -f "$HOST_FILE"
    
    echo "Sync complete. Exiting."
}

# Catch normal exits and the SIGTERM sent by SLURM's --signal command
trap cleanup EXIT SIGTERM

# 4. Run Singularity as a background process
echo "Starting Qdrant..."
singularity run \
    --pwd /qdrant \
    --writable-tmpfs \
    --env QDRANT__SERVICE__HOST="0.0.0.0" \
    --bind "${LOCAL_STORAGE}:/qdrant/storage" \
    "${QDRANT_SIF}" &

QDRANT_PID=$!

wait "$QDRANT_PID" 2>/dev/null || true

# Kill job with scancel --signal=B:TERM <job_id>