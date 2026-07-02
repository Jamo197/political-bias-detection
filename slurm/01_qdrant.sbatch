#!/bin/bash
#SBATCH --job-name=qdrant
#SBATCH --partition=cpu            # <-- ADJUST: a long-lived CPU partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/qdrant_%j.out
#SBATCH --error=logs/slurm/qdrant_%j.err

# Long-running Qdrant server from qdrant.sif (Singularity).
# Keep this job RUNNING for the whole ingestion campaign. It writes its
# hostname to logs/slurm/qdrant_host.txt so the ingest jobs can find it.
set -euo pipefail

module load singularity || module load apptainer || true

cd "${PROJECT_ROOT:-$PWD}"
mkdir -p logs/slurm

# Persisted storage on BeeGFS scratch (reproducible data). Survives restarts.
QDRANT_SIF="${QDRANT_SIF:-$PWD/qdrant.sif}"
QDRANT_STORAGE="${QDRANT_STORAGE:-$PWD/qdrant_storage}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
mkdir -p "$QDRANT_STORAGE"

echo "$(hostname):${QDRANT_PORT}" > logs/slurm/qdrant_host.txt
echo "Qdrant host -> $(cat logs/slurm/qdrant_host.txt)"
echo "Storage     -> $QDRANT_STORAGE"
echo "Image       -> $QDRANT_SIF"

# Bind a writable storage dir into the container's /qdrant/storage.
exec singularity run \
    --bind "${QDRANT_STORAGE}:/qdrant/storage" \
    "${QDRANT_SIF}"
