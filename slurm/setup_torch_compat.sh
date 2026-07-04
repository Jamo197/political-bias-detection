#!/bin/bash
# One-time setup: install a CUDA-12.8-compatible torch to a writable prefix
# so it overrides the container's too-new torch via PYTHONPATH.
#
# The vllm.sif container ships a torch compiled for CUDA >=12.9, but the HPC
# driver only supports 12.8.  We install the SAME torch version from the cu128
# index (or cu124 as fallback) to $PWD/.pip_compat.  The ingest sbatch scripts
# prepend this directory to PYTHONPATH, so Python finds our torch first.
#
# Run this ONCE on a GPU node:
#   srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G \
#       --time=00:30:00 bash slurm/setup_torch_compat.sh
set -euo pipefail

module load singularity || module load apptainer || true
cd "${PROJECT_ROOT:-$PWD}"

VLLM_SIF="${VLLM_SIF:-$PWD/vllm.sif}"
PIP_COMPAT="${PIP_COMPAT:-$PWD/.pip_compat}"
PY_VER="cp312"  # container uses Python 3.12

echo "=========================================================="
echo " Step 1: Check container's current torch / CUDA"
echo "=========================================================="
singularity exec --nv \
    --bind "$PWD:$PWD" --pwd "$PWD" \
    "$VLLM_SIF" \
    python -c "
import torch
print(f'  torch version : {torch.__version__}')
print(f'  cuda avail    : {torch.cuda.is_available()}')
print(f'  cuda (compile): {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'  GPU           : {torch.cuda.get_device_name(0)}')
"

if [[ -d "$PIP_COMPAT/torch" ]]; then
    echo ""
    echo "  Compatible torch already installed at $PIP_COMPAT"
    echo "  Skipping install. (Delete $PIP_COMPAT to re-run.)"
else
    echo ""
    echo "=========================================================="
    echo " Step 2: Install CUDA-12.8-compatible torch"
    echo "=========================================================="
    mkdir -p "$PIP_COMPAT"

    # Extract the container's torch version (strip the +cuXXX suffix).
    TORCH_VER=$(singularity exec \
        --bind "$PWD:$PWD" --pwd "$PWD" \
        "$VLLM_SIF" \
        python -c "import torch; print(torch.__version__.split('+')[0])")
    echo "  Container has torch ${TORCH_VER}"
    echo "  Installing same version from cu128 index (matches driver 12.8)..."

    # Try cu128 first (exact match for driver 12.8), then cu124 (backward compat).
    # --no-deps: only install torch itself; CUDA runtime libs (nvidia-*) and
    # other deps from the container are ABI-compatible and remain usable.
    install_torch() {
        local index_url="$1"
        local label="$2"
        echo "  Trying ${label}..."
        if singularity exec \
            --bind "$PWD:$PWD" --pwd "$PWD" \
            "$VLLM_SIF" \
            pip install --target "$PIP_COMPAT" --no-deps \
                "torch==${TORCH_VER}" \
                --index-url "$index_url"; then
            echo "  SUCCESS: installed torch==${TORCH_VER} from ${label}"
            return 0
        fi
        return 1
    }

    install_torch "https://download.pytorch.org/whl/cu128" "cu128 (exact version)" \
        || install_torch "https://download.pytorch.org/whl/cu124" "cu124 (exact version)" \
        || {
            echo "  Exact version ${TORCH_VER} not on cu128/cu124."
            echo "  Trying latest cu128 (may have minor API differences)..."
            singularity exec \
                --bind "$PWD:$PWD" --pwd "$PWD" \
                "$VLLM_SIF" \
                pip install --target "$PIP_COMPAT" --no-deps \
                    torch \
                    --index-url https://download.pytorch.org/whl/cu128
        }
fi

echo ""
echo "=========================================================="
echo " Step 3: Verify CUDA works with the compatible torch"
echo "=========================================================="
export PYTHONPATH="$PIP_COMPAT:${PYTHONPATH:-}"

singularity exec --nv \
    --bind "$PWD:$PWD" --pwd "$PWD" \
    --env PYTHONPATH="$PYTHONPATH" \
    "$VLLM_SIF" \
    python -c "
import torch
print(f'  torch version : {torch.__version__}')
print(f'  cuda avail    : {torch.cuda.is_available()}')
print(f'  cuda (compile): {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'  GPU           : {torch.cuda.get_device_name(0)}')
    x = torch.randn(3, 3, device='cuda')
    y = x @ x.T
    print(f'  GPU matmul    : OK ({y.shape})')
    print()
    print('  *** CUDA is working! ***')
    print('  The ingest sbatch scripts will auto-detect this prefix.')
else:
    print()
    print('  *** CUDA STILL NOT AVAILABLE ***')
    print('  Check that you ran this on a GPU node (--gres=gpu:1).')
    exit(1)
"

echo ""
echo "Done. Compatible torch is at: $PIP_COMPAT"
echo "The ingest sbatch scripts (10/11/12) will automatically use it."
