#!/bin/bash
# One-time setup: create a Python 3.12 venv with CUDA-12.8-compatible torch.
#
# The vllm.sif container ships torch compiled for CUDA >=12.9, but the HPC
# driver only supports 12.8.  Instead of fighting the container, we create a
# standalone venv with torch==2.7.0+cu128 (which pulls in all nvidia-* runtime
# libraries) and use it directly for e5 / bge / jina ingestion — no Singularity.
#
# Run this ONCE on a LOGIN NODE (needs internet for pip):
#   bash slurm/setup_venv.sh
set -euo pipefail
cd "${PROJECT_ROOT:-$PWD}"

VENV_DIR="${VENV_DIR:-$PWD/.venv-hpc}"
TORCH_VERSION="${TORCH_VERSION:-2.7.0}"
CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "=========================================================="
echo " Creating HPC venv at: $VENV_DIR"
echo " torch: $TORCH_VERSION+cu128  (matches driver 12.8)"
echo "=========================================================="

# --- 1. Find Python 3.12 ----------------------------------------------------
PY_BIN=""
for candidate in python3.12 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        VER=$("$candidate" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || echo "0")
        if [[ "$VER" == "3.12" ]]; then
            PY_BIN="$candidate"
            break
        fi
    fi
done

# Try module system if no system Python 3.12.
if [[ -z "$PY_BIN" ]]; then
    echo "  No system python3.12 found. Trying module system..."
    module load python/3.12 2>/dev/null \
        || module load python3/3.12 2>/dev/null \
        || module load python 2>/dev/null \
        || true
    for candidate in python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            VER=$("$candidate" -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null || echo "0")
            if [[ "$VER" == "3.12" ]]; then
                PY_BIN="$candidate"
                break
            fi
        fi
    done
fi

if [[ -z "$PY_BIN" ]]; then
    echo ""
    echo "ERROR: Python 3.12 not found."
    echo "  Available pythons:"
    for p in python3.12 python3 python; do
        command -v "$p" >/dev/null 2>&1 && echo "    $p -> $($p --version 2>&1)"
    done
    echo ""
    echo "  Try: module load python/3.12  (or whatever your HPC names it)"
    exit 1
fi

echo "  Python: $PY_BIN ($($PY_BIN --version 2>&1))"

# --- 2. Create venv ---------------------------------------------------------
if [[ -d "$VENV_DIR" && -f "$VENV_DIR/bin/python" ]]; then
    echo "  venv already exists at $VENV_DIR — reusing."
else
    "$PY_BIN" -m venv "$VENV_DIR"
fi

# --- 3. Install torch from cu128 index (WITH deps -> nvidia-* libs) ---------
echo ""
echo "=========================================================="
echo " Installing torch==${TORCH_VERSION}+cu128 (with CUDA runtime libs)"
echo "=========================================================="
"$VENV_DIR/bin/pip" install -U pip
"$VENV_DIR/bin/pip" install \
    "torch==${TORCH_VERSION}" \
    --index-url "$CUDA_INDEX"

# --- 4. Install the rest of requirements ------------------------------------
echo ""
echo "=========================================================="
echo " Installing remaining requirements"
echo "=========================================================="
"$VENV_DIR/bin/pip" install -r requirements-hpc.txt

# --- 5. Verify ---------------------------------------------------------------
echo ""
echo "=========================================================="
echo " Verifying torch + CUDA"
echo "=========================================================="
"$VENV_DIR/bin/python" -c "
import torch
print(f'  torch version : {torch.__version__}')
print(f'  cuda (compile): {torch.version.cuda}')
print(f'  cuda avail    : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU           : {torch.cuda.get_device_name(0)}')
    x = torch.randn(3, 3, device='cuda')
    y = x @ x.T
    print(f'  GPU matmul    : OK ({y.shape})')
    print()
    print('  *** Setup complete! CUDA is working. ***')
else:
    print()
    print('  WARNING: torch.cuda.is_available() is False.')
    print('  This is expected on a login node without GPU.')
    print('  The ingest jobs (submitted to --partition=gpu) will see the GPU.')
    print('  Setup is still complete — venv is ready.')
"

echo ""
echo "Done. venv at: $VENV_DIR"
echo "sbatch scripts 10/11/12 will auto-detect and use it."
