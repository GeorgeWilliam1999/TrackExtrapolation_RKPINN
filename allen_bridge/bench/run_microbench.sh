#!/usr/bin/env bash
# run_microbench.sh — HTCondor GPU-slot worker wrapper for Tier-1.
# Compiles (NVRTC) + runs the throughput micro-bench. No external nvcc/CUDA
# toolkit is needed: cupy + the conda env's bundled NVRTC do everything.
set -uo pipefail

REPO=/data/bfys/gscriven/track-extrapolation-pinn
BENCH="$REPO/allen_bridge/bench"
PY=/data/bfys/gscriven/conda/envs/TE/bin/python
SP=/data/bfys/gscriven/conda/envs/TE/lib/python3.10/site-packages

export CUPY_CACHE_DIR="$BENCH/.cupy_cache"
export PYTHONPATH="$BENCH/_pyenv:${PYTHONPATH:-}"
# Make cupy's bundled-CUDA discovery robust (cuda-pathfinder also finds these).
for comp in cuda_nvrtc cuda_runtime nvjitlink cuda_cupti; do
  if [ -d "$SP/nvidia/$comp/lib" ]; then
    export LD_LIBRARY_PATH="$SP/nvidia/$comp/lib:${LD_LIBRARY_PATH:-}"
  fi
done
mkdir -p "$CUPY_CACHE_DIR"

echo "=========================================="
echo "  Tier-1 throughput micro-bench"
echo "  host: $(hostname)   date: $(date)"
echo "=========================================="
nvidia-smi || echo "WARNING: nvidia-smi failed"
echo "--- GPU clocks (locking is best-effort; usually unprivileged on shared pool) ---"
nvidia-smi -q -d CLOCK 2>/dev/null | grep -A4 "Clocks$" | head -10 || true

exec "$PY" "$BENCH/microbench.py" "$@"
