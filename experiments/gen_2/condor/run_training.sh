#!/bin/bash
# =============================================================================
# HTCondor worker script for gen_2 training jobs
# =============================================================================
# Called by condor as: run_training.sh <config_path>
#
# Environment:
#   - Nikhef STBC cluster (stoomboot)
#   - GPU: Tesla V100-PCIE-32GB or NVIDIA L40S
#   - Shared NFS: /data/bfys/gscriven/
# =============================================================================

set -euo pipefail

CONFIG_PATH="$1"

echo "========================================="
echo "gen_2 Training Job"
echo "========================================="
echo "Config:   ${CONFIG_PATH}"
echo "Host:     $(hostname)"
echo "Date:     $(date)"
echo "GPU:      ${CUDA_VISIBLE_DEVICES:-none}"
echo "========================================="

# Activate conda environment
source /data/bfys/gscriven/conda/etc/profile.d/conda.sh
conda activate TE

# Verify GPU
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Run training
cd /data/bfys/gscriven/TrackExtrapolation/experiments/gen_2/models
python train.py --config "${CONFIG_PATH}"

echo ""
echo "========================================="
echo "Job completed at $(date)"
echo "========================================="
