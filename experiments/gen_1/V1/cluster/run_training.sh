#!/bin/bash
#===============================================================================
# Training Runner Script for HTCondor
#
# This script is executed by HTCondor jobs to train models.
# It sets up the environment and runs the training script.
#
# Author: G. Scriven
# Date: January 2026
#===============================================================================

set -e

# Parse arguments
MODEL_TYPE=$1
PRESET=$2
EXPERIMENT_NAME=$3
EXTRA_ARGS="${@:4}"

# Paths
BASE_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation"
DATA_PATH="${BASE_DIR}/data_generation/data/training_50M.npz"
MODELS_DIR="${BASE_DIR}/models"
CHECKPOINT_DIR="${BASE_DIR}/trained_models"

# Activate conda environment
echo "=============================================="
echo "Setting up environment..."
echo "=============================================="
source /data/bfys/gscriven/conda/etc/profile.d/conda.sh
conda activate TE

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# Check for GPU using PyTorch (more reliable than nvidia-smi on cluster)
GPU_AVAILABLE=$(python -c 'import torch; print("yes" if torch.cuda.is_available() else "no")')
if [ "$GPU_AVAILABLE" = "yes" ]; then
    echo "GPU available via PyTorch CUDA:"
    python -c 'import torch; print(f"  Device: {torch.cuda.get_device_name(0)}"); print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")'
    DEVICE="cuda"
else
    echo "No GPU detected via PyTorch, using CPU"
    DEVICE="cpu"
fi

# Create output directory
mkdir -p "${CHECKPOINT_DIR}"

echo ""
echo "=============================================="
echo "Starting training..."
echo "=============================================="
echo "Model type: ${MODEL_TYPE}"
echo "Preset: ${PRESET}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Data: ${DATA_PATH}"
echo "Device: ${DEVICE}"
echo ""

cd "${MODELS_DIR}"

# Run training
python train.py \
    --model "${MODEL_TYPE}" \
    --preset "${PRESET}" \
    --name "${EXPERIMENT_NAME}" \
    --data_path "${DATA_PATH}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --device "${DEVICE}" \
    ${EXTRA_ARGS}

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="

# Export to ONNX if training succeeded
BEST_MODEL="${CHECKPOINT_DIR}/${EXPERIMENT_NAME}/best_model.pt"
if [ -f "${BEST_MODEL}" ]; then
    echo ""
    echo "Exporting to ONNX..."
    python export_onnx.py \
        --checkpoint "${BEST_MODEL}" \
        --output "${CHECKPOINT_DIR}/${EXPERIMENT_NAME}/exports" \
        --name "${EXPERIMENT_NAME}" \
        --generate_cpp
fi

echo "Done!"
