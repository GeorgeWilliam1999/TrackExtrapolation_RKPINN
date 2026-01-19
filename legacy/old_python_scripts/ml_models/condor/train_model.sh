#!/bin/bash
#
# Wrapper script for HTCondor GPU training job
#
# This script is called by HTCondor to train a single model on GPU
#
# Usage: ./train_model.sh <model_name> <hidden_layers>
#
# Example: ./train_model.sh large 512-512-256-128

set -e  # Exit on error

MODEL_NAME=$1
HIDDEN_LAYERS=$2
BASE_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/ml_models"
PYTHON_DIR="${BASE_DIR}/python"
DATA_DIR="${BASE_DIR}/data"
OUTPUT_DIR="${BASE_DIR}/models"

# Activate conda environment
source /data/bfys/gscriven/conda/bin/activate TE

echo "=========================================="
echo "HTCondor GPU Training Job"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Architecture: ${HIDDEN_LAYERS}"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found!"
fi

echo "=========================================="

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Convert hidden layers format (512-512-256-128 -> 512 512 256 128)
HIDDEN_ARGS=$(echo ${HIDDEN_LAYERS} | tr '-' ' ')

# Train model
cd "${PYTHON_DIR}"

python train_on_gpu.py \
    --data "${DATA_DIR}" \
    --hidden ${HIDDEN_ARGS} \
    --epochs 2000 \
    --batch 512 \
    --lr 0.001 \
    --output "${OUTPUT_DIR}" \
    --name "mlp_${MODEL_NAME}_condor"

echo "=========================================="
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}/mlp_${MODEL_NAME}_condor.bin"
echo "=========================================="
