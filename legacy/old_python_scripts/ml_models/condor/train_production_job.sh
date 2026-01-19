#!/bin/bash
#
# Wrapper script for training production models on HTCondor GPU nodes
#
# Usage: ./train_production_job.sh <model_name> <hidden_layers>

set -e

MODEL_NAME=$1
HIDDEN_LAYERS=$2
BASE_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/ml_models"
PYTHON_DIR="${BASE_DIR}/python"
DATA_DIR="${BASE_DIR}/data"
OUTPUT_DIR="${BASE_DIR}/models/production"

# Activate conda environment
source /data/bfys/gscriven/conda/bin/activate TE

echo "=========================================="
echo "Production Model Training Job"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Architecture: ${HIDDEN_LAYERS}"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=========================================="

mkdir -p "${OUTPUT_DIR}"
cd "${PYTHON_DIR}"

# Convert hidden layers format (512-512-256-128 -> 512 512 256 128)
HIDDEN_ARGS=$(echo ${HIDDEN_LAYERS} | tr '-' ' ')

# Train model with full 2000 epochs
python train_on_gpu.py \
    --data "${DATA_DIR}" \
    --dataset analysis \
    --hidden ${HIDDEN_ARGS} \
    --epochs 2000 \
    --batch 512 \
    --lr 0.001 \
    --output "${OUTPUT_DIR}" \
    --name "${MODEL_NAME}"

echo "=========================================="
echo "Training complete!"
echo "Model saved to: ${OUTPUT_DIR}/${MODEL_NAME}.bin"
echo "Metadata: ${OUTPUT_DIR}/${MODEL_NAME}_metadata.json"
echo "=========================================="
