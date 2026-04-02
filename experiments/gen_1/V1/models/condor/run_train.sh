#!/bin/bash
# =============================================================================
# run_train.sh — HTCondor worker for MLP training
#
# Called by train_sweep.sub with arguments:
#   $1 = experiment name
#   $2 = hidden_dims (e.g. "64_64" or "256_256_128", underscore-separated)
#   $3 = max_samples (e.g. 5000000 or "all")
# =============================================================================
set -e

# Ensure Python output is unbuffered (visible in HTCondor logs immediately)
export PYTHONUNBUFFERED=1

EXP_NAME="$1"
HIDDEN_DIMS_CSV="$2"
MAX_SAMPLES="$3"

CONDA_PREFIX="/data/bfys/gscriven/conda"
ENV_NAME="TE"
MODELS_DIR="/data/bfys/gscriven/TrackExtrapolation/experiments/gen_1/V1/models"

# Convert underscore-separated hidden dims to space-separated for argparse
HIDDEN_DIMS=$(echo "$HIDDEN_DIMS_CSV" | tr '_' ' ')

echo "======================================="
echo "  MLP Training"
echo "  Name:       $EXP_NAME"
echo "  Hidden:     $HIDDEN_DIMS"
echo "  Samples:    $MAX_SAMPLES"
echo "  Host:       $(hostname)"
echo "  Date:       $(date)"
echo "======================================="

# Activate conda
eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi

cd "$MODELS_DIR"

# Build command
CMD="python train.py --model mlp --name $EXP_NAME --hidden_dims $HIDDEN_DIMS"
CMD="$CMD --epochs 500 --patience 30 --min_delta 1e-7"
CMD="$CMD --batch_size 32768 --lr 1e-3 --weight_decay 1e-4"
CMD="$CMD --activation silu --num_workers 0"
CMD="$CMD --mlflow --mlflow-experiment V1_MLP_sweep"

if [ "$MAX_SAMPLES" != "all" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

echo ""
echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "Training $EXP_NAME completed at $(date)"
