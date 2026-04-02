#!/bin/bash
# =============================================================================
# run_train_cpu.sh — HTCondor worker for MLP training (CPU-only)
#
# For small models that don't saturate GPU and trigger the GPU watchdog.
# =============================================================================
set -e
export PYTHONUNBUFFERED=1

EXP_NAME="$1"
HIDDEN_DIMS_CSV="$2"
MAX_SAMPLES="$3"

CONDA_PREFIX="/data/bfys/gscriven/conda"
ENV_NAME="TE"
MODELS_DIR="/data/bfys/gscriven/TrackExtrapolation/experiments/gen_1/V1/models"

HIDDEN_DIMS=$(echo "$HIDDEN_DIMS_CSV" | tr '_' ' ')

echo "======================================="
echo "  MLP Training (CPU)"
echo "  Name:       $EXP_NAME"
echo "  Hidden:     $HIDDEN_DIMS"
echo "  Samples:    $MAX_SAMPLES"
echo "  Host:       $(hostname)"
echo "  Date:       $(date)"
echo "======================================="

eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

cd "$MODELS_DIR"

CMD="python train.py --model mlp --name $EXP_NAME --hidden_dims $HIDDEN_DIMS"
CMD="$CMD --device cpu"
CMD="$CMD --epochs 500 --patience 30 --min_delta 1e-7"
CMD="$CMD --batch_size 4096 --lr 1e-3 --weight_decay 1e-4"
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
