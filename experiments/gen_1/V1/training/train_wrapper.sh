#!/bin/bash
# =============================================================================
# Training Job Wrapper Script
# Executed on HTCondor worker node
# =============================================================================

# Arguments:
#   $1 = model type (mlp, pinn, rk_pinn)
#   $2 = experiment name
#   $3 = hidden dims (e.g., "64,64" or "256,256,128")
#   $4 = lambda_pde (default 1.0)
#   $5 = lambda_ic (default 1.0)
#   $6 = data_path (optional, for momentum studies)

MODEL_TYPE="$1"
EXP_NAME="$2"
HIDDEN_DIMS="$3"
LAMBDA_PDE="${4:-1.0}"
LAMBDA_IC="${5:-1.0}"
DATA_PATH="${6:-}"

# Base directory
BASE_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation"
cd "$BASE_DIR"

echo "=============================================="
echo "Training Job: $EXP_NAME"
echo "=============================================="
echo "Model type:   $MODEL_TYPE"
echo "Hidden dims:  $HIDDEN_DIMS"
echo "Lambda PDE:   $LAMBDA_PDE"
echo "Lambda IC:    $LAMBDA_IC"
echo "Data path:    ${DATA_PATH:-default}"
echo "Hostname:     $(hostname)"
echo "Date:         $(date)"
echo "=============================================="

# Activate environment
source /data/bfys/gscriven/conda/etc/profile.d/conda.sh
conda activate TE

# Convert hidden dims from comma-separated to space-separated
HIDDEN_DIMS_ARGS=$(echo "$HIDDEN_DIMS" | tr ',' ' ')

# Build command
CMD="python models/train.py"
CMD="$CMD --model $MODEL_TYPE"
CMD="$CMD --name $EXP_NAME"
CMD="$CMD --hidden_dims $HIDDEN_DIMS_ARGS"
CMD="$CMD --activation silu"
CMD="$CMD --epochs 10"
CMD="$CMD --patience 30"
CMD="$CMD --batch_size 4096"
CMD="$CMD --lr 1e-3"
CMD="$CMD --checkpoint_dir trained_models"

# Add physics loss weights for PINN/RK_PINN
if [ "$MODEL_TYPE" != "mlp" ]; then
    CMD="$CMD --lambda_pde $LAMBDA_PDE"
    CMD="$CMD --lambda_ic $LAMBDA_IC"
fi

# Add data path (use default if not specified)
if [ -n "$DATA_PATH" ]; then
    CMD="$CMD --data_path $DATA_PATH"
else
    CMD="$CMD --data_path $BASE_DIR/data_generation/data/training_50M.npz"
fi

echo ""
echo "Command: $CMD"
echo ""

# Run training
$CMD
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job finished with exit code: $EXIT_CODE"
echo "Date: $(date)"
echo "=============================================="

exit $EXIT_CODE
