#!/bin/bash
# =============================================================================
# Field Map NN v2 -- Training Job Wrapper
# Loss function comparison on silu 2L 128H architecture
#
# Usage:
#   ./train_field_nn_v2_wrapper.sh <config_name>
# =============================================================================

CONFIG_NAME="$1"

if [ -z "$CONFIG_NAME" ]; then
    echo "ERROR: No config name provided"
    echo "Usage: $0 <config_name>"
    exit 1
fi

BASE_DIR="/data/bfys/gscriven/TrackExtrapolation/experiments/field_maps/field_nn"
cd "$BASE_DIR" || { echo "ERROR: Cannot cd to $BASE_DIR"; exit 1; }

CONFIG_FILE="configs_v2/${CONFIG_NAME}.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls configs_v2/*.json 2>/dev/null | head -20
    exit 1
fi

echo "=============================================="
echo "Field Map NN v2: $CONFIG_NAME"
echo "=============================================="
echo "Hostname:     $(hostname)"
echo "Date:         $(date)"
echo "GPU:          ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Config:       $CONFIG_FILE"
echo "=============================================="

# Activate conda environment
source /data/bfys/gscriven/conda/etc/profile.d/conda.sh
conda activate TE

echo ""
echo "Python:       $(which python)"
echo "PyTorch:      $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:   $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "GPU device:   $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null
else
    echo "WARNING: No GPU available, training will use CPU"
fi

FIELD_MAP="/data/bfys/gscriven/TrackExtrapolation/experiments/field_maps/twodip.rtf"
if [ -f "$FIELD_MAP" ]; then
    echo "Field map:    OK ($FIELD_MAP)"
else
    echo "ERROR: Field map not found: $FIELD_MAP"
    exit 1
fi

echo ""
echo "Starting training: $CONFIG_NAME"
echo "=============================================="
echo ""

python training/train_field_nn_v2.py --config "$CONFIG_FILE"
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Job finished: $CONFIG_NAME"
echo "Exit code:    $EXIT_CODE"
echo "Date:         $(date)"
echo "=============================================="

exit $EXIT_CODE
