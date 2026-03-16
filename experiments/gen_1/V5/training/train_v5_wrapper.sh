#!/bin/bash
# =============================================================================
# V5 Training Job Wrapper Script
# Executed on HTCondor worker node
#
# Activates conda environment and runs train_v5.py with the specified config.
#
# Usage:
#   ./train_v5_wrapper.sh <config_name>
#   e.g. ./train_v5_wrapper.sh pde_pure_v5_2L_1024_512
# =============================================================================

CONFIG_NAME="$1"

if [ -z "$CONFIG_NAME" ]; then
    echo "ERROR: No config name provided"
    echo "Usage: $0 <config_name>"
    exit 1
fi

# Base directory (shared filesystem, no file transfer needed)
BASE_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation"
cd "$BASE_DIR" || { echo "ERROR: Cannot cd to $BASE_DIR"; exit 1; }

CONFIG_FILE="V5/training/configs/${CONFIG_NAME}.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls V5/training/configs/*.json 2>/dev/null | head -30
    exit 1
fi

echo "=============================================="
echo "V5 Training Job: $CONFIG_NAME"
echo "=============================================="
echo "Hostname:     $(hostname)"
echo "Date:         $(date)"
echo "GPU:          ${CUDA_VISIBLE_DEVICES:-not set}"
echo "Config:       $CONFIG_FILE"
echo "=============================================="

# Activate conda environment
source /data/bfys/gscriven/conda/etc/profile.d/conda.sh
conda activate TE

# Verify environment
echo ""
echo "Python:       $(which python)"
echo "PyTorch:      $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:   $(python -c 'import torch; print(torch.cuda.is_available())')"

if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "GPU device:   $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null
else
    echo "WARNING: No GPU available, training will use CPU (very slow)"
fi

# Check field map for PDE models
if [[ "$CONFIG_NAME" == pde_* ]]; then
    FIELD_MAP="field_maps/twodip.rtf"
    if [ -f "$FIELD_MAP" ]; then
        echo "Field map:    OK ($FIELD_MAP)"
    else
        echo "WARNING: Field map not found at $FIELD_MAP — PDE models will use Gaussian approximation"
    fi
fi

echo ""
echo "Starting training: $CONFIG_NAME"
echo "=============================================="
echo ""

# Run training
python V5/training/train_v5.py --config "$CONFIG_FILE"
EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training finished: $CONFIG_NAME"
echo "Exit code: $EXIT_CODE"
echo "Date: $(date)"
echo "=============================================="

exit $EXIT_CODE
