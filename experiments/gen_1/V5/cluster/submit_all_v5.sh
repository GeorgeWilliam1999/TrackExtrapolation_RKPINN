#!/bin/bash
# =============================================================================
# V5 Master Submission Script
# Validates environment and submits all V5 training jobs to HTCondor
#
# Usage:
#   cd experiments/next_generation/
#   bash V5/cluster/submit_all_v5.sh          # dry run (default)
#   bash V5/cluster/submit_all_v5.sh --submit # actually submit
# =============================================================================

set -e

BASE_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation"
cd "$BASE_DIR"

SUBMIT_FILE="V5/cluster/submit_v5_training.sub"
JOB_FILE="V5/cluster/v5_jobs.txt"
WRAPPER="V5/training/train_v5_wrapper.sh"
CONFIG_DIR="V5/training/configs"
LOG_DIR="V5/cluster/logs"

DRY_RUN=true
if [ "$1" = "--submit" ]; then
    DRY_RUN=false
fi

echo "=============================================="
echo "V5 Training Submission"
echo "=============================================="
echo "Date:       $(date)"
echo "Mode:       $([ "$DRY_RUN" = true ] && echo 'DRY RUN' || echo 'SUBMITTING')"
echo "=============================================="

# ---- Pre-flight checks ----

ERRORS=0

# Check submit file
if [ ! -f "$SUBMIT_FILE" ]; then
    echo "ERROR: Submit file not found: $SUBMIT_FILE"
    ERRORS=$((ERRORS + 1))
fi

# Check job file
if [ ! -f "$JOB_FILE" ]; then
    echo "ERROR: Job file not found: $JOB_FILE"
    ERRORS=$((ERRORS + 1))
fi

# Check wrapper
if [ ! -x "$WRAPPER" ]; then
    echo "WARNING: Wrapper not executable: $WRAPPER (will attempt chmod)"
    chmod +x "$WRAPPER" 2>/dev/null && echo "  Fixed: $WRAPPER is now executable" || ERRORS=$((ERRORS + 1))
fi

# Check training script
if [ ! -f "V5/training/train_v5.py" ]; then
    echo "ERROR: Training script not found: V5/training/train_v5.py"
    ERRORS=$((ERRORS + 1))
fi

# Count and validate configs
echo ""
echo "Checking config files..."
N_JOBS=0
N_MISSING=0
while IFS= read -r line; do
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue

    CONFIG_NAME="$line"
    CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}.json"

    if [ -f "$CONFIG_PATH" ]; then
        echo "  OK: $CONFIG_NAME"
        N_JOBS=$((N_JOBS + 1))
    else
        echo "  MISSING: $CONFIG_NAME -> $CONFIG_PATH"
        N_MISSING=$((N_MISSING + 1))
        ERRORS=$((ERRORS + 1))
    fi
done < "$JOB_FILE"

echo ""
echo "Jobs found:    $N_JOBS"
echo "Jobs missing:  $N_MISSING"

# Check training data
echo ""
echo "Checking training data..."
MLP_DATA="V3/data/training_mlp_v3_100M_v2.npz"
PINN_DATA="V3/data/training_pinn_v3_col10_v2.npz"

if [ -f "$MLP_DATA" ]; then
    MLP_SIZE=$(du -h "$MLP_DATA" | cut -f1)
    echo "  OK: MLP data ($MLP_SIZE) - $MLP_DATA"
else
    echo "  MISSING: $MLP_DATA"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "$PINN_DATA" ]; then
    PINN_SIZE=$(du -h "$PINN_DATA" | cut -f1)
    echo "  OK: PINN data ($PINN_SIZE) - $PINN_DATA"
else
    echo "  MISSING: $PINN_DATA"
    ERRORS=$((ERRORS + 1))
fi

# Check field map (needed for PDE models)
echo ""
echo "Checking magnetic field map..."
FIELD_MAP="field_maps/twodip.rtf"
if [ -f "$FIELD_MAP" ]; then
    FIELD_SIZE=$(du -h "$FIELD_MAP" | cut -f1)
    echo "  OK: Field map ($FIELD_SIZE) - $FIELD_MAP"
else
    echo "  WARNING: Field map not found ($FIELD_MAP) — PDE models will use Gaussian approximation"
fi

# Check output directories
echo ""
echo "Checking output directories..."
mkdir -p "$LOG_DIR"
echo "  OK: Log directory: $LOG_DIR"

# Create output model directories
while IFS= read -r line; do
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue

    CONFIG_NAME="$line"
    CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}.json"

    if [ -f "$CONFIG_PATH" ]; then
        OUTPUT_DIR=$(python -c "import json; c=json.load(open('$CONFIG_PATH')); print(c['output']['dir'])" 2>/dev/null)
        if [ -n "$OUTPUT_DIR" ]; then
            mkdir -p "$OUTPUT_DIR"
        fi
    fi
done < "$JOB_FILE"
echo "  OK: Output model directories created"

# Check for existing outputs
echo ""
echo "Checking for existing trained models..."
N_EXISTING=0
while IFS= read -r line; do
    [[ "$line" =~ ^#.*$ ]] && continue
    [[ -z "$line" ]] && continue

    CONFIG_NAME="$line"
    CONFIG_PATH="${CONFIG_DIR}/${CONFIG_NAME}.json"

    if [ -f "$CONFIG_PATH" ]; then
        OUTPUT_DIR=$(python -c "import json; c=json.load(open('$CONFIG_PATH')); print(c['output']['dir'])" 2>/dev/null)
        if [ -n "$OUTPUT_DIR" ] && [ -f "${OUTPUT_DIR}/best_model.pt" ]; then
            echo "  WARNING: Already trained: $CONFIG_NAME"
            N_EXISTING=$((N_EXISTING + 1))
        fi
    fi
done < "$JOB_FILE"

if [ $N_EXISTING -gt 0 ]; then
    echo "  $N_EXISTING models already have trained weights (will be overwritten)"
fi

# ---- Summary ----

echo ""
echo "=============================================="
echo "V5 Summary"
echo "=============================================="
echo ""
echo "  Model types:"
echo "    1. MLP baseline      (mlp)"
echo "    2. QuadraticResidual (quadratic)"
echo "    3. ZFracInput PINN   (zfrac)"
echo "    4. PDE-Residual PINN (pde — pure & hybrid)"
echo "    5. Compositional     (compositional — N=8 & N=16)"
echo ""
echo "  Total jobs:       $N_JOBS"
echo "  Already trained:  $N_EXISTING"
echo "  Errors:           $ERRORS"
echo ""

if [ $ERRORS -gt 0 ]; then
    echo "ABORTING: $ERRORS error(s) found. Fix them before submitting."
    exit 1
fi

# ---- Submit ----

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN: Would submit $N_JOBS jobs with:"
    echo "  condor_submit $SUBMIT_FILE"
    echo ""
    echo "To actually submit, run:"
    echo "  bash V5/cluster/submit_all_v5.sh --submit"
else
    echo "Submitting $N_JOBS jobs..."
    condor_submit "$SUBMIT_FILE"
    echo ""
    echo "Jobs submitted. Monitor with:"
    echo "  condor_q -nobatch"
    echo "  watch -n 30 'condor_q -nobatch'"
fi

echo ""
echo "Done."
