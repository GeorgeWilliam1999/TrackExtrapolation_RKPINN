#!/bin/bash
# =============================================================================
# Field Map NN -- Master Submission Script
#
# Usage:
#   bash cluster/submit_all.sh          # dry run (default)
#   bash cluster/submit_all.sh --submit # actually submit
# =============================================================================

set -e

BASE_DIR="/data/bfys/gscriven/TrackExtrapolation/experiments/field_maps/field_nn"
cd "$BASE_DIR"

SUBMIT_FILE="cluster/submit_field_nn.sub"
JOB_FILE="cluster/field_nn_jobs.txt"
WRAPPER="training/train_field_nn_wrapper.sh"
TRAIN_SCRIPT="training/train_field_nn.py"
CONFIG_DIR="configs"
LOG_DIR="cluster/logs"
FIELD_MAP="/data/bfys/gscriven/TrackExtrapolation/experiments/field_maps/twodip.rtf"

DRY_RUN=true
if [ "$1" = "--submit" ]; then
    DRY_RUN=false
fi

echo "=============================================="
echo "Field Map NN Grid Search -- Submission"
echo "=============================================="
echo "Date:       $(date)"
echo "Mode:       $([ "$DRY_RUN" = true ] && echo 'DRY RUN (add --submit to actually submit)' || echo 'SUBMITTING')"
echo "=============================================="

ERRORS=0

for FILE in "$SUBMIT_FILE" "$JOB_FILE" "$TRAIN_SCRIPT"; do
    if [ ! -f "$FILE" ]; then
        echo "ERROR: File not found: $FILE"
        ERRORS=$((ERRORS + 1))
    else
        echo "  OK: $FILE"
    fi
done

if [ ! -x "$WRAPPER" ]; then
    echo "  WARNING: Wrapper not executable, fixing..."
    chmod +x "$WRAPPER" 2>/dev/null && echo "  Fixed: $WRAPPER" || ERRORS=$((ERRORS + 1))
else
    echo "  OK: $WRAPPER (executable)"
fi

if [ -f "$FIELD_MAP" ]; then
    FSIZE=$(du -h "$FIELD_MAP" | cut -f1)
    echo "  OK: Field map ($FSIZE)"
else
    echo "  ERROR: Field map not found: $FIELD_MAP"
    ERRORS=$((ERRORS + 1))
fi

if [ -f "training/trilinear.py" ]; then
    echo "  OK: trilinear.py"
else
    echo "  ERROR: trilinear.py not found"
    ERRORS=$((ERRORS + 1))
fi

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

mkdir -p "$LOG_DIR"

echo ""
echo "Quick import test..."
python -c "
import sys; sys.path.insert(0, 'training')
from trilinear import TrilinearGrid
from train_field_nn import FieldMLP, count_params, count_flops
m = FieldMLP([64], 'relu')
print(f'  FieldMLP([64]): {count_params(m)} params, {count_flops([64])} FLOPs')
print('  Import test: PASSED')
" 2>&1 || { echo "  Import test: FAILED"; ERRORS=$((ERRORS + 1)); }

echo ""
echo "=============================================="

if [ $ERRORS -gt 0 ]; then
    echo "ERRORS: $ERRORS -- fix before submitting"
    exit 1
fi

echo "Pre-flight checks: ALL PASSED"
echo "Ready to submit $N_JOBS jobs"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "DRY RUN -- no jobs submitted."
    echo "Run with --submit to actually submit:"
    echo "  bash cluster/submit_all.sh --submit"
else
    echo ""
    echo "Submitting to HTCondor..."
    condor_submit "$SUBMIT_FILE"
    echo ""
    echo "Jobs submitted. Monitor with:"
    echo "  condor_q -nobatch"
fi
