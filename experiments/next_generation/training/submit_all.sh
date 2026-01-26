#!/bin/bash
# =============================================================================
# HTCondor Training Job Submission Script
# Submits all 29 experiments to HTCondor
# =============================================================================

SUBMIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SUBMIT_DIR"

echo "=============================================="
echo "Submitting Training Jobs to HTCondor"
echo "=============================================="
echo ""

# Count submissions
TOTAL=0
SUBMITTED=0

# Submit all jobs
for sub_file in jobs/*.sub; do
    if [ -f "$sub_file" ]; then
        TOTAL=$((TOTAL + 1))
        JOB_NAME=$(basename "$sub_file" .sub)
        echo "Submitting: $JOB_NAME"
        condor_submit "$sub_file"
        if [ $? -eq 0 ]; then
            SUBMITTED=$((SUBMITTED + 1))
        else
            echo "  ERROR: Failed to submit $JOB_NAME"
        fi
    fi
done

echo ""
echo "=============================================="
echo "Submitted $SUBMITTED / $TOTAL jobs"
echo "=============================================="
echo ""
echo "Monitor with: condor_q"
echo "Check logs in: logs/"
