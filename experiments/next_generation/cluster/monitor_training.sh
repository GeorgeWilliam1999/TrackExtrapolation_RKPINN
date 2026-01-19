#!/bin/bash
# Monitor GPU training jobs

LOGS_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/cluster/logs"
TRAINED_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation/trained_models"

echo "=============================================="
echo "GPU Training Monitor - $(date)"
echo "=============================================="
echo ""

# Job status
echo "=== HTCondor Job Status ==="
condor_q -submitter gscriven 2>/dev/null | tail -3
echo ""

# Epoch distribution
echo "=== Epoch Distribution ==="
grep -h "^Epoch" ${LOGS_DIR}/gpu_suite_3839198_*.out 2>/dev/null | awk -F'/' '{print $1}' | sort | uniq -c | sort -t' ' -k3 -n
echo ""

# Best position errors
echo "=== Top 10 Best Position Errors ==="
for out in ${LOGS_DIR}/gpu_suite_3839198_*.out; do
    job=$(basename "$out" | sed 's/gpu_suite_3839198_//' | sed 's/.out//')
    model=$(grep "^Experiment:" "$out" 2>/dev/null | head -1 | awk '{print $2}')
    best=$(grep "val_pos=" "$out" 2>/dev/null | awk -F'val_pos=' '{print $2}' | awk -F'mm' '{print $1}' | sort -n | head -1)
    if [ ! -z "$best" ]; then
        echo "$best mm - $model (job $job)"
    fi
done | sort -n | head -10
echo ""

# Check for completed models
echo "=== Completed Models ==="
for d in ${TRAINED_DIR}/*/; do
    if [ -f "$d/results.json" ]; then
        name=$(basename "$d")
        pos_err=$(python3 -c "import json; r=json.load(open('${d}/results.json')); print(f\"{r['test_pos_error_mm']:.4f}mm\")" 2>/dev/null)
        echo "$name: $pos_err"
    fi
done | sort -t: -k2 -n | head -10

echo ""
echo "=============================================="
echo "Run: bash monitor_training.sh"
echo "=============================================="
