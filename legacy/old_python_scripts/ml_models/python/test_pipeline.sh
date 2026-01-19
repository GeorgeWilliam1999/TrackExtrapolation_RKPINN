#!/bin/bash
#
# Test script to verify the training pipeline works
#
# This script runs a minimal test:
# 1. Generate 1000 training samples
# 2. Train a small model for 10 epochs
# 3. Verify all files are created
#
# Usage: ./test_pipeline.sh

set -e

echo "========================================"
echo "Training Pipeline Test"
echo "========================================"

BASE_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/ml_models"
cd "${BASE_DIR}/python"

# Activate environment
source /data/bfys/gscriven/conda/bin/activate TE

echo ""
echo "Step 1: Generate test data (1000 samples)..."
python generate_training_data.py \
    --samples 1000 \
    --output ../data/ \
    --name test

echo ""
echo "Step 2: Train test model (10 epochs)..."
python train_on_gpu.py \
    --data ../data/ \
    --dataset test \
    --hidden 64 64 32 \
    --epochs 10 \
    --batch 64 \
    --output ../models/ \
    --name test_model

echo ""
echo "Step 3: Verify files..."

if [ -f "../data/X_test.npy" ]; then
    echo "✓ Data generated: ../data/X_test.npy"
else
    echo "✗ FAILED: Data not found"
    exit 1
fi

if [ -f "../models/test_model.bin" ]; then
    echo "✓ Model saved: ../models/test_model.bin"
else
    echo "✗ FAILED: Model not saved"
    exit 1
fi

if [ -f "../models/test_model_metadata.json" ]; then
    echo "✓ Metadata saved: ../models/test_model_metadata.json"
    
    # Print summary
    echo ""
    echo "Model Summary:"
    python -c "
import json
with open('../models/test_model_metadata.json') as f:
    meta = json.load(f)
    print(f\"  Architecture: {meta['architecture']}\")
    print(f\"  Parameters: {meta['performance']['total_params']:,}\")
    print(f\"  Mean Error: {meta['performance']['mean_error']:.4f} mm\")
    print(f\"  P95 Error: {meta['performance']['p95_error']:.4f} mm\")
    print(f\"  Time/track: {meta['performance']['time_per_track_us']:.2f} μs\")
"
else
    echo "✗ FAILED: Metadata not saved"
    exit 1
fi

echo ""
echo "========================================"
echo "✓ Pipeline test PASSED!"
echo "========================================"
echo ""
echo "You can now:"
echo "  1. Generate full dataset: python generate_training_data.py --samples 100000"
echo "  2. Train production model: python train_on_gpu.py --data ../data/ --model large"
echo "  3. Submit to cluster: cd ../condor && condor_submit train_models.sub"
echo ""
