#!/bin/bash
# V3 Data Generation Pipeline
#
# Generates all training data for V3 experiments:
# 1. Full trajectories (5mm resolution)
# 2. MLP training samples (100M)
# 3. PINN training samples with varying collocation points
#
# Usage:
#   ./run_datagen_pipeline.sh
#
# Or submit to condor:
#   condor_submit submit_datagen_full.sub

set -e  # Exit on error

cd "$(dirname "$0")/.."  # Go to V3 directory

echo "=============================================================="
echo "V3 Data Generation Pipeline"
echo "=============================================================="
echo "Started at: $(date)"
echo ""

# Create data directory
mkdir -p data

# ============================================================
# Step 1: Generate trajectories
# ============================================================
echo "[1/6] Generating 10k trajectories..."
python data_generation/generate_trajectories.py \
    --n_trajectories 10000 \
    --z_start 0 \
    --z_end 15000 \
    --step_size 5 \
    --p_min 0.5 \
    --p_max 100 \
    --workers 16 \
    --output data/trajectories_10k.npz

echo ""

# ============================================================
# Step 2: Extract MLP samples
# ============================================================
echo "[2/6] Extracting 100M MLP samples..."
python data_generation/extract_segments.py \
    --input data/trajectories_10k.npz \
    --n_samples 100000000 \
    --dz_min 500 \
    --dz_max 12000 \
    --output data/training_mlp_v3.npz

echo ""

# ============================================================
# Step 3-6: Extract PINN samples with varying collocation
# ============================================================
for n_col in 5 10 20 50; do
    echo "[Step] Extracting 10M PINN samples with ${n_col} collocation points..."
    python data_generation/extract_segments.py \
        --input data/trajectories_10k.npz \
        --n_samples 10000000 \
        --dz_min 500 \
        --dz_max 12000 \
        --collocation_points ${n_col} \
        --output data/training_pinn_v3_col${n_col}.npz
    echo ""
done

# ============================================================
# Summary
# ============================================================
echo "=============================================================="
echo "Data Generation Complete!"
echo "=============================================================="
echo "Generated files:"
ls -lh data/
echo ""
echo "Finished at: $(date)"
