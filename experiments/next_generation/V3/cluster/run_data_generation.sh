#!/bin/bash
#===============================================================================
# Data Generation Job Script
# Runs on CPU nodes with many cores for parallel trajectory generation
#===============================================================================

set -e

# Setup environment
source /data/bfys/gscriven/conda/etc/profile.d/conda.sh
conda activate TE

cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation

echo "=============================================="
echo "V3 Data Generation - Realistic IC Ranges"
echo "=============================================="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CPUs: $1"
echo ""

#===============================================================================
# Step 1: Generate trajectories (10k high-res trajectories)
#===============================================================================
echo "[1/3] Generating 10k trajectories..."
echo "Parameters: x=±1500mm, y=±1200mm, tx=±0.4, ty=±0.35"

python V3/data_generation/generate_trajectories.py \
    --n_trajectories 10000 \
    --z_start 0 \
    --z_end 15000 \
    --step_size 5 \
    --p_min 0.5 \
    --p_max 100 \
    --workers $1 \
    --seed 42 \
    --output V3/data/trajectories_10k_v2.npz

echo "Done: V3/data/trajectories_10k_v2.npz"
ls -lh V3/data/trajectories_10k_v2.npz

#===============================================================================
# Step 2: Extract MLP training data (100M samples)
#===============================================================================
echo ""
echo "[2/3] Extracting 100M MLP training samples..."

python V3/data_generation/extract_segments.py \
    --input V3/data/trajectories_10k_v2.npz \
    --n_samples 100000000 \
    --dz_min 500 \
    --dz_max 12000 \
    --seed 123 \
    --output V3/data/training_mlp_v3_100M_v2.npz

echo "Done: V3/data/training_mlp_v3_100M_v2.npz"
ls -lh V3/data/training_mlp_v3_100M_v2.npz

#===============================================================================
# Step 3: Extract PINN training data (10M samples with collocation)
#===============================================================================
echo ""
echo "[3/3] Extracting PINN training samples with collocation points..."

for NCOL in 5 10 20 50; do
    echo "  Generating PINN data with ${NCOL} collocation points..."
    python V3/data_generation/extract_segments.py \
        --input V3/data/trajectories_10k_v2.npz \
        --n_samples 10000000 \
        --dz_min 500 \
        --dz_max 12000 \
        --collocation_points $NCOL \
        --seed $((200 + NCOL)) \
        --output V3/data/training_pinn_v3_col${NCOL}_v2.npz
    ls -lh V3/data/training_pinn_v3_col${NCOL}_v2.npz
done

echo ""
echo "=============================================="
echo "Data generation complete!"
echo "=============================================="
echo ""
echo "Generated files:"
ls -lh V3/data/*_v2.npz
