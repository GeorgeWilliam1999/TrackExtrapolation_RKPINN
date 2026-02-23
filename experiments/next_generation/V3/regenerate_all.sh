#!/bin/bash
#===============================================================================
# V3 Full Retrain Pipeline
# 
# Regenerates training data with realistic IC ranges and retrains all models.
#
# Parameter changes:
#   - x: ±300 → ±1500 mm  
#   - y: ±250 → ±1200 mm
#   - tx: ±0.15 → ±0.4
#   - ty: ±0.15 → ±0.35
#
# This matches the LHCb test coverage which uses x,y up to ±900mm and 
# slopes up to ±0.3
#===============================================================================

set -e

WORKDIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation"
cd $WORKDIR

# Activate conda environment
source /data/bfys/gscriven/conda/etc/profile.d/conda.sh
conda activate TE

echo "=============================================="
echo "V3 Full Retrain Pipeline"
echo "=============================================="
echo "Working directory: $WORKDIR"
echo ""

#===============================================================================
# Step 1: Generate trajectories (10k high-res trajectories)
#===============================================================================
echo "[1/5] Generating trajectories with realistic IC ranges..."

python V3/data_generation/generate_trajectories.py \
    --n_trajectories 10000 \
    --z_start 0 \
    --z_end 15000 \
    --step_size 5 \
    --p_min 0.5 \
    --p_max 100 \
    --n_workers 8 \
    --output V3/data/trajectories_10k_v2.npz

echo "Trajectories generated: V3/data/trajectories_10k_v2.npz"

#===============================================================================
# Step 2: Extract MLP training data (100M samples)
#===============================================================================
echo ""
echo "[2/5] Extracting MLP training samples..."

python V3/data_generation/extract_segments.py \
    --trajectories V3/data/trajectories_10k_v2.npz \
    --n_samples 100000000 \
    --dz_min 500 \
    --dz_max 12000 \
    --output V3/data/training_mlp_v3_100M_v2.npz

echo "MLP data generated: V3/data/training_mlp_v3_100M_v2.npz"

#===============================================================================
# Step 3: Extract PINN training data (10M samples with collocation)
#===============================================================================
echo ""
echo "[3/5] Extracting PINN training samples with collocation points..."

for NCOL in 5 10 20; do
    echo "  Generating PINN data with ${NCOL} collocation points..."
    python V3/data_generation/extract_segments.py \
        --trajectories V3/data/trajectories_10k_v2.npz \
        --n_samples 10000000 \
        --dz_min 500 \
        --dz_max 12000 \
        --n_collocation $NCOL \
        --output V3/data/training_pinn_v3_col${NCOL}_v2.npz
done

echo "PINN data generated for col5, col10, col20"

#===============================================================================
# Step 4: Update training configs
#===============================================================================
echo ""
echo "[4/5] Updating training configurations..."

# MLP configs - update data paths
for CONFIG in V3/training/configs/mlp_v3_*.json; do
    sed -i 's|training_mlp_v3_100M.npz|training_mlp_v3_100M_v2.npz|g' $CONFIG
done

# PINN configs - update data paths
for NCOL in 5 10 20; do
    CONFIG="V3/training/configs/pinn_v3_res_256_col${NCOL}.json"
    sed -i "s|training_pinn_v3_col${NCOL}.npz|training_pinn_v3_col${NCOL}_v2.npz|g" $CONFIG
done

echo "Configs updated"

#===============================================================================
# Step 5: Clean old models and prepare output dirs
#===============================================================================
echo ""
echo "[5/5] Cleaning old models..."

rm -rf V3/trained_models/mlp_v3_*
rm -rf V3/trained_models/pinn_v3_*

echo "Old models removed"
echo ""
echo "=============================================="
echo "Data generation complete!"
echo "=============================================="
echo ""
echo "Generated files:"
ls -lh V3/data/*_v2.npz
echo ""
echo "Next step: Submit training jobs with:"
echo "  cd V3/cluster && condor_submit submit_all_v2.sub"
