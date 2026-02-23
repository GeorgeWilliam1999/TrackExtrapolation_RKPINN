#!/bin/bash
#===============================================================================
# Data Extraction Job Script - extracts training data from existing trajectories
#===============================================================================

set -e

# Setup environment
source /data/bfys/gscriven/conda/etc/profile.d/conda.sh
conda activate TE

cd /data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/experiments/next_generation

echo "=============================================="
echo "V3 Data Extraction"
echo "=============================================="
echo "Host: $(hostname)"
echo "Date: $(date)"

# Check trajectories exist
if [ ! -f V3/data/trajectories_10k_v2.npz ]; then
    echo "ERROR: trajectories_10k_v2.npz not found!"
    exit 1
fi

echo "Using existing trajectories: $(ls -lh V3/data/trajectories_10k_v2.npz)"

#===============================================================================
# Extract MLP training data (100M samples)
#===============================================================================
echo ""
echo "[1/2] Extracting 100M MLP training samples..."

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
# Extract PINN training data (10M samples with collocation)
#===============================================================================
echo ""
echo "[2/2] Extracting PINN training samples with collocation points..."

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
echo "Data extraction complete!"
echo "=============================================="
ls -lh V3/data/*_v2.npz
