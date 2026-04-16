#!/bin/bash
# =============================================================================
# Generate gen_2 training dataset: 50M tracks with dz_min=25mm
# Covers VELO spacing (~25mm) through full detector (~10000mm)
# Extended momentum range: 1-200 GeV/c
# =============================================================================

set -euo pipefail

echo "========================================="
echo "gen_2 Data Generation"
echo "Host:     $(hostname)"
echo "Date:     $(date)"
echo "========================================="

source /data/bfys/gscriven/conda/etc/profile.d/conda.sh
conda activate TE

cd /data/bfys/gscriven/TrackExtrapolation/experiments/gen_1/V1/data_generation

python generate_data.py \
    --n-tracks 50000000 \
    --name train_50M_dz25 \
    --output-dir /data/bfys/gscriven/TrackExtrapolation/experiments/gen_2/data \
    --dz-min 25 \
    --dz-max 10000 \
    --z-min 0 \
    --z-max 14000 \
    --p-min 1.0 \
    --p-max 200.0 \
    --step-size 5.0 \
    --polarity -1 \
    --workers 16 \
    --seed 42

echo ""
echo "========================================="
echo "Data generation completed at $(date)"
echo "========================================="
