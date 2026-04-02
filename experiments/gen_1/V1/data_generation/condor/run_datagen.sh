#!/bin/bash
# =============================================================================
# run_datagen.sh — HTCondor worker for track data generation
#
# Called by datagen.sub with arguments:
#   $1 = batch_id       (0-based integer from $(Process))
#   $2 = n_tracks       (number of tracks per batch)
#   $3 = output_dir     (absolute path to datasets directory)
# =============================================================================
set -e

BATCH_ID="$1"
N_TRACKS="$2"
OUTPUT_DIR="$3"

# ── Paths ──
CONDA_PREFIX="/data/bfys/gscriven/conda"
ENV_NAME="TE"
DATAGEN_DIR="/data/bfys/gscriven/TrackExtrapolation/experiments/gen_1/V1/data_generation"

# ── Batch name (zero-padded) ──
BATCH_NAME=$(printf "batch_%04d" "$BATCH_ID")

echo "======================================="
echo "  Data Generation Batch"
echo "  Batch:     $BATCH_NAME (id=$BATCH_ID)"
echo "  Tracks:    $N_TRACKS"
echo "  Output:    $OUTPUT_DIR"
echo "  Host:      $(hostname)"
echo "  CPUs:      $(nproc)"
echo "  Date:      $(date)"
echo "======================================="

# ── Activate conda environment ──
eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "Python: $(which python)"
echo "NumPy:  $(python -c 'import numpy; print(numpy.__version__)')"

# ── Run data generation ──
# Use 2 workers per job; unique seed per batch for reproducibility
SEED=$((42 + BATCH_ID * 7919))  # large prime spacing to avoid overlap

python "$DATAGEN_DIR/generate_data.py" \
    --n-tracks "$N_TRACKS" \
    --name "$BATCH_NAME" \
    --output-dir "$OUTPUT_DIR" \
    --workers 2 \
    --seed "$SEED" \
    --z-min 0 \
    --z-max 14000 \
    --dz-min 100 \
    --dz-max 10000 \
    --p-min 1.0 \
    --p-max 100.0 \
    --step-size 5.0 \
    --polarity -1

echo ""
echo "Batch $BATCH_NAME completed at $(date)"
