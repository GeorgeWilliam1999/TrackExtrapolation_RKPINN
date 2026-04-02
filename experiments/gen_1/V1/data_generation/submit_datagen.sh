#!/bin/bash
# =============================================================================
# submit_datagen.sh — Submit data generation jobs to HTCondor
#
# Run from the Nikhef login node:
#   cd /data/bfys/gscriven/TrackExtrapolation/experiments/gen_1/V1/data_generation
#   bash submit_datagen.sh
#
# Configuration:
#   - 2000 jobs × 25K tracks = 50 million tracks
#   - 2 CPUs per job, 4 GB RAM, "medium" category (<8h)
#   - Estimated ~4.6h per job at ~1.5 tracks/s
#   - Output: datasets/batch_XXXX.npz
#
# After completion, merge with:
#   python merge_batches.py --input "datasets/batch_*.npz" \
#       --output datasets/train_50M.npz --verify
# =============================================================================
set -e

DATAGEN_DIR="/data/bfys/gscriven/TrackExtrapolation/experiments/gen_1/V1/data_generation"

echo "============================================"
echo "  Track Extrapolation Data Generation"
echo "  50M tracks via HTCondor"
echo "============================================"
echo ""

# ── Step 1: Create output and log directories ────────────────────
echo "[1/3] Creating directories ..."
mkdir -p "$DATAGEN_DIR/datasets"
mkdir -p "$DATAGEN_DIR/condor/logs"
echo "  datasets:  $DATAGEN_DIR/datasets"
echo "  logs:      $DATAGEN_DIR/condor/logs"
echo ""

# ── Step 2: Make worker script executable ────────────────────────
chmod +x "$DATAGEN_DIR/condor/run_datagen.sh"

# ── Step 3: Submit to HTCondor ───────────────────────────────────
echo "[2/3] Submitting 2000 jobs (25K tracks each = 50M total) ..."
echo "  Config: 2 CPUs, 4 GB RAM, medium priority"
echo ""

condor_submit "$DATAGEN_DIR/condor/datagen.sub"

echo ""
echo "[3/3] Submitted! Monitor with:"
echo "  condor_q                          # your jobs"
echo "  condor_q -analyze                 # why jobs are idle"
echo "  condor_q -af ClusterId ProcId JobStatus RemoteHost"
echo "  tail -f condor/logs/batch_0.out   # first job output"
echo ""
echo "After completion, merge batches:"
echo "  python merge_batches.py --input 'datasets/batch_*.npz' \\"
echo "      --output datasets/train_50M.npz --verify"
echo ""
