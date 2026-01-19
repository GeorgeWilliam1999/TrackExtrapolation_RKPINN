#!/bin/bash
#
# Wrapper script for HTCondor data generation job
#
# This script is called by HTCondor with a process ID argument
# to generate training data in parallel
#
# Usage: ./generate_data.sh <process_id>

set -e  # Exit on error

PROCESS_ID=$1
BASE_DIR="/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/ml_models"
PYTHON_DIR="${BASE_DIR}/python"
DATA_DIR="${BASE_DIR}/data"

# Activate conda environment
source /data/bfys/gscriven/conda/bin/activate TE

echo "=========================================="
echo "HTCondor Data Generation Job"
echo "=========================================="
echo "Process ID: ${PROCESS_ID}"
echo "Hostname: $(hostname)"
echo "Date: $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "=========================================="

# Create data directory
mkdir -p "${DATA_DIR}"

# Generate data (100K samples per job)
# Each job uses different random seed based on process ID
cd "${PYTHON_DIR}"

python generate_training_data.py \
    --samples 100000 \
    --workers 8 \
    --output "${DATA_DIR}" \
    --name "train_part${PROCESS_ID}"

echo "=========================================="
echo "Data generation complete!"
echo "=========================================="
