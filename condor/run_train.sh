#!/usr/bin/env bash
# condor/run_train.sh — HTCondor worker wrapper for gen-3 training runs.
#
# Arguments:
#   $1  path to yaml config file
#   $2  job tag (Cluster.Process, used only for logging)
#
# The script activates the TE conda environment and runs train.py.
# stdout/stderr are captured by condor's output/error directives.
set -euo pipefail

CONFIG="$1"
JOB_TAG="${2:-local}"

CONDA_PYTHON="/data/bfys/gscriven/conda/envs/TE/bin/python3"
TRAIN_SCRIPT="/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/models/train.py"

echo "=========================================="
echo "  gen-3 training job: $JOB_TAG"
echo "  config: $CONFIG"
echo "  host:   $(hostname)"
echo "  date:   $(date)"
echo "=========================================="

exec "$CONDA_PYTHON" "$TRAIN_SCRIPT" --config "$CONFIG"
