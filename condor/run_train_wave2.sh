#!/usr/bin/env bash
# Wave-2 training worker: runs the REPO train.py; data/checkpoints in the lab (TE_LAB).
set -euo pipefail
CONFIG="$1"; JOB_TAG="${2:-local}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TE_LAB="${TE_LAB:-/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3}"
CONDA_PYTHON="/data/bfys/gscriven/conda/envs/TE/bin/python3"
echo "=== wave-2 train: $JOB_TAG  config=$CONFIG  host=$(hostname)  $(date) ==="
exec "$CONDA_PYTHON" "${REPO_DIR}/models/train.py" --config "${REPO_DIR}/$CONFIG"
