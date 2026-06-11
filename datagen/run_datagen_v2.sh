#!/usr/bin/env bash
set -euo pipefail
# Code lives in the repo; big data stays in the lab (TE_LAB).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TE_LAB="${TE_LAB:-/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3}"
exec /data/bfys/gscriven/conda/envs/TE/bin/python \
  "${REPO_DIR}/datagen/generate_data_v2.py" \
  "$1" 100000 "${TE_LAB}/data/gen4_shards"
# Gen-4 corpus: 100 shards x 100k = 10M tracks. v8r1 field, kappa=1e-3, 70/30 pointing mix.
