#!/usr/bin/env bash
set -euo pipefail
# Wave-2 UT->T-focused corpus shard. Code in the repo; data in the lab (TE_LAB).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TE_LAB="${TE_LAB:-/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3}"
exec /data/bfys/gscriven/conda/envs/TE/bin/python \
  "${REPO_DIR}/datagen/generate_utt_focused.py" \
  "$1" 95000 "${TE_LAB}/data/utt_focused_shards"
