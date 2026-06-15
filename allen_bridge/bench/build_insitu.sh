#!/usr/bin/env bash
# build_insitu.sh — compile the Tier-2 CPU in-situ harness (g++, no GPU).
set -euo pipefail
ALLEN_DIR=${ALLEN_DIR:-/data/bfys/gscriven/Allen}
STACK_DIR=${STACK_DIR:-/data/bfys/gscriven/TE_stack}
HERE=$(cd "$(dirname "$0")" && pwd)
STANDALONE="$HERE/../standalone"

bash "$HERE/generate_snippets.sh"  # ensure extraputt_snippet.inc is fresh

g++ -std=c++20 -O3 -march=native -funroll-loops -Wall -Wno-unused-variable -include cassert \
  -I"$HERE" -I"$STANDALONE" \
  -I"$ALLEN_DIR/device/kalman/ParKalman/include" \
  -I"$ALLEN_DIR/device/event_model/kalman/include" \
  -I"$ALLEN_DIR/device/event_model/common/include" \
  -I"$STACK_DIR/Allen/main/include" \
  "$HERE/insitu_parkalman.cpp" -o "$HERE/insitu_parkalman" -lm
echo "built: $HERE/insitu_parkalman"
