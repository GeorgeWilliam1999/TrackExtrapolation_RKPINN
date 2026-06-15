#!/usr/bin/env bash
# build_bench_host.sh — compile the CPU-only host dumper for extrapUTT chart
# tables. No GPU / nvcc needed; runs on the login node.
#
# Usage: ALLEN_DIR=/path/to/Allen STACK_DIR=/path/to/TE_stack ./build_bench_host.sh
set -euo pipefail
ALLEN_DIR=${ALLEN_DIR:-/data/bfys/gscriven/Allen}
STACK_DIR=${STACK_DIR:-/data/bfys/gscriven/TE_stack}
HERE=$(cd "$(dirname "$0")" && pwd)
STANDALONE="$HERE/../standalone"

g++ -std=c++20 -O2 -Wall -Wno-unused-variable -include cassert \
  -I"$HERE" -I"$STANDALONE" \
  -I"$ALLEN_DIR/device/event_model/kalman/include" \
  -I"$ALLEN_DIR/device/event_model/common/include" \
  -I"$STACK_DIR/Allen/main/include" \
  "$HERE/dump_utt_params.cpp" -o "$HERE/dump_utt_params" -lm
echo "built: $HERE/dump_utt_params"
