#!/usr/bin/env bash
# Build the extrapUTT bake-off driver against an UNMODIFIED Allen checkout.
# Usage: ALLEN_DIR=/path/to/Allen STACK_DIR=/path/to/TE_stack ./build_extraputt.sh
set -euo pipefail
ALLEN_DIR=${ALLEN_DIR:-/data/bfys/gscriven/Allen}
STACK_DIR=${STACK_DIR:-/data/bfys/gscriven/TE_stack}
HERE=$(cd "$(dirname "$0")" && pwd)
# verbatim extraction of extrapUTT from the (read-only) Allen header
awk '/^__device__ inline void extrapUTT\(/{f=1} f&&/^__device__ inline void ExtrapolateUTT\(/{exit} f{print}' \
  "$ALLEN_DIR/device/kalman/ParKalman/include/ParKalmanMethods.cuh" > "$HERE/extraputt_snippet.inc"
g++ -std=c++20 -O2 -Wall -Wno-unused-variable -include cassert \
  -I"$HERE" -I"$HERE/standalone" \
  -I"$ALLEN_DIR/device/event_model/kalman/include" \
  -I"$ALLEN_DIR/device/event_model/common/include" \
  -I"$STACK_DIR/Allen/main/include" \
  "$HERE/extraputt_baseline.cpp" -o "$HERE/extraputt_baseline" -lm
echo "built: $HERE/extraputt_baseline  (param dir e.g. $STACK_DIR/Allen/build.ci-cpu-master/external/ParamFiles/data)"
