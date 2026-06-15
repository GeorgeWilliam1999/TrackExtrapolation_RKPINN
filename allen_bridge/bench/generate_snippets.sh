#!/usr/bin/env bash
# generate_snippets.sh — extract the two device functions we time VERBATIM from the
# read-only Allen headers, so the micro-bench never hand-copies physics code.
#
#   compute_state_snippet.inc : KalmanParametrizations::compute_state<> template
#                               (device polynomial chart contraction)
#   extraputt_snippet.inc     : the device extrapUTT() function
#
# Allen is never modified; we only read from it.
set -euo pipefail
ALLEN_DIR=${ALLEN_DIR:-/data/bfys/gscriven/Allen}
HERE=$(cd "$(dirname "$0")" && pwd)
KP="$ALLEN_DIR/device/event_model/kalman/include/KalmanParametrizations.cuh"
PM="$ALLEN_DIR/device/kalman/ParKalman/include/ParKalmanMethods.cuh"
RK="$ALLEN_DIR/device/kalman/ParKalman/include/RungeKuttaExtrapolator.cuh"

# CashKarp RungeKuttaExtrapolator struct ONLY (the production extrapolate_states
# path). The rest of RungeKuttaExtrapolator.cuh (the Nyström struct) uses C++20
# requires/std::convertible_to that NVRTC's freestanding mode can't satisfy, and
# we don't call it — so we lift just the struct we time, verbatim.
awk '/template<typename ftype = float, typename Table = ButcherTableau::CashKarp/{f=1} f{print} f&&/^  };$/{exit}' "$RK" \
  > "$HERE/rk_cashkarp_snippet.inc"

# compute_state: from the template line up to and including its closing brace at
# column 4 (the next member, UttOffset, then follows).
awk '/^    template<int DEG0, int DEG1>/{f=1} f{print} f&&/^    }$/{exit}' "$KP" \
  > "$HERE/compute_state_snippet.inc"

# extrapUTT: from its signature up to (not including) ExtrapolateUTT.
awk '/^__device__ inline void extrapUTT\(/{f=1} f&&/^__device__ inline void ExtrapolateUTT\(/{exit} f{print}' "$PM" \
  > "$HERE/extraputt_snippet.inc"

echo "rk_cashkarp_snippet.inc   : $(wc -l < "$HERE/rk_cashkarp_snippet.inc") lines"
echo "compute_state_snippet.inc : $(wc -l < "$HERE/compute_state_snippet.inc") lines"
echo "extraputt_snippet.inc     : $(wc -l < "$HERE/extraputt_snippet.inc") lines"
