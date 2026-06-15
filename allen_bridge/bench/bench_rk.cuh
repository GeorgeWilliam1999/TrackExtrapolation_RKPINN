// bench_rk.cuh — NVRTC-safe assembly of the production CashKarp Runge-Kutta
// extrapolator (Allen's extrapolate_states path).
//
// ExtrapolatorCommon.cuh (State, derivative, c_light) and ButcherTableau.cuh
// (CashKarp) are included VERBATIM from the read-only Allen headers. The
// RungeKuttaExtrapolator struct itself is #included verbatim via
// generate_snippets.sh (rk_cashkarp_snippet.inc) rather than the whole
// RungeKuttaExtrapolator.cuh, whose Nyström struct uses C++20 requires /
// std::convertible_to that NVRTC's freestanding mode cannot resolve and which we
// never call. The device math of the path we time is byte-identical to Allen.
#pragma once

#include "BackendCommon.h"        // shim: UNROLL
#include "ButcherTableau.cuh"     // REAL Allen
#include "ExtrapolatorCommon.cuh" // REAL Allen (opens namespace Extrapolators; State, derivative)
#include "MagneticField.cuh"      // shim: verbatim texture field

namespace Extrapolators {
#include "rk_cashkarp_snippet.inc"  // verbatim RungeKuttaExtrapolator<ftype, Table>
} // namespace Extrapolators
