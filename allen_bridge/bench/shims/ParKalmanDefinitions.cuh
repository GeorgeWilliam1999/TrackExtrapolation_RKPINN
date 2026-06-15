// shims/ParKalmanDefinitions.cuh — NVRTC-safe subset of Allen's
// device/event_model/kalman/include/ParKalmanDefinitions.cuh.
//
// The real header pulls in BackendCommon.h / States.cuh / ParKalmanMath.cuh
// (host-heavy, not NVRTC-compilable). The device code we time (PINN_V2_UTT.cuh
// and the verbatim extrapUTT in bench_extraputt.cuh) needs only the KalmanFloat
// alias and the chart dimensioning constants — reproduced here verbatim with the
// production values (KALMAN_DOUBLE_PRECISION is OFF in the deployed build, so
// KalmanFloat = float; all three methods are timed in fp32).
#pragma once

namespace ParKalmanFilter {
  using KalmanFloat = float;

  constexpr int nBinXMax = 60;
  constexpr int nBinYMax = 50;

  constexpr int DEGx1 = 7;
  constexpr int DEGx2 = 9;
  constexpr int DEGy1 = 5;
  constexpr int DEGy2 = 7;
} // namespace ParKalmanFilter
