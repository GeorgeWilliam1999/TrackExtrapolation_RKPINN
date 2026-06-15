// bench_extraputt.cuh — NVRTC-safe assembly of the production UT->T polynomial
// extrapolator (extrapUTT) and the chart parametrization it contracts.
//
// The two device functions (KalmanParametrizations::compute_state<> and
// extrapUTT) are #included VERBATIM from the read-only Allen headers via
// generate_snippets.sh; only the struct's *data layout* and a minimal trackInfo
// are reproduced here, because the real KalmanParametrizations.cuh pulls in
// host-only <fstream> machinery NVRTC cannot compile.
//
// The data-member declarations below are byte-for-byte ABI-identical to the real
// struct: the host dumper (dump_utt_params.cpp) writes the loaded struct raw, and
// the micro-bench uploads those bytes straight into a device buffer reinterpreted
// as this struct. Layout is verified at runtime (ZINI/Nbinx offsets + values).
#pragma once

#include "ParKalmanDefinitions.cuh"  // KalmanFloat, nBin*Max, DEG* (production values)

namespace ParKalmanFilter {

  // extrapUTT touches only m_polarity of the real (much larger) trackInfo.
  struct trackInfo {
    KalmanFloat m_polarity;
  };

  struct KalmanParametrizations {
    // --- data members: identical declaration order/types to the real struct ---
    float x00[nBinXMax * nBinYMax * DEGx2];
    float x10[nBinXMax * nBinYMax * DEGx1];
    float x01[nBinXMax * nBinYMax * DEGx1];
    float tx00[nBinXMax * nBinYMax * DEGx2];
    float tx10[nBinXMax * nBinYMax * DEGx1];
    float tx01[nBinXMax * nBinYMax * DEGx1];

    float y00[nBinXMax * nBinYMax * DEGy2];
    float y10[nBinXMax * nBinYMax * DEGy1];
    float y01[nBinXMax * nBinYMax * DEGy1];
    float ty00[nBinXMax * nBinYMax * DEGy2];
    float ty10[nBinXMax * nBinYMax * DEGy1];
    float ty01[nBinXMax * nBinYMax * DEGy1];

    float ZINI, ZFIN;
    float PMIN;
    float BENDX, BENDX_X2, BENDX_Y2, BENDY_XY;
    float Txmax, Tymax, XFmax, Xmax, Ymax;
    float Dtxy;
    float step;

    int Nbinx, Nbiny;
    int XGridOption, YGridOption;
    int DEGX1, DEGX2, DEGY1, DEGY2;

    bool paramsLoaded;

    // --- verbatim Allen device methods ---
    __host__ __device__ inline int UttOffset(const int& x_bin, const int& y_bin, const int& deg) const
    { // simple row-major access pattern
      return deg * (nBinYMax * x_bin + y_bin);
    }

#include "compute_state_snippet.inc"  // KalmanParametrizations::compute_state<DEG0,DEG1>
  };

#include "extraputt_snippet.inc"  // __device__ void extrapUTT(...)

} // namespace ParKalmanFilter
