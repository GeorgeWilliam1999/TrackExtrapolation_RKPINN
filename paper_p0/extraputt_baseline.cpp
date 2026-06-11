// extraputt_baseline.cpp — P0.1 incumbent-baseline bake-off driver.
//
// Runs the PRODUCTION UT->T polynomial extrapolation (extrapUTT, extracted
// VERBATIM from ParKalmanMethods.cuh at build time -> extraputt_snippet.inc)
// on a CSV of states at z=ZINI, writing predicted states at z=ZFIN.
//
// The parametrization is loaded with the production loader
// (KalmanParametrizations::SetParameters -> ParametrizedKalmanFit/25v0/
//  params_UTT_v0.tab); dev_UTT_META is filled from the loaded struct in the
// order documented inside extrapUTT (indices 00 ZINI ... 18 DEGY2).
//
// qop conventions: the input CSV carries the training-corpus qop
// (qop_corpus = 299.792458 * q/p[1/MeV]); ParKalman states carry plain
// q/p[1/MeV] (PMIN=1500 MeV in the tab makes fq = qop*PMIN dimensionless).
// We convert: qop_MeV = qop_corpus / 299.792458.
//
// Usage: ./extraputt_baseline <ParamFiles/data dir> <in.csv> <out.csv> <polarity>
//
// Note: we call extrapUTT directly (not the ExtrapolateUTT wrapper), i.e.
// without the dev_pars[18]/[19] qopHere energy-loss-style correction — the
// RK ground truth is field-only, so the comparison stays like-for-like.

#include "cuda_compat.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "KalmanParametrizations.cuh"

using namespace ParKalmanFilter;

// Minimal trackInfo: extrapUTT touches only m_polarity. (The real struct in
// ParKalmanMethods.cuh carries Jacobians/chi2s irrelevant to this driver.)
namespace ParKalmanFilter {
  struct trackInfo {
    KalmanFloat m_polarity;
  };
} // namespace ParKalmanFilter

// The production polynomial, extracted verbatim at build time:
#include "extraputt_snippet.inc"

static constexpr double QOP_CORPUS_TO_MEV = 1.0 / 299.792458;

int main(int argc, char** argv)
{
  if (argc != 5) {
    std::cerr << "usage: " << argv[0] << " <param_data_dir> <in.csv> <out.csv> <polarity(+1|-1)>\n";
    return 1;
  }
  const std::string param_dir = argv[1];
  const std::string in_csv = argv[2];
  const std::string out_csv = argv[3];
  const float polarity = std::strtof(argv[4], nullptr);

  // --- load the production parametrization (heap: struct is ~1 MB) ---
  auto* params = new KalmanParametrizations();
  params->SetParameters(param_dir);
  if (!params->paramsLoaded) {
    std::cerr << "ERROR: parametrization not loaded from " << param_dir << "\n";
    return 2;
  }
  std::printf("loaded params_UTT_v0.tab: ZINI=%.2f ZFIN=%.2f PMIN=%.1f Nbin=%dx%d DEG=(%d,%d,%d,%d)\n",
              params->ZINI, params->ZFIN, params->PMIN, params->Nbinx, params->Nbiny,
              params->DEGX1, params->DEGX2, params->DEGY1, params->DEGY2);

  // --- dev_UTT_META in the order documented in extrapUTT ---
  float META[19] = {
    params->ZINI,                      // 00
    params->ZFIN,                      // 01
    params->PMIN,                      // 02
    params->BENDX,                     // 03
    params->BENDX_X2,                  // 04
    params->BENDX_Y2,                  // 05
    params->BENDY_XY,                  // 06
    params->Txmax,                     // 07
    params->Tymax,                     // 08
    params->XFmax,                     // 09
    params->Dtxy,                      // 10
    (float) params->Nbinx,             // 11
    (float) params->Nbiny,             // 12
    (float) params->XGridOption,       // 13
    (float) params->YGridOption,       // 14
    (float) params->DEGX1,             // 15
    (float) params->DEGX2,             // 16
    (float) params->DEGY1,             // 17
    (float) params->DEGY2              // 18
  };

  std::ifstream fin(in_csv);
  if (!fin) { std::cerr << "ERROR: cannot open " << in_csv << "\n"; return 3; }
  std::ofstream fout(out_csv);
  fout << "x,y,tx,ty\n";

  std::string line;
  std::getline(fin, line); // header
  size_t n = 0;
  while (std::getline(fin, line)) {
    if (line.empty()) continue;
    std::stringstream ss(line);
    double v[5];
    char comma;
    for (int i = 0; i < 5; i++) { ss >> v[i]; if (i < 4) ss >> comma; }

    KalmanFloat x = (KalmanFloat) v[0];
    KalmanFloat y = (KalmanFloat) v[1];
    KalmanFloat tx = (KalmanFloat) v[2];
    KalmanFloat ty = (KalmanFloat) v[3];
    const KalmanFloat qop_mev = (KalmanFloat)(v[4] * QOP_CORPUS_TO_MEV);

    trackInfo tI;
    tI.m_polarity = (KalmanFloat) polarity;
    KalmanFloat der_tx[4], der_ty[4], der_qop[4];

    extrapUTT(params, META, x, y, tx, ty, qop_mev, der_tx, der_ty, der_qop, tI);

    fout << x << "," << y << "," << tx << "," << ty << "\n";
    n++;
  }
  std::printf("wrote %zu predictions -> %s (polarity %+.0f)\n", n, out_csv.c_str(), polarity);
  delete params;
  return 0;
}
