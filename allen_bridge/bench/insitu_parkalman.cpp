// insitu_parkalman.cpp — Tier-2 in-situ reality check (CPU).
//
// Times the TWO production UT->T device functions, compiled verbatim from the
// read-only Allen headers, over the SAME gen-4 tracks used by the Tier-1 GPU
// micro-bench:
//   - extrapUTT          : the 19-param chart polynomial (state + Jacobian ders)
//   - pinn_v2_utt_state  : the locked PINN_V2 forward pass
//
// Rationale: the deployed m_use_nn_utt branch is a HYBRID — ExtrapolateUTT always
// runs extrapUTT (for the covariance Jacobian F/Q) and, when use_nn_utt is true,
// ADDS pinn_v2_utt_state for the state. So the in-situ "stock vs NN" delta is
// exactly the cost of one PINN forward pass, and the quantity that must agree with
// Tier-1 (within 2x, platform-invariant because it is a ratio of two device
// functions on the same hardware) is extrapUTT : PINN per track.
//
// The built Allen is a CPU target (TARGET_DEVICE=CPU), so this CPU timing is the
// faithful in-situ analogue; insitu_parkalman.md documents the GPU sequence-monitor
// recipe for a CUDA Allen build. Timing: std::chrono steady_clock, >=200 warm-up
// reps discarded, >=30 timed reps, median + IQR; outputs are summed into a sink so
// the optimiser cannot elide the work.
//
// Build: build_bench_host.sh-style flags (see build_insitu.sh).

#include "cuda_compat.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "KalmanParametrizations.cuh"           // verbatim Allen (chart struct + loader)
#include "PINN_V2_UTT.cuh"                       // verbatim Allen (NN forward pass)

using namespace ParKalmanFilter;

// extrapUTT touches only m_polarity of the real trackInfo.
namespace ParKalmanFilter {
  struct trackInfo {
    KalmanFloat m_polarity;
  };
}
#include "extraputt_snippet.inc"                 // verbatim Allen extrapUTT

static constexpr double QOP_CORPUS_TO_MEV = 1.0 / 299.792458;

static double median(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static double pct(std::vector<double> v, double p)
{
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, (size_t)(p / 100.0 * v.size()))];
}

int main(int argc, char** argv)
{
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s <param_dir> <tracks_f32.bin> <utt_meta.bin> "
                         "[warmup=200] [reps=50] [out.json]\n", argv[0]);
    return 1;
  }
  const std::string param_dir = argv[1];
  const std::string tracks_path = argv[2];
  const std::string meta_path = argv[3];
  const int warmup = argc > 4 ? std::atoi(argv[4]) : 200;
  const int reps = argc > 5 ? std::atoi(argv[5]) : 50;
  const std::string out_json = argc > 6 ? argv[6] : "";

  // --- load tracks (N x 7 float32) ---
  std::ifstream ft(tracks_path, std::ios::binary | std::ios::ate);
  if (!ft) { std::fprintf(stderr, "cannot open %s\n", tracks_path.c_str()); return 2; }
  size_t bytes = ft.tellg();
  ft.seekg(0);
  size_t N = bytes / (7 * sizeof(float));
  std::vector<float> T(N * 7);
  ft.read(reinterpret_cast<char*>(T.data()), N * 7 * sizeof(float));

  // --- load chart parametrization + META ---
  auto* params = new KalmanParametrizations();
  params->SetParameters(param_dir);
  if (!params->paramsLoaded) { std::fprintf(stderr, "params not loaded\n"); return 3; }
  std::vector<float> META(19);
  std::ifstream fm(meta_path, std::ios::binary);
  fm.read(reinterpret_cast<char*>(META.data()), 19 * sizeof(float));

  // split-out columns
  std::vector<float> X(N), Y(N), TX(N), TY(N), QOP(N), DZ(N);
  for (size_t i = 0; i < N; i++) {
    X[i] = T[i * 7 + 0]; Y[i] = T[i * 7 + 1]; TX[i] = T[i * 7 + 2];
    TY[i] = T[i * 7 + 3]; QOP[i] = T[i * 7 + 4]; DZ[i] = T[i * 7 + 6];
  }
  const float dz_nn = META[1] - META[0];  // fixed UT->T span used in production
  volatile double sink = 0;

  auto run_extraputt = [&]() {
    double s = 0;
    for (size_t i = 0; i < N; i++) {
      KalmanFloat x = X[i], y = Y[i], tx = TX[i], ty = TY[i];
      KalmanFloat qop_mev = (KalmanFloat)(QOP[i] * QOP_CORPUS_TO_MEV);
      trackInfo tI; tI.m_polarity = (KalmanFloat) -1.0;
      KalmanFloat dtx[4], dty[4], dq[4];
      extrapUTT(params, META.data(), x, y, tx, ty, qop_mev, dtx, dty, dq, tI);
      s += x + y + tx + ty;
    }
    return s;
  };
  auto run_pinn = [&]() {
    double s = 0;
    for (size_t i = 0; i < N; i++) {
      KalmanFloat xo, yo, txo, tyo;
      pinn_v2_utt_state(X[i], Y[i], TX[i], TY[i], QOP[i], dz_nn, xo, yo, txo, tyo);
      s += xo + yo + txo + tyo;
    }
    return s;
  };

  auto bench = [&](const char* name, auto fn) {
    for (int r = 0; r < warmup / 10 + 1; r++) sink += fn();  // warm caches/branch predictor
    std::vector<double> us_per_track;
    for (int r = 0; r < reps; r++) {
      auto t0 = std::chrono::steady_clock::now();
      sink += fn();
      auto t1 = std::chrono::steady_clock::now();
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      us_per_track.push_back(us / N);
    }
    double med = median(us_per_track), p25 = pct(us_per_track, 25), p75 = pct(us_per_track, 75);
    std::printf("%-14s  %.5f us/track (median)  IQR=[%.5f,%.5f]  rel_iqr=%.2f%%  %.3e tracks/s\n",
                name, med, p25, p75, 100.0 * (p75 - p25) / med, 1e6 / med);
    return std::vector<double>{med, p25, p75};
  };

  std::printf("Tier-2 in-situ (CPU), N=%zu tracks, warmup=%d reps=%d\n", N, warmup, reps);
  auto e = bench("extrapUTT", run_extraputt);
  auto p = bench("pinn_v2_utt", run_pinn);
  double ratio = e[0] / p[0];
  std::printf("extrapUTT/PINN (CPU, in-situ) = %.3f\n", ratio);
  std::printf("(hybrid in-situ overhead of use_nn_utt = +1 PINN pass = %.5f us/track)\n", p[0]);

  if (!out_json.empty()) {
    std::ofstream f(out_json);
    f << "{\n";
    f << "  \"tier\": 2,\n";
    f << "  \"platform\": \"CPU (built Allen TARGET_DEVICE=CPU); host g++ build of verbatim Allen device fns\",\n";
    f << "  \"timing\": \"std::chrono steady_clock, median+IQR\",\n";
    f << "  \"n_tracks\": " << N << ",\n  \"warmup\": " << warmup << ",\n  \"reps\": " << reps << ",\n";
    f << "  \"dtype\": \"fp32 (KalmanFloat=float)\",\n";
    f << "  \"pinn_dz_mm\": " << dz_nn << ",\n";
    f << "  \"extrapUTT_us_per_track\": {\"median\": " << e[0] << ", \"p25\": " << e[1] << ", \"p75\": " << e[2] << "},\n";
    f << "  \"pinn_us_per_track\": {\"median\": " << p[0] << ", \"p25\": " << p[1] << ", \"p75\": " << p[2] << "},\n";
    f << "  \"extrapUTT_div_PINN\": " << ratio << ",\n";
    f << "  \"hybrid_use_nn_utt_overhead_us_per_track\": " << p[0] << ",\n";
    f << "  \"note\": \"use_nn_utt is a hybrid: extrapUTT always runs (Jacobian F/Q); the NN is ADDED for the state. The cross-check vs Tier-1 is the extrapUTT:PINN ratio (platform-invariant).\"\n";
    f << "}\n";
    std::printf("wrote %s\n", out_json.c_str());
  }
  (void) sink;
  delete params;
  return 0;
}
