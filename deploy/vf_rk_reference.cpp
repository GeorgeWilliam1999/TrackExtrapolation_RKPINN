// Same-machine cost reference for the vertex-fit speed gate: the production
// extrapolation SHAPE — Cash-Karp RK45 with fixed 100 mm steps and a real
// v8r1 trilinear field-map lookup per stage (6 per step) — over the same leg
// population as the NN kernel benchmark.
//
// Mirrors allen_bridge/bench/bench_kernels.cu::rk_kernel (fixed step_dz =
// 100 mm, CashKarp 6 stages, stop within 0.5 mm of the target), with the
// standard Cash-Karp tableau and the gen-4 equation of motion
// (datagen/vf_engine.cpp verbatim semantics; corpus qop convention,
// kappa = 1e-3 * qop). This is a COST reference for like-for-like timing on
// this machine, not a byte-parity port of Allen; the per-call work profile
// (~6 lookups / 100 mm, tens of lookups per vertex-fit leg) matches the
// production profiling that attributes ~90 % of TrackMasterExtrapolator's
// 8.35 us/call to ~69 field lookups.
//
// Field blob (written by gates/run_vf_cpp_gate.py from core/field_v8r1.py,
// grids PRE-multiplied by the map scale):
//   magic "VFF1", int32 N[3] (Nx, Ny, Nz), float min[3], invD[3],
//   Bx[Nz*Ny*Nx], By[...], Bz[...]   (all little-endian fp32)
//
// API: rkref_load(path); rkref_propagate(in7, out4);
//      rkref_bench(n, in7, checksum) -> total ns.
//
// Build: g++ -O3 -march=native -ffast-math -shared -fPIC -o librkref.so
//        vf_rk_reference.cpp

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

namespace {

constexpr float KAPPA = 1.0e-3f;
constexpr float STEP_DZ = 100.0f;   // bench_kernels.cu rk_kernel step size
constexpr int MAX_STEPS = 128;

int NX = 0, NY = 0, NZ = 0;
float mn[3], invD[3];
std::vector<float> Bx, By, Bz;

inline void field_eval(float x, float y, float z, float B[3]) {
  // Trilinear with the FieldV8R1 clamp semantics (truncate-toward-zero cell
  // index clamped to [0, N-2]; fractions clamped to [0, 1]).
  float fx = (x - mn[0]) * invD[0];
  float fy = (y - mn[1]) * invD[1];
  float fz = (z - mn[2]) * invD[2];
  int ix = int(fx), iy = int(fy), iz = int(fz);
  ix = ix < 0 ? 0 : (ix > NX - 2 ? NX - 2 : ix);
  iy = iy < 0 ? 0 : (iy > NY - 2 ? NY - 2 : iy);
  iz = iz < 0 ? 0 : (iz > NZ - 2 ? NZ - 2 : iz);
  float tx = fx - ix, ty = fy - iy, tz = fz - iz;
  tx = tx < 0 ? 0 : (tx > 1 ? 1 : tx);
  ty = ty < 0 ? 0 : (ty > 1 ? 1 : ty);
  tz = tz < 0 ? 0 : (tz > 1 ? 1 : tz);
  const size_t sxy = size_t(NX) * NY;
  const size_t base = size_t(iz) * sxy + size_t(iy) * NX + ix;
  const float* G[3] = {Bx.data(), By.data(), Bz.data()};
  for (int c = 0; c < 3; ++c) {
    const float* g = G[c];
    const float c000 = g[base], c100 = g[base + 1];
    const float c010 = g[base + NX], c110 = g[base + NX + 1];
    const float c001 = g[base + sxy], c101 = g[base + sxy + 1];
    const float c011 = g[base + sxy + NX], c111 = g[base + sxy + NX + 1];
    const float c00 = c000 * (1 - tx) + c100 * tx;
    const float c10 = c010 * (1 - tx) + c110 * tx;
    const float c01 = c001 * (1 - tx) + c101 * tx;
    const float c11 = c011 * (1 - tx) + c111 * tx;
    const float c0 = c00 * (1 - ty) + c10 * ty;
    const float c1 = c01 * (1 - ty) + c11 * ty;
    B[c] = c0 * (1 - tz) + c1 * tz;
  }
}

struct S5 {
  float x, y, tx, ty, qop;
};

inline void deriv(const S5& s, float z, float k[4]) {
  float B[3];
  field_eval(s.x, s.y, z, B);
  const float kq = KAPPA * s.qop;
  const float N = std::sqrt(1.0f + s.tx * s.tx + s.ty * s.ty);
  k[0] = s.tx;
  k[1] = s.ty;
  k[2] = kq * N * (s.tx * s.ty * B[0] - (1.0f + s.tx * s.tx) * B[1] + s.ty * B[2]);
  k[3] = kq * N * ((1.0f + s.ty * s.ty) * B[0] - s.tx * s.ty * B[1] - s.tx * B[2]);
}

// Standard Cash-Karp tableau (6 stages)
constexpr float A[6][5] = {
    {0, 0, 0, 0, 0},
    {1.f / 5, 0, 0, 0, 0},
    {3.f / 40, 9.f / 40, 0, 0, 0},
    {3.f / 10, -9.f / 10, 6.f / 5, 0, 0},
    {-11.f / 54, 5.f / 2, -70.f / 27, 35.f / 27, 0},
    {1631.f / 55296, 175.f / 512, 575.f / 13824, 44275.f / 110592, 253.f / 4096}};
constexpr float BW[6] = {37.f / 378, 0, 250.f / 621, 125.f / 594, 0, 512.f / 1771};
constexpr float C[6] = {0, 1.f / 5, 3.f / 10, 3.f / 5, 1.f, 7.f / 8};

inline void ck_step(S5& s, float z, float h) {
  float k[6][4];
  for (int st = 0; st < 6; ++st) {
    S5 t = s;
    for (int i = 0; i < st; ++i) {
      t.x += h * A[st][i] * k[i][0];
      t.y += h * A[st][i] * k[i][1];
      t.tx += h * A[st][i] * k[i][2];
      t.ty += h * A[st][i] * k[i][3];
    }
    deriv(t, z + C[st] * h, k[st]);
  }
  for (int st = 0; st < 6; ++st) {
    s.x += h * BW[st] * k[st][0];
    s.y += h * BW[st] * k[st][1];
    s.tx += h * BW[st] * k[st][2];
    s.ty += h * BW[st] * k[st][3];
  }
}

}  // namespace

extern "C" {

int rkref_load(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return 1;
  char magic[4];
  int32_t n[3];
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "VFF1", 4) != 0 ||
      std::fread(n, 4, 3, f) != 3) {
    std::fclose(f);
    return 2;
  }
  NX = n[0];
  NY = n[1];
  NZ = n[2];
  const size_t tot = size_t(NX) * NY * NZ;
  bool ok = std::fread(mn, 4, 3, f) == 3 && std::fread(invD, 4, 3, f) == 3;
  Bx.resize(tot);
  By.resize(tot);
  Bz.resize(tot);
  ok = ok && std::fread(Bx.data(), 4, tot, f) == tot &&
       std::fread(By.data(), 4, tot, f) == tot &&
       std::fread(Bz.data(), 4, tot, f) == tot;
  std::fclose(f);
  return ok ? 0 : 3;
}

// in7 = [x, y, tx, ty, qop, z0, dz] (corpus row); out4 = [x, y, tx, ty] at z1.
void rkref_propagate(const double* in7, double* out4) {
  S5 s{float(in7[0]), float(in7[1]), float(in7[2]), float(in7[3]), float(in7[4])};
  float z = float(in7[5]);
  const float target = z + float(in7[6]);
  const float dir = in7[6] >= 0 ? 1.0f : -1.0f;
  for (int step = 0; step < MAX_STEPS; ++step) {
    const float remaining = target - z;
    if (std::fabs(remaining) < 0.5f) break;
    const float h = dir * std::fmin(STEP_DZ, std::fabs(remaining));
    ck_step(s, z, h);
    z += h;
  }
  out4[0] = s.x;
  out4[1] = s.y;
  out4[2] = s.tx;
  out4[3] = s.ty;
}

double rkref_bench(long n, const double* in7, double* checksum) {
  double out[4];
  double sink = 0.0;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (long i = 0; i < n; ++i) {
    rkref_propagate(in7 + 7 * i, out);
    sink += out[0];
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (checksum) *checksum = sink;
  return double(t1.tv_sec - t0.tv_sec) * 1e9 + double(t1.tv_nsec - t0.tv_nsec);
}

}  // extern "C"
