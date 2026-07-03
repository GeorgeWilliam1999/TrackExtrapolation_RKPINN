// vf_engine.cpp — fast fp64 RK4 track propagation + Jacobian for the
// vertex-fit corpus (PLAN 3915d544-b9d9-81b7, engine v1.1).
//
// Replaces the torch forward-mode Jacobian path of generate_vertexfit_v1.py
// (~0.7 s/row/core) with a C++ implementation of the SAME discrete map:
//   - identical EOM (Allen gamma(); kappa = 1e-3 * qop, B in Tesla),
//   - identical trilinear field interpolation (trunc-toward-zero indexing,
//     clip to [0, N-2], fractional clip to [0,1] — matching FieldV8R1 and the
//     torch mirror bit-for-bit in exact arithmetic),
//   - identical RK4 stepping (fixed 5 mm, remainder step, same loop bounds).
//
// Jacobian: exact tangent of the DISCRETE RK4 map (what autodiff computes),
// propagated alongside the state:
//   K1 = A(S1,z)      * T          S1 = S
//   K2 = A(S2,z+h/2)  * (T + h/2 K1)   S2 = S + h/2 k1
//   K3 = A(S3,z+h/2)  * (T + h/2 K2)   S3 = S + h/2 k2
//   K4 = A(S4,z+h)    * (T + h  K3)    S4 = S + h  k3
//   T <- T + h/6 (K1 + 2K2 + 2K3 + K4)
// with A = df/dS (5x5), including the analytic in-cell gradient of the
// trilinear interpolation (dB/dx, dB/dy). Outside the grid the interpolation
// clips, so the corresponding gradient is masked to zero — the same
// derivative the torch clamp produces. The A-matrix mirrors the structure of
// Allen's RungeKuttaExtrapolator.cuh incrementJacobian (Nystrom) adapted to
// this first-order form:
//   f2 = k N P,  P = tx ty Bx - (1+tx^2) By + ty Bz
//   f3 = k N Q,  Q = (1+ty^2) Bx - tx ty By - tx Bz      (k = KAPPA qop)
//   df2/dtx = k( tx/N P + N( ty Bx - 2 tx By) )   df2/dty = k( ty/N P + N( tx Bx + Bz) )
//   df3/dtx = k( tx/N Q + N(-ty By -   Bz) )     df3/dty = k( ty/N Q + N(2 ty Bx - tx By) )
//   df{2,3}/d{x,y} via dB/d{x,y};  df{2,3}/dqop = KAPPA N {P,Q}
// Correctness is enforced by the generator's gate battery (G0c/G1c/G2c/G3c/
// G4c/G5 in generate_vertexfit_v1.py --selftest): agreement with the numpy
// engine, the exact uniform-field closed forms (state AND full 5x5 J), the
// torch autodiff J, and central finite differences.
//
// Build (done automatically by the generator if the .so is missing/stale):
//   g++ -O3 -march=native -fopenmp -shared -fPIC vf_engine.cpp -o vf_engine.so
//
// C ABI:
//   void*  vf_field_load(const char* path)   — v8r1 .bin (Allen magfield fmt)
//   void*  vf_field_const(double Bx, By, Bz) — uniform field (gates)
//   void   vf_field_free(void*)
//   double vf_field_peak_by(void*)           — raw peak |By| (sanity)
//   int    vf_propagate(void* fld, const double* S0 (n*5),
//                       const double* z0 (n), const double* z1 (n), int n,
//                       double step, double* Y (n*5), double* J (n*25|NULL))
// Returns 0 on success. OpenMP-parallel over rows.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const double KAPPA = 1.0e-3;

struct Field {
  bool is_const;
  double B0[3];
  // grid
  int64_t N[3];
  double invD[3], mn[3], scale;
  double *Bx, *By, *Bz; // [(iz*Ny + iy)*Nx + ix]
};

extern "C" void* vf_field_const(double bx, double by, double bz) {
  Field* f = new Field();
  f->is_const = true;
  f->B0[0] = bx; f->B0[1] = by; f->B0[2] = bz;
  f->Bx = f->By = f->Bz = nullptr;
  return f;
}

extern "C" void* vf_field_load(const char* path) {
  FILE* fp = std::fopen(path, "rb");
  if (!fp) return nullptr;
  float head[12];
  if (std::fread(head, sizeof(float), 12, fp) != 12) { std::fclose(fp); return nullptr; }
  Field* f = new Field();
  f->is_const = false;
  for (int i = 0; i < 3; ++i) f->invD[i] = (double)head[i];
  int32_t Ni[4];
  std::memcpy(Ni, head + 4, 4 * sizeof(int32_t));
  for (int i = 0; i < 3; ++i) f->N[i] = (int64_t)Ni[i];
  for (int i = 0; i < 3; ++i) f->mn[i] = (double)head[8 + i];
  const int64_t ncell = f->N[0] * f->N[1] * f->N[2];
  float* raw = (float*)std::malloc(sizeof(float) * (size_t)ncell * 4);
  if (std::fread(raw, sizeof(float), (size_t)ncell * 4, fp) != (size_t)ncell * 4) {
    std::free(raw); std::fclose(fp); delete f; return nullptr;
  }
  std::fclose(fp);
  f->Bx = new double[ncell]; f->By = new double[ncell]; f->Bz = new double[ncell];
  double peak = 0.0;
  for (int64_t i = 0; i < ncell; ++i) {
    f->Bx[i] = raw[4 * i + 0];
    f->By[i] = raw[4 * i + 1];
    f->Bz[i] = raw[4 * i + 2];
    const double a = std::fabs(f->By[i]);
    if (a > peak) peak = a;
  }
  std::free(raw);
  // unit detection — same rule as FieldV8R1 (Tesla vs Gaudi 1T=1e-3)
  if (peak > 0.1 && peak < 10.0) f->scale = 1.0;
  else if (peak > 1e-4 && peak < 1e-2) f->scale = 1.0 / 1e-3;
  else { delete[] f->Bx; delete[] f->By; delete[] f->Bz; delete f; return nullptr; }
  f->B0[1] = peak; // stash raw peak for sanity
  return f;
}

extern "C" double vf_field_peak_by(void* vf) { return ((Field*)vf)->B0[1]; }

extern "C" void vf_field_free(void* vf) {
  Field* f = (Field*)vf;
  if (!f->is_const) { delete[] f->Bx; delete[] f->By; delete[] f->Bz; }
  delete f;
}

// Trilinear field + in-cell gradient (dB/dx, dB/dy). Matches FieldV8R1:
// ix = trunc(fx) clipped to [0,N-2]; t = clip(fx-ix, 0, 1). Gradient masked
// to zero on any axis whose fraction was clipped (point outside the grid).
static inline void field_eval(const Field* f, double x, double y, double z,
                              double B[3], double dBdx[3], double dBdy[3],
                              bool want_grad) {
  if (f->is_const) {
    B[0] = f->B0[0]; B[1] = f->B0[1]; B[2] = f->B0[2];
    if (want_grad) for (int c = 0; c < 3; ++c) { dBdx[c] = 0.0; dBdy[c] = 0.0; }
    return;
  }
  const double fx = (x - f->mn[0]) * f->invD[0];
  const double fy = (y - f->mn[1]) * f->invD[1];
  const double fz = (z - f->mn[2]) * f->invD[2];
  int64_t ix = (int64_t)fx, iy = (int64_t)fy, iz = (int64_t)fz; // trunc toward 0
  if (ix < 0) ix = 0; if (ix > f->N[0] - 2) ix = f->N[0] - 2;
  if (iy < 0) iy = 0; if (iy > f->N[1] - 2) iy = f->N[1] - 2;
  if (iz < 0) iz = 0; if (iz > f->N[2] - 2) iz = f->N[2] - 2;
  double tx = fx - ix, ty = fy - iy, tz = fz - iz;
  const bool okx = (tx >= 0.0 && tx <= 1.0), oky = (ty >= 0.0 && ty <= 1.0);
  if (tx < 0) tx = 0; if (tx > 1) tx = 1;
  if (ty < 0) ty = 0; if (ty > 1) ty = 1;
  if (tz < 0) tz = 0; if (tz > 1) tz = 1;
  const int64_t nx = f->N[0], ny = f->N[1];
  const int64_t i000 = (iz * ny + iy) * nx + ix;
  const int64_t i010 = i000 + nx, i001 = i000 + ny * nx, i011 = i001 + nx;
  const double* G[3] = { f->Bx, f->By, f->Bz };
  for (int c = 0; c < 3; ++c) {
    const double* g = G[c];
    const double c000 = g[i000],     c100 = g[i000 + 1];
    const double c010 = g[i010],     c110 = g[i010 + 1];
    const double c001 = g[i001],     c101 = g[i001 + 1];
    const double c011 = g[i011],     c111 = g[i011 + 1];
    const double c00 = c000 * (1 - tx) + c100 * tx;
    const double c10 = c010 * (1 - tx) + c110 * tx;
    const double c01 = c001 * (1 - tx) + c101 * tx;
    const double c11 = c011 * (1 - tx) + c111 * tx;
    const double c0 = c00 * (1 - ty) + c10 * ty;
    const double c1 = c01 * (1 - ty) + c11 * ty;
    B[c] = (c0 * (1 - tz) + c1 * tz) * f->scale;
    if (want_grad) {
      // d/dtx: bilinear interp (ty,tz) of x-differences; then * invD * scale
      const double dx00 = c100 - c000, dx10 = c110 - c010;
      const double dx01 = c101 - c001, dx11 = c111 - c011;
      const double dx0 = dx00 * (1 - ty) + dx10 * ty;
      const double dx1 = dx01 * (1 - ty) + dx11 * ty;
      dBdx[c] = okx ? (dx0 * (1 - tz) + dx1 * tz) * f->invD[0] * f->scale : 0.0;
      const double dy0 = c10 - c00, dy1 = c11 - c01;
      dBdy[c] = oky ? (dy0 * (1 - tz) + dy1 * tz) * f->invD[1] * f->scale : 0.0;
    }
  }
}

// EOM derivative f(S,z) and (optionally) A = df/dS.
static inline void eom(const Field* fld, const double S[5], double z,
                       double dS[5], double A[25], bool want_A) {
  const double x = S[0], y = S[1], tx = S[2], ty = S[3], qop = S[4];
  double B[3], dBdx[3], dBdy[3];
  field_eval(fld, x, y, z, B, dBdx, dBdy, want_A);
  const double k = KAPPA * qop;
  const double N2 = 1.0 + tx * tx + ty * ty;
  const double N = std::sqrt(N2);
  const double P = tx * ty * B[0] - (1.0 + tx * tx) * B[1] + ty * B[2];
  const double Q = (1.0 + ty * ty) * B[0] - tx * ty * B[1] - tx * B[2];
  dS[0] = tx; dS[1] = ty; dS[2] = k * N * P; dS[3] = k * N * Q; dS[4] = 0.0;
  if (!want_A) return;
  std::memset(A, 0, 25 * sizeof(double));
  A[0 * 5 + 2] = 1.0;                       // dx'/dtx
  A[1 * 5 + 3] = 1.0;                       // dy'/dty
  const double Px = tx * ty * dBdx[0] - (1.0 + tx * tx) * dBdx[1] + ty * dBdx[2];
  const double Py = tx * ty * dBdy[0] - (1.0 + tx * tx) * dBdy[1] + ty * dBdy[2];
  const double Qx = (1.0 + ty * ty) * dBdx[0] - tx * ty * dBdx[1] - tx * dBdx[2];
  const double Qy = (1.0 + ty * ty) * dBdy[0] - tx * ty * dBdy[1] - tx * dBdy[2];
  A[2 * 5 + 0] = k * N * Px;
  A[2 * 5 + 1] = k * N * Py;
  A[2 * 5 + 2] = k * ((tx / N) * P + N * (ty * B[0] - 2.0 * tx * B[1]));
  A[2 * 5 + 3] = k * ((ty / N) * P + N * (tx * B[0] + B[2]));
  A[2 * 5 + 4] = KAPPA * N * P;
  A[3 * 5 + 0] = k * N * Qx;
  A[3 * 5 + 1] = k * N * Qy;
  A[3 * 5 + 2] = k * ((tx / N) * Q + N * (-ty * B[1] - B[2]));
  A[3 * 5 + 3] = k * ((ty / N) * Q + N * (2.0 * ty * B[0] - tx * B[1]));
  A[3 * 5 + 4] = KAPPA * N * Q;
}

// K = A * T  (5x5 each). Exploits A's sparsity (rows 0,1 pick rows 2,3 of T;
// row 4 is zero; rows 2,3 dense).
static inline void amul(const double A[25], const double T[25], double K[25]) {
  for (int j = 0; j < 5; ++j) {
    K[0 * 5 + j] = T[2 * 5 + j];
    K[1 * 5 + j] = T[3 * 5 + j];
    K[4 * 5 + j] = 0.0;
  }
  for (int r = 2; r <= 3; ++r)
    for (int j = 0; j < 5; ++j) {
      double s = 0.0;
      for (int c = 0; c < 5; ++c) s += A[r * 5 + c] * T[c * 5 + j];
      K[r * 5 + j] = s;
    }
}

// One RK4 step of size h at z, updating S (and T if J requested).
static inline void rk4_step(const Field* fld, double S[5], double T[25],
                            double z, double h, bool want_J) {
  double k1[5], k2[5], k3[5], k4[5], Sw[5];
  double A1[25], A2[25], A3[25], A4[25];
  double K1[25], K2[25], K3[25], K4[25], Tw[25];

  eom(fld, S, z, k1, A1, want_J);
  for (int i = 0; i < 5; ++i) Sw[i] = S[i] + 0.5 * h * k1[i];
  eom(fld, Sw, z + 0.5 * h, k2, A2, want_J);
  for (int i = 0; i < 5; ++i) Sw[i] = S[i] + 0.5 * h * k2[i];
  eom(fld, Sw, z + 0.5 * h, k3, A3, want_J);
  for (int i = 0; i < 5; ++i) Sw[i] = S[i] + h * k3[i];
  eom(fld, Sw, z + h, k4, A4, want_J);

  if (want_J) {
    amul(A1, T, K1);
    for (int i = 0; i < 25; ++i) Tw[i] = T[i] + 0.5 * h * K1[i];
    amul(A2, Tw, K2);
    for (int i = 0; i < 25; ++i) Tw[i] = T[i] + 0.5 * h * K2[i];
    amul(A3, Tw, K3);
    for (int i = 0; i < 25; ++i) Tw[i] = T[i] + h * K3[i];
    amul(A4, Tw, K4);
    for (int i = 0; i < 25; ++i)
      T[i] += (h / 6.0) * (K1[i] + 2.0 * K2[i] + 2.0 * K3[i] + K4[i]);
  }
  for (int i = 0; i < 5; ++i)
    S[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
}

extern "C" int vf_propagate(void* vfld, const double* S0, const double* z0s,
                            const double* z1s, int n, double step,
                            double* Y, double* J) {
  const Field* fld = (const Field*)vfld;
  const bool want_J = (J != nullptr);
#pragma omp parallel for schedule(dynamic, 16)
  for (int i = 0; i < n; ++i) {
    double S[5], T[25];
    for (int c = 0; c < 5; ++c) S[c] = S0[i * 5 + c];
    if (want_J) {
      std::memset(T, 0, sizeof(T));
      for (int c = 0; c < 5; ++c) T[c * 5 + c] = 1.0;
    }
    const double z0 = z0s[i], z1 = z1s[i];
    double z = z0;
    if (z1 != z0) {
      const double h = (z1 > z0) ? step : -step;
      const double sgn = (h > 0) ? 1.0 : -1.0;
      // same loop bounds as the numpy/torch engines
      while ((z1 - z) * sgn > std::fabs(h)) {
        rk4_step(fld, S, T, z, h, want_J);
        z += h;
      }
      const double r = z1 - z;
      if (std::fabs(r) > 1e-12) rk4_step(fld, S, T, z, r, want_J);
    }
    for (int c = 0; c < 5; ++c) Y[i * 5 + c] = S[c];
    if (want_J) for (int c = 0; c < 25; ++c) J[i * 25 + c] = T[c];
  }
  return 0;
}
