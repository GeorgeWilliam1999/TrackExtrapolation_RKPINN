// bench_kernels.cu — Tier-1 throughput micro-bench kernels (NVRTC translation unit).
//
// Three device paths, timed over the SAME N tracks drawn from the real gen-4
// (state, z0, dz) population:
//   rk_kernel        : Allen RungeKuttaExtrapolator<float, CashKarp> + v8r1 field-map
//                      texture lookups (6 tex3D per Cash-Karp step), crossing the
//                      given dz in fixed step_dz=100 mm sub-steps (production
//                      extrapolate_states step size). Per-track step count varies
//                      with |dz| -> real warp divergence and path-length cost.
//   extraputt_kernel : the production 19-param chart polynomial extrapUTT (UT->T).
//   pinn_kernel      : our locked PINN_V2_UTT forward pass (UT->T).
//
// Allen device code (RungeKuttaExtrapolator.cuh, ButcherTableau.cuh,
// ExtrapolatorCommon.cuh, PINN_V2_UTT.cuh, compute_state, extrapUTT) is included
// VERBATIM. Only NVRTC-safe shims (BackendCommon.h, FloatOperations.cuh,
// MagneticField.cuh, ParKalmanDefinitions.cuh) stand in for host-heavy headers;
// the device-executed instructions are unchanged. fp32 throughout (KalmanFloat=float).

#include "bench_rk.cuh"                // verbatim CashKarp RK + field (ExtrapolatorCommon/ButcherTableau)

#include "ParKalmanDefinitions.cuh"    // shim: KalmanFloat + chart dims
#include "bench_extraputt.cuh"         // verbatim compute_state + extrapUTT
#include "PINN_V2_UTT.cuh"             // REAL Allen (self-contained PINN forward pass)

// corpus qop (= 0.299792458 * q/p[GeV]) -> q/p[MeV] expected by extrapUTT.
#define QOP_CORPUS_TO_MEV (1.0f / 299.792458f)

extern "C" __global__ void rk_kernel(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ Z0,
  const float* __restrict__ DZ, const int N, const float step_dz, const int max_steps,
  const cudaTextureObject_t tex_Bx, const cudaTextureObject_t tex_By, const cudaTextureObject_t tex_Bz,
  const float minX, const float minY, const float minZ, const float invDx, const float invDy,
  const float invDz, float* __restrict__ OX, float* __restrict__ OY, float* __restrict__ OTX,
  float* __restrict__ OTY)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;

  MagneticField::Magfield field;
  field.tex_Bx = tex_Bx;
  field.tex_By = tex_By;
  field.tex_Bz = tex_Bz;
  field.minX = minX;
  field.minY = minY;
  field.minZ = minZ;
  field.invDx = invDx;
  field.invDy = invDy;
  field.invDz = invDz;

  // State.qop is the corpus qop directly (= input.qop()*c_light*eplus in Allen).
  Extrapolators::State s {X[i], Y[i], Z0[i], TX[i], TY[i], QOP[i]};
  const float target = Z0[i] + DZ[i];
  const float dir = (DZ[i] >= 0.f) ? 1.f : -1.f;

  for (int step = 0; step < max_steps; step++) {
    const float remaining = target - s.z;
    if (fabsf(remaining) < 0.5f) break;
    const float h = dir * fminf(step_dz, fabsf(remaining));
    Extrapolators::State::Error err;
    Extrapolators::RungeKuttaExtrapolator<float, ButcherTableau::CashKarp<float>>::propagate(s, err, h, field);
  }

  OX[i] = s.x;
  OY[i] = s.y;
  OTX[i] = s.tx;
  OTY[i] = s.ty;
}

extern "C" __global__ void extraputt_kernel(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const int N,
  const ParKalmanFilter::KalmanParametrizations* __restrict__ kp, const float* __restrict__ META,
  const float polarity, float* __restrict__ OX, float* __restrict__ OY, float* __restrict__ OTX,
  float* __restrict__ OTY)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  using namespace ParKalmanFilter;

  KalmanFloat x = X[i], y = Y[i], tx = TX[i], ty = TY[i];
  const KalmanFloat qop_mev = QOP[i] * QOP_CORPUS_TO_MEV;
  trackInfo tI;
  tI.m_polarity = polarity;  // convention: extrapUTT pairs with m_polarity = -1
  KalmanFloat der_tx[4], der_ty[4], der_qop[4];

  extrapUTT(kp, META, x, y, tx, ty, qop_mev, der_tx, der_ty, der_qop, tI);

  OX[i] = x;
  OY[i] = y;
  OTX[i] = tx;
  OTY[i] = ty;
}

extern "C" __global__ void pinn_kernel(
  const float* __restrict__ X, const float* __restrict__ Y, const float* __restrict__ TX,
  const float* __restrict__ TY, const float* __restrict__ QOP, const float* __restrict__ DZ,
  const int N, float* __restrict__ OX, float* __restrict__ OY, float* __restrict__ OTX,
  float* __restrict__ OTY)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N) return;
  using namespace ParKalmanFilter;

  KalmanFloat xo, yo, txo, tyo;
  // qop is the corpus qop (training convention); dz is the per-track span.
  pinn_v2_utt_state(X[i], Y[i], TX[i], TY[i], QOP[i], DZ[i], xo, yo, txo, tyo);

  OX[i] = xo;
  OY[i] = yo;
  OTX[i] = txo;
  OTY[i] = tyo;
}

// ABI / upload self-test: read back chart scalars so the host can confirm the raw
// struct blit landed on identical member offsets (expect ZINI=2665, ZFIN=7826,
// Nbinx=60).
extern "C" __global__ void read_kp_scalars(
  const ParKalmanFilter::KalmanParametrizations* __restrict__ kp, float* __restrict__ out)
{
  if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
  out[0] = kp->ZINI;
  out[1] = kp->ZFIN;
  out[2] = (float) kp->Nbinx;
  out[3] = (float) kp->Nbiny;
  out[4] = kp->PMIN;
  out[5] = (float) kp->DEGX2;
}
