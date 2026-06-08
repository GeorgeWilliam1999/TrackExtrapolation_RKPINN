// Standalone numerical harness for NRKExtrapolator invariants.
// Compiles with plain g++ -- no Allen, no CUDA, no Catch2.
// Reproduces the five invariants from
// test/unit_tests/generic/src/TestNRKExtrapolator.cu using the exact same
// math from device/kalman/ParKalman/include/NRKExtrapolator.cuh.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

// ---- shims for CUDA-only types -------------------------------------------
struct float2 {
  float x, y;
};
struct float3 {
  float x, y, z;
};
inline float2 make_float2(float a, float b) { return {a, b}; }
inline float3 make_float3(float a, float b, float c) { return {a, b, c}; }

#define __host__
#define __device__

// ---- replicate ExtrapolatorCommon.cuh::State (relevant subset) -----------
namespace Extrapolators {
  struct State {
    float x{0}, y{0}, z{0}, tx{0}, ty{0}, qop{0};
  };

  // ---- replicate NRKExtrapolatorConstants ---------------------------------
  namespace NRKExtrapolatorConstants {
    constexpr int n_stages = 4;
    constexpr float stage_offsets[n_stages] = {0.f, 0.5f, 0.5f, 1.f};
  }

  // ---- replicate NRKExtrapolator (verbatim modulo includes) ---------------
  struct NRKExtrapolator {
  private:
    static inline float2 gamma(const float2 t, const float3 B)
    {
      const float norm = sqrtf(1.f + t.x * t.x + t.y * t.y);
      return make_float2(
        norm * (t.x * t.y * B.x - (1.f + t.x * t.x) * B.y + t.y * B.z),
        norm * ((1.f + t.y * t.y) * B.x - t.x * t.y * B.y - t.x * B.z));
    }

    static inline void
    fillStages(float2 k[NRKExtrapolatorConstants::n_stages], const float2 t, const float qop, const float3 B,
               const float dz)
    {
      k[0] = gamma(t, B);
      k[0].x *= qop;
      k[0].y *= qop;
      for (int stage = 1; stage < NRKExtrapolatorConstants::n_stages; stage++) {
        const float c = NRKExtrapolatorConstants::stage_offsets[stage];
        const float2 t_eff = make_float2(t.x + k[stage - 1].x * dz * c, t.y + k[stage - 1].y * dz * c);
        k[stage] = gamma(t_eff, B);
        k[stage].x *= qop;
        k[stage].y *= qop;
      }
    }

  public:
    static inline void make_step(State& state, const float dz, const float3 B)
    {
      const float2 tn = make_float2(state.tx, state.ty);
      float2 k[NRKExtrapolatorConstants::n_stages];
      fillStages(k, tn, state.qop, B, dz);

      const float2 dRn = make_float2(
        tn.x * dz + (k[0].x + k[1].x + k[2].x) * (dz * dz * (1.f / 6.f)),
        tn.y * dz + (k[0].y + k[1].y + k[2].y) * (dz * dz * (1.f / 6.f)));
      const float2 dTn = make_float2(
        (k[0].x + 2.f * k[1].x + 2.f * k[2].x + k[3].x) * (dz * (1.f / 6.f)),
        (k[0].y + 2.f * k[1].y + 2.f * k[2].y + k[3].y) * (dz * (1.f / 6.f)));

      state.x += dRn.x;
      state.y += dRn.y;
      state.z += dz;
      state.tx += dTn.x;
      state.ty += dTn.y;
    }
  };
} // namespace Extrapolators

using Extrapolators::NRKExtrapolator;
using Extrapolators::State;

static int n_pass = 0, n_fail = 0;

static bool close(float a, float b, float tol) { return std::fabs(a - b) <= tol; }

#define CHECK(cond, msg)                                                                                              \
  do {                                                                                                                \
    if (cond) {                                                                                                       \
      ++n_pass;                                                                                                       \
      std::printf("  PASS  %s\n", msg);                                                                               \
    } else {                                                                                                          \
      ++n_fail;                                                                                                       \
      std::printf("  FAIL  %s\n", msg);                                                                               \
    }                                                                                                                 \
  } while (0)

#define CHECK_CLOSE(a, b, tol, msg)                                                                                   \
  do {                                                                                                                \
    const float aa = (a), bb = (b);                                                                                   \
    if (close(aa, bb, tol)) {                                                                                         \
      ++n_pass;                                                                                                       \
      std::printf("  PASS  %s   (%.9g vs %.9g, |delta|=%.3g, tol=%.3g)\n", msg, aa, bb, std::fabs(aa - bb), tol);     \
    } else {                                                                                                          \
      ++n_fail;                                                                                                       \
      std::printf("  FAIL  %s   (%.9g vs %.9g, |delta|=%.3g, tol=%.3g)\n", msg, aa, bb, std::fabs(aa - bb), tol);     \
    }                                                                                                                 \
  } while (0)

int main()
{
  const float tol = 1e-5f;

  // --- A2: zero-field straight line ----------------------------------------
  std::printf("[A2] zero-field straight line\n");
  {
    const float3 B = make_float3(0.f, 0.f, 0.f);
    const float dz = 100.f, qop = 1e-3f;
    State s{1.0f, 2.0f, 0.0f, 0.1f, -0.05f, qop};
    const State s0 = s;
    NRKExtrapolator::make_step(s, dz, B);
    CHECK_CLOSE(s.x, s0.x + s0.tx * dz, tol, "x");
    CHECK_CLOSE(s.y, s0.y + s0.ty * dz, tol, "y");
    CHECK_CLOSE(s.z, s0.z + dz, tol, "z");
    CHECK_CLOSE(s.tx, s0.tx, tol, "tx");
    CHECK_CLOSE(s.ty, s0.ty, tol, "ty");
    CHECK_CLOSE(s.qop, s0.qop, 0.f, "qop");
  }

  // --- A4: sign symmetry under (x,tx)->(-x,-tx) ----------------------------
  std::printf("[A4] sign symmetry (zero field)\n");
  {
    const float3 B = make_float3(0.f, 0.f, 0.f);
    const float dz = 50.f, qop = 1e-3f;
    State sa{0.5f, 0.3f, 0.0f, 0.02f, 0.01f, qop};
    State sb{-0.5f, 0.3f, 0.0f, -0.02f, 0.01f, qop};
    NRKExtrapolator::make_step(sa, dz, B);
    NRKExtrapolator::make_step(sb, dz, B);
    CHECK_CLOSE(sa.x, -sb.x, tol, "x  flips");
    CHECK_CLOSE(sa.tx, -sb.tx, tol, "tx flips");
    CHECK_CLOSE(sa.y, sb.y, tol, "y  invariant");
    CHECK_CLOSE(sa.ty, sb.ty, tol, "ty invariant");
    CHECK_CLOSE(sa.z, sb.z, tol, "z  invariant");
  }

  // --- A1: qop invariance --------------------------------------------------
  std::printf("[A1] qop invariance under propagation\n");
  {
    const float3 B = make_float3(0.1f, -0.2f, 0.05f);
    const float dz = 25.f;
    for (float qop : {-1e-3f, 0.f, 1e-3f, 5e-3f}) {
      State s{0.0f, 0.0f, 0.0f, 0.01f, -0.02f, qop};
      NRKExtrapolator::make_step(s, dz, B);
      CHECK_CLOSE(s.qop, qop, 0.f, "qop");
    }
  }

  // --- A3: qop linearity at small step -------------------------------------
  std::printf("[A3] qop linearity (small dz)\n");
  {
    const float3 B = make_float3(0.0f, 0.5f, 0.0f);
    const float dz = 1.f;
    State s1{0.f, 0.f, 0.f, 0.f, 0.f, 1e-4f};
    State s2{0.f, 0.f, 0.f, 0.f, 0.f, 2e-4f};
    NRKExtrapolator::make_step(s1, dz, B);
    NRKExtrapolator::make_step(s2, dz, B);
    // epsilon = 1e-3 of |s2.tx|
    CHECK_CLOSE(s2.tx, 2.f * s1.tx, std::fabs(2.f * s1.tx) * 1e-3f + 1e-9f, "tx ~ 2x");
    CHECK_CLOSE(s2.ty, 2.f * s1.ty, std::fabs(2.f * s1.ty) * 1e-3f + 1e-9f, "ty ~ 2x");
  }

  // --- Smoke: dz = 0 is identity ------------------------------------------
  std::printf("[smoke] dz=0 idempotent\n");
  {
    const float3 B = make_float3(0.3f, -0.1f, 0.05f);
    State s{1.f, -2.f, 10.f, 0.05f, 0.03f, 1e-3f};
    const State s0 = s;
    NRKExtrapolator::make_step(s, 0.f, B);
    CHECK_CLOSE(s.x, s0.x, 0.f, "x");
    CHECK_CLOSE(s.y, s0.y, 0.f, "y");
    CHECK_CLOSE(s.z, s0.z, 0.f, "z");
    CHECK_CLOSE(s.tx, s0.tx, 0.f, "tx");
    CHECK_CLOSE(s.ty, s0.ty, 0.f, "ty");
    CHECK_CLOSE(s.qop, s0.qop, 0.f, "qop");
  }

  std::printf("\n=========================================\n");
  std::printf("  Total: %d PASS, %d FAIL\n", n_pass, n_fail);
  std::printf("=========================================\n");
  return n_fail == 0 ? 0 : 1;
}
