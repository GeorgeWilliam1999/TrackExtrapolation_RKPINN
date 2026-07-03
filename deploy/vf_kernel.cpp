// Standalone CPU inference kernel for the vertex-fit PINN_v2 surrogate
// (deployed configuration: 8-feature encoder [x,y,tx,ty,qop norm, z_frac=1,
// z0n, dzn], two tanh hidden layers of width H, 4-channel correction head,
// kick-scaled output head, kick order 1, single shot).
//
// Math mirrors models/architectures.py::PINN_v2.forward (fp32), including the
// analytic forward-mode Jacobian of the 5-state output w.r.t. the 5-state
// input (same object train._model_state_jacobian computes with torch.func.jvp;
// z0 and dz are call geometry, not state, so their columns do not appear).
//
// tanh uses the float-accurate rational approximation (~2 ulp for fp32, the
// same accuracy class as libm tanhf) so the whole kernel auto-vectorises;
// parity vs the torch model is enforced by gates/run_vf_cpp_gate.py.
//
// API (extern "C", double in/out, fp32 arithmetic inside):
//   vf_load(path)                       load the VFK1 blob written by
//                                       deploy/vf_export_weights.py
//   vf_propagate(in7, out5, J25)        one call; J25 may be null; J is
//                                       row-major d(out5)/d(in5), row 4 = e5
//   vf_propagate_batch(n, in7, out5, J) loop over rows (parity/throughput)
//   vf_bench(n, in7, wantJ, checksum)   in-library timing, returns total ns
//
// Build: g++ -O3 -march=native -ffast-math -shared -fPIC -o libvfkernel.so
//        vf_kernel.cpp

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

namespace {

constexpr float KAPPA = 1.0e-3f;  // models/architectures.py::_ALLEN_KAPPA_PREFACTOR
constexpr int HMAX = 256;
int H = 0;
std::vector<float> mean7, std7, W1, b1, W2, b2, W3, b3, g4;

inline bool read_block(FILE* f, std::vector<float>& v, size_t n) {
  v.resize(n);
  return std::fread(v.data(), sizeof(float), n, f) == n;
}

// 8-lane float vector (GCC/Clang extension) — the tangent lane type. Using
// the extension instead of auto-vectorisation makes the 8-wide FMA explicit.
typedef float v8f __attribute__((vector_size(32)));

// Float-accurate tanh (rational P13/Q6, clamped) — ~2 ulp over the fp32
// range, fully vectorisable. Same polynomial family as Eigen/XLA use.
inline float tanh_fast(float x) {
  const float c = x < -9.0f ? -9.0f : (x > 9.0f ? 9.0f : x);
  const float x2 = c * c;
  float num = -2.76076847742355e-16f;
  num = x2 * num + 2.00018790482477e-13f;
  num = x2 * num + -8.60467152213735e-11f;
  num = x2 * num + 5.12229709037114e-08f;
  num = x2 * num + 1.48572235717979e-05f;
  num = x2 * num + 6.37261928875436e-04f;
  num = x2 * num + 4.89352455891786e-03f;
  num = c * num;
  float den = 1.19825839466702e-06f;
  den = x2 * den + 1.18534705686654e-04f;
  den = x2 * den + 2.26843463243900e-03f;
  den = x2 * den + 4.89352518554385e-03f;
  return num / den;
}

}  // namespace

extern "C" {

int vf_load(const char* path) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return 1;
  char magic[4];
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "VFK1", 4) != 0) {
    std::fclose(f);
    return 2;
  }
  int32_t h = 0;
  if (std::fread(&h, sizeof(h), 1, f) != 1 || h <= 0 || h > HMAX) {
    std::fclose(f);
    return 3;
  }
  H = h;
  bool ok = read_block(f, mean7, 7) && read_block(f, std7, 7) &&
            read_block(f, W1, size_t(H) * 8) && read_block(f, b1, H) &&
            read_block(f, W2, size_t(H) * H) && read_block(f, b2, H) &&
            read_block(f, W3, size_t(4) * H) && read_block(f, b3, 4) &&
            read_block(f, g4, 4);
  std::fclose(f);
  return ok ? 0 : 4;
}

// in7 = [x, y, tx, ty, qop, z0, dz]; out5 = [x', y', tx', ty', qop];
// J25 (optional) = row-major 5x5 d(out)/d(state5).
// jmode: 0 = state only, 1 = exact forward-mode J (tangent passes),
//        2 = head-only J (straight-line block + the momentum column, which
//            the kick head gives EXACTLY for free: d(out)/d(qop) =
//            correction/qop since kappa is linear in qop; the dropped
//            d(corr)/d(state) terms are tier-2- and fit-level negligible,
//            see gates/run_vf_p4_fitharness.py NNH arm).
void vf_propagate_mode(const double* in7, double* out5, double* J25, int jmode) {
  const float x = float(in7[0]), y = float(in7[1]), tx = float(in7[2]),
              ty = float(in7[3]), qop = float(in7[4]), z0 = float(in7[5]),
              dz = float(in7[6]);

  // encoder input (z_frac = 1 in the deployed single-shot configuration)
  float e[8];
  e[0] = (x - mean7[0]) / std7[0];
  e[1] = (y - mean7[1]) / std7[1];
  e[2] = (tx - mean7[2]) / std7[2];
  e[3] = (ty - mean7[3]) / std7[3];
  e[4] = (qop - mean7[4]) / std7[4];
  e[5] = 1.0f;
  e[6] = (z0 - mean7[5]) / std7[5];
  e[7] = (dz - mean7[6]) / std7[6];

  const bool wantJ = J25 != nullptr && jmode == 1;   // tangent passes needed?
  const bool anyJ = J25 != nullptr && jmode != 0;

  // Tangents are stored lane-packed: T[i*8 + j] holds tangent j of unit i
  // (lanes 5..7 are zero padding). One broadcast-FMA per weight then carries
  // ALL five state tangents at once, so the whole Jacobian costs about one
  // extra forward pass instead of five.
  alignas(64) float h1[HMAX], h2[HMAX];
  alignas(64) float T1[HMAX * 8], T2[HMAX * 8];
  alignas(32) float inv_std8[8] = {};
  if (wantJ)
    for (int j = 0; j < 5; ++j) inv_std8[j] = 1.0f / std7[j];

  // ---- layer 1: h1 = tanh(W1 e + b1);  T1[i] = (1-h1^2) W1[i, 0:5] / std
  // (the unit state tangent e_j is 1/std_j in the normalised coordinate)
  for (int i = 0; i < H; ++i) {
    const float* w = &W1[size_t(i) * 8];
    float a = b1[i];
    for (int k = 0; k < 8; ++k) a += w[k] * e[k];
    const float t = tanh_fast(a);
    h1[i] = t;
    if (wantJ) {
      const float d = 1.0f - t * t;
      float* t1 = &T1[size_t(i) * 8];
      for (int j = 0; j < 8; ++j) t1[j] = d * w[j] * inv_std8[j];
    }
  }

  // ---- layer 2: h2 = tanh(W2 h1 + b2);  T2[i] = (1-h2^2) sum_k W2[i,k] T1[k]
  const v8f* T1v = reinterpret_cast<const v8f*>(T1);
  v8f* T2v = reinterpret_cast<v8f*>(T2);
  for (int i = 0; i < H; ++i) {
    const float* w = &W2[size_t(i) * H];
    float a = b2[i];
    if (wantJ) {
      v8f acc = {};
      for (int k = 0; k < H; ++k) {
        const float wk = w[k];
        a += wk * h1[k];
        acc += wk * T1v[k];
      }
      const float t = tanh_fast(a);
      h2[i] = t;
      T2v[i] = (1.0f - t * t) * acc;
    } else {
      for (int k = 0; k < H; ++k) a += w[k] * h1[k];
      h2[i] = tanh_fast(a);
    }
  }

  // ---- head: c = W3 h2 + b3;  dc[i] = sum_k W3[i,k] T2[k]  (lane-packed)
  float c[4];
  alignas(32) v8f dc[4] = {};
  for (int i = 0; i < 4; ++i) {
    const float* w = &W3[size_t(i) * H];
    float a = b3[i];
    if (wantJ) {
      v8f acc = {};
      for (int k = 0; k < H; ++k) {
        const float wk = w[k];
        a += wk * h2[k];
        acc += wk * T2v[k];
      }
      dc[i] = acc;
    } else {
      for (int k = 0; k < H; ++k) a += w[k] * h2[k];
    }
    c[i] = a;
  }

  // ---- kick-scaled output head
  const float kd = KAPPA * qop * dz;   // kappa * dz
  out5[0] = double(x + tx * dz + g4[2] * kd * dz * c[2]);
  out5[1] = double(y + ty * dz + g4[3] * kd * dz * c[3]);
  out5[2] = double(tx + g4[0] * kd * c[0]);
  out5[3] = double(ty + g4[1] * kd * c[1]);
  out5[4] = double(qop);

  if (!anyJ) return;
  // jmode 2 reaches the assembly below with dc = 0: the straight-line block
  // plus the exact momentum column (dkd terms), nothing else.

  const float dkd = KAPPA * dz;        // d(kd)/d(qop)
  for (int j = 0; j < 5; ++j) {
    const float dkd_j = (j == 4) ? dkd : 0.0f;
    float v = (j == 0 ? 1.0f : 0.0f) + (j == 2 ? dz : 0.0f);
    v += g4[2] * dz * (dkd_j * c[2] + kd * dc[2][j]);
    J25[0 * 5 + j] = double(v);
    v = (j == 1 ? 1.0f : 0.0f) + (j == 3 ? dz : 0.0f);
    v += g4[3] * dz * (dkd_j * c[3] + kd * dc[3][j]);
    J25[1 * 5 + j] = double(v);
    v = (j == 2 ? 1.0f : 0.0f) + g4[0] * (dkd_j * c[0] + kd * dc[0][j]);
    J25[2 * 5 + j] = double(v);
    v = (j == 3 ? 1.0f : 0.0f) + g4[1] * (dkd_j * c[1] + kd * dc[1][j]);
    J25[3 * 5 + j] = double(v);
    J25[4 * 5 + j] = (j == 4) ? 1.0 : 0.0;
  }
}

// Back-compat entry point: exact J when J25 is provided.
void vf_propagate(const double* in7, double* out5, double* J25) {
  vf_propagate_mode(in7, out5, J25, J25 ? 1 : 0);
}

void vf_propagate_batch(long n, const double* in7, double* out5, double* J25) {
  for (long i = 0; i < n; ++i)
    vf_propagate(in7 + 7 * i, out5 + 5 * i, J25 ? J25 + 25 * i : nullptr);
}

void vf_propagate_batch_mode(long n, const double* in7, double* out5,
                             double* J25, int jmode) {
  for (long i = 0; i < n; ++i)
    vf_propagate_mode(in7 + 7 * i, out5 + 5 * i,
                      J25 ? J25 + 25 * i : nullptr, jmode);
}

// In-library single-call benchmark (keeps ctypes call overhead out of the
// measurement): n sequential calls over the provided rows, returns total ns.
// jmode as in vf_propagate_mode (0 state, 1 exact J, 2 head-only J).
// The checksum sink prevents dead-code elimination.
double vf_bench(long n, const double* in7, int jmode, double* checksum) {
  double out[5], J[25];
  double sink = 0.0;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  for (long i = 0; i < n; ++i) {
    vf_propagate_mode(in7 + 7 * i, out, jmode ? J : nullptr, jmode);
    sink += out[0] + (jmode ? J[2 * 5 + 4] : 0.0);
  }
  clock_gettime(CLOCK_MONOTONIC, &t1);
  if (checksum) *checksum = sink;
  return double(t1.tv_sec - t0.tv_sec) * 1e9 + double(t1.tv_nsec - t0.tv_nsec);
}

}  // extern "C"
