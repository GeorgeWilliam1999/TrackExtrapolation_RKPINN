// =============================================================================
// field_bench.cpp — Native C++ micro-benchmark for field map sub-components
//
// Measures the wall-clock cost of:
//   1. Trilinear interpolation on a grid matching the LHCb field map
//   2. Neural network inference (1-hidden-layer SiLU, 32 neurons)
//   3. A single Runge-Kutta derivative evaluation (Lorentz force)
//   4. A single trilinear + derivative combo (= 1 RK stage)
//   5. Full 6-stage CashKarp step emulation
//
// All timing uses std::chrono::high_resolution_clock inside C++.
// Python is NOT in the measurement loop.
//
// Compile:
//   g++ -O2 -std=c++17 -march=native -o field_bench field_bench.cpp -lm
// Run:
//   ./field_bench          # prints CSV to stdout
//   ./field_bench 100000   # custom iteration count
// =============================================================================
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <immintrin.h>  // AVX2 intrinsics

// ─── Grid dimensions matching LHCb twodip.rtf ───────────────────────────────
static constexpr int NX = 81, NY = 81, NZ = 146;
static constexpr int NPTS = NX * NY * NZ;          // 957 906
static constexpr double X_MIN = -4000.0, X_MAX = 4000.0;
static constexpr double Y_MIN = -4000.0, Y_MAX = 4000.0;
static constexpr double Z_MIN = -500.0,  Z_MAX = 14000.0;
static constexpr double DX = 100.0, DY = 100.0, DZ = 100.0;

// ─── Grid storage ────────────────────────────────────────────────────────────
// [ix][iy][iz][component], stored flat for cache-realistic access pattern.
static float grid[NX * NY * NZ * 3];

// Fill with pseudo-realistic field values (dipole-like, not random noise)
static void fill_grid() {
    for (int ix = 0; ix < NX; ++ix)
    for (int iy = 0; iy < NY; ++iy)
    for (int iz = 0; iz < NZ; ++iz) {
        double x = X_MIN + ix * DX;
        double y = Y_MIN + iy * DY;
        double z = Z_MIN + iz * DZ;
        // Simple dipole-like field: peaked around z=5000, y=-3900
        double r2 = x*x + (y+3900)*(y+3900);
        double zp = (z - 5000.0) / 2000.0;
        double mag = 33000.0 * std::exp(-r2 / (2000.0*2000.0)) * std::exp(-zp*zp);
        int idx = ((ix * NY + iy) * NZ + iz) * 3;
        grid[idx + 0] = float(mag * 0.01);      // Bx (small)
        grid[idx + 1] = float(-mag);              // By (dominant)
        grid[idx + 2] = float(mag * 0.005);      // Bz (small)
    }
}

// ─── Trilinear interpolation ─────────────────────────────────────────────────
// Matches the LHCb fieldVectorLinearInterpolation implementation:
// 8 corner lookups + weighted average
static inline void trilinear(double x, double y, double z,
                              float& bx, float& by, float& bz) {
    // Scale to grid indices
    double fx = (x - X_MIN) / DX;
    double fy = (y - Y_MIN) / DY;
    double fz = (z - Z_MIN) / DZ;

    int ix = int(fx);  if (ix < 0) ix = 0; if (ix >= NX-1) ix = NX-2;
    int iy = int(fy);  if (iy < 0) iy = 0; if (iy >= NY-1) iy = NY-2;
    int iz = int(fz);  if (iz < 0) iz = 0; if (iz >= NZ-1) iz = NZ-2;

    double dx = fx - ix;
    double dy = fy - iy;
    double dz = fz - iz;

    // 8 corner indices
    int i000 = ((ix   * NY + iy  ) * NZ + iz  ) * 3;
    int i001 = ((ix   * NY + iy  ) * NZ + iz+1) * 3;
    int i010 = ((ix   * NY + iy+1) * NZ + iz  ) * 3;
    int i011 = ((ix   * NY + iy+1) * NZ + iz+1) * 3;
    int i100 = (((ix+1)*NY + iy  ) * NZ + iz  ) * 3;
    int i101 = (((ix+1)*NY + iy  ) * NZ + iz+1) * 3;
    int i110 = (((ix+1)*NY + iy+1) * NZ + iz  ) * 3;
    int i111 = (((ix+1)*NY + iy+1) * NZ + iz+1) * 3;

    double w000 = (1-dx)*(1-dy)*(1-dz);
    double w001 = (1-dx)*(1-dy)*dz;
    double w010 = (1-dx)*dy*(1-dz);
    double w011 = (1-dx)*dy*dz;
    double w100 = dx*(1-dy)*(1-dz);
    double w101 = dx*(1-dy)*dz;
    double w110 = dx*dy*(1-dz);
    double w111 = dx*dy*dz;

    bx = float(w000*grid[i000+0] + w001*grid[i001+0] + w010*grid[i010+0] + w011*grid[i011+0]
              + w100*grid[i100+0] + w101*grid[i101+0] + w110*grid[i110+0] + w111*grid[i111+0]);
    by = float(w000*grid[i000+1] + w001*grid[i001+1] + w010*grid[i010+1] + w011*grid[i011+1]
              + w100*grid[i100+1] + w101*grid[i101+1] + w110*grid[i110+1] + w111*grid[i111+1]);
    bz = float(w000*grid[i000+2] + w001*grid[i001+2] + w010*grid[i010+2] + w011*grid[i011+2]
              + w100*grid[i100+2] + w101*grid[i101+2] + w110*grid[i110+2] + w111*grid[i111+2]);
}

// ─── Neural network field (1-hidden-layer SiLU, 32 neurons) ─────────────────
// Exact copy of FieldMapNNWeights.h evaluate() logic.
// Uses random but fixed weights (we only care about timing, not accuracy).
static float nn_w0[96], nn_b0[32], nn_w2[96], nn_b2[3];
static float nn_in_mean[3], nn_in_std[3], nn_out_mean[3], nn_out_std[3];

static void init_nn_weights() {
    std::mt19937 gen(12345);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    for (int i = 0; i < 96; ++i) nn_w0[i] = dist(gen);
    for (int i = 0; i < 32; ++i) nn_b0[i] = dist(gen);
    for (int i = 0; i < 96; ++i) nn_w2[i] = dist(gen);
    for (int i = 0; i < 3;  ++i) nn_b2[i] = dist(gen);
    nn_in_mean[0] = 0.0f; nn_in_mean[1] = 0.0f; nn_in_mean[2] = 6750.0f;
    nn_in_std[0]  = 2338.0f; nn_in_std[1] = 2338.0f; nn_in_std[2] = 4214.0f;
    nn_out_mean[0] = 0.0f; nn_out_mean[1] = 0.12f; nn_out_mean[2] = 0.0f;
    nn_out_std[0]  = 22.15f; nn_out_std[1] = 71.61f; nn_out_std[2] = 22.15f;
}

static inline void nn_field(float x, float y, float z,
                             float& bx, float& by, float& bz) {
    float in0 = (x - nn_in_mean[0]) / nn_in_std[0];
    float in1 = (y - nn_in_mean[1]) / nn_in_std[1];
    float in2 = (z - nn_in_mean[2]) / nn_in_std[2];
    float h[32];
    for (int i = 0; i < 32; ++i) {
        h[i] = nn_w0[i*3+0]*in0 + nn_w0[i*3+1]*in1 + nn_w0[i*3+2]*in2 + nn_b0[i];
        h[i] = h[i] / (1.0f + std::exp(-h[i]));  // SiLU
    }
    float out[3] = {nn_b2[0], nn_b2[1], nn_b2[2]};
    for (int i = 0; i < 32; ++i) {
        out[0] += nn_w2[0*32+i] * h[i];
        out[1] += nn_w2[1*32+i] * h[i];
        out[2] += nn_w2[2*32+i] * h[i];
    }
    bx = out[0]*nn_out_std[0] + nn_out_mean[0];
    by = out[1]*nn_out_std[1] + nn_out_mean[1];
    bz = out[2]*nn_out_std[2] + nn_out_mean[2];
}

// ─── Neural network field (1-hidden-layer ReLU, 32 neurons) ─────────────────
// Same architecture & weights as nn_field, but with ReLU activation.
// This isolates the cost of the activation function (no std::exp).
static inline void nn_field_relu(float x, float y, float z,
                                  float& bx, float& by, float& bz) {
    float in0 = (x - nn_in_mean[0]) / nn_in_std[0];
    float in1 = (y - nn_in_mean[1]) / nn_in_std[1];
    float in2 = (z - nn_in_mean[2]) / nn_in_std[2];
    float h[32];
    for (int i = 0; i < 32; ++i) {
        h[i] = nn_w0[i*3+0]*in0 + nn_w0[i*3+1]*in1 + nn_w0[i*3+2]*in2 + nn_b0[i];
        h[i] = h[i] > 0.0f ? h[i] : 0.0f;  // ReLU
    }
    float out[3] = {nn_b2[0], nn_b2[1], nn_b2[2]};
    for (int i = 0; i < 32; ++i) {
        out[0] += nn_w2[0*32+i] * h[i];
        out[1] += nn_w2[1*32+i] * h[i];
        out[2] += nn_w2[2*32+i] * h[i];
    }
    bx = out[0]*nn_out_std[0] + nn_out_mean[0];
    by = out[1]*nn_out_std[1] + nn_out_mean[1];
    bz = out[2]*nn_out_std[2] + nn_out_mean[2];
}

// ─── NN field [128, 128] SiLU (2-layer) ─────────────────────────────────────
// This is the architecture used in the v2 loss-function experiments
static float nn2_w0[128*3], nn2_b0[128];
static float nn2_w1[128*128], nn2_b1[128];
static float nn2_w2[3*128], nn2_b2_2L[3];

static void init_nn2_weights() {
    std::mt19937 gen(54321);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    for (auto& w : nn2_w0) w = dist(gen);
    for (auto& b : nn2_b0) b = dist(gen);
    for (auto& w : nn2_w1) w = dist(gen);
    for (auto& b : nn2_b1) b = dist(gen);
    for (auto& w : nn2_w2) w = dist(gen);
    for (auto& b : nn2_b2_2L) b = dist(gen);
}

static inline void nn2_field(float x, float y, float z,
                              float& bx, float& by, float& bz) {
    float in0 = (x - nn_in_mean[0]) / nn_in_std[0];
    float in1 = (y - nn_in_mean[1]) / nn_in_std[1];
    float in2 = (z - nn_in_mean[2]) / nn_in_std[2];
    // Layer 1: 3 -> 128
    float h1[128];
    for (int i = 0; i < 128; ++i) {
        h1[i] = nn2_w0[i*3+0]*in0 + nn2_w0[i*3+1]*in1 + nn2_w0[i*3+2]*in2 + nn2_b0[i];
        h1[i] = h1[i] / (1.0f + std::exp(-h1[i]));
    }
    // Layer 2: 128 -> 128
    float h2[128];
    for (int i = 0; i < 128; ++i) {
        float s = nn2_b1[i];
        for (int j = 0; j < 128; ++j) s += nn2_w1[i*128+j] * h1[j];
        h2[i] = s / (1.0f + std::exp(-s));
    }
    // Output: 128 -> 3
    float out[3] = {nn2_b2_2L[0], nn2_b2_2L[1], nn2_b2_2L[2]};
    for (int i = 0; i < 128; ++i) {
        out[0] += nn2_w2[0*128+i] * h2[i];
        out[1] += nn2_w2[1*128+i] * h2[i];
        out[2] += nn2_w2[2*128+i] * h2[i];
    }
    bx = out[0]*nn_out_std[0] + nn_out_mean[0];
    by = out[1]*nn_out_std[1] + nn_out_mean[1];
    bz = out[2]*nn_out_std[2] + nn_out_mean[2];
}

// ─── NN field [32, 32] ReLU (2-layer, 32 neurons each) ─────────────────────
// Smaller 2-layer network: 1283 parameters total
// (3*32 + 32) + (32*32 + 32) + (32*3 + 3) = 128 + 1056 + 99 = 1283
static float nn3_w0[32*3], nn3_b0[32];
static float nn3_w1[32*32], nn3_b1[32];
static float nn3_w2[3*32], nn3_b2[3];

static void init_nn3_weights() {
    std::mt19937 gen(99999);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    for (auto& w : nn3_w0) w = dist(gen);
    for (auto& b : nn3_b0) b = dist(gen);
    for (auto& w : nn3_w1) w = dist(gen);
    for (auto& b : nn3_b1) b = dist(gen);
    for (auto& w : nn3_w2) w = dist(gen);
    for (auto& b : nn3_b2) b = dist(gen);
}

static inline void nn3_field(float x, float y, float z,
                              float& bx, float& by, float& bz) {
    float in0 = (x - nn_in_mean[0]) / nn_in_std[0];
    float in1 = (y - nn_in_mean[1]) / nn_in_std[1];
    float in2 = (z - nn_in_mean[2]) / nn_in_std[2];
    // Layer 1: 3 -> 32
    float h1[32];
    for (int i = 0; i < 32; ++i) {
        h1[i] = nn3_w0[i*3+0]*in0 + nn3_w0[i*3+1]*in1 + nn3_w0[i*3+2]*in2 + nn3_b0[i];
        h1[i] = h1[i] > 0.0f ? h1[i] : 0.0f;  // ReLU
    }
    // Layer 2: 32 -> 32
    float h2[32];
    for (int i = 0; i < 32; ++i) {
        float s = nn3_b1[i];
        for (int j = 0; j < 32; ++j) s += nn3_w1[i*32+j] * h1[j];
        h2[i] = s > 0.0f ? s : 0.0f;  // ReLU
    }
    // Output: 32 -> 3
    float out[3] = {nn3_b2[0], nn3_b2[1], nn3_b2[2]};
    for (int i = 0; i < 32; ++i) {
        out[0] += nn3_w2[0*32+i] * h2[i];
        out[1] += nn3_w2[1*32+i] * h2[i];
        out[2] += nn3_w2[2*32+i] * h2[i];
    }
    bx = out[0]*nn_out_std[0] + nn_out_mean[0];
    by = out[1]*nn_out_std[1] + nn_out_mean[1];
    bz = out[2]*nn_out_std[2] + nn_out_mean[2];
}

// =============================================================================
// OPTIMISED NN VARIANTS — Exploiting parallelism & SIMD
// =============================================================================

// ─── OPT 1: Batch-6 [32] ReLU ──────────────────────────────────────────────
// Evaluate 6 field queries at once. Key optimisation: weights are loaded into
// registers ONCE and reused across all 6 inputs. This eliminates 5/6 of the
// weight-loading traffic.
//
// Memory layout: inputs[6][3], outputs[6][3]
// Weight reuse: W0 (96 floats = 384 bytes) loaded once, applied 6 times
//               W2 (96 floats = 384 bytes) loaded once, applied 6 times
static inline void nn_field_relu_batch6(
    const float inputs[6][3], float outputs[6][3]) {
    // Pre-compute inverse std (division is expensive, do once)
    float inv_std0 = 1.0f / nn_in_std[0];
    float inv_std1 = 1.0f / nn_in_std[1];
    float inv_std2 = 1.0f / nn_in_std[2];

    // Normalise all 6 inputs
    float in_norm[6][3];
    for (int b = 0; b < 6; ++b) {
        in_norm[b][0] = (inputs[b][0] - nn_in_mean[0]) * inv_std0;
        in_norm[b][1] = (inputs[b][1] - nn_in_mean[1]) * inv_std1;
        in_norm[b][2] = (inputs[b][2] - nn_in_mean[2]) * inv_std2;
    }

    // Hidden layer: for each neuron, compute across all 6 inputs
    // Key: weight[i] is loaded once and applied to all 6 inputs
    float h[6][32];
    for (int i = 0; i < 32; ++i) {
        float w0 = nn_w0[i*3+0], w1 = nn_w0[i*3+1], w2 = nn_w0[i*3+2];
        float b = nn_b0[i];
        for (int k = 0; k < 6; ++k) {
            float val = w0*in_norm[k][0] + w1*in_norm[k][1] + w2*in_norm[k][2] + b;
            h[k][i] = val > 0.0f ? val : 0.0f;  // ReLU
        }
    }

    // Output layer: for each output component, accumulate across hidden
    for (int b = 0; b < 6; ++b) {
        float o0 = nn_b2[0], o1 = nn_b2[1], o2 = nn_b2[2];
        for (int i = 0; i < 32; ++i) {
            o0 += nn_w2[0*32+i] * h[b][i];
            o1 += nn_w2[1*32+i] * h[b][i];
            o2 += nn_w2[2*32+i] * h[b][i];
        }
        outputs[b][0] = o0*nn_out_std[0] + nn_out_mean[0];
        outputs[b][1] = o1*nn_out_std[1] + nn_out_mean[1];
        outputs[b][2] = o2*nn_out_std[2] + nn_out_mean[2];
    }
}

// ─── OPT 2: AVX2 [32] ReLU (single call) ───────────────────────────────────
// Hand-coded AVX2 intrinsics for max throughput on a single evaluation.
// Processes 8 neurons per SIMD iteration (32 neurons = 4 iterations).
// Uses FMA (fused multiply-add) where possible.
//
// Key tricks:
//   - _mm256_fmadd_ps: fused multiply-add (1 cycle vs 2 for separate mul+add)
//   - _mm256_max_ps: branchless ReLU (no misprediction penalty)
//   - _mm256_broadcast_ss: replicate a scalar to all 8 lanes
//   - Horizontal reduction via hadd + permute for the 3 output sums
static inline void nn_field_relu_avx2(float x, float y, float z,
                                       float& bx, float& by, float& bz) {
    float in0 = (x - nn_in_mean[0]) / nn_in_std[0];
    float in1 = (y - nn_in_mean[1]) / nn_in_std[1];
    float in2 = (z - nn_in_mean[2]) / nn_in_std[2];

    __m256 vin0 = _mm256_broadcast_ss(&in0);
    __m256 vin1 = _mm256_broadcast_ss(&in1);
    __m256 vin2 = _mm256_broadcast_ss(&in2);
    __m256 vzero = _mm256_setzero_ps();

    // Hidden layer: process 8 neurons at a time (4 iterations for 32)
    // W0 is stored as [neuron][input], stride = 3 per neuron
    // We need to gather W0[:,0], W0[:,1], W0[:,2] for 8 neurons
    // Since stride is 3, we must load-and-shuffle manually
    alignas(32) float h_buf[32];

    for (int blk = 0; blk < 4; ++blk) {
        int base = blk * 8;
        // Load weights for 8 neurons — each neuron has 3 weights at stride 3
        alignas(32) float wt0[8], wt1[8], wt2[8], bias[8];
        for (int j = 0; j < 8; ++j) {
            wt0[j] = nn_w0[(base+j)*3+0];
            wt1[j] = nn_w0[(base+j)*3+1];
            wt2[j] = nn_w0[(base+j)*3+2];
            bias[j] = nn_b0[base+j];
        }
        __m256 vw0 = _mm256_load_ps(wt0);
        __m256 vw1 = _mm256_load_ps(wt1);
        __m256 vw2 = _mm256_load_ps(wt2);
        __m256 vb  = _mm256_load_ps(bias);

        // h = W0*in + b, then ReLU
        __m256 vh = _mm256_fmadd_ps(vw0, vin0, vb);        // w0*in0 + b
        vh = _mm256_fmadd_ps(vw1, vin1, vh);                // + w1*in1
        vh = _mm256_fmadd_ps(vw2, vin2, vh);                // + w2*in2
        vh = _mm256_max_ps(vh, vzero);                       // ReLU

        _mm256_store_ps(h_buf + base, vh);
    }

    // Output layer: 3 dot products of length 32
    // out[c] = sum_i(W2[c*32+i] * h[i]) + b2[c]
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    for (int blk = 0; blk < 4; ++blk) {
        __m256 vh = _mm256_load_ps(h_buf + blk*8);
        __m256 vw_o0 = _mm256_loadu_ps(nn_w2 + 0*32 + blk*8);
        __m256 vw_o1 = _mm256_loadu_ps(nn_w2 + 1*32 + blk*8);
        __m256 vw_o2 = _mm256_loadu_ps(nn_w2 + 2*32 + blk*8);
        acc0 = _mm256_fmadd_ps(vw_o0, vh, acc0);
        acc1 = _mm256_fmadd_ps(vw_o1, vh, acc1);
        acc2 = _mm256_fmadd_ps(vw_o2, vh, acc2);
    }

    // Horizontal sum: reduce 8 lanes to scalar
    // hadd pairs adjacent lanes, then we add the two 128-bit halves
    auto hsum = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);              // 4 values
        s = _mm_hadd_ps(s, s);                       // 2 values
        s = _mm_hadd_ps(s, s);                       // 1 value
        return _mm_cvtss_f32(s);
    };

    bx = (hsum(acc0) + nn_b2[0]) * nn_out_std[0] + nn_out_mean[0];
    by = (hsum(acc1) + nn_b2[1]) * nn_out_std[1] + nn_out_mean[1];
    bz = (hsum(acc2) + nn_b2[2]) * nn_out_std[2] + nn_out_mean[2];
}

// ─── OPT 3: AVX2 Batch-6 [32] ReLU ─────────────────────────────────────────
// Combines SIMD (8 neurons in parallel) with batching (6 inputs sharing weights).
// The weight registers are loaded ONCE and applied to all 6 inputs before
// moving to the next group of 8 neurons.
//
// This achieves:
//   - Weight reuse: each W0 load serves 6 inputs (6× amortisation)
//   - SIMD parallelism: 8 neurons per instruction (8× throughput)
//   - Combined: 48× parallelism vs scalar single-call
static inline void nn_field_relu_avx2_batch6(
    const float inputs[6][3], float outputs[6][3]) {
    float inv_std0 = 1.0f / nn_in_std[0];
    float inv_std1 = 1.0f / nn_in_std[1];
    float inv_std2 = 1.0f / nn_in_std[2];

    __m256 vzero = _mm256_setzero_ps();

    // Normalise inputs and broadcast
    __m256 vin0[6], vin1[6], vin2[6];
    for (int b = 0; b < 6; ++b) {
        float n0 = (inputs[b][0] - nn_in_mean[0]) * inv_std0;
        float n1 = (inputs[b][1] - nn_in_mean[1]) * inv_std1;
        float n2 = (inputs[b][2] - nn_in_mean[2]) * inv_std2;
        vin0[b] = _mm256_broadcast_ss(&n0);
        vin1[b] = _mm256_broadcast_ss(&n1);
        vin2[b] = _mm256_broadcast_ss(&n2);
    }

    // Hidden layer: process 8 neurons at a time
    alignas(32) float h_buf[6][32];
    for (int blk = 0; blk < 4; ++blk) {
        int base = blk * 8;
        // Gather weights for 8 neurons
        alignas(32) float wt0[8], wt1[8], wt2[8], bias[8];
        for (int j = 0; j < 8; ++j) {
            wt0[j] = nn_w0[(base+j)*3+0];
            wt1[j] = nn_w0[(base+j)*3+1];
            wt2[j] = nn_w0[(base+j)*3+2];
            bias[j] = nn_b0[base+j];
        }
        __m256 vw0 = _mm256_load_ps(wt0);
        __m256 vw1 = _mm256_load_ps(wt1);
        __m256 vw2 = _mm256_load_ps(wt2);
        __m256 vb  = _mm256_load_ps(bias);

        // Apply same weights to all 6 inputs
        for (int b = 0; b < 6; ++b) {
            __m256 vh = _mm256_fmadd_ps(vw0, vin0[b], vb);
            vh = _mm256_fmadd_ps(vw1, vin1[b], vh);
            vh = _mm256_fmadd_ps(vw2, vin2[b], vh);
            vh = _mm256_max_ps(vh, vzero);
            _mm256_store_ps(h_buf[b] + base, vh);
        }
    }

    // Output layer
    auto hsum = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    };

    for (int b = 0; b < 6; ++b) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        for (int blk = 0; blk < 4; ++blk) {
            __m256 vh = _mm256_load_ps(h_buf[b] + blk*8);
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(nn_w2 + 0*32 + blk*8), vh, acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(nn_w2 + 1*32 + blk*8), vh, acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(nn_w2 + 2*32 + blk*8), vh, acc2);
        }
        outputs[b][0] = (hsum(acc0) + nn_b2[0]) * nn_out_std[0] + nn_out_mean[0];
        outputs[b][1] = (hsum(acc1) + nn_b2[1]) * nn_out_std[1] + nn_out_mean[1];
        outputs[b][2] = (hsum(acc2) + nn_b2[2]) * nn_out_std[2] + nn_out_mean[2];
    }
}

// ─── OPT 4: Fully unrolled [32] ReLU (no loops at all) ─────────────────────
// The compiler sometimes fails to fully unroll the 32-iteration loop.
// This version manually unrolls everything for zero loop overhead.
// Uses __attribute__((always_inline)) to force inlining even at -O2.
__attribute__((always_inline))
static inline void nn_field_relu_unrolled(float x, float y, float z,
                                           float& bx, float& by, float& bz) {
    float in0 = (x - nn_in_mean[0]) / nn_in_std[0];
    float in1 = (y - nn_in_mean[1]) / nn_in_std[1];
    float in2 = (z - nn_in_mean[2]) / nn_in_std[2];

    // Hidden layer — fully unrolled macro
    #define NEURON(i) \
        float h##i = nn_w0[(i)*3+0]*in0 + nn_w0[(i)*3+1]*in1 + nn_w0[(i)*3+2]*in2 + nn_b0[i]; \
        h##i = h##i > 0.0f ? h##i : 0.0f;

    NEURON(0)  NEURON(1)  NEURON(2)  NEURON(3)
    NEURON(4)  NEURON(5)  NEURON(6)  NEURON(7)
    NEURON(8)  NEURON(9)  NEURON(10) NEURON(11)
    NEURON(12) NEURON(13) NEURON(14) NEURON(15)
    NEURON(16) NEURON(17) NEURON(18) NEURON(19)
    NEURON(20) NEURON(21) NEURON(22) NEURON(23)
    NEURON(24) NEURON(25) NEURON(26) NEURON(27)
    NEURON(28) NEURON(29) NEURON(30) NEURON(31)
    #undef NEURON

    // Output layer — accumulate in 3 sums
    #define OUT(i) \
        nn_w2[0*32+(i)]*h##i, nn_w2[1*32+(i)]*h##i, nn_w2[2*32+(i)]*h##i

    float o0 = nn_b2[0], o1 = nn_b2[1], o2 = nn_b2[2];
    #define ACC(i) o0 += nn_w2[0*32+(i)]*h##i; o1 += nn_w2[1*32+(i)]*h##i; o2 += nn_w2[2*32+(i)]*h##i;
    ACC(0)  ACC(1)  ACC(2)  ACC(3)  ACC(4)  ACC(5)  ACC(6)  ACC(7)
    ACC(8)  ACC(9)  ACC(10) ACC(11) ACC(12) ACC(13) ACC(14) ACC(15)
    ACC(16) ACC(17) ACC(18) ACC(19) ACC(20) ACC(21) ACC(22) ACC(23)
    ACC(24) ACC(25) ACC(26) ACC(27) ACC(28) ACC(29) ACC(30) ACC(31)
    #undef ACC

    bx = o0*nn_out_std[0] + nn_out_mean[0];
    by = o1*nn_out_std[1] + nn_out_mean[1];
    bz = o2*nn_out_std[2] + nn_out_mean[2];
}

// ─── OPT 5: Aligned weights for OPT2/OPT3 ─────────────────────────────────
// Restructure nn_w0 from [neuron][input] to [input][neuron_block8] for
// contiguous AVX2 loads. Pre-transposed at init time.
alignas(32) static float nn_w0_t[3][32];  // transposed: [input_dim][neurons]
alignas(32) static float nn_b0_a[32];     // aligned copy of biases
alignas(32) static float nn_w2_a[3][32];  // aligned copy of output weights

static void init_nn_weights_transposed() {
    for (int i = 0; i < 32; ++i) {
        nn_w0_t[0][i] = nn_w0[i*3+0];
        nn_w0_t[1][i] = nn_w0[i*3+1];
        nn_w0_t[2][i] = nn_w0[i*3+2];
        nn_b0_a[i] = nn_b0[i];
        nn_w2_a[0][i] = nn_w2[0*32+i];
        nn_w2_a[1][i] = nn_w2[1*32+i];
        nn_w2_a[2][i] = nn_w2[2*32+i];
    }
}

// ─── OPT 6: AVX2 with pre-transposed aligned weights ───────────────────────
// Same SIMD approach as OPT 2, but weights are pre-arranged for contiguous
// 256-bit loads (no gather/scatter needed). This is the fastest single-call.
static inline void nn_field_relu_avx2_aligned(float x, float y, float z,
                                               float& bx, float& by, float& bz) {
    float in0 = (x - nn_in_mean[0]) / nn_in_std[0];
    float in1 = (y - nn_in_mean[1]) / nn_in_std[1];
    float in2 = (z - nn_in_mean[2]) / nn_in_std[2];

    __m256 vin0 = _mm256_set1_ps(in0);
    __m256 vin1 = _mm256_set1_ps(in1);
    __m256 vin2 = _mm256_set1_ps(in2);
    __m256 vzero = _mm256_setzero_ps();

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();

    for (int blk = 0; blk < 4; ++blk) {
        int off = blk * 8;
        // Contiguous aligned loads — zero gather overhead
        __m256 vw0 = _mm256_load_ps(nn_w0_t[0] + off);
        __m256 vw1 = _mm256_load_ps(nn_w0_t[1] + off);
        __m256 vw2 = _mm256_load_ps(nn_w0_t[2] + off);
        __m256 vb  = _mm256_load_ps(nn_b0_a + off);

        __m256 vh = _mm256_fmadd_ps(vw0, vin0, vb);
        vh = _mm256_fmadd_ps(vw1, vin1, vh);
        vh = _mm256_fmadd_ps(vw2, vin2, vh);
        vh = _mm256_max_ps(vh, vzero);  // ReLU

        // Fuse output accumulation — multiply h by output weights immediately
        // This avoids storing h to memory and reloading it
        acc0 = _mm256_fmadd_ps(_mm256_load_ps(nn_w2_a[0] + off), vh, acc0);
        acc1 = _mm256_fmadd_ps(_mm256_load_ps(nn_w2_a[1] + off), vh, acc1);
        acc2 = _mm256_fmadd_ps(_mm256_load_ps(nn_w2_a[2] + off), vh, acc2);
    }

    auto hsum = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    };

    bx = (hsum(acc0) + nn_b2[0]) * nn_out_std[0] + nn_out_mean[0];
    by = (hsum(acc1) + nn_b2[1]) * nn_out_std[1] + nn_out_mean[1];
    bz = (hsum(acc2) + nn_b2[2]) * nn_out_std[2] + nn_out_mean[2];
}

// ─── OPT 7: AVX2 aligned batch-6 — the ultimate version ────────────────────
// Combines: pre-transposed aligned weights + AVX2 FMA + batch-6 weight reuse
// + fused output accumulation (no intermediate h buffer store).
static inline void nn_field_relu_avx2_aligned_batch6(
    const float inputs[6][3], float outputs[6][3]) {
    float inv_std0 = 1.0f / nn_in_std[0];
    float inv_std1 = 1.0f / nn_in_std[1];
    float inv_std2 = 1.0f / nn_in_std[2];

    __m256 vzero = _mm256_setzero_ps();

    // Normalise and broadcast
    __m256 vin0[6], vin1[6], vin2[6];
    for (int b = 0; b < 6; ++b) {
        float n0 = (inputs[b][0] - nn_in_mean[0]) * inv_std0;
        float n1 = (inputs[b][1] - nn_in_mean[1]) * inv_std1;
        float n2 = (inputs[b][2] - nn_in_mean[2]) * inv_std2;
        vin0[b] = _mm256_set1_ps(n0);
        vin1[b] = _mm256_set1_ps(n1);
        vin2[b] = _mm256_set1_ps(n2);
    }

    // Per-batch output accumulators
    __m256 acc[6][3];
    for (int b = 0; b < 6; ++b)
        acc[b][0] = acc[b][1] = acc[b][2] = _mm256_setzero_ps();

    // Process 8 neurons at a time — weights loaded once per block
    for (int blk = 0; blk < 4; ++blk) {
        int off = blk * 8;
        __m256 vw0 = _mm256_load_ps(nn_w0_t[0] + off);
        __m256 vw1 = _mm256_load_ps(nn_w0_t[1] + off);
        __m256 vw2 = _mm256_load_ps(nn_w0_t[2] + off);
        __m256 vb  = _mm256_load_ps(nn_b0_a + off);
        __m256 vout0 = _mm256_load_ps(nn_w2_a[0] + off);
        __m256 vout1 = _mm256_load_ps(nn_w2_a[1] + off);
        __m256 vout2 = _mm256_load_ps(nn_w2_a[2] + off);

        for (int b = 0; b < 6; ++b) {
            __m256 vh = _mm256_fmadd_ps(vw0, vin0[b], vb);
            vh = _mm256_fmadd_ps(vw1, vin1[b], vh);
            vh = _mm256_fmadd_ps(vw2, vin2[b], vh);
            vh = _mm256_max_ps(vh, vzero);

            // Fused output accumulation — no store to h_buf
            acc[b][0] = _mm256_fmadd_ps(vout0, vh, acc[b][0]);
            acc[b][1] = _mm256_fmadd_ps(vout1, vh, acc[b][1]);
            acc[b][2] = _mm256_fmadd_ps(vout2, vh, acc[b][2]);
        }
    }

    auto hsum = [](__m256 v) -> float {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        return _mm_cvtss_f32(s);
    };

    for (int b = 0; b < 6; ++b) {
        outputs[b][0] = (hsum(acc[b][0]) + nn_b2[0]) * nn_out_std[0] + nn_out_mean[0];
        outputs[b][1] = (hsum(acc[b][1]) + nn_b2[1]) * nn_out_std[1] + nn_out_mean[1];
        outputs[b][2] = (hsum(acc[b][2]) + nn_b2[2]) * nn_out_std[2] + nn_out_mean[2];
    }
}

// ─── RK derivative evaluation (Lorentz force) ───────────────────────────────
// Matches TrackRungeKuttaExtrapolator::evaluateDerivatives exactly
struct Deriv { double dtx; double dty; };

static inline Deriv eval_deriv(double tx, double ty, double qop,
                                float Bx, float By, float Bz) {
    double tx2 = tx * tx;
    double ty2 = ty * ty;
    double norm = std::sqrt(1.0 + tx2 + ty2);
    double Ax = norm * (ty * (tx * Bx + Bz) - (1.0 + tx2) * By);
    double Ay = norm * (-tx * (ty * By + Bz) + (1.0 + ty2) * Bx);
    return {qop * Ax, qop * Ay};
}

// ─── CashKarp step emulation ─────────────────────────────────────────────────
// Emulates one full CashKarp step: 6 stages, each with field lookup + deriv
// Uses trilinear interpolation for field lookups
static inline void cashkarp_step_trilinear(
    double x, double y, double z, double tx, double ty, double qop, double dz) {
    // CashKarp Butcher c coefficients
    static constexpr double c[6] = {0.0, 1.0/5, 3.0/10, 3.0/5, 1.0, 7.0/8};
    float bx, by, bz;
    Deriv k[6];
    for (int s = 0; s < 6; ++s) {
        double zi = z + c[s] * dz;
        double xi = x + c[s] * dz * tx;  // simplified intermediate position
        double yi = y + c[s] * dz * ty;
        trilinear(xi, yi, zi, bx, by, bz);
        k[s] = eval_deriv(tx, ty, qop, bx, by, bz);
    }
    // Consume k to prevent optimizer from eliminating the computation
    volatile double sink = k[0].dtx + k[5].dty;
    (void)sink;
}

// Same but using NN field lookup
static inline void cashkarp_step_nn(
    double x, double y, double z, double tx, double ty, double qop, double dz) {
    static constexpr double c[6] = {0.0, 1.0/5, 3.0/10, 3.0/5, 1.0, 7.0/8};
    float bx, by, bz;
    Deriv k[6];
    for (int s = 0; s < 6; ++s) {
        double zi = z + c[s] * dz;
        double xi = x + c[s] * dz * tx;
        double yi = y + c[s] * dz * ty;
        nn_field(float(xi), float(yi), float(zi), bx, by, bz);
        k[s] = eval_deriv(tx, ty, qop, bx, by, bz);
    }
    volatile double sink = k[0].dtx + k[5].dty;
    (void)sink;
}

// Using nn2 (128,128) field lookup
static inline void cashkarp_step_nn2(
    double x, double y, double z, double tx, double ty, double qop, double dz) {
    static constexpr double c[6] = {0.0, 1.0/5, 3.0/10, 3.0/5, 1.0, 7.0/8};
    float bx, by, bz;
    Deriv k[6];
    for (int s = 0; s < 6; ++s) {
        double zi = z + c[s] * dz;
        double xi = x + c[s] * dz * tx;
        double yi = y + c[s] * dz * ty;
        nn2_field(float(xi), float(yi), float(zi), bx, by, bz);
        k[s] = eval_deriv(tx, ty, qop, bx, by, bz);
    }
    volatile double sink = k[0].dtx + k[5].dty;
    (void)sink;
}

// Using nn_field_relu [32] ReLU field lookup
static inline void cashkarp_step_nn_relu(
    double x, double y, double z, double tx, double ty, double qop, double dz) {
    static constexpr double c[6] = {0.0, 1.0/5, 3.0/10, 3.0/5, 1.0, 7.0/8};
    float bx, by, bz;
    Deriv k[6];
    for (int s = 0; s < 6; ++s) {
        double zi = z + c[s] * dz;
        double xi = x + c[s] * dz * tx;
        double yi = y + c[s] * dz * ty;
        nn_field_relu(float(xi), float(yi), float(zi), bx, by, bz);
        k[s] = eval_deriv(tx, ty, qop, bx, by, bz);
    }
    volatile double sink2 = k[0].dtx + k[5].dty;
    (void)sink2;
}

// Using nn3 [32,32] ReLU field lookup
static inline void cashkarp_step_nn3(
    double x, double y, double z, double tx, double ty, double qop, double dz) {
    static constexpr double c[6] = {0.0, 1.0/5, 3.0/10, 3.0/5, 1.0, 7.0/8};
    float bx, by, bz;
    Deriv k[6];
    for (int s = 0; s < 6; ++s) {
        double zi = z + c[s] * dz;
        double xi = x + c[s] * dz * tx;
        double yi = y + c[s] * dz * ty;
        nn3_field(float(xi), float(yi), float(zi), bx, by, bz);
        k[s] = eval_deriv(tx, ty, qop, bx, by, bz);
    }
    volatile double sink3 = k[0].dtx + k[5].dty;
    (void)sink3;
}

// ─── CashKarp step with batched/optimised NN field ──────────────────────────
// All 6 field lookups computed at once, then derivatives computed sequentially.
// This is the key insight: the 6 NN field evaluations are INDEPENDENT.
// Only the derivative accumulation (Butcher tableau) is sequential.

static inline void cashkarp_step_nn_relu_batch6(
    double x, double y, double z, double tx, double ty, double qop, double dz) {
    static constexpr double c[6] = {0.0, 1.0/5, 3.0/10, 3.0/5, 1.0, 7.0/8};

    float inputs[6][3];
    for (int s = 0; s < 6; ++s) {
        inputs[s][0] = float(x + c[s] * dz * tx);
        inputs[s][1] = float(y + c[s] * dz * ty);
        inputs[s][2] = float(z + c[s] * dz);
    }

    float B[6][3];
    nn_field_relu_batch6(inputs, B);

    Deriv k[6];
    for (int s = 0; s < 6; ++s) {
        k[s] = eval_deriv(tx, ty, qop, B[s][0], B[s][1], B[s][2]);
    }
    volatile double sink = k[0].dtx + k[5].dty;
    (void)sink;
}

static inline void cashkarp_step_nn_relu_avx2(
    double x, double y, double z, double tx, double ty, double qop, double dz) {
    static constexpr double c[6] = {0.0, 1.0/5, 3.0/10, 3.0/5, 1.0, 7.0/8};
    float lbx, lby, lbz;
    Deriv k[6];
    for (int s = 0; s < 6; ++s) {
        float xi = float(x + c[s] * dz * tx);
        float yi = float(y + c[s] * dz * ty);
        float zi = float(z + c[s] * dz);
        nn_field_relu_avx2(xi, yi, zi, lbx, lby, lbz);
        k[s] = eval_deriv(tx, ty, qop, lbx, lby, lbz);
    }
    volatile double sink = k[0].dtx + k[5].dty;
    (void)sink;
}

static inline void cashkarp_step_nn_relu_avx2_aligned(
    double x, double y, double z, double tx, double ty, double qop, double dz) {
    static constexpr double c[6] = {0.0, 1.0/5, 3.0/10, 3.0/5, 1.0, 7.0/8};
    float lbx, lby, lbz;
    Deriv k[6];
    for (int s = 0; s < 6; ++s) {
        float xi = float(x + c[s] * dz * tx);
        float yi = float(y + c[s] * dz * ty);
        float zi = float(z + c[s] * dz);
        nn_field_relu_avx2_aligned(xi, yi, zi, lbx, lby, lbz);
        k[s] = eval_deriv(tx, ty, qop, lbx, lby, lbz);
    }
    volatile double sink = k[0].dtx + k[5].dty;
    (void)sink;
}

static inline void cashkarp_step_nn_relu_avx2_batch6(
    double x, double y, double z, double tx, double ty, double qop, double dz) {
    static constexpr double c[6] = {0.0, 1.0/5, 3.0/10, 3.0/5, 1.0, 7.0/8};

    float inputs[6][3];
    for (int s = 0; s < 6; ++s) {
        inputs[s][0] = float(x + c[s] * dz * tx);
        inputs[s][1] = float(y + c[s] * dz * ty);
        inputs[s][2] = float(z + c[s] * dz);
    }

    float B[6][3];
    nn_field_relu_avx2_aligned_batch6(inputs, B);

    Deriv k[6];
    for (int s = 0; s < 6; ++s) {
        k[s] = eval_deriv(tx, ty, qop, B[s][0], B[s][1], B[s][2]);
    }
    volatile double sink = k[0].dtx + k[5].dty;
    (void)sink;
}

static inline void cashkarp_step_nn_relu_unrolled(
    double x, double y, double z, double tx, double ty, double qop, double dz) {
    static constexpr double c[6] = {0.0, 1.0/5, 3.0/10, 3.0/5, 1.0, 7.0/8};
    float lbx, lby, lbz;
    Deriv k[6];
    for (int s = 0; s < 6; ++s) {
        float xi = float(x + c[s] * dz * tx);
        float yi = float(y + c[s] * dz * ty);
        float zi = float(z + c[s] * dz);
        nn_field_relu_unrolled(xi, yi, zi, lbx, lby, lbz);
        k[s] = eval_deriv(tx, ty, qop, lbx, lby, lbz);
    }
    volatile double sink = k[0].dtx + k[5].dty;
    (void)sink;
}

// ─── Cache pollution buffer ──────────────────────────────────────────────────
// Used to evict the field-map grid from cache between trilinear calls,
// simulating realistic production conditions where other reconstruction
// code competes for cache space.
static constexpr size_t POLLUTE_SIZE = 32 * 1024 * 1024;  // 32 MB > L3/CCX (8 MB)
static char* pollute_buf = nullptr;

static void init_pollute_buf() {
    pollute_buf = new char[POLLUTE_SIZE];
    std::memset(pollute_buf, 1, POLLUTE_SIZE);
}

// Read through the entire pollution buffer to flush all cache levels.
// Returns a volatile sum to prevent dead-code elimination.
static inline int pollute_cache() {
    volatile int sum = 0;
    // Stride by cache line (64 bytes) to touch every line without wasting time
    for (size_t i = 0; i < POLLUTE_SIZE; i += 64) {
        sum += pollute_buf[i];
    }
    return sum;
}

// ─── Benchmark harness ──────────────────────────────────────────────────────
using Clock = std::chrono::high_resolution_clock;

template<typename F>
double bench(F&& func, int N) {
    // Warmup
    for (int i = 0; i < N/10 + 100; ++i) func(i);
    // Timed run
    auto t0 = Clock::now();
    for (int i = 0; i < N; ++i) func(i);
    auto t1 = Clock::now();
    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
    return ns / N;
}

// Benchmark with cache pollution: calls pollute_cache() between each
// invocation, but times ONLY the func() call itself.
// This gives the cost of func() with a cold cache (grid evicted).
template<typename F>
double bench_cold(F&& func, int N) {
    // Warmup
    for (int i = 0; i < 50; ++i) { pollute_cache(); func(i); }

    // Timed run: measure each call individually after cache flush
    double total_ns = 0.0;
    for (int i = 0; i < N; ++i) {
        pollute_cache();                          // evict grid from cache
        auto t0 = Clock::now();
        func(i);
        auto t1 = Clock::now();
        total_ns += std::chrono::duration<double, std::nano>(t1 - t0).count();
    }
    return total_ns / N;
}

int main(int argc, char** argv) {
    int N = 100000;
    if (argc > 1) N = std::atoi(argv[1]);

    // Initialise
    fill_grid();
    init_nn_weights();
    init_nn2_weights();
    init_nn3_weights();
    init_nn_weights_transposed();
    init_pollute_buf();

    // Random query points inside the grid volume
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dx(X_MIN+200, X_MAX-200);
    std::uniform_real_distribution<double> dy(Y_MIN+200, Y_MAX-200);
    std::uniform_real_distribution<double> dz_dist(Z_MIN+200, Z_MAX-200);
    std::vector<double> xs(N), ys(N), zs(N);
    for (int i = 0; i < N; ++i) { xs[i] = dx(gen); ys[i] = dy(gen); zs[i] = dz_dist(gen); }

    // Track parameters for derivative evaluation
    double tx = 0.15, ty = -0.1, qop = 2e-4;  // 5 GeV track
    double step_dz = 400.0;  // typical adaptive step size in mm

    fprintf(stdout, "component,time_ns,iterations\n");

    // 1. Trilinear interpolation
    float bx, by, bz;
    double t_trilin = bench([&](int i){ trilinear(xs[i%N], ys[i%N], zs[i%N], bx, by, bz); }, N);
    fprintf(stdout, "trilinear_interp,%.2f,%d\n", t_trilin, N);

    // 2. NN [32] SiLU inference
    double t_nn = bench([&](int i){ nn_field(float(xs[i%N]), float(ys[i%N]), float(zs[i%N]), bx, by, bz); }, N);
    fprintf(stdout, "nn_1L_32H_silu,%.2f,%d\n", t_nn, N);

    // 3. NN [128,128] SiLU inference
    double t_nn2 = bench([&](int i){ nn2_field(float(xs[i%N]), float(ys[i%N]), float(zs[i%N]), bx, by, bz); }, N);
    fprintf(stdout, "nn_2L_128H_silu,%.2f,%d\n", t_nn2, N);

    // 3b. NN [32] ReLU inference
    double t_nn_relu = bench([&](int i){ nn_field_relu(float(xs[i%N]), float(ys[i%N]), float(zs[i%N]), bx, by, bz); }, N);
    fprintf(stdout, "nn_1L_32H_relu,%.2f,%d\n", t_nn_relu, N);

    // 3c. NN [32,32] ReLU inference
    double t_nn3 = bench([&](int i){ nn3_field(float(xs[i%N]), float(ys[i%N]), float(zs[i%N]), bx, by, bz); }, N);
    fprintf(stdout, "nn_2L_32H_relu,%.2f,%d\n", t_nn3, N);

    // ─── OPTIMISED NN [32] ReLU VARIANTS ───────────────────────────────

    // OPT: NN [32] ReLU AVX2 (single-call, gathered weights)
    double t_nn_relu_avx2 = bench([&](int i){ nn_field_relu_avx2(float(xs[i%N]), float(ys[i%N]), float(zs[i%N]), bx, by, bz); }, N);
    fprintf(stdout, "nn_1L_32H_relu_avx2,%.2f,%d\n", t_nn_relu_avx2, N);

    // OPT: NN [32] ReLU AVX2 aligned (pre-transposed weights)
    double t_nn_relu_avx2a = bench([&](int i){ nn_field_relu_avx2_aligned(float(xs[i%N]), float(ys[i%N]), float(zs[i%N]), bx, by, bz); }, N);
    fprintf(stdout, "nn_1L_32H_relu_avx2_aligned,%.2f,%d\n", t_nn_relu_avx2a, N);

    // OPT: NN [32] ReLU fully unrolled
    double t_nn_relu_unroll = bench([&](int i){ nn_field_relu_unrolled(float(xs[i%N]), float(ys[i%N]), float(zs[i%N]), bx, by, bz); }, N);
    fprintf(stdout, "nn_1L_32H_relu_unrolled,%.2f,%d\n", t_nn_relu_unroll, N);

    // OPT: NN [32] ReLU batch-6 (amortised over 6 calls, report per-call)
    double t_nn_relu_b6 = bench([&](int i){
        float inp[6][3], out[6][3];
        for (int s = 0; s < 6; ++s) {
            int j = (i*6+s) % N;
            inp[s][0] = float(xs[j]); inp[s][1] = float(ys[j]); inp[s][2] = float(zs[j]);
        }
        nn_field_relu_batch6(inp, out);
        volatile float sink_b6 = out[0][0] + out[5][2];  // prevent DCE
        (void)sink_b6;
    }, N);
    fprintf(stdout, "nn_1L_32H_relu_batch6,%.2f,%d\n", t_nn_relu_b6 / 6.0, N);  // per-call
    fprintf(stdout, "nn_1L_32H_relu_batch6_total,%.2f,%d\n", t_nn_relu_b6, N);   // total for 6

    // OPT: NN [32] ReLU AVX2 aligned batch-6
    double t_nn_relu_avx2b6 = bench([&](int i){
        float inp[6][3], out[6][3];
        for (int s = 0; s < 6; ++s) {
            int j = (i*6+s) % N;
            inp[s][0] = float(xs[j]); inp[s][1] = float(ys[j]); inp[s][2] = float(zs[j]);
        }
        nn_field_relu_avx2_aligned_batch6(inp, out);
        volatile float sink_b6 = out[0][0] + out[5][2];  // prevent DCE
        (void)sink_b6;
    }, N);
    fprintf(stdout, "nn_1L_32H_relu_avx2_batch6,%.2f,%d\n", t_nn_relu_avx2b6 / 6.0, N);  // per-call
    fprintf(stdout, "nn_1L_32H_relu_avx2_batch6_total,%.2f,%d\n", t_nn_relu_avx2b6, N);

    // 4. Derivative evaluation only (no field lookup)
    double t_deriv = bench([&](int i){
        volatile auto d = eval_deriv(tx+0.001*i/N, ty, qop, 100.0f, -33000.0f, 50.0f);
        (void)d;
    }, N);
    fprintf(stdout, "rk_derivative,%.2f,%d\n", t_deriv, N);

    // 5. Single CashKarp stage = trilinear + derivative
    double t_stage = bench([&](int i){
        float lbx, lby, lbz;
        trilinear(xs[i%N], ys[i%N], zs[i%N], lbx, lby, lbz);
        volatile auto d = eval_deriv(tx, ty, qop, lbx, lby, lbz);
        (void)d;
    }, N);
    fprintf(stdout, "ck_single_stage_trilinear,%.2f,%d\n", t_stage, N);

    // 6. Full CashKarp step (6 stages) with trilinear
    double t_ck_trilin = bench([&](int i){
        cashkarp_step_trilinear(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, N);
    fprintf(stdout, "ck_full_step_trilinear,%.2f,%d\n", t_ck_trilin, N);

    // 7. Full CashKarp step with NN [32] field
    double t_ck_nn = bench([&](int i){
        cashkarp_step_nn(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, N);
    fprintf(stdout, "ck_full_step_nn_1L_32H,%.2f,%d\n", t_ck_nn, N);

    // 8. Full CashKarp step with NN [128,128] field
    double t_ck_nn2 = bench([&](int i){
        cashkarp_step_nn2(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, N);
    fprintf(stdout, "ck_full_step_nn_2L_128H,%.2f,%d\n", t_ck_nn2, N);

    // 8b. Full CashKarp step with NN [32] ReLU
    double t_ck_nn_relu = bench([&](int i){
        cashkarp_step_nn_relu(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, N);
    fprintf(stdout, "ck_full_step_nn_1L_32H_relu,%.2f,%d\n", t_ck_nn_relu, N);

    // 8c. Full CashKarp step with NN [32,32] ReLU
    double t_ck_nn3 = bench([&](int i){
        cashkarp_step_nn3(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, N);
    fprintf(stdout, "ck_full_step_nn_2L_32H_relu,%.2f,%d\n", t_ck_nn3, N);

    // ─── OPTIMISED CashKarp full step variants ─────────────────────────

    // OPT: CK step with batch-6 NN ReLU
    double t_ck_b6 = bench([&](int i){
        cashkarp_step_nn_relu_batch6(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, N);
    fprintf(stdout, "ck_full_step_nn_relu_batch6,%.2f,%d\n", t_ck_b6, N);

    // OPT: CK step with AVX2 NN ReLU (sequential 6 calls)
    double t_ck_avx2 = bench([&](int i){
        cashkarp_step_nn_relu_avx2(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, N);
    fprintf(stdout, "ck_full_step_nn_relu_avx2,%.2f,%d\n", t_ck_avx2, N);

    // OPT: CK step with AVX2 aligned NN ReLU (sequential 6 calls)
    double t_ck_avx2a = bench([&](int i){
        cashkarp_step_nn_relu_avx2_aligned(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, N);
    fprintf(stdout, "ck_full_step_nn_relu_avx2_aligned,%.2f,%d\n", t_ck_avx2a, N);

    // OPT: CK step with AVX2 batch-6 (the "everything" variant)
    double t_ck_avx2b6 = bench([&](int i){
        cashkarp_step_nn_relu_avx2_batch6(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, N);
    fprintf(stdout, "ck_full_step_nn_relu_avx2_batch6,%.2f,%d\n", t_ck_avx2b6, N);

    // OPT: CK step with unrolled NN ReLU
    double t_ck_unroll = bench([&](int i){
        cashkarp_step_nn_relu_unrolled(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, N);
    fprintf(stdout, "ck_full_step_nn_relu_unrolled,%.2f,%d\n", t_ck_unroll, N);

    // 9. Noop (measure overhead)
    volatile double sink = 0;
    double t_noop = bench([&](int i){ sink += i; }, N);
    fprintf(stdout, "noop_baseline,%.2f,%d\n", t_noop, N);

    // 10. std::sqrt (for reference)
    double t_sqrt = bench([&](int i){
        volatile double v = std::sqrt(1.0 + 0.01*(i%1000));
        (void)v;
    }, N);
    fprintf(stdout, "std_sqrt,%.2f,%d\n", t_sqrt, N);

    // 11. std::exp (for reference, this is inside SiLU)
    double t_exp = bench([&](int i){
        volatile float v = std::exp(-0.5f*(i%1000)/1000.0f);
        (void)v;
    }, N);
    fprintf(stdout, "std_exp,%.2f,%d\n", t_exp, N);

    // 12. Emulate full track extrapolation (10 CashKarp steps = ~4m at 400mm/step)
    int nsteps = 10;
    double t_full_trilin = bench([&](int i){
        double lx = xs[i%N], ly = ys[i%N], lz = 3000.0;
        for (int s = 0; s < nsteps; ++s) {
            cashkarp_step_trilinear(lx, ly, lz, tx, ty, qop, step_dz);
            lz += step_dz;
        }
    }, N/10);
    fprintf(stdout, "full_track_10steps_trilinear,%.2f,%d\n", t_full_trilin, N/10);

    double t_full_nn = bench([&](int i){
        double lx = xs[i%N], ly = ys[i%N], lz = 3000.0;
        for (int s = 0; s < nsteps; ++s) {
            cashkarp_step_nn(lx, ly, lz, tx, ty, qop, step_dz);
            lz += step_dz;
        }
    }, N/10);
    fprintf(stdout, "full_track_10steps_nn_1L_32H,%.2f,%d\n", t_full_nn, N/10);

    double t_full_nn2 = bench([&](int i){
        double lx = xs[i%N], ly = ys[i%N], lz = 3000.0;
        for (int s = 0; s < nsteps; ++s) {
            cashkarp_step_nn2(lx, ly, lz, tx, ty, qop, step_dz);
            lz += step_dz;
        }
    }, N/10);
    fprintf(stdout, "full_track_10steps_nn_2L_128H,%.2f,%d\n", t_full_nn2, N/10);

    // Full track with NN [32] ReLU
    double t_full_nn_relu = bench([&](int i){
        double lx = xs[i%N], ly = ys[i%N], lz = 3000.0;
        for (int s = 0; s < nsteps; ++s) {
            cashkarp_step_nn_relu(lx, ly, lz, tx, ty, qop, step_dz);
            lz += step_dz;
        }
    }, N/10);
    fprintf(stdout, "full_track_10steps_nn_1L_32H_relu,%.2f,%d\n", t_full_nn_relu, N/10);

    // Full track with NN [32,32] ReLU
    double t_full_nn3 = bench([&](int i){
        double lx = xs[i%N], ly = ys[i%N], lz = 3000.0;
        for (int s = 0; s < nsteps; ++s) {
            cashkarp_step_nn3(lx, ly, lz, tx, ty, qop, step_dz);
            lz += step_dz;
        }
    }, N/10);
    fprintf(stdout, "full_track_10steps_nn_2L_32H_relu,%.2f,%d\n", t_full_nn3, N/10);

    // OPT: Full track with batch-6 NN ReLU
    double t_full_b6 = bench([&](int i){
        double lx = xs[i%N], ly = ys[i%N], lz = 3000.0;
        for (int s = 0; s < nsteps; ++s) {
            cashkarp_step_nn_relu_batch6(lx, ly, lz, tx, ty, qop, step_dz);
            lz += step_dz;
        }
    }, N/10);
    fprintf(stdout, "full_track_10steps_nn_relu_batch6,%.2f,%d\n", t_full_b6, N/10);

    // OPT: Full track with AVX2 batch-6 NN ReLU
    double t_full_avx2b6 = bench([&](int i){
        double lx = xs[i%N], ly = ys[i%N], lz = 3000.0;
        for (int s = 0; s < nsteps; ++s) {
            cashkarp_step_nn_relu_avx2_batch6(lx, ly, lz, tx, ty, qop, step_dz);
            lz += step_dz;
        }
    }, N/10);
    fprintf(stdout, "full_track_10steps_nn_relu_avx2_batch6,%.2f,%d\n", t_full_avx2b6, N/10);

    // OPT: Full track with AVX2 aligned NN ReLU (sequential)
    double t_full_avx2a = bench([&](int i){
        double lx = xs[i%N], ly = ys[i%N], lz = 3000.0;
        for (int s = 0; s < nsteps; ++s) {
            cashkarp_step_nn_relu_avx2_aligned(lx, ly, lz, tx, ty, qop, step_dz);
            lz += step_dz;
        }
    }, N/10);
    fprintf(stdout, "full_track_10steps_nn_relu_avx2_aligned,%.2f,%d\n", t_full_avx2a, N/10);

    // OPT: Full track with unrolled NN ReLU
    double t_full_unroll = bench([&](int i){
        double lx = xs[i%N], ly = ys[i%N], lz = 3000.0;
        for (int s = 0; s < nsteps; ++s) {
            cashkarp_step_nn_relu_unrolled(lx, ly, lz, tx, ty, qop, step_dz);
            lz += step_dz;
        }
    }, N/10);
    fprintf(stdout, "full_track_10steps_nn_relu_unrolled,%.2f,%d\n", t_full_unroll, N/10);

    // ─── Cache-cold benchmarks ──────────────────────────────────────────
    // These simulate production conditions where the 11 MB grid is NOT
    // resident in cache (evicted by other reconstruction code between calls).
    // We flush all cache levels with a 32 MB buffer read between each call.
    int Ncold = std::min(N/10, 5000);  // fewer iterations (each takes ~ms with pollution)

    // Cache-cold trilinear
    double t_trilin_cold = bench_cold([&](int i){
        trilinear(xs[i%N], ys[i%N], zs[i%N], bx, by, bz);
    }, Ncold);
    fprintf(stdout, "trilinear_interp_cold,%.2f,%d\n", t_trilin_cold, Ncold);

    // Cache-cold NN [32] (should be unaffected — weights are 0.9 KB, fit in L1)
    double t_nn_cold = bench_cold([&](int i){
        nn_field(float(xs[i%N]), float(ys[i%N]), float(zs[i%N]), bx, by, bz);
    }, Ncold);
    fprintf(stdout, "nn_1L_32H_silu_cold,%.2f,%d\n", t_nn_cold, Ncold);

    // Cache-cold full CashKarp step with trilinear
    double t_ck_trilin_cold = bench_cold([&](int i){
        cashkarp_step_trilinear(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, Ncold);
    fprintf(stdout, "ck_full_step_trilinear_cold,%.2f,%d\n", t_ck_trilin_cold, Ncold);

    // Cache-cold full CashKarp step with NN [32]
    double t_ck_nn_cold = bench_cold([&](int i){
        cashkarp_step_nn(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, Ncold);
    fprintf(stdout, "ck_full_step_nn_1L_32H_cold,%.2f,%d\n", t_ck_nn_cold, Ncold);

    // Cache-cold NN [32] ReLU inference
    double t_nn_relu_cold = bench_cold([&](int i){
        nn_field_relu(float(xs[i%N]), float(ys[i%N]), float(zs[i%N]), bx, by, bz);
    }, Ncold);
    fprintf(stdout, "nn_1L_32H_relu_cold,%.2f,%d\n", t_nn_relu_cold, Ncold);

    // Cache-cold NN [32,32] ReLU inference
    double t_nn3_cold = bench_cold([&](int i){
        nn3_field(float(xs[i%N]), float(ys[i%N]), float(zs[i%N]), bx, by, bz);
    }, Ncold);
    fprintf(stdout, "nn_2L_32H_relu_cold,%.2f,%d\n", t_nn3_cold, Ncold);

    // Cache-cold CK step with NN [32] ReLU
    double t_ck_nn_relu_cold = bench_cold([&](int i){
        cashkarp_step_nn_relu(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, Ncold);
    fprintf(stdout, "ck_full_step_nn_1L_32H_relu_cold,%.2f,%d\n", t_ck_nn_relu_cold, Ncold);

    // Cache-cold CK step with NN [32,32] ReLU
    double t_ck_nn3_cold = bench_cold([&](int i){
        cashkarp_step_nn3(xs[i%N], ys[i%N], zs[i%N], tx, ty, qop, step_dz);
    }, Ncold);
    fprintf(stdout, "ck_full_step_nn_2L_32H_relu_cold,%.2f,%d\n", t_ck_nn3_cold, Ncold);

    // Cache-cold full track (trilinear)
    double t_full_trilin_cold = bench_cold([&](int i){
        double lx = xs[i%N], ly = ys[i%N], lz = 3000.0;
        for (int s = 0; s < nsteps; ++s) {
            cashkarp_step_trilinear(lx, ly, lz, tx, ty, qop, step_dz);
            lz += step_dz;
        }
    }, Ncold / 5);
    fprintf(stdout, "full_track_10steps_trilinear_cold,%.2f,%d\n", t_full_trilin_cold, Ncold/5);

    // Cache-cold full track (NN [32])
    double t_full_nn_cold = bench_cold([&](int i){
        double lx = xs[i%N], ly = ys[i%N], lz = 3000.0;
        for (int s = 0; s < nsteps; ++s) {
            cashkarp_step_nn(lx, ly, lz, tx, ty, qop, step_dz);
            lz += step_dz;
        }
    }, Ncold / 5);
    fprintf(stdout, "full_track_10steps_nn_1L_32H_cold,%.2f,%d\n", t_full_nn_cold, Ncold/5);

    // Clean up
    delete[] pollute_buf;

    return 0;
}
