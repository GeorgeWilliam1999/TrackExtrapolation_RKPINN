#pragma once
// =============================================================================
// FieldMapNNReLU -- Raw C inference for [32] relu network
// Parameters: 227, FLOPs: 416
// Weight memory: 0.9 KB
// Field units: Gauss (same as twodip.rtf)
//
// Trained ReLU [32] model: field_nn_relu_1L_32H_log_space (200 epochs, MAE=0.58G)
// =============================================================================
#include <algorithm>  // std::max
#include <immintrin.h> // AVX2 intrinsics

namespace LHCb::FieldNNReLU {

// --- Row-major weights (32x3) for scalar evaluation ---
alignas(32) static constexpr float nn_net_0_weight[96] = {
   0.00020931f,  0.50834560f, -0.26127818f,  0.00654700f, -0.23641640f,  0.69892234f, -0.14622632f,  0.13053103f,
   0.21513081f, -0.00142998f,  0.21394065f,  0.30076745f,  0.00073086f,  0.31068990f, -0.29031965f,  0.28556794f,
   0.05260484f,  0.53892076f,  0.15442370f,  0.22901873f, -0.03492390f, -0.09837230f, -0.10437506f,  0.61267126f,
   0.10705148f,  0.20012955f,  0.22367023f, -0.00526302f,  0.22082677f,  0.44769013f,  0.15810645f,  0.30524850f,
  -0.01331578f, -0.40296316f, -0.02654241f, -0.21413609f,  0.33683375f, -0.11548026f, -0.31519157f,  0.00746151f,
   0.17844453f,  0.11926135f,  0.11170960f, -0.10880635f,  0.63487041f, -0.28048915f,  0.04874155f,  0.51730049f,
  -0.10551948f, -0.25090417f, -0.84493536f,  0.29409039f, -0.05288889f,  0.53900975f, -0.12182222f,  0.13960102f,
   0.72300237f, -0.28000379f,  0.22782840f,  0.11033140f, -0.79043114f,  0.06903400f, -0.41217840f, -0.12242242f,
  -0.26268566f,  0.04942993f, -0.13056892f,  0.14346728f,  0.09486693f, -0.43075597f,  0.23261443f, -0.01728059f,
  -0.03912291f, -0.41553295f, -0.14722218f, -0.11737398f, -0.24121706f, -0.11709028f, -0.50764215f,  0.31425679f,
   0.17292041f, -0.14265996f, -0.16680133f, -0.88132012f, -0.23389155f,  0.10296381f,  0.48345581f,  0.34696430f,
   0.06773844f, -0.57889014f, -0.07228037f,  0.00436828f, -0.35809711f, -0.09517280f,  0.18109605f,  0.63137066f,
};

alignas(32) static constexpr float nn_net_0_bias[32] = {
  -0.76773810f,  0.03433918f,  0.08085171f,  0.40807828f, -0.63983190f,  0.21286280f,  0.37988815f, -0.05887919f,
   0.60604584f,  0.58338535f,  0.41918975f, -0.09595847f, -0.15190750f,  0.35113135f, -0.05540320f,  0.20840876f,
  -0.04564619f,  0.21034686f, -0.02581194f,  0.35194996f, -0.19153549f,  0.65244901f,  0.58595347f, -0.50370520f,
  -0.63158065f, -0.48004577f,  0.64468944f,  0.04868887f,  0.18774731f, -0.22354369f,  0.72891366f,  0.03006827f,
};

alignas(32) static constexpr float nn_net_2_weight[96] = {
  -0.01788078f, -0.01254037f,  0.09961220f, -0.00017342f,  0.00967762f,  0.21138234f, -0.17334203f, -0.07049391f,
   0.23563379f, -0.02214710f,  0.06135469f,  0.31693175f, -0.14085521f, -0.08982719f,  0.09503601f, -0.14878125f,
   0.02744759f, -0.19310015f,  0.11454522f, -0.16753201f, -0.18224949f,  0.14239776f, -0.07782006f,  0.09831212f,
   0.06644019f, -0.26609024f,  0.18697160f, -0.07842058f, -0.04149115f,  0.20541987f,  0.06513534f, -0.08792736f,
   0.15387128f,  0.05261374f,  0.01482316f,  0.14400955f, -0.24356773f,  0.01033242f, -0.11960935f, -0.03468945f,
   0.15570527f, -0.13108425f,  0.09838774f, -0.01059496f, -0.00026339f, -0.36230481f, -0.03765261f,  0.05478710f,
   0.03356842f,  0.02271971f, -0.06115111f,  0.04879285f,  0.00978004f, -0.03305691f,  0.15384698f, -0.01083772f,
   0.18358327f, -0.22113580f, -0.03416439f, -0.03636909f, -0.04350078f,  0.01060732f,  0.02663256f,  0.06845409f,
  -0.08385966f, -0.22009797f, -0.07482409f,  0.06657937f,  0.28074759f,  0.29009813f,  0.00495094f,  0.31180856f,
   0.17913230f, -0.19138078f,  0.00583839f,  0.02508154f,  0.04605021f, -0.38706866f,  0.29484037f,  0.14817497f,
   0.18567383f, -0.27456611f, -0.40605482f,  0.05108217f, -0.01572737f, -0.19957617f, -0.11180279f,  0.00324184f,
   0.12458245f, -0.13793343f, -0.03220843f, -0.30339295f,  0.11967878f, -0.20511355f,  0.38921833f,  0.27593538f,
};

alignas(32) static constexpr float nn_net_2_bias[3] = {
  -0.17744060f, -0.05229685f, -0.00571826f,
};

static constexpr float nn_input_mean[3] = {0.00000000f, 0.00000000f, 6749.99951172f};
static constexpr float nn_input_std[3] = {2338.09155273f, 2338.09155273f, 4214.56005859f};
static constexpr float nn_output_mean[3] = {0.00000001f, 0.12292606f, -0.00000000f};
static constexpr float nn_output_std[3] = {22.14661598f, 71.60929108f, 22.14662170f};

static constexpr int NN_HIDDEN = 32;

// --- Transposed input weights (3 x 32) for AVX2: contiguous across neurons ---
alignas(32) static constexpr float nn_net_0_weight_t[3][32] = {
  // input 0 (x) weights for all 32 neurons
  {nn_net_0_weight[0], nn_net_0_weight[3], nn_net_0_weight[6], nn_net_0_weight[9],
   nn_net_0_weight[12], nn_net_0_weight[15], nn_net_0_weight[18], nn_net_0_weight[21],
   nn_net_0_weight[24], nn_net_0_weight[27], nn_net_0_weight[30], nn_net_0_weight[33],
   nn_net_0_weight[36], nn_net_0_weight[39], nn_net_0_weight[42], nn_net_0_weight[45],
   nn_net_0_weight[48], nn_net_0_weight[51], nn_net_0_weight[54], nn_net_0_weight[57],
   nn_net_0_weight[60], nn_net_0_weight[63], nn_net_0_weight[66], nn_net_0_weight[69],
   nn_net_0_weight[72], nn_net_0_weight[75], nn_net_0_weight[78], nn_net_0_weight[81],
   nn_net_0_weight[84], nn_net_0_weight[87], nn_net_0_weight[90], nn_net_0_weight[93]},
  // input 1 (y) weights for all 32 neurons
  {nn_net_0_weight[1], nn_net_0_weight[4], nn_net_0_weight[7], nn_net_0_weight[10],
   nn_net_0_weight[13], nn_net_0_weight[16], nn_net_0_weight[19], nn_net_0_weight[22],
   nn_net_0_weight[25], nn_net_0_weight[28], nn_net_0_weight[31], nn_net_0_weight[34],
   nn_net_0_weight[37], nn_net_0_weight[40], nn_net_0_weight[43], nn_net_0_weight[46],
   nn_net_0_weight[49], nn_net_0_weight[52], nn_net_0_weight[55], nn_net_0_weight[58],
   nn_net_0_weight[61], nn_net_0_weight[64], nn_net_0_weight[67], nn_net_0_weight[70],
   nn_net_0_weight[73], nn_net_0_weight[76], nn_net_0_weight[79], nn_net_0_weight[82],
   nn_net_0_weight[85], nn_net_0_weight[88], nn_net_0_weight[91], nn_net_0_weight[94]},
  // input 2 (z) weights for all 32 neurons
  {nn_net_0_weight[2], nn_net_0_weight[5], nn_net_0_weight[8], nn_net_0_weight[11],
   nn_net_0_weight[14], nn_net_0_weight[17], nn_net_0_weight[20], nn_net_0_weight[23],
   nn_net_0_weight[26], nn_net_0_weight[29], nn_net_0_weight[32], nn_net_0_weight[35],
   nn_net_0_weight[38], nn_net_0_weight[41], nn_net_0_weight[44], nn_net_0_weight[47],
   nn_net_0_weight[50], nn_net_0_weight[53], nn_net_0_weight[56], nn_net_0_weight[59],
   nn_net_0_weight[62], nn_net_0_weight[65], nn_net_0_weight[68], nn_net_0_weight[71],
   nn_net_0_weight[74], nn_net_0_weight[77], nn_net_0_weight[80], nn_net_0_weight[83],
   nn_net_0_weight[86], nn_net_0_weight[89], nn_net_0_weight[92], nn_net_0_weight[95]},
};

// --- Output weights re-arranged for contiguous AVX2 loads ---
alignas(32) static constexpr float nn_net_2_weight_a[3][32] = {
  // output 0 (Bx) weights
  {nn_net_2_weight[0], nn_net_2_weight[1], nn_net_2_weight[2], nn_net_2_weight[3],
   nn_net_2_weight[4], nn_net_2_weight[5], nn_net_2_weight[6], nn_net_2_weight[7],
   nn_net_2_weight[8], nn_net_2_weight[9], nn_net_2_weight[10], nn_net_2_weight[11],
   nn_net_2_weight[12], nn_net_2_weight[13], nn_net_2_weight[14], nn_net_2_weight[15],
   nn_net_2_weight[16], nn_net_2_weight[17], nn_net_2_weight[18], nn_net_2_weight[19],
   nn_net_2_weight[20], nn_net_2_weight[21], nn_net_2_weight[22], nn_net_2_weight[23],
   nn_net_2_weight[24], nn_net_2_weight[25], nn_net_2_weight[26], nn_net_2_weight[27],
   nn_net_2_weight[28], nn_net_2_weight[29], nn_net_2_weight[30], nn_net_2_weight[31]},
  // output 1 (By) weights
  {nn_net_2_weight[32], nn_net_2_weight[33], nn_net_2_weight[34], nn_net_2_weight[35],
   nn_net_2_weight[36], nn_net_2_weight[37], nn_net_2_weight[38], nn_net_2_weight[39],
   nn_net_2_weight[40], nn_net_2_weight[41], nn_net_2_weight[42], nn_net_2_weight[43],
   nn_net_2_weight[44], nn_net_2_weight[45], nn_net_2_weight[46], nn_net_2_weight[47],
   nn_net_2_weight[48], nn_net_2_weight[49], nn_net_2_weight[50], nn_net_2_weight[51],
   nn_net_2_weight[52], nn_net_2_weight[53], nn_net_2_weight[54], nn_net_2_weight[55],
   nn_net_2_weight[56], nn_net_2_weight[57], nn_net_2_weight[58], nn_net_2_weight[59],
   nn_net_2_weight[60], nn_net_2_weight[61], nn_net_2_weight[62], nn_net_2_weight[63]},
  // output 2 (Bz) weights
  {nn_net_2_weight[64], nn_net_2_weight[65], nn_net_2_weight[66], nn_net_2_weight[67],
   nn_net_2_weight[68], nn_net_2_weight[69], nn_net_2_weight[70], nn_net_2_weight[71],
   nn_net_2_weight[72], nn_net_2_weight[73], nn_net_2_weight[74], nn_net_2_weight[75],
   nn_net_2_weight[76], nn_net_2_weight[77], nn_net_2_weight[78], nn_net_2_weight[79],
   nn_net_2_weight[80], nn_net_2_weight[81], nn_net_2_weight[82], nn_net_2_weight[83],
   nn_net_2_weight[84], nn_net_2_weight[85], nn_net_2_weight[86], nn_net_2_weight[87],
   nn_net_2_weight[88], nn_net_2_weight[89], nn_net_2_weight[90], nn_net_2_weight[91],
   nn_net_2_weight[92], nn_net_2_weight[93], nn_net_2_weight[94], nn_net_2_weight[95]},
};

/// Evaluate NN field at (x,y,z) mm -> (Bx,By,Bz) Gauss.
/// Single-hidden-layer ReLU — scalar implementation.
inline void evaluate_relu(float x, float y, float z,
                          float& bx, float& by, float& bz) {
  const float in0 = (x - nn_input_mean[0]) / nn_input_std[0];
  const float in1 = (y - nn_input_mean[1]) / nn_input_std[1];
  const float in2 = (z - nn_input_mean[2]) / nn_input_std[2];
  float h[NN_HIDDEN];
  for (int i = 0; i < NN_HIDDEN; ++i) {
    h[i] = nn_net_0_weight[i*3+0]*in0 + nn_net_0_weight[i*3+1]*in1
         + nn_net_0_weight[i*3+2]*in2 + nn_net_0_bias[i];
    h[i] = std::max(0.0f, h[i]);  // ReLU
  }
  float out[3] = {nn_net_2_bias[0], nn_net_2_bias[1], nn_net_2_bias[2]};
  for (int i = 0; i < NN_HIDDEN; ++i) {
    out[0] += nn_net_2_weight[0*NN_HIDDEN+i] * h[i];
    out[1] += nn_net_2_weight[1*NN_HIDDEN+i] * h[i];
    out[2] += nn_net_2_weight[2*NN_HIDDEN+i] * h[i];
  }
  bx = out[0]*nn_output_std[0] + nn_output_mean[0];
  by = out[1]*nn_output_std[1] + nn_output_mean[1];
  bz = out[2]*nn_output_std[2] + nn_output_mean[2];
}

/// Evaluate NN field at (x,y,z) mm -> (Bx,By,Bz) Gauss.
/// AVX2 ReLU with pre-transposed aligned weights.
/// Processes 8 neurons per iteration (32/8 = 4 iterations).
#pragma GCC push_options
#pragma GCC target("avx2,fma")
inline void evaluate_relu_avx2(float x, float y, float z,
                               float& bx, float& by, float& bz) {
  const float in0 = (x - nn_input_mean[0]) / nn_input_std[0];
  const float in1 = (y - nn_input_mean[1]) / nn_input_std[1];
  const float in2 = (z - nn_input_mean[2]) / nn_input_std[2];

  __m256 vin0 = _mm256_set1_ps(in0);
  __m256 vin1 = _mm256_set1_ps(in1);
  __m256 vin2 = _mm256_set1_ps(in2);
  __m256 vzero = _mm256_setzero_ps();

  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();

  for (int blk = 0; blk < 4; ++blk) {
    int off = blk * 8;
    __m256 vw0 = _mm256_load_ps(nn_net_0_weight_t[0] + off);
    __m256 vw1 = _mm256_load_ps(nn_net_0_weight_t[1] + off);
    __m256 vw2 = _mm256_load_ps(nn_net_0_weight_t[2] + off);
    __m256 vb  = _mm256_load_ps(nn_net_0_bias + off);

    __m256 vh = _mm256_fmadd_ps(vw0, vin0, vb);   // w0*in0 + bias
    vh = _mm256_fmadd_ps(vw1, vin1, vh);            // + w1*in1
    vh = _mm256_fmadd_ps(vw2, vin2, vh);            // + w2*in2
    vh = _mm256_max_ps(vh, vzero);                   // ReLU

    // Fused output accumulation — no intermediate h store
    acc0 = _mm256_fmadd_ps(_mm256_load_ps(nn_net_2_weight_a[0] + off), vh, acc0);
    acc1 = _mm256_fmadd_ps(_mm256_load_ps(nn_net_2_weight_a[1] + off), vh, acc1);
    acc2 = _mm256_fmadd_ps(_mm256_load_ps(nn_net_2_weight_a[2] + off), vh, acc2);
  }

  // Horizontal sum: reduce 8 floats -> 1
  auto hsum = [](__m256 v) -> float {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
  };

  bx = (hsum(acc0) + nn_net_2_bias[0]) * nn_output_std[0] + nn_output_mean[0];
  by = (hsum(acc1) + nn_net_2_bias[1]) * nn_output_std[1] + nn_output_mean[1];
  bz = (hsum(acc2) + nn_net_2_bias[2]) * nn_output_std[2] + nn_output_mean[2];
}

/// Batch-evaluate N field points sharing weight register loads.
/// Each point: (inputs[i][0..2]) -> (outputs[i][0..2]) in Gauss.
/// N should be small (e.g. 4-8) for register pressure.
/// Used for track-level batching where multiple independent tracks
/// need field evaluation at the same CK stage simultaneously.
template<int N>
inline void evaluate_relu_avx2_batch(const float inputs[N][3], float outputs[N][3]) {
  // Pre-normalize all inputs
  float norm_in[N][3];
  for (int b = 0; b < N; ++b) {
    norm_in[b][0] = (inputs[b][0] - nn_input_mean[0]) / nn_input_std[0];
    norm_in[b][1] = (inputs[b][1] - nn_input_mean[1]) / nn_input_std[1];
    norm_in[b][2] = (inputs[b][2] - nn_input_mean[2]) / nn_input_std[2];
  }

  // Broadcast inputs for each batch element
  __m256 vin0[N], vin1[N], vin2[N];
  for (int b = 0; b < N; ++b) {
    vin0[b] = _mm256_set1_ps(norm_in[b][0]);
    vin1[b] = _mm256_set1_ps(norm_in[b][1]);
    vin2[b] = _mm256_set1_ps(norm_in[b][2]);
  }
  __m256 vzero = _mm256_setzero_ps();

  // Output accumulators for each batch element × 3 outputs
  __m256 acc[N][3];
  for (int b = 0; b < N; ++b) {
    acc[b][0] = _mm256_setzero_ps();
    acc[b][1] = _mm256_setzero_ps();
    acc[b][2] = _mm256_setzero_ps();
  }

  // Process 8 neurons per block, sharing weight loads across all batch elements
  for (int blk = 0; blk < 4; ++blk) {
    int off = blk * 8;
    // Load weights ONCE — shared across all N batch elements
    __m256 vw0 = _mm256_load_ps(nn_net_0_weight_t[0] + off);
    __m256 vw1 = _mm256_load_ps(nn_net_0_weight_t[1] + off);
    __m256 vw2 = _mm256_load_ps(nn_net_0_weight_t[2] + off);
    __m256 vb  = _mm256_load_ps(nn_net_0_bias + off);
    __m256 vo0 = _mm256_load_ps(nn_net_2_weight_a[0] + off);
    __m256 vo1 = _mm256_load_ps(nn_net_2_weight_a[1] + off);
    __m256 vo2 = _mm256_load_ps(nn_net_2_weight_a[2] + off);

    for (int b = 0; b < N; ++b) {
      __m256 vh = _mm256_fmadd_ps(vw0, vin0[b], vb);
      vh = _mm256_fmadd_ps(vw1, vin1[b], vh);
      vh = _mm256_fmadd_ps(vw2, vin2[b], vh);
      vh = _mm256_max_ps(vh, vzero);  // ReLU

      // Fused output accumulation
      acc[b][0] = _mm256_fmadd_ps(vo0, vh, acc[b][0]);
      acc[b][1] = _mm256_fmadd_ps(vo1, vh, acc[b][1]);
      acc[b][2] = _mm256_fmadd_ps(vo2, vh, acc[b][2]);
    }
  }

  // Horizontal sum and denormalize
  auto hsum = [](__m256 v) -> float {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
  };

  for (int b = 0; b < N; ++b) {
    outputs[b][0] = (hsum(acc[b][0]) + nn_net_2_bias[0]) * nn_output_std[0] + nn_output_mean[0];
    outputs[b][1] = (hsum(acc[b][1]) + nn_net_2_bias[1]) * nn_output_std[1] + nn_output_mean[1];
    outputs[b][2] = (hsum(acc[b][2]) + nn_net_2_bias[2]) * nn_output_std[2] + nn_output_mean[2];
  }
}
#pragma GCC pop_options

} // namespace LHCb::FieldNNReLU
