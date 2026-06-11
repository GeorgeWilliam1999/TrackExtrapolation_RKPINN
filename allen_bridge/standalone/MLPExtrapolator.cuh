// MLPExtrapolator.cuh — Standalone CPU-compatible MLP extrapolator
//
// Loads binary model files produced by TrackMLPExtrapolator (double precision)
// and evaluates them in float32 for compatibility with the ParKalman harness.
//
// Binary format (V2, matching TrackMLPExtrapolator.cpp):
//   int model_type         (0=MLP, 1=PINN residual)
//   int num_layers
//   per layer:
//     int rows, cols
//     double weights[rows*cols]  (row-major, numpy C-order)
//     double biases[rows]
//   int input_size
//   double input_mean[input_size]
//   double input_std[input_size]
//   int output_size
//   double output_mean[output_size]
//   double output_std[output_size]
//   int activation_len       (optional)
//   char activation[activation_len]
//
// If first int > 1, it is the old V1 format where first int is num_layers directly.
//
// Usage:
//   MLPModel model;
//   model.load("model.bin");
//   Extrapolators::State s{...};
//   MLPExtrapolator::propagate(s, dz, field, model);

#pragma once

#include "ExtrapolatorCommon.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <vector>
#include <string>

// ═══════════════════════════════════════════════════════════════════════════
// MLPModel — stores weights, biases, and normalization parameters
// ═══════════════════════════════════════════════════════════════════════════
struct MLPModel {
  // Layer weights stored as flat float arrays (row-major after conversion)
  struct Layer {
    int rows, cols;
    std::vector<float> weights;  // rows * cols, row-major
    std::vector<float> biases;   // rows
  };

  std::vector<Layer> layers;
  std::vector<float> input_mean, input_std;
  std::vector<float> output_mean, output_std;
  std::string activation = "tanh";
  bool is_residual = false;  // PINN residual mode
  bool loaded = false;

  bool load(const char* filepath)
  {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
      fprintf(stderr, "MLPModel: cannot open '%s'\n", filepath);
      return false;
    }

    // Read first int — model_type (0 or 1) or num_layers (old format)
    int first_int;
    file.read(reinterpret_cast<char*>(&first_int), sizeof(int));

    int model_type = 0;
    int num_layers;
    if (first_int <= 1) {
      model_type = first_int;
      file.read(reinterpret_cast<char*>(&num_layers), sizeof(int));
    } else {
      num_layers = first_int;
    }
    is_residual = (model_type == 1);

    if (num_layers <= 0 || num_layers > 100) {
      fprintf(stderr, "MLPModel: invalid num_layers=%d in '%s'\n", num_layers, filepath);
      return false;
    }

    layers.resize(num_layers);
    for (int l = 0; l < num_layers; l++) {
      int rows, cols;
      file.read(reinterpret_cast<char*>(&rows), sizeof(int));
      file.read(reinterpret_cast<char*>(&cols), sizeof(int));

      if (rows <= 0 || cols <= 0 || rows > 10000 || cols > 10000) {
        fprintf(stderr, "MLPModel: invalid layer %d dims %dx%d\n", l, rows, cols);
        return false;
      }

      layers[l].rows = rows;
      layers[l].cols = cols;
      layers[l].weights.resize(rows * cols);
      layers[l].biases.resize(rows);

      // Read weights as doubles, convert to float
      // Gen_1 export_to_cpp.py writes row-major (numpy C-order),
      // so we read directly as row-major: weights[r * cols + c]
      std::vector<double> w_double(rows * cols);
      file.read(reinterpret_cast<char*>(w_double.data()), rows * cols * sizeof(double));
      for (int i = 0; i < rows * cols; i++)
        layers[l].weights[i] = static_cast<float>(w_double[i]);

      // Read biases as doubles
      std::vector<double> b_double(rows);
      file.read(reinterpret_cast<char*>(b_double.data()), rows * sizeof(double));
      for (int r = 0; r < rows; r++)
        layers[l].biases[r] = static_cast<float>(b_double[r]);
    }

    // Read normalization
    auto read_norm = [&](std::vector<float>& mean, std::vector<float>& std_v) {
      int size;
      file.read(reinterpret_cast<char*>(&size), sizeof(int));
      mean.resize(size);
      std_v.resize(size);
      std::vector<double> tmp(size);
      file.read(reinterpret_cast<char*>(tmp.data()), size * sizeof(double));
      for (int i = 0; i < size; i++) mean[i] = static_cast<float>(tmp[i]);
      file.read(reinterpret_cast<char*>(tmp.data()), size * sizeof(double));
      for (int i = 0; i < size; i++) std_v[i] = static_cast<float>(tmp[i]);
    };

    read_norm(input_mean, input_std);
    read_norm(output_mean, output_std);

    // Read activation (optional)
    int act_len;
    if (file.read(reinterpret_cast<char*>(&act_len), sizeof(int)) && act_len > 0 && act_len < 32) {
      std::vector<char> act_buf(act_len + 1, '\0');
      file.read(act_buf.data(), act_len);
      activation = std::string(act_buf.data());
    }

    loaded = file.good() || file.eof();
    if (loaded) {
      printf("MLPModel: loaded %d layers from '%s' (activation=%s, residual=%d)\n",
             num_layers, filepath, activation.c_str(), is_residual);
      printf("  input_dim=%d, output_dim=%d\n",
             layers.empty() ? 0 : layers[0].cols,
             layers.empty() ? 0 : layers.back().rows);
      for (int l = 0; l < num_layers; l++)
        printf("  layer %d: %d x %d\n", l, layers[l].rows, layers[l].cols);
    }
    return loaded;
  }

  // Set up a dummy model (identity-like) for testing without a trained model.
  // Input: [x, y, tx, ty, qop, dz], Output: [x', y', tx', ty']
  void init_linear_fallback()
  {
    layers.clear();

    // Single layer: output = W * input + b
    // For linear extrapolation: x' = x + tx*dz, y' = y + ty*dz, tx'=tx, ty'=ty
    // Input order: [x, y, tx, ty, qop, dz]
    //   x'  = 1*x + 0*y + 0*tx + 0*ty + 0*qop + 0*dz   (wrong, needs tx*dz)
    // Can't express tx*dz as linear — use identity and let propagate() handle fallback.

    // Instead: just mark as not loaded
    loaded = false;
    printf("MLPModel: no model file, using linear extrapolation fallback\n");
  }

  // ── Forward pass (float32) ───────────────────────────────────────────
  // x_in: input vector of size layers[0].cols
  // x_out: output buffer of size layers.back().rows
  void forward(const float* x_in, float* x_out) const
  {
    int input_dim = layers[0].cols;
    int max_dim = 0;
    for (auto& l : layers) {
      if (l.cols > max_dim) max_dim = l.cols;
      if (l.rows > max_dim) max_dim = l.rows;
    }

    // Use two alternating buffers for intermediate results
    std::vector<float> buf_a(max_dim), buf_b(max_dim);
    float* cur = buf_a.data();
    float* nxt = buf_b.data();

    // Normalize input
    for (int i = 0; i < input_dim; i++)
      cur[i] = (x_in[i] - input_mean[i]) / input_std[i];

    int n_layers = static_cast<int>(layers.size());
    for (int l = 0; l < n_layers; l++) {
      const auto& layer = layers[l];
      // nxt = W * cur + b
      for (int r = 0; r < layer.rows; r++) {
        float sum = layer.biases[r];
        for (int c = 0; c < layer.cols; c++)
          sum += layer.weights[r * layer.cols + c] * cur[c];

        // Apply activation for hidden layers (not the last)
        if (l < n_layers - 1)
          sum = apply_activation(sum);

        nxt[r] = sum;
      }
      // Swap buffers
      float* tmp = cur;
      cur = nxt;
      nxt = tmp;
    }

    // Denormalize output
    int output_dim = layers.back().rows;
    for (int i = 0; i < output_dim; i++)
      x_out[i] = cur[i] * output_std[i] + output_mean[i];
  }

private:
  float apply_activation(float x) const
  {
    if (activation == "relu")
      return x > 0.0f ? x : 0.0f;
    if (activation == "silu" || activation == "swish")
      return x / (1.0f + expf(-x));
    if (activation == "sigmoid")
      return 1.0f / (1.0f + expf(-x));
    // Default: tanh
    return tanhf(x);
  }
};


// ═══════════════════════════════════════════════════════════════════════════
// MLPExtrapolator — propagates State using the loaded MLPModel
// ═══════════════════════════════════════════════════════════════════════════
namespace MLPExtrapolator {

  // Propagate state from z to z+dz using the MLP model.
  // Input features:  [x, y, tx, ty, qop, dz]
  // Output features: [x', y', tx', ty']  (qop unchanged)
  inline void propagate(
    Extrapolators::State& state,
    float dz,
    const MagneticField::Magfield& /*field*/,
    const MLPModel& model)
  {
    if (!model.loaded) {
      // Linear extrapolation fallback
      state.x  += state.tx * dz;
      state.y  += state.ty * dz;
      state.z  += dz;
      return;
    }

    float input[6] = {state.x, state.y, state.tx, state.ty, state.qop, dz};
    float output[4];
    model.forward(input, output);

    state.x  = output[0];
    state.y  = output[1];
    state.tx = output[2];
    state.ty = output[3];
    // qop unchanged
    state.z += dz;
  }

  // Multi-step propagation from current z to z_target.
  // Subdivides into steps of size `step_size` for models trained on fixed dz ranges.
  inline void propagate_multistep(
    Extrapolators::State& state,
    float z_target,
    const MagneticField::Magfield& field,
    const MLPModel& model,
    float step_size = 500.0f)
  {
    float remaining = z_target - state.z;
    float sign = (remaining > 0) ? 1.0f : -1.0f;
    remaining = std::abs(remaining);

    while (remaining > 0.5f) {
      float dz = std::min(step_size, remaining) * sign;
      propagate(state, dz, field, model);
      remaining -= std::abs(dz);
    }
  }

  // Compute Jacobian by central finite differences.
  // F[i][j] = d(x'_i)/d(x_j)  for i,j in {x, y, tx, ty, qop}
  inline void compute_jacobian(
    const float x[5], float z_from, float z_to,
    float F[5][5],
    const MagneticField::Magfield& field,
    const MLPModel& model)
  {
    const float dz = z_to - z_from;

    // Perturbation sizes
    const float eps[5] = {
      1e-3f, 1e-3f, 1e-6f, 1e-6f,
      std::max(std::abs(x[4]) * 1e-4f, 1e-8f)
    };

    for (int j = 0; j < 5; j++) {
      // Forward perturbation
      Extrapolators::State sp{x[0], x[1], z_from, x[2], x[3], x[4]};
      // Backward perturbation
      Extrapolators::State sm = sp;

      // Apply perturbation to the j-th state component
      switch (j) {
        case 0: sp.x  += eps[j]; sm.x  -= eps[j]; break;
        case 1: sp.y  += eps[j]; sm.y  -= eps[j]; break;
        case 2: sp.tx += eps[j]; sm.tx -= eps[j]; break;
        case 3: sp.ty += eps[j]; sm.ty -= eps[j]; break;
        case 4: sp.qop += eps[j]; sm.qop -= eps[j]; break;
      }

      propagate(sp, dz, field, model);
      propagate(sm, dz, field, model);

      float fp[5] = {sp.x, sp.y, sp.tx, sp.ty, sp.qop};
      float fm[5] = {sm.x, sm.y, sm.tx, sm.ty, sm.qop};

      for (int i = 0; i < 5; i++)
        F[i][j] = (fp[i] - fm[i]) / (2.0f * eps[j]);
    }
  }

} // namespace MLPExtrapolator
