/**
 * Standalone C++ benchmark for V3 ML extrapolator models + baselines.
 * 
 * Loads binary model files and the real field map, then times:
 *   - Linear extrapolation (no field)
 *   - Parabolic extrapolation (single field eval at midpoint)
 *   - RK4 extrapolation (full integration with field)
 *   - MLP inference from binary weights
 *   - PINN-Residual inference from binary weights
 * 
 * Compile: g++ -O2 -std=c++17 -o benchmark_cpp benchmark_cpp.cpp -lm
 * Usage:   ./benchmark_cpp <field_map.rtf> <model1.bin> [model2.bin ...]
 * 
 * Output is JSON for easy parsing from Python/notebook.
 * 
 * Author: G. Scriven, Feb 2026
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>

// ============================================================================
// Field Map
// ============================================================================

struct FieldMap {
    // Grid extents
    double x_min, x_max, y_min, y_max, z_min, z_max;
    double dx, dy, dz;
    int nx, ny, nz;
    
    // Field components stored as flat arrays [ix * ny * nz + iy * nz + iz]
    // Actually: file order is (z, x, y) so we reshape to (nx, ny, nz)
    std::vector<double> Bx, By, Bz;
    
    bool load(const char* path) {
        FILE* f = fopen(path, "r");
        if (!f) { fprintf(stderr, "Cannot open field map: %s\n", path); return false; }
        
        // First pass: find grid extents
        std::vector<double> xs, ys, zs;
        std::vector<double> bx_raw, by_raw, bz_raw;
        double x, y, z, bx, by, bz;
        while (fscanf(f, "%lf %lf %lf %lf %lf %lf", &x, &y, &z, &bx, &by, &bz) == 6) {
            xs.push_back(x); ys.push_back(y); zs.push_back(z);
            bx_raw.push_back(bx); by_raw.push_back(by); bz_raw.push_back(bz);
        }
        fclose(f);
        
        // Get unique sorted coordinates
        auto unique_sorted = [](std::vector<double>& v) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        };
        std::vector<double> ux = xs, uy = ys, uz = zs;
        unique_sorted(ux); unique_sorted(uy); unique_sorted(uz);
        
        nx = ux.size(); ny = uy.size(); nz = uz.size();
        x_min = ux.front(); x_max = ux.back();
        y_min = uy.front(); y_max = uy.back();
        z_min = uz.front(); z_max = uz.back();
        dx = (x_max - x_min) / (nx - 1);
        dy = (y_max - y_min) / (ny - 1);
        dz = (z_max - z_min) / (nz - 1);
        
        // Reshape: file is ordered (z, x, y) -> we want [ix][iy][iz]
        Bx.resize(nx * ny * nz, 0.0);
        By.resize(nx * ny * nz, 0.0);
        Bz.resize(nx * ny * nz, 0.0);
        
        for (size_t i = 0; i < xs.size(); ++i) {
            int ix = (int)std::round((xs[i] - x_min) / dx);
            int iy = (int)std::round((ys[i] - y_min) / dy);
            int iz = (int)std::round((zs[i] - z_min) / this->dz);
            ix = std::clamp(ix, 0, nx-1);
            iy = std::clamp(iy, 0, ny-1);
            iz = std::clamp(iz, 0, nz-1);
            int idx = ix * ny * nz + iy * nz + iz;
            this->Bx[idx] = bx_raw[i];
            this->By[idx] = by_raw[i];
            this->Bz[idx] = bz_raw[i];
        }
        
        fprintf(stderr, "Field map loaded: %d x %d x %d = %d points\n", nx, ny, nz, nx*ny*nz);
        fprintf(stderr, "  x: [%.0f, %.0f] mm, y: [%.0f, %.0f] mm, z: [%.0f, %.0f] mm\n",
                x_min, x_max, y_min, y_max, z_min, z_max);
        return true;
    }
    
    // Trilinear interpolation
    void get_field(double x, double y, double z, double& bx, double& by, double& bz) const {
        // Normalize to grid coordinates
        double fx = (x - x_min) / dx;
        double fy = (y - y_min) / dy;
        double fz = (z - z_min) / this->dz;
        
        // Clamp to grid
        fx = std::clamp(fx, 0.0, (double)(nx - 1));
        fy = std::clamp(fy, 0.0, (double)(ny - 1));
        fz = std::clamp(fz, 0.0, (double)(nz - 1));
        
        int ix = (int)fx; int iy = (int)fy; int iz = (int)fz;
        if (ix >= nx - 1) ix = nx - 2;
        if (iy >= ny - 1) iy = ny - 2;
        if (iz >= nz - 1) iz = nz - 2;
        
        double wx = fx - ix, wy = fy - iy, wz = fz - iz;
        
        // 8 corners
        auto idx = [&](int i, int j, int k) { return i * ny * nz + j * nz + k; };
        
        double c000, c001, c010, c011, c100, c101, c110, c111;
        
        // Bx
        c000 = Bx[idx(ix,iy,iz)];     c001 = Bx[idx(ix,iy,iz+1)];
        c010 = Bx[idx(ix,iy+1,iz)];   c011 = Bx[idx(ix,iy+1,iz+1)];
        c100 = Bx[idx(ix+1,iy,iz)];   c101 = Bx[idx(ix+1,iy,iz+1)];
        c110 = Bx[idx(ix+1,iy+1,iz)]; c111 = Bx[idx(ix+1,iy+1,iz+1)];
        bx = (1-wx)*(1-wy)*(1-wz)*c000 + (1-wx)*(1-wy)*wz*c001
           + (1-wx)*wy*(1-wz)*c010     + (1-wx)*wy*wz*c011
           + wx*(1-wy)*(1-wz)*c100     + wx*(1-wy)*wz*c101
           + wx*wy*(1-wz)*c110         + wx*wy*wz*c111;
        
        // By
        c000 = By[idx(ix,iy,iz)];     c001 = By[idx(ix,iy,iz+1)];
        c010 = By[idx(ix,iy+1,iz)];   c011 = By[idx(ix,iy+1,iz+1)];
        c100 = By[idx(ix+1,iy,iz)];   c101 = By[idx(ix+1,iy,iz+1)];
        c110 = By[idx(ix+1,iy+1,iz)]; c111 = By[idx(ix+1,iy+1,iz+1)];
        by = (1-wx)*(1-wy)*(1-wz)*c000 + (1-wx)*(1-wy)*wz*c001
           + (1-wx)*wy*(1-wz)*c010     + (1-wx)*wy*wz*c011
           + wx*(1-wy)*(1-wz)*c100     + wx*(1-wy)*wz*c101
           + wx*wy*(1-wz)*c110         + wx*wy*wz*c111;
        
        // Bz
        c000 = this->Bz[idx(ix,iy,iz)];     c001 = this->Bz[idx(ix,iy,iz+1)];
        c010 = this->Bz[idx(ix,iy+1,iz)];   c011 = this->Bz[idx(ix,iy+1,iz+1)];
        c100 = this->Bz[idx(ix+1,iy,iz)];   c101 = this->Bz[idx(ix+1,iy,iz+1)];
        c110 = this->Bz[idx(ix+1,iy+1,iz)]; c111 = this->Bz[idx(ix+1,iy+1,iz+1)];
        bz = (1-wx)*(1-wy)*(1-wz)*c000 + (1-wx)*(1-wy)*wz*c001
           + (1-wx)*wy*(1-wz)*c010     + (1-wx)*wy*wz*c011
           + wx*(1-wy)*(1-wz)*c100     + wx*(1-wy)*wz*c101
           + wx*wy*(1-wz)*c110         + wx*wy*wz*c111;
    }
};

// ============================================================================
// Traditional Extrapolators
// ============================================================================

static const double C_LIGHT = 2.99792458e-4;  // eplus * c in LHCb units

// Linear: x += tx*dz, y += ty*dz, slopes unchanged
void linear_extrapolate(const double* input, double* output) {
    double x = input[0], y = input[1], tx = input[2], ty = input[3];
    double dz = input[5];
    output[0] = x + tx * dz;
    output[1] = y + ty * dz;
    output[2] = tx;
    output[3] = ty;
}

// Parabolic: single B eval at midpoint (matches TrackParabolicExtrapolator)
void parabolic_extrapolate(const double* input, double* output,
                           const FieldMap& field, double z_start) {
    double x = input[0], y = input[1], tx = input[2], ty = input[3];
    double qop = input[4], dz = input[5];
    
    double x_mid = x + 0.5 * tx * dz;
    double y_mid = y + 0.5 * ty * dz;
    double z_mid = z_start + 0.5 * dz;
    
    double bx, by, bz;
    field.get_field(x_mid, y_mid, z_mid, bx, by, bz);
    
    double nTx2 = 1.0 + tx * tx;
    double nTy2 = 1.0 + ty * ty;
    double norm = std::sqrt(nTx2 + nTy2 - 1.0);
    
    double ax = norm * (ty * (tx * bx + bz) - nTx2 * by);
    double ay = norm * (-tx * (ty * by + bz) + nTy2 * bx);
    
    double fact = C_LIGHT * dz * qop;
    
    output[0] = x + dz * (tx + 0.5 * ax * fact);
    output[1] = y + dz * (ty + 0.5 * ay * fact);
    output[2] = tx + ax * fact;
    output[3] = ty + ay * fact;
}

// RK4: full integration with field
struct RK4State { double x, y, tx, ty, qop; };

void rk4_derivatives(const RK4State& s, double z, const FieldMap& field,
                     double* dx, double* dy, double* dtx, double* dty) {
    double bx, by, bz;
    field.get_field(s.x, s.y, z, bx, by, bz);
    
    double kappa = C_LIGHT * s.qop;
    double N = std::sqrt(1.0 + s.tx*s.tx + s.ty*s.ty);
    
    *dx  = s.tx;
    *dy  = s.ty;
    *dtx = kappa * N * (s.tx * s.ty * bx - (1.0 + s.tx*s.tx) * by + s.ty * bz);
    *dty = kappa * N * ((1.0 + s.ty*s.ty) * bx - s.tx * s.ty * by - s.tx * bz);
}

void rk4_extrapolate(const double* input, double* output,
                     const FieldMap& field, double z_start, double step_size = 10.0) {
    RK4State s{input[0], input[1], input[2], input[3], input[4]};
    double dz_total = input[5];
    double z_end = z_start + dz_total;
    double step = (dz_total > 0) ? step_size : -step_size;
    double z = z_start;
    
    while (std::abs(z - z_end) > std::abs(step)) {
        double dx1, dy1, dtx1, dty1;
        double dx2, dy2, dtx2, dty2;
        double dx3, dy3, dtx3, dty3;
        double dx4, dy4, dtx4, dty4;
        
        rk4_derivatives(s, z, field, &dx1, &dy1, &dtx1, &dty1);
        
        RK4State s2{s.x + 0.5*step*dx1, s.y + 0.5*step*dy1,
                    s.tx + 0.5*step*dtx1, s.ty + 0.5*step*dty1, s.qop};
        rk4_derivatives(s2, z + 0.5*step, field, &dx2, &dy2, &dtx2, &dty2);
        
        RK4State s3{s.x + 0.5*step*dx2, s.y + 0.5*step*dy2,
                    s.tx + 0.5*step*dtx2, s.ty + 0.5*step*dty2, s.qop};
        rk4_derivatives(s3, z + 0.5*step, field, &dx3, &dy3, &dtx3, &dty3);
        
        RK4State s4{s.x + step*dx3, s.y + step*dy3,
                    s.tx + step*dtx3, s.ty + step*dty3, s.qop};
        rk4_derivatives(s4, z + step, field, &dx4, &dy4, &dtx4, &dty4);
        
        s.x  += (step/6.0) * (dx1  + 2*dx2  + 2*dx3  + dx4);
        s.y  += (step/6.0) * (dy1  + 2*dy2  + 2*dy3  + dy4);
        s.tx += (step/6.0) * (dtx1 + 2*dtx2 + 2*dtx3 + dtx4);
        s.ty += (step/6.0) * (dty1 + 2*dty2 + 2*dty3 + dty4);
        z += step;
    }
    
    // Final fractional step
    double remaining = z_end - z;
    if (std::abs(remaining) > 1e-6) {
        double dx1, dy1, dtx1, dty1, dx2, dy2, dtx2, dty2;
        double dx3, dy3, dtx3, dty3, dx4, dy4, dtx4, dty4;
        
        rk4_derivatives(s, z, field, &dx1, &dy1, &dtx1, &dty1);
        RK4State s2{s.x + 0.5*remaining*dx1, s.y + 0.5*remaining*dy1,
                    s.tx + 0.5*remaining*dtx1, s.ty + 0.5*remaining*dty1, s.qop};
        rk4_derivatives(s2, z + 0.5*remaining, field, &dx2, &dy2, &dtx2, &dty2);
        RK4State s3{s.x + 0.5*remaining*dx2, s.y + 0.5*remaining*dy2,
                    s.tx + 0.5*remaining*dtx2, s.ty + 0.5*remaining*dty2, s.qop};
        rk4_derivatives(s3, z + 0.5*remaining, field, &dx3, &dy3, &dtx3, &dty3);
        RK4State s4{s.x + remaining*dx3, s.y + remaining*dy3,
                    s.tx + remaining*dtx3, s.ty + remaining*dty3, s.qop};
        rk4_derivatives(s4, z + remaining, field, &dx4, &dy4, &dtx4, &dty4);
        
        s.x  += (remaining/6.0) * (dx1  + 2*dx2  + 2*dx3  + dx4);
        s.y  += (remaining/6.0) * (dy1  + 2*dy2  + 2*dy3  + dy4);
        s.tx += (remaining/6.0) * (dtx1 + 2*dtx2 + 2*dtx3 + dtx4);
        s.ty += (remaining/6.0) * (dty1 + 2*dty2 + 2*dty3 + dty4);
    }
    
    output[0] = s.x;
    output[1] = s.y;
    output[2] = s.tx;
    output[3] = s.ty;
}

// ============================================================================
// Binary Model Loader + Inference
// ============================================================================

struct Layer {
    int rows, cols;
    std::vector<double> weights;  // row-major: [rows x cols]
    std::vector<double> biases;   // [rows]
};

struct BinaryModel {
    std::string name;
    int model_type;  // 0=MLP, 1=PINN residual
    std::vector<Layer> layers;
    int input_size, output_size;
    std::vector<double> input_mean, input_std;
    std::vector<double> output_mean, output_std;
    std::string activation;
    int n_params;
    
    bool load(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "Cannot open model: %s\n", path); return false; }
        
        // Extract name from path
        name = path;
        size_t slash = name.rfind('/');
        if (slash != std::string::npos) name = name.substr(slash + 1);
        size_t dot = name.rfind('.');
        if (dot != std::string::npos) name = name.substr(0, dot);
        
        fread(&model_type, sizeof(int), 1, f);
        
        int num_layers;
        fread(&num_layers, sizeof(int), 1, f);
        
        n_params = 0;
        layers.resize(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            fread(&layers[i].rows, sizeof(int), 1, f);
            fread(&layers[i].cols, sizeof(int), 1, f);
            int n_w = layers[i].rows * layers[i].cols;
            layers[i].weights.resize(n_w);
            layers[i].biases.resize(layers[i].rows);
            fread(layers[i].weights.data(), sizeof(double), n_w, f);
            fread(layers[i].biases.data(), sizeof(double), layers[i].rows, f);
            n_params += n_w + layers[i].rows;
        }
        
        fread(&input_size, sizeof(int), 1, f);
        input_mean.resize(input_size); input_std.resize(input_size);
        fread(input_mean.data(), sizeof(double), input_size, f);
        fread(input_std.data(), sizeof(double), input_size, f);
        
        fread(&output_size, sizeof(int), 1, f);
        output_mean.resize(output_size); output_std.resize(output_size);
        fread(output_mean.data(), sizeof(double), output_size, f);
        fread(output_std.data(), sizeof(double), output_size, f);
        
        int act_len;
        fread(&act_len, sizeof(int), 1, f);
        activation.resize(act_len);
        fread(&activation[0], 1, act_len, f);
        
        fclose(f);
        
        fprintf(stderr, "Model %s: type=%d, layers=%d, params=%d, act=%s\n",
                name.c_str(), model_type, num_layers, n_params, activation.c_str());
        return true;
    }
    
    // SiLU activation: x * sigmoid(x)
    static inline double silu(double x) {
        return x / (1.0 + std::exp(-x));
    }
    
    void predict(const double* input, double* output) const {
        // Allocate temp buffers (small, on stack for speed)
        double buf_a[1024], buf_b[1024];
        double* cur = buf_a;
        double* next = buf_b;
        
        // Normalize input
        for (int i = 0; i < input_size; ++i) {
            cur[i] = (input[i] - input_mean[i]) / input_std[i];
        }
        
        // Save normalized initial state for PINN residual connection
        double norm_init[4];
        if (model_type == 1) {
            for (int i = 0; i < 4; ++i)
                norm_init[i] = cur[i];
        }
        
        // Forward pass
        for (size_t l = 0; l < layers.size(); ++l) {
            const Layer& layer = layers[l];
            // Matrix-vector: next[i] = sum_j W[i,j] * cur[j] + b[i]
            for (int i = 0; i < layer.rows; ++i) {
                double sum = layer.biases[i];
                const double* w = &layer.weights[i * layer.cols];
                for (int j = 0; j < layer.cols; ++j) {
                    sum += w[j] * cur[j];
                }
                next[i] = sum;
            }
            
            // Activation on all but last layer
            if (l < layers.size() - 1) {
                if (activation == "silu") {
                    for (int i = 0; i < layer.rows; ++i)
                        next[i] = silu(next[i]);
                } else if (activation == "relu") {
                    for (int i = 0; i < layer.rows; ++i)
                        next[i] = std::max(0.0, next[i]);
                } else if (activation == "tanh") {
                    for (int i = 0; i < layer.rows; ++i)
                        next[i] = std::tanh(next[i]);
                }
            }
            
            // Swap buffers
            std::swap(cur, next);
        }
        
        if (model_type == 1) {
            // PINN residual: add residual in normalized space, THEN denormalize
            // output = (correction + normalized_input[:4]) * output_std + output_mean
            for (int i = 0; i < output_size; ++i) {
                output[i] = (cur[i] + norm_init[i]) * output_std[i] + output_mean[i];
            }
        } else {
            // MLP: just denormalize
            for (int i = 0; i < output_size; ++i) {
                output[i] = cur[i] * output_std[i] + output_mean[i];
            }
        }
    }
};

// ============================================================================
// Timing utilities
// ============================================================================

struct TimingResult {
    double median_ns;
    double mean_ns;
    double std_ns;
    double p5_ns;
    double p95_ns;
};

TimingResult compute_stats(std::vector<double>& times_ns) {
    std::sort(times_ns.begin(), times_ns.end());
    int n = times_ns.size();
    
    double mean = std::accumulate(times_ns.begin(), times_ns.end(), 0.0) / n;
    double sq_sum = 0;
    for (double t : times_ns) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / n);
    
    return {
        times_ns[n / 2],           // median
        mean,                       // mean
        stddev,                     // std
        times_ns[(int)(n * 0.05)],  // p5
        times_ns[(int)(n * 0.95)],  // p95
    };
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <field_map.rtf> <model1.bin> [model2.bin ...]\n", argv[0]);
        fprintf(stderr, "  Also reads test_inputs.bin (N x 7 doubles: x,y,tx,ty,qop,dz,z_start)\n");
        return 1;
    }
    
    // Load field map
    FieldMap field;
    if (!field.load(argv[1])) return 1;
    
    // Load models
    std::vector<BinaryModel> models;
    for (int i = 2; i < argc; ++i) {
        BinaryModel m;
        if (m.load(argv[i])) models.push_back(std::move(m));
    }
    
    // Load test inputs from binary file (written by Python notebook)
    // Format: int N, then N rows of 7 doubles [x, y, tx, ty, qop, dz, z_start]
    FILE* fin = fopen("test_inputs.bin", "rb");
    if (!fin) { fprintf(stderr, "Cannot open test_inputs.bin\n"); return 1; }
    int N;
    fread(&N, sizeof(int), 1, fin);
    std::vector<std::vector<double>> inputs(N, std::vector<double>(7));
    for (int i = 0; i < N; ++i) {
        fread(inputs[i].data(), sizeof(double), 7, fin);
    }
    fclose(fin);
    fprintf(stderr, "Loaded %d test inputs\n", N);
    
    const int N_ITER = 5000;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, N - 1);
    
    // Pre-generate random indices
    std::vector<int> indices(N_ITER);
    for (int i = 0; i < N_ITER; ++i) indices[i] = dist(rng);
    
    double output[4];
    
    // Start JSON output
    printf("{\n");
    
    // Also compute predictions for accuracy (output to binary file)
    // Format: for each model, N rows of 4 doubles
    
    // ---- Linear ----
    {
        // Warmup
        for (int i = 0; i < 200; ++i) linear_extrapolate(inputs[indices[i]].data(), output);
        
        std::vector<double> times(N_ITER);
        for (int i = 0; i < N_ITER; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            linear_extrapolate(inputs[indices[i]].data(), output);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }
        auto s = compute_stats(times);
        printf("  \"Linear\": {\"median_ns\": %.1f, \"mean_ns\": %.1f, \"std_ns\": %.1f, \"p5_ns\": %.1f, \"p95_ns\": %.1f},\n",
               s.median_ns, s.mean_ns, s.std_ns, s.p5_ns, s.p95_ns);
    }
    
    // ---- Parabolic ----
    {
        for (int i = 0; i < 200; ++i) {
            parabolic_extrapolate(inputs[indices[i]].data(), output, field, inputs[indices[i]][6]);
        }
        
        std::vector<double> times(N_ITER);
        for (int i = 0; i < N_ITER; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            parabolic_extrapolate(inputs[indices[i]].data(), output, field, inputs[indices[i]][6]);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }
        auto s = compute_stats(times);
        printf("  \"Parabolic\": {\"median_ns\": %.1f, \"mean_ns\": %.1f, \"std_ns\": %.1f, \"p5_ns\": %.1f, \"p95_ns\": %.1f},\n",
               s.median_ns, s.mean_ns, s.std_ns, s.p5_ns, s.p95_ns);
    }
    
    // ---- RK4 (step_size=10mm) ----
    {
        for (int i = 0; i < 50; ++i) {
            rk4_extrapolate(inputs[indices[i]].data(), output, field, inputs[indices[i]][6], 10.0);
        }
        
        std::vector<double> times(N_ITER);
        for (int i = 0; i < N_ITER; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            rk4_extrapolate(inputs[indices[i]].data(), output, field, inputs[indices[i]][6], 10.0);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }
        auto s = compute_stats(times);
        printf("  \"RK4\": {\"median_ns\": %.1f, \"mean_ns\": %.1f, \"std_ns\": %.1f, \"p5_ns\": %.1f, \"p95_ns\": %.1f},\n",
               s.median_ns, s.mean_ns, s.std_ns, s.p5_ns, s.p95_ns);
    }
    
    // ---- ML Models ----
    for (size_t m = 0; m < models.size(); ++m) {
        // Warmup
        for (int i = 0; i < 200; ++i) models[m].predict(inputs[indices[i]].data(), output);
        
        std::vector<double> times(N_ITER);
        for (int i = 0; i < N_ITER; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            models[m].predict(inputs[indices[i]].data(), output);
            auto t1 = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::nano>(t1 - t0).count();
        }
        auto s = compute_stats(times);
        bool last = (m == models.size() - 1);
        printf("  \"%s\": {\"median_ns\": %.1f, \"mean_ns\": %.1f, \"std_ns\": %.1f, \"p5_ns\": %.1f, \"p95_ns\": %.1f}%s\n",
               models[m].name.c_str(), s.median_ns, s.mean_ns, s.std_ns, s.p5_ns, s.p95_ns,
               last ? "" : ",");
    }
    
    printf("}\n");
    
    // Write predictions for accuracy evaluation
    // Output: predictions_cpp.bin with format:
    //   int num_models
    //   for each model:
    //     int name_len, char[name_len] name
    //     N x 4 doubles (predictions)
    //   then 3 baselines: Linear, Parabolic, RK4
    FILE* fout = fopen("predictions_cpp.bin", "wb");
    if (!fout) { fprintf(stderr, "Cannot open predictions_cpp.bin for writing\n"); return 1; }
    
    int total_models = models.size() + 3;  // +Linear, Parabolic, RK4
    fwrite(&N, sizeof(int), 1, fout);
    fwrite(&total_models, sizeof(int), 1, fout);
    
    // Helper to write model name + predictions
    auto write_predictions = [&](const char* model_name, auto predict_fn) {
        int name_len = strlen(model_name);
        fwrite(&name_len, sizeof(int), 1, fout);
        fwrite(model_name, 1, name_len, fout);
        for (int i = 0; i < N; ++i) {
            predict_fn(inputs[i].data(), output, i);
            fwrite(output, sizeof(double), 4, fout);
        }
    };
    
    write_predictions("Linear", [](const double* in, double* out, int) {
        linear_extrapolate(in, out);
    });
    
    write_predictions("Parabolic", [&](const double* in, double* out, int i) {
        parabolic_extrapolate(in, out, field, inputs[i][6]);
    });
    
    write_predictions("RK4", [&](const double* in, double* out, int i) {
        rk4_extrapolate(in, out, field, inputs[i][6], 5.0);  // 5mm step for ground truth
    });
    
    for (auto& m : models) {
        write_predictions(m.name.c_str(), [&](const double* in, double* out, int) {
            m.predict(in, out);
        });
    }
    
    fclose(fout);
    fprintf(stderr, "Predictions written to predictions_cpp.bin\n");
    
    return 0;
}
