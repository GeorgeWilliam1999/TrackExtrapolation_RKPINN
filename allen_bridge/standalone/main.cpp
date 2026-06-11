// main.cpp — Standalone driver that exercises the ACTUAL Allen extrapolator
// headers  (ButcherTableau.cuh, ExtrapolatorCommon.cuh, ParabolicExtrapolator.cuh,
//            RungeKuttaExtrapolator.cuh)  with a CPU-compiled build.
//
// Outputs:
//   trajectories.csv   — per-step state for every method
//   kalman_results.csv  — per-track Kalman-fit quality metrics
//   kalman_hits.csv     — per-hit residuals / pulls
//
// Compile:
//   g++ -std=c++20 -O2 -I. -I../../device/kalman/ParKalman/include main.cpp -o run_extrapolators -lm
//
// The include search order is:  this directory (compat shims) then
// ../../device/kalman/ParKalman/include (actual Allen headers).
// The only headers that are NOT the originals are:
//   MagneticField.cuh, BackendCommon.h, FloatOperations.cuh, cuda_compat.h

#include "cuda_compat.h"            // must come first — defines __device__, float3 etc.
#include "ButcherTableau.cuh"       // ← actual Allen header
#include "ExtrapolatorCommon.cuh"   // ← actual Allen header
#include "ParabolicExtrapolator.cuh"// ← actual Allen header
#include "RungeKuttaExtrapolator.cuh" // ← actual Allen header (incl. Nyström)
#include "MLPExtrapolator.cuh"      // MLP-based extrapolator

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <array>
#include <algorithm>

using State = Extrapolators::State;
using Extrap = Extrapolators::RungeKuttaNystromExtrapolator;
static MagneticField::Magfield g_field;
static MLPModel g_mlp_model;

// ═══════════════════════════════════════════════════════════════════════════
// 1.  TRAJECTORY COMPARISON — propagate one track with all three methods
// ═══════════════════════════════════════════════════════════════════════════

static void write_trajectories(const char* fname)
{
    FILE* fp = fopen(fname, "w");
    fprintf(fp, "method,step_mm,z,x,y,tx,ty,qop\n");

    const float p_GeV  = 10.0f;
    const float qop0   = Extrapolators::c_light / (p_GeV * 1e3f);

    auto run = [&](const char* name, auto propagate_fn, float step) {
        State s{1.0f, 0.5f, 0.0f, 0.01f, 0.005f, qop0};
        fprintf(fp, "%s,%.0f,%.2f,%.6f,%.6f,%.8f,%.8f,%.10f\n",
                name, step, s.z, s.x, s.y, s.tx, s.ty, s.qop);
        const float z_end = 9500.0f;
        while (s.z < z_end - 0.5f) {
            float dz = std::min(step, z_end - s.z);
            propagate_fn(s, dz);
            fprintf(fp, "%s,%.0f,%.2f,%.6f,%.6f,%.8f,%.8f,%.10f\n",
                    name, step, s.z, s.x, s.y, s.tx, s.ty, s.qop);
        }
    };

    // Truth: Cash-Karp, 2 mm steps
    run("CashKarp_2mm", [](State& s, float dz) {
        State::Error err;
        Extrapolators::RungeKuttaExtrapolator<float, ButcherTableau::CashKarp<float>>
            ::propagate(s, err, dz, g_field);
    }, 2.0f);

    // Parabolic, 10 mm steps (small steps to keep stable)
    run("Parabolic_10mm", [](State& s, float dz) {
        Extrapolators::ParabolicExtrapolator<float>::propagate(s, dz, g_field);
    }, 10.0f);

    // RK4, 100 mm steps (RK4 has no b_star, so use the generic RK path
    // and collect stages manually — matching how the CUDA code handles
    // tableaux without an embedded error estimate)
    run("RK4_100mm", [](State& s, float dz) {
        using Table = ButcherTableau::RK4<float>;
        constexpr int N = Table::N_stages;
        State::Derivative k[N];
        for (int stage = 0; stage < N; stage++) {
            State ss = s;
            for (int i = 0; i < stage; i++)
                ss = ss + k[i] * Table::a(stage, i);
            float3 B = g_field.fieldVectorLinearInterpolation(
                make_float3(ss.x, ss.y, ss.z));
            k[stage] = derivative(ss, B) * dz;
        }
        for (int i = 0; i < N; i++)
            s = s + k[i] * Table::b(i);
    }, 100.0f);

    // Cash-Karp, 100 mm steps
    run("CashKarp_100mm", [](State& s, float dz) {
        State::Error err;
        Extrapolators::RungeKuttaExtrapolator<float, ButcherTableau::CashKarp<float>>
            ::propagate(s, err, dz, g_field);
    }, 100.0f);

    // RKN, 100 mm steps
    run("RKN_100mm", [](State& s, float dz) {
        Extrap::make_fast_step(s, dz, g_field);
    }, 100.0f);

    // RKN via propagate() with 500 mm step size to z=9500
    {
        State s{1.0f, 0.5f, 0.0f, 0.01f, 0.005f, qop0};
        fprintf(fp, "RKN_propagate,500,%.2f,%.6f,%.6f,%.8f,%.8f,%.10f\n",
                s.z, s.x, s.y, s.tx, s.ty, s.qop);
        for (float zt = 500.0f; zt <= 9500.0f; zt += 500.0f) {
            Extrap::propagate(s, zt, g_field, 500.0f, 100);
            fprintf(fp, "RKN_propagate,500,%.2f,%.6f,%.6f,%.8f,%.8f,%.10f\n",
                    s.z, s.x, s.y, s.tx, s.ty, s.qop);
        }
    }

    // MLP extrapolator — single big step
    if (g_mlp_model.loaded) {
        run("MLP_single", [](State& s, float dz) {
            MLPExtrapolator::propagate(s, dz, g_field, g_mlp_model);
        }, 9500.0f);

        // MLP with 500 mm steps
        run("MLP_500mm", [](State& s, float dz) {
            MLPExtrapolator::propagate(s, dz, g_field, g_mlp_model);
        }, 500.0f);

        // MLP with 100 mm steps
        run("MLP_100mm", [](State& s, float dz) {
            MLPExtrapolator::propagate(s, dz, g_field, g_mlp_model);
        }, 100.0f);
    } else {
        printf("  (MLP model not loaded — skipping MLP trajectories)\n");
    }

    fclose(fp);
    printf("  -> %s written\n", fname);
}

// ═══════════════════════════════════════════════════════════════════════════
// 2.  KALMAN FILTER using the RKN extrapolator + analytic Jacobian
// ═══════════════════════════════════════════════════════════════════════════

// Minimal 5×5 matrix type satisfying the Jacobian template constraint
struct Mat5 {
    float data[5][5]{};
    float& operator()(unsigned i, unsigned j) { return data[i][j]; }
    float  operator()(unsigned i, unsigned j) const { return data[i][j]; }
};

// Detector layers
static const int N_VELO  = 26;
static const int N_UT    =  4;
static const int N_SCIFI = 12;
static const int N_TOTAL = N_VELO + N_UT + N_SCIFI;

static float LAYER_Z[N_TOTAL];
static float LAYER_SX[N_TOTAL];   // sigma_x
static float LAYER_SY[N_TOTAL];   // sigma_y

static void init_detector()
{
    for (int i = 0; i < N_VELO; i++) {
        LAYER_Z[i]  = 0.0f + 750.0f * i / (N_VELO - 1);
        LAYER_SX[i] = 0.012f;
        LAYER_SY[i] = 0.012f;
    }
    float ut_z[] = {2315.0f, 2405.0f, 2530.0f, 2642.5f};
    for (int i = 0; i < N_UT; i++) {
        LAYER_Z [N_VELO + i] = ut_z[i];
        LAYER_SX[N_VELO + i] = 0.050f;
        LAYER_SY[N_VELO + i] = 5.0f;
    }
    for (int i = 0; i < N_SCIFI; i++) {
        LAYER_Z [N_VELO + N_UT + i] = 7620.0f + 1790.0f * i / (N_SCIFI - 1);
        LAYER_SX[N_VELO + N_UT + i] = 0.060f;
        LAYER_SY[N_VELO + N_UT + i] = 0.500f;
    }
}

// 5×5 helpers (inline, simple)
static void mat_zero(float C[5][5]) { memset(C, 0, 25*sizeof(float)); }
static void mat_eye (float C[5][5]) {
    mat_zero(C);
    for (int i = 0; i < 5; i++) C[i][i] = 1.0f;
}
static void mat_mul(const float A[5][5], const float B[5][5], float R[5][5]) {
    mat_zero(R);
    for (int i = 0; i < 5; i++)
      for (int j = 0; j < 5; j++)
        for (int k = 0; k < 5; k++)
          R[i][j] += A[i][k] * B[k][j];
}
static void mat_add(float A[5][5], const float B[5][5]) {
    for (int i = 0; i < 5; i++)
      for (int j = 0; j < 5; j++)
        A[i][j] += B[i][j];
}
static void mat_transpose(const float A[5][5], float T[5][5]) {
    for (int i = 0; i < 5; i++)
      for (int j = 0; j < 5; j++)
        T[i][j] = A[j][i];
}

static void process_noise(float dz, float qop, float Q[5][5])
{
    mat_zero(Q);
    float p_inv = std::abs(qop) / Extrapolators::c_light * 1e3f;
    float scat = 0.0136f * p_inv * std::sqrt(0.5f / 93.6f);
    float sig_x = std::abs(dz) * scat * 0.5f;
    Q[0][0] = sig_x * sig_x;
    Q[1][1] = sig_x * sig_x;
    Q[2][2] = scat * scat;
    Q[3][3] = scat * scat;
    Q[0][2] = 0.5f * sig_x * scat;  Q[2][0] = Q[0][2];
    Q[1][3] = 0.5f * sig_x * scat;  Q[3][1] = Q[1][3];
    // Energy-loss fluctuation noise on qop
    float dz_m = std::abs(dz) * 1e-3f;  // mm -> m
    float sig_qop = 0.01f * std::abs(qop) * std::sqrt(dz_m / 0.1f);  // ~1% per 10cm
    Q[4][4] = sig_qop * sig_qop;
}

// Propagate state from z_from to z_to using Allen's RKN extrapolator.
// Returns the transported state.
static void extrapolate_state(float x[5], float z_from, float z_to, float step = 50.0f)
{
    State s{x[0], x[1], z_from, x[2], x[3], x[4]};
    Extrap::propagate(s, z_to, g_field, step, 300);
    x[0] = s.x; x[1] = s.y; x[2] = s.tx; x[3] = s.ty; x[4] = s.qop;
}

// Compute the Jacobian F_ij = d(x'_i)/d(x_j) by central finite differences.
// State propagation uses the actual Allen RKN extrapolator.
static void compute_jacobian(
    const float x[5], float z_from, float z_to,
    float F[5][5], float step = 50.0f)
{
    // Perturbation sizes — must be well above float32 noise floor.
    // For qop, use relative perturbation since |qop| ~ 0.006–0.06.
    const float eps[5] = {1e-3f, 1e-3f, 1e-6f, 1e-6f,
                          std::max(std::abs(x[4]) * 1e-4f, 1e-8f)};

    for (int j = 0; j < 5; j++) {
        float xp[5], xm[5], fp[5], fm[5];
        memcpy(xp, x, 5*sizeof(float));
        memcpy(xm, x, 5*sizeof(float));
        xp[j] += eps[j];
        xm[j] -= eps[j];
        memcpy(fp, xp, 5*sizeof(float));
        memcpy(fm, xm, 5*sizeof(float));
        extrapolate_state(fp, z_from, z_to, step);
        extrapolate_state(fm, z_from, z_to, step);
        for (int i = 0; i < 5; i++)
            F[i][j] = (fp[i] - fm[i]) / (2.0f * eps[j]);
    }
}

// ── MLP-based extrapolation functions ──────────────────────────────────────
static void extrapolate_state_mlp(float x[5], float z_from, float z_to)
{
    State s{x[0], x[1], z_from, x[2], x[3], x[4]};
    MLPExtrapolator::propagate_multistep(s, z_to, g_field, g_mlp_model, 500.0f);
    x[0] = s.x; x[1] = s.y; x[2] = s.tx; x[3] = s.ty; x[4] = s.qop;
}

static void compute_jacobian_mlp(
    const float x[5], float z_from, float z_to, float F[5][5])
{
    MLPExtrapolator::compute_jacobian(x, z_from, z_to, F, g_field, g_mlp_model);
}

struct TrackResult {
    float p_true, p_fit;
    float chi2;
    int   ndof;
    float pull_x, pull_y, pull_tx, pull_ty, pull_qop;
    float res_x,  res_y,  res_tx,  res_ty,  res_qop;
};

struct HitResult {
    int   track_id;
    int   layer;
    float z;
    float res_x, res_y;     // meas - truth
    float filt_res_x;       // filtered state - truth
    float filt_res_y;
    float pull_x, pull_y;   // filtered residual / sqrt(cov)
    const char* region;
};

// Function pointer types for swappable extrapolation
using ExtrapFn   = void (*)(float x[5], float z_from, float z_to);
using JacobianFn = void (*)(const float x[5], float z_from, float z_to, float F[5][5]);

static TrackResult fit_track(
    const State& s0, std::mt19937& rng,
    std::vector<HitResult>& hit_results, int track_id,
    ExtrapFn extrap_fn = nullptr,
    JacobianFn jacobian_fn = nullptr)
{
    // Default to RKN-based functions
    if (!extrap_fn) {
        extrap_fn = [](float x[5], float zf, float zt) {
            extrapolate_state(x, zf, zt, 50.0f);
        };
    }
    if (!jacobian_fn) {
        jacobian_fn = [](const float x[5], float zf, float zt, float F[5][5]) {
            compute_jacobian(x, zf, zt, F, 50.0f);
        };
    }

    // ── Generate truth hits ────────────────────────────────────────────────
    float true_x[N_TOTAL], true_y[N_TOTAL];
    float meas_x[N_TOTAL], meas_y[N_TOTAL];
    {
        State s = s0;
        for (int i = 0; i < N_TOTAL; i++) {
            Extrap::propagate(s, LAYER_Z[i], g_field, 50.0f, 200);
            true_x[i] = s.x;
            true_y[i] = s.y;
            std::normal_distribution<float> nx(0, LAYER_SX[i]);
            std::normal_distribution<float> ny(0, LAYER_SY[i]);
            meas_x[i] = true_x[i] + nx(rng);
            meas_y[i] = true_y[i] + ny(rng);
        }
    }

    // ── Propagate truth to last layer for reference ────────────────────────
    State s_truth = s0;
    Extrap::propagate(s_truth, LAYER_Z[N_TOTAL-1], g_field, 50.0f, 200);

    // ── Seed state from first + last VELO hit ──────────────────────────────
    float x[5];
    x[0] = meas_x[0];
    x[1] = meas_y[0];
    x[2] = (meas_x[N_VELO-1] - meas_x[0]) / (LAYER_Z[N_VELO-1] - LAYER_Z[0]);
    x[3] = (meas_y[N_VELO-1] - meas_y[0]) / (LAYER_Z[N_VELO-1] - LAYER_Z[0]);
    // Seed qop with 20% smearing
    std::normal_distribution<float> qop_smear(0, 0.2f);
    x[4] = s0.qop * (1.0f + qop_smear(rng));

    float C[5][5];
    mat_zero(C);
    C[0][0] = LAYER_SX[0] * LAYER_SX[0];
    C[1][1] = LAYER_SY[0] * LAYER_SY[0];
    C[2][2] = 0.01f;
    C[3][3] = 0.01f;
    C[4][4] = 0.09f * x[4] * x[4];

    float z_cur = LAYER_Z[0];
    float chi2 = 0;
    int ndof_contrib = 0;

    // ── Forward fit ────────────────────────────────────────────────────────
    bool diverged = false;
    for (int i = 1; i < N_TOTAL; i++) {
        float dz = LAYER_Z[i] - z_cur;

        // Predict
        float F[5][5], Q[5][5];
        jacobian_fn(x, z_cur, LAYER_Z[i], F);
        extrap_fn(x, z_cur, LAYER_Z[i]);
        if (std::isnan(x[0]) || std::abs(x[2]) > 5.0f || std::abs(x[3]) > 5.0f) {
            diverged = true;
            break;
        }
        process_noise(dz, x[4], Q);

        // C = F C F^T + Q
        float FT[5][5], tmp[5][5], C_new[5][5];
        mat_transpose(F, FT);
        mat_mul(F, C, tmp);
        mat_mul(tmp, FT, C_new);
        mat_add(C_new, Q);
        memcpy(C, C_new, 25*sizeof(float));

        z_cur = LAYER_Z[i];

        // Update
        float sx2 = LAYER_SX[i] * LAYER_SX[i];
        float sy2 = LAYER_SY[i] * LAYER_SY[i];

        // Innovation
        float rx = meas_x[i] - x[0];
        float ry = meas_y[i] - x[1];

        // S = H C H^T + V  (2×2, H selects x,y)
        float S00 = C[0][0] + sx2;
        float S01 = C[0][1];
        float S10 = C[1][0];
        float S11 = C[1][1] + sy2;
        float detS = S00 * S11 - S01 * S10;
        float iS00 =  S11 / detS;
        float iS01 = -S01 / detS;
        float iS10 = -S10 / detS;
        float iS11 =  S00 / detS;

        // K = C H^T S^{-1}  (5×2)
        float K[5][2];
        for (int ii = 0; ii < 5; ii++) {
            K[ii][0] = C[ii][0] * iS00 + C[ii][1] * iS10;
            K[ii][1] = C[ii][0] * iS01 + C[ii][1] * iS11;
        }

        // chi2 += r^T S^{-1} r
        chi2 += rx * (iS00 * rx + iS01 * ry) + ry * (iS10 * rx + iS11 * ry);
        ndof_contrib += 2;

        // x += K r
        for (int ii = 0; ii < 5; ii++)
            x[ii] += K[ii][0] * rx + K[ii][1] * ry;

        // C = (I - K H) C
        float C_tmp[5][5];
        memcpy(C_tmp, C, 25*sizeof(float));
        for (int ii = 0; ii < 5; ii++)
          for (int jj = 0; jj < 5; jj++)
            C[ii][jj] = C_tmp[ii][jj] - K[ii][0] * C_tmp[0][jj] - K[ii][1] * C_tmp[1][jj];

        // Record per-hit results
        const char* reg = (i < N_VELO) ? "VELO" :
                          (i < N_VELO + N_UT) ? "UT" : "SciFi";
        float filt_res_x = x[0] - true_x[i];
        float filt_res_y = x[1] - true_y[i];
        float sig_filt_x = std::sqrt(std::abs(C[0][0]));
        float sig_filt_y = std::sqrt(std::abs(C[1][1]));

        hit_results.push_back({
            track_id, i, LAYER_Z[i],
            meas_x[i] - true_x[i],
            meas_y[i] - true_y[i],
            filt_res_x, filt_res_y,
            sig_filt_x > 1e-10f ? filt_res_x / sig_filt_x : 0.0f,
            sig_filt_y > 1e-10f ? filt_res_y / sig_filt_y : 0.0f,
            reg
        });
    }

    // ── Build result ───────────────────────────────────────────────────────
    TrackResult r;
    if (diverged) {
        r.p_true = Extrapolators::c_light / (std::abs(s0.qop) * 1e3f);
        r.p_fit = NAN; r.chi2 = NAN; r.ndof = 0;
        r.res_x = r.res_y = r.res_tx = r.res_ty = r.res_qop = NAN;
        r.pull_x = r.pull_y = r.pull_tx = r.pull_ty = r.pull_qop = NAN;
        return r;
    }
    r.p_true = Extrapolators::c_light / (std::abs(s0.qop) * 1e3f);
    r.p_fit  = Extrapolators::c_light / (std::abs(x[4])   * 1e3f);
    r.chi2   = chi2;
    r.ndof   = ndof_contrib - 5;

    float true_end[5] = {s_truth.x, s_truth.y, s_truth.tx, s_truth.ty, s_truth.qop};
    for (int j = 0; j < 5; j++) {
        float sig = std::sqrt(std::abs(C[j][j]));
        float res = x[j] - true_end[j];
        switch (j) {
            case 0: r.res_x   = res; r.pull_x   = sig > 1e-12f ? res/sig : 0; break;
            case 1: r.res_y   = res; r.pull_y   = sig > 1e-12f ? res/sig : 0; break;
            case 2: r.res_tx  = res; r.pull_tx  = sig > 1e-12f ? res/sig : 0; break;
            case 3: r.res_ty  = res; r.pull_ty  = sig > 1e-12f ? res/sig : 0; break;
            case 4: r.res_qop = res; r.pull_qop = sig > 1e-12f ? res/sig : 0; break;
        }
    }
    return r;
}

// ═══════════════════════════════════════════════════════════════════════════
// 3.  MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[])
{
    printf("ParKalman Standalone — using actual Allen headers\n");
    printf("  Extrapolators::c_light = %.6f\n", Extrapolators::c_light);
    printf("  Butcher CashKarp stages = %d\n", ButcherTableau::CashKarp<float>::N_stages);

    // Parse optional model path: --model <path>
    const char* model_path = nullptr;
    for (int i = 1; i < argc; i++) {
        if ((strcmp(argv[i], "--model") == 0 || strcmp(argv[i], "-m") == 0) && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    if (model_path) {
        printf("  Loading MLP model: %s\n", model_path);
        if (!g_mlp_model.load(model_path)) {
            fprintf(stderr, "ERROR: failed to load MLP model from '%s'\n", model_path);
            return 1;
        }
    } else {
        printf("  No --model <path> specified — MLP tests will be skipped\n");
        printf("  Usage: %s [--model <path/to/model.bin>]\n", argv[0]);
    }
    printf("\n");

    // ── Part 1: Trajectory comparison ──────────────────────────────────────
    printf("[1/2] Writing trajectory comparison...\n");
    write_trajectories("trajectories.csv");

    // ── Part 2: Kalman filter on many tracks ───────────────────────────────
    printf("[2/3] Running Kalman filter (RKN) on 500 tracks...\n");
    init_detector();

    const int N_TRACKS = 500;
    std::mt19937 rng(42);

    std::vector<TrackResult> results;
    std::vector<HitResult>   all_hits;
    results.reserve(N_TRACKS);
    all_hits.reserve(N_TRACKS * N_TOTAL);

    for (int t = 0; t < N_TRACKS; t++) {
        if ((t + 1) % 100 == 0)
            printf("  Track %d/%d\n", t + 1, N_TRACKS);

        std::uniform_real_distribution<float> p_dist(5.0f, 50.0f);
        std::uniform_int_distribution<int>    q_dist(0, 1);
        std::uniform_real_distribution<float> tx_dist(-0.1f, 0.1f);
        std::uniform_real_distribution<float> ty_dist(-0.05f, 0.05f);
        std::uniform_real_distribution<float> xy_dist(-5.0f, 5.0f);

        float p = p_dist(rng);
        float charge = q_dist(rng) ? 1.0f : -1.0f;
        float qop = charge * Extrapolators::c_light / (p * 1e3f);

        State s0{xy_dist(rng), xy_dist(rng), LAYER_Z[0],
                 tx_dist(rng), ty_dist(rng), qop};

        results.push_back(fit_track(s0, rng, all_hits, t));
    }

    // Write track results
    {
        FILE* fp = fopen("kalman_results.csv", "w");
        fprintf(fp, "track_id,p_true,p_fit,chi2,ndof,chi2_ndof,"
                    "res_x,res_y,res_tx,res_ty,res_qop,"
                    "pull_x,pull_y,pull_tx,pull_ty,pull_qop\n");
        for (int t = 0; t < N_TRACKS; t++) {
            auto& r = results[t];
            fprintf(fp, "%d,%.6f,%.6f,%.4f,%d,%.4f,"
                        "%.8f,%.8f,%.10f,%.10f,%.12f,"
                        "%.6f,%.6f,%.6f,%.6f,%.6f\n",
                    t, r.p_true, r.p_fit, r.chi2, r.ndof,
                    r.ndof > 0 ? r.chi2 / r.ndof : 0.0f,
                    r.res_x, r.res_y, r.res_tx, r.res_ty, r.res_qop,
                    r.pull_x, r.pull_y, r.pull_tx, r.pull_ty, r.pull_qop);
        }
        fclose(fp);
        printf("  -> kalman_results.csv written (%d tracks)\n", N_TRACKS);
    }

    // Write hit-level results
    {
        FILE* fp = fopen("kalman_hits.csv", "w");
        fprintf(fp, "track_id,layer,z,region,meas_res_x,meas_res_y,"
                    "filt_res_x,filt_res_y,pull_x,pull_y\n");
        for (auto& h : all_hits) {
            fprintf(fp, "%d,%d,%.2f,%s,%.8f,%.8f,%.8f,%.8f,%.6f,%.6f\n",
                    h.track_id, h.layer, h.z, h.region,
                    h.res_x, h.res_y, h.filt_res_x, h.filt_res_y,
                    h.pull_x, h.pull_y);
        }
        fclose(fp);
        printf("  -> kalman_hits.csv written (%zu entries)\n", all_hits.size());
    }

    // ── Part 3: MLP Kalman filter (if model loaded) ────────────────────────
    if (g_mlp_model.loaded) {
        printf("[3/3] Running Kalman filter (MLP) on 500 tracks...\n");
        std::mt19937 rng_mlp(42);  // same seed for fair comparison

        std::vector<TrackResult> mlp_results;
        std::vector<HitResult>   mlp_hits;
        mlp_results.reserve(N_TRACKS);
        mlp_hits.reserve(N_TRACKS * N_TOTAL);

        for (int t = 0; t < N_TRACKS; t++) {
            if ((t + 1) % 100 == 0)
                printf("  Track %d/%d\n", t + 1, N_TRACKS);

            std::uniform_real_distribution<float> p_dist(5.0f, 50.0f);
            std::uniform_int_distribution<int>    q_dist(0, 1);
            std::uniform_real_distribution<float> tx_dist(-0.1f, 0.1f);
            std::uniform_real_distribution<float> ty_dist(-0.05f, 0.05f);
            std::uniform_real_distribution<float> xy_dist(-5.0f, 5.0f);

            float p = p_dist(rng_mlp);
            float charge = q_dist(rng_mlp) ? 1.0f : -1.0f;
            float qop = charge * Extrapolators::c_light / (p * 1e3f);

            State s0{xy_dist(rng_mlp), xy_dist(rng_mlp), LAYER_Z[0],
                     tx_dist(rng_mlp), ty_dist(rng_mlp), qop};

            mlp_results.push_back(fit_track(s0, rng_mlp, mlp_hits, t,
                                            extrapolate_state_mlp,
                                            compute_jacobian_mlp));
        }

        // Write MLP track results
        {
            FILE* fp = fopen("kalman_results_mlp.csv", "w");
            fprintf(fp, "track_id,p_true,p_fit,chi2,ndof,chi2_ndof,"
                        "res_x,res_y,res_tx,res_ty,res_qop,"
                        "pull_x,pull_y,pull_tx,pull_ty,pull_qop\n");
            for (int t = 0; t < N_TRACKS; t++) {
                auto& r = mlp_results[t];
                fprintf(fp, "%d,%.6f,%.6f,%.4f,%d,%.4f,"
                            "%.8f,%.8f,%.10f,%.10f,%.12f,"
                            "%.6f,%.6f,%.6f,%.6f,%.6f\n",
                        t, r.p_true, r.p_fit, r.chi2, r.ndof,
                        r.ndof > 0 ? r.chi2 / r.ndof : 0.0f,
                        r.res_x, r.res_y, r.res_tx, r.res_ty, r.res_qop,
                        r.pull_x, r.pull_y, r.pull_tx, r.pull_ty, r.pull_qop);
            }
            fclose(fp);
            printf("  -> kalman_results_mlp.csv written (%d tracks)\n", N_TRACKS);
        }

        // Write MLP hit-level results
        {
            FILE* fp = fopen("kalman_hits_mlp.csv", "w");
            fprintf(fp, "track_id,layer,z,region,meas_res_x,meas_res_y,"
                        "filt_res_x,filt_res_y,pull_x,pull_y\n");
            for (auto& h : mlp_hits) {
                fprintf(fp, "%d,%d,%.2f,%s,%.8f,%.8f,%.8f,%.8f,%.6f,%.6f\n",
                        h.track_id, h.layer, h.z, h.region,
                        h.res_x, h.res_y, h.filt_res_x, h.filt_res_y,
                        h.pull_x, h.pull_y);
            }
            fclose(fp);
            printf("  -> kalman_hits_mlp.csv written (%zu entries)\n", mlp_hits.size());
        }
    } else {
        printf("[3/3] Skipping MLP Kalman fit (no model loaded)\n");
    }

    printf("\nDone.\n");
    return 0;
}
