# V4: Wide Architectures + PINN Training Investigation

**Status:** 📋 Planning  
**Goal:** Achieve sub-mm position accuracy with variable dz using very wide, shallow networks; diagnose and fix PINN training failures  
**Key Innovation:** Systematic width scaling study + root-cause analysis of PINN position errors

> **📖 COMPREHENSIVE DOCUMENTATION**  
> This directory includes extensive documentation of the PINN architecture investigation:
> - **[PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md)** - Complete technical analysis (11,000 words)
> - **[DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md)** - Navigation guide and document roadmap
> - **[README_DOCUMENTATION.md](README_DOCUMENTATION.md)** - Documentation summary and statistics
>
> **Quick Links:**
> - [V2/QUICK_REFERENCE.md](../V2/QUICK_REFERENCE.md) - One-page visual summary
> - [V2/PINN_ARCHITECTURE_ISSUE.md](../V2/PINN_ARCHITECTURE_ISSUE.md) - Executive summary
> - [physics_exploration.ipynb](../physics_exploration.ipynb) Section 8 - Interactive analysis

---

## 📋 Table of Contents

1. [V3 Summary & Lessons Learned](#v3-summary--lessons-learned)
2. [Improvement Opportunities](#improvement-opportunities)
3. [V4 Strategy](#v4-strategy)
4. [Experiment Plan: Architecture Width Sweep](#experiment-plan-architecture-width-sweep)
5. [Experiment Plan: PINN Training Investigation](#experiment-plan-pinn-training-investigation)
6. [Infrastructure & Tooling](#infrastructure--tooling)
7. [Success Criteria](#success-criteria)

---

## V3 Summary & Lessons Learned

### V3 Results (after bug fixes)

> **⚠️ Note:** V3 PINN models show large position errors (~50mm) due to architectural limitations.
> See [PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md) for root cause analysis.
> V4 implements fixes that are expected to reduce this to <1mm.

| Model | Params | C++ Time (µs) | Speedup vs RK4 | Pos RMSE (mm) | Slope RMSE |
|-------|--------|---------------|-----------------|---------------|------------|
| Linear | 0 | 0.04 | 2131× | 1.63 | 0.000308 |
| Parabolic | 0 | 0.09 | 947× | 1.63 | 0.000285 |
| **RK4** | 0 | **85.2** | **1.0×** | **0 (truth)** | **0** |
| MLP deep_128 | 59k | 65.0 | 1.3× | 1.01 | 0.0115 |
| MLP shallow_256 | 101k | 116.2 | 0.7× | 1.01 | 0.0092 |
| MLP deep_256 | 233k | 265.7 | 0.3× | 1.06 | 0.0117 |
| MLP shallow_512 | 399k | 465.6 | 0.2× | 0.96 | 0.0094 |
| PINN col5 | 69k | 79.3 | 1.1× | 49.4 | 0.000249 |
| PINN col10 | 69k | 79.3 | 1.1× | 54.4 | 0.000249 |

### V2 Best Results (fixed dz=8000mm, for reference)

| Model | Params | PyTorch Time (µs) | Pos Mean (mm) | Speedup vs C++ RK4 |
|-------|--------|--------------------|---------------|---------------------|
| mlp_v2_single_256 | ~12k | 0.83 | 0.065 | 3.0× |
| mlp_v2_shallow_256 | ~72k | 1.50 | 0.044 | 1.7× |
| mlp_v2_shallow_512_256 | ~140k | 1.93 | 0.028 | 1.3× |

**⚠️ V2 timings are PyTorch batched (10k tracks), not C++ single-sample — not directly comparable.**

### Key Lessons

1. **Width > Depth for this problem.** V2's [512, 256] 2-layer network (0.028mm) massively outperformed V3's [128, 128, 128, 128, 64] 5-layer network (1.01mm) despite fewer parameters. The extrapolation function is smooth and physics-governed — wide layers can approximate it with fewer compositions.

2. **Variable dz is ~10× harder than fixed dz.** Even comparing architectures of similar width, V3's best (0.96mm) is ~35× worse than V2's best (0.028mm). The model must now learn a *family* of extrapolations parameterised by dz.

3. **PINN residual architecture has a fundamental position error.** The `Output = IC + z_frac × Correction` formula forces linear interpolation in z_frac, but positions evolve nonlinearly through the variable magnetic field. The encoder did NOT include z_frac as input, preventing position-dependent corrections. This produces ~50mm position RMSE while slopes (which are approximately linear) get excellent ~0.00025 RMSE. **Full analysis:** [PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md)

4. **C++ single-sample inference is expensive.** The V3 benchmark showed that single-track C++ inference of even a [256, 256] network takes ~80µs — comparable to RK4 at 85µs. The matrix-vector multiply dominates. Batching, SIMD, or algorithmic simplifications are needed for speed gains.

---

## Improvement Opportunities

### Where can we beat existing extrapolators?

The benchmark reveals several axes for improvement, each with different constraints:

### 1. Inference Speed (currently the bottleneck)

**Current situation:**
- Parabolic: **0.09 µs** (1 field lookup + algebra)
- RK4 (5mm step): **85 µs** (~4 × dz/step field lookups, ~3400 for dz=8500mm)
- MLP [256,256]: **79 µs** (~68k FMAs + 512 exp() calls)
- MLP [128,128,128,128,64]: **65 µs** (~59k FMAs + some exp())

**Improvement space:**
| Optimisation | Estimated Gain | Complexity |
|-------------|---------------|------------|
| **Single hidden layer** (no activation overhead between layers) | 10-20% | Easy — architecture change |
| **ReLU instead of SiLU** (avoid 512 `exp()` calls) | 10-15% | Easy — but may hurt accuracy |
| **float32 instead of float64** (2× throughput, halved memory bandwidth) | ~2× | Medium — needs accuracy validation |
| **SIMD vectorisation** (AVX2: 4 doubles at once) | 2-4× | Medium — manual or compiler hints |
| **Batched inference** (process N tracks at once) | 5-50× | Medium — needs API change in LHCb framework |
| **Smaller network** ([256] single layer = ~1.5k FMAs) | 50× | Easy — if accuracy is sufficient |
| **Quantisation** (INT8 weights) | 4-8× | Hard — needs careful calibration |

**Key insight:** A single-layer [1024] network has 6×1024 + 1024×4 = 10,240 FMAs — **6.6× fewer** than a [256, 256] network (68k FMAs) while having more representational capacity per layer.

### 2. Accuracy

**Current situation:**
- Parabolic: ~1.6mm position RMSE, ~0.0003 slope RMSE
- Best MLP: ~0.96mm position RMSE, ~0.009 slope RMSE
- Best PINN: ~49mm position RMSE (broken), ~0.00025 slope RMSE (excellent)

**Improvement space:**
- MLPs already beat Parabolic on position (~1.0mm vs ~1.6mm) — good
- MLP slope accuracy (~0.009) is 30× worse than Parabolic (~0.0003) — room to improve
- Wider networks should reduce both errors
- PINN fix could give best-of-both-worlds: MLP-level position + physics-level slopes

### 3. Collocation Points & Step Amounts

**Current situation:** V3 PINNs were trained with 5, 10, 20, 50 collocation points. All gave nearly identical results because the architecture (not the data) was the bottleneck.

**Improvement space:**
- After fixing the PINN architecture, collocation count becomes a meaningful hyperparameter
- More collocation = smoother learned trajectory = potentially better generalisation
- But more collocation = more memory per batch = smaller effective batch size
- **Physical collocation** (PDE residual loss, no ground truth needed) would be the ultimate goal — eliminates need for pre-computed trajectory data

### 4. B-Field Reading Times

**Current situation:** Each trilinear interpolation of the field map costs ~50-100ns. RK4 does ~3400 lookups for dz=8500mm (step=10mm). This is ~170-340µs of field lookups alone.

**Improvement space:**
- ML models **bypass the field map entirely** — this is the fundamental advantage
- The ML model implicitly learns the integrated effect of the field
- **Larger RK4 steps** (e.g. 50mm) reduce lookups 5× but lose accuracy
- The Parabolic extrapolator uses only 1 field lookup and achieves 1.6mm accuracy — proving that a single field evaluation captures most physics
- A hybrid approach (ML correction on top of Parabolic) could be interesting

### 5. Variable vs Fixed dz

**Current situation:** V3 uses variable dz ∈ [500, 12000]mm, making the problem ~10× harder than V2's fixed dz=8000mm.

**Improvement space:**
- **Binned models:** Train separate models for dz ranges (e.g. 0-2000, 2000-6000, 6000-12000) — each model has a simpler mapping to learn
- **dz as auxiliary input, not primary:** Explore architectures where dz modulates the network (e.g. FiLM conditioning) rather than being just another input feature
- **dz normalisation:** Currently dz has std=2893, mean=3855. Could normalise to [0,1] using the known range

---

## V4 Strategy

V4 has two parallel tracks:

### Track A: MLP Width Scaling (production-focused)

**Goal:** Find the Pareto-optimal MLP width for variable-dz extrapolation.
- Fix architecture to **single or two hidden layers** (shallow)
- Sweep width systematically
- Benchmark in C++ for honest timing
- Target: sub-mm position accuracy at ≤ RK4 speed

### Track B: PINN Training Investigation (research-focused)

**Goal:** Understand and fix the PINN position error.
- Diagnose the root cause experimentally
- Test architectural fixes
- If successful, PINNs could offer better generalisation + slope accuracy than MLPs

---

## Experiment Plan: Architecture Width Sweep

### Phase 1: Single-Layer Width Scan

Train 6 single-hidden-layer MLPs to isolate the effect of width:

| Model ID | Architecture | Params | Est. FMAs | Est. C++ Time |
|----------|-------------|--------|-----------|---------------|
| `mlp_v4_single_256` | [256] | 2,564 | 2,560 | ~2 µs |
| `mlp_v4_single_512` | [512] | 5,124 | 5,120 | ~4 µs |
| `mlp_v4_single_1024` | [1024] | 10,244 | 10,240 | ~8 µs |
| `mlp_v4_single_2048` | [2048] | 20,484 | 20,480 | ~16 µs |
| `mlp_v4_single_4096` | [4096] | 40,964 | 40,960 | ~32 µs |
| `mlp_v4_single_8192` | [8192] | 81,924 | 81,920 | ~65 µs |

**Hypothesis:** Width of 2048-4096 should achieve sub-mm accuracy with a single layer, at 16-32µs — potentially 2-5× faster than RK4.

### Phase 2: Two-Layer Width Scan

For the widths that show promising results in Phase 1, add a second layer:

| Model ID | Architecture | Params | Est. FMAs |
|----------|-------------|--------|-----------|
| `mlp_v4_wide_1024_512` | [1024, 512] | 524k | 524k |
| `mlp_v4_wide_2048_512` | [2048, 512] | 1.0M | 1.0M |
| `mlp_v4_wide_2048_1024` | [2048, 1024] | 2.1M | 2.1M |
| `mlp_v4_wide_1024_256` | [1024, 256] | 268k | 268k |
| `mlp_v4_wide_512_256` | [512, 256] | 134k | 134k |

**These replicate V2's winning strategy** ([512, 256] was V2's best) but with more width headroom.

### Phase 3: Activation Function Comparison

For the best Phase 1/2 architectures, compare activations:

| Activation | Cost per neuron | Expected accuracy |
|-----------|----------------|-------------------|
| **SiLU** (current) | 1 exp() + 2 mul + 1 div | Best (smooth) |
| **ReLU** | 1 comparison | 10-15% faster, possibly rougher |
| **GELU** | 1 erf() + mul | Similar to SiLU, slower |
| **Swish-1** | Same as SiLU | Same as SiLU |

### Training Configuration

All models use the same training setup for fair comparison:

```yaml
data: V3/data/training_mlp_v3_100M_v2.npz  # 100M samples, variable dz
batch_size: 4096
epochs: 100 (with early stopping, patience=10)
optimizer: AdamW
learning_rate: 0.001
scheduler: cosine annealing with 5-epoch warmup
loss: MSE (physical space, denormalized)
activation: SiLU (default, compare in Phase 3)
train/val/test: 80/10/10
```

### Deliverables

- Position RMSE vs width plot (single-layer)
- Position RMSE vs FLOPs plot (all architectures)
- C++ timing vs width plot
- Speed-accuracy Pareto frontier (V4 vs V3 vs V2 vs baselines)

---

## Experiment Plan: PINN Training Investigation

> **📖 COMPREHENSIVE ANALYSIS AVAILABLE**  
> See [PINN_ARCHITECTURE_DIAGNOSIS.md](PINN_ARCHITECTURE_DIAGNOSIS.md) for complete root cause analysis,
> mathematical proofs, experimental evidence, and detailed implementation of all three V4 solutions.

### Executive Summary of PINN Issue

**Root Cause Identified (Feb 2026):**
- V2/V3 PINN encoder did NOT include z_frac as input
- Network predicts single correction vector per track
- Correction scaled linearly: `output = IC + z_frac × correction`
- Variable magnetic field requires position-dependent corrections
- Linear ansatz cannot represent nonlinear trajectories → 50mm position errors

**V4 Solutions:**
1. **PINNZFracInput**: Add z_frac as 7th encoder input ⭐ **Recommended**
2. **QuadraticResidual**: Polynomial basis (z + z²) for corrections
3. **PDE-Residual PINN**: True physics loss via autodiff (future work)

**Expected Improvements:**
- Position: 50mm → <1mm (50× improvement)
- Slopes: Maintain ~0.0003 accuracy (PINN advantage preserved)

### Diagnosis: Why does the PINN fail on position?

#### Experiment P1: Verify the linear z_frac hypothesis

**Setup:** Take the trained `pinn_v3_res_256_col10` model and evaluate it at multiple z_frac values for a set of test tracks. Compare predictions vs ground truth trajectories.

**Expected result:** Position predictions should fall on a straight line between IC and endpoint (because the architecture forces `output = IC + z_frac × correction`), while true trajectories curve through the magnet.

**Metric:** Plot predicted vs true trajectory for 10 representative tracks. Compute maximum deviation from linear interpolation.

#### Experiment P2: Remove collocation — endpoint-only PINN

**Setup:** Train the same `PINNResidual` architecture but with `lambda_collocation = 0`. This makes the PINN equivalent to an MLP with a skip connection (output = IC + network_output at z_frac=1).

**Hypothesis:** Without the collocation loss fighting the endpoint loss for positions, position accuracy may improve. The collocation loss creates a conflicting gradient signal: it wants the correction vector to give good *intermediate* positions (which requires nonlinear z_frac dependence), but the architecture only allows *linear* z_frac dependence.

| Model | Collocation | Expected Position | Expected Slope |
|-------|------------|-------------------|----------------|
| `pinn_v4_nocol` | None | ~1mm (MLP-like) | ~0.01 (MLP-like) |
| `pinn_v4_col10` | 10 points | ~50mm (unchanged) | ~0.0003 (good) |

#### Experiment P3: PINNWithZFracInput architecture

**Setup:** Use the `PINNWithZFracInput` variant already defined in [pinn_residual.py](experiments/next_generation/V3/models/pinn_residual.py). This feeds z_frac as a 7th input to the core network, allowing the network to learn *nonlinear* z_frac dependence.

**Architecture:** `core(x, y, tx, ty, qop, dz, z_frac)` → 4-dim output. Output = IC + z_frac × core_output.

**Hypothesis:** The network can now modulate its correction based on z_frac, enabling quadratic (and higher) trajectories. Position accuracy should improve dramatically.

| Model | Architecture | Input Dim |
|-------|-------------|-----------|
| `pinn_v4_zfrac_256` | PINNWithZFracInput [256, 256] | 7 |
| `pinn_v4_zfrac_512` | PINNWithZFracInput [512, 512] | 7 |

#### Experiment P4: Quadratic residual architecture

**Setup:** Modify `PINNResidual` to output TWO correction vectors:

```python
# New: output = IC + z_frac × correction_1 + z_frac² × correction_2
correction_1 = self.core_linear(state_dz)    # [B, 4]
correction_2 = self.core_quadratic(state_dz) # [B, 4]
output = initial + z_frac * correction_1 + z_frac**2 * correction_2
```

**Hypothesis:** The quadratic term captures the dominant nonlinearity in position (magnetic bending is approximately parabolic). This preserves the IC guarantee while allowing accurate intermediate states.

| Model | Architecture | Output Dim |
|-------|-------------|------------|
| `pinn_v4_quad_256` | Quadratic residual [256, 256] | 8 (2×4) |

#### Experiment P5: Loss function investigation

**Setup:** Keep the `PINNResidual` architecture but modify the loss:

| Variant | Change | Hypothesis |
|---------|--------|-----------|
| `pinn_v4_posweight` | Weight position terms 10× in endpoint loss | Forces model to prioritise position accuracy |
| `pinn_v4_physloss` | Replace supervised collocation with ODE residual loss | True physics-informed loss, no trajectory data needed |
| `pinn_v4_physloss_no_col` | Only endpoint + ODE residual (no collocation) | Purest PINN approach |

The **ODE residual loss** evaluates the Lorentz force equations at collocation points:
```
dtx/dz_predicted - κ·N·[tx·ty·Bx - (1+tx²)·By + ty·Bz] = 0
dty/dz_predicted - κ·N·[(1+ty²)·Bx - tx·ty·By - tx·Bz] = 0
```
This requires autodiff through the model w.r.t. z_frac and a field map lookup at each collocation point during training. Much more expensive per sample but eliminates the need for pre-computed trajectory ground truth.

#### Experiment P6: Head-to-head MLP vs PINN at matched width

**Setup:** Train both an MLP and a fixed PINN at the same width, same data, same epochs, to isolate the architecture effect:

| Model | Type | Architecture | Training |
|-------|------|-------------|----------|
| `mlp_v4_matched_512` | MLP | [512, 512] | MSE endpoint only |
| `pinn_v4_matched_512_nocol` | PINN | [512, 512] residual, no collocation | MSE endpoint only |
| `pinn_v4_matched_512_col10` | PINN | [512, 512] residual, 10 col | MSE endpoint + collocation |
| `pinn_v4_zfrac_matched_512` | PINN | [512, 512] z_frac input, 10 col | MSE endpoint + collocation |

This is the **controlled experiment** that isolates whether the PINN architecture itself helps or hurts.

---

## Infrastructure & Tooling

### Training

- **Cluster:** Condor GPU cluster at Nikhef
- **Environment:** conda env `TE` (PyTorch, CUDA)
- **Data:** Reuse V3 training data (`training_mlp_v3_100M_v2.npz`, 100M samples)
- **PINN data:** Reuse V3 trajectory data with collocation (`training_pinn_v3_col*_v2.npz`)

### Benchmarking

- **C++ benchmark:** Extend `V3/analysis/benchmark_cpp.cpp` for V4 models
- **Timing:** C++ single-sample with `-O2`, same methodology as V3
- **Accuracy:** 100 test tracks, forward dz ∈ [500, 12000]mm, RK4 ground truth

### Directory Structure

```
V4/
├── README.md              ← This file
├── models/                ← Architecture definitions
├── training/              ← Training scripts
├── trained_models/        ← Checkpoints + configs + histories
├── cluster/               ← Condor submission scripts
├── analysis/              ← Benchmark notebook + results
├── deployment/            ← Export to C++ binary
└── data_generation/       ← (symlink to V3/data, reuse same data)
```

---

## Success Criteria

### Minimum Success (Track A — MLP width sweep)

- [ ] At least one MLP architecture achieves **< 0.5mm position RMSE** with variable dz
- [ ] At least one MLP architecture achieves **≥ 1× speedup vs RK4** (< 85µs C++ single-sample)
- [ ] Both simultaneously: **< 1mm position AND < 85µs** → viable RK4 replacement

### Stretch Goals

- [ ] Sub-0.1mm position RMSE (matching V2's fixed-dz accuracy)
- [ ] 2× faster than RK4 (< 42µs)
- [ ] PINN matching MLP on position while maintaining superior slope accuracy

### Track B — PINN Investigation

- [ ] Root-cause for position error confirmed experimentally (P1)
- [ ] At least one PINN variant achieves < 5mm position RMSE (P2-P4)
- [ ] Head-to-head comparison document: MLP vs PINN at matched width (P6)
- [ ] Recommendation: use PINN or MLP for V5 production?

---

## Execution Priority

| Priority | Experiment | Effort | Impact |
|----------|-----------|--------|--------|
| 🔴 **High** | Phase 1: Single-layer width scan (6 models) | 2 days | Identifies optimal width |
| 🔴 **High** | P1: Verify z_frac linearity issue | 0.5 day | Confirms root cause |
| 🟡 **Medium** | P2: Endpoint-only PINN | 1 day | Quick PINN fix attempt |
| 🟡 **Medium** | Phase 2: Two-layer width scan (5 models) | 2 days | Refines best architecture |
| 🟡 **Medium** | P3: PINNWithZFracInput | 1 day | Simplest architectural fix |
| 🟢 **Low** | Phase 3: Activation comparison | 1 day | ~10% speed improvement |
| 🟢 **Low** | P4: Quadratic residual | 1 day | Elegant fix, more complex |
| 🟢 **Low** | P5: Physics residual loss | 2 days | True PINN, harder to implement |
| 🟢 **Low** | P6: Full head-to-head | 2 days | Final comparison for paper |

**Estimated total: ~12 days of cluster training + 3-4 days of analysis**

---

## Authors

- G. Scriven (LHCb Collaboration)
- February 2026
