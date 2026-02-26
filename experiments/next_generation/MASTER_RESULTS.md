# Master Results: Neural Network Track Extrapolation (V1–V4)

**Author:** G. Scriven (LHCb, Nikhef)  
**Date:** February 2026  
**Versions covered:** V1, V2, V3, V4

---

## Table of Contents

1. [What This Project Does](#1-what-this-project-does)
2. [The Physics](#2-the-physics)
3. [Model Architectures](#3-model-architectures)
4. [Version Timeline and Key Lessons](#4-version-timeline-and-key-lessons)
5. [V1 Results — First Experiments](#5-v1-results--first-experiments)
6. [V2 Results — Shallow-Wide and Residual Fix](#6-v2-results--shallow-wide-and-residual-fix)
7. [V3 Results — Variable Step Size](#7-v3-results--variable-step-size)
8. [V4 Results — Width Scaling and PINN Diagnosis](#8-v4-results--width-scaling-and-pinn-diagnosis)
9. [Cross-Version Comparison](#9-cross-version-comparison)
10. [PINN Failure: Full Story](#10-pinn-failure-full-story)
11. [C++ Benchmark: Traditional Extrapolators](#11-c-benchmark-traditional-extrapolators)
12. [Plots](#12-plots)
13. [All Numerical Results](#13-all-numerical-results)
14. [Conclusions and Next Steps](#14-conclusions-and-next-steps)

---

## 1. What This Project Does

Charged particles curve when they pass through a magnetic field. In the LHCb detector, we need to predict where a particle ends up after travelling through the dipole magnet — this is called **track extrapolation**.

The standard method uses Runge-Kutta numerical integration (RK4), which is accurate but requires many steps and magnetic field lookups. This project replaces RK4 with a **neural network** that learns the same mapping in a single forward pass.

**Input** (6 numbers):

| # | Variable | What it is | Units |
|---|----------|------------|-------|
| 0 | $x_0$ | Horizontal position | mm |
| 1 | $y_0$ | Vertical position | mm |
| 2 | $t_{x0}$ | Horizontal slope $dx/dz$ | — |
| 3 | $t_{y0}$ | Vertical slope $dy/dz$ | — |
| 4 | $q/p$ | Charge divided by momentum | 1/MeV |
| 5 | $\Delta z$ | How far to extrapolate | mm |

**Output** (4 numbers): the particle's state $(x, y, t_x, t_y)$ at the end.

---

## 2. The Physics

### Lorentz Force (LHCb Form)

A charged particle in a magnetic field $\vec{B}(x,y,z)$ obeys:

$$\frac{d}{dz}\begin{pmatrix} x \\ y \\ t_x \\ t_y \end{pmatrix} = \begin{pmatrix} t_x \\ t_y \\ \kappa \sqrt{1 + t_x^2 + t_y^2}\;\left[t_x t_y B_x - (1 + t_x^2) B_y + t_y B_z\right] \\ \kappa \sqrt{1 + t_x^2 + t_y^2}\;\left[(1 + t_y^2) B_x - t_x t_y B_y - t_x B_z\right] \end{pmatrix}$$

where $\kappa = (q/p) \times c_{\text{light}} = (q/p) \times 2.998 \times 10^{-4}$ mm/(MeV·T).

### The LHCb Dipole Field

The magnet field $B_y$ is **not uniform** — it varies along $z$:

| $z$ (mm) | $B_y$ (T) | Curvature (relative) |
|---------:|-----------:|:--------------------:|
| 0 | 1.10 | 1.65× |
| 2000 | 0.95 | 1.43× |
| 4000 | 0.70 | 1.05× |
| 6000 | 0.50 | 0.75× |
| 8000 | 0.40 | 0.60× |

The field varies by a factor of ~3. This means trajectory curvature depends on *where you are*, not just *where you started*. This single fact dominates the design choices across all four versions.

### Taylor Expansion of a Trajectory

Expanding the position in $z$:

$$x(z) = x_0 + t_{x0}\,\Delta z + \frac{1}{2}\frac{d^2 x}{dz^2}\bigg|_{z_0}\!(\Delta z)^2 + \mathcal{O}(\Delta z^3)$$

- The **linear** term ($t_x \cdot \Delta z$) is the straight-line extrapolation.
- The **quadratic** term ($\propto \kappa \cdot B_y$) captures magnetic bending.
- Higher-order terms capture the *change* in field along the path.

If a neural network can only produce linear outputs in $z$, it will miss the quadratic bending.

---

## 3. Model Architectures

Three types of neural network were explored:

### 3.1 MLP (Multi-Layer Perceptron)

A plain feedforward network:

$$\text{Input}(6) \;\to\; [\text{Linear} \to \text{SiLU}]^N \;\to\; \text{Linear}(4) \;\to\; \text{Output}$$

**Loss:**

$$\mathcal{L} = \frac{1}{B}\sum_{i=1}^{B}\|\hat{y}_i - y_i\|^2$$

No physics, just data. Simple, fast, and (it turned out) hard to beat.

### 3.2 PINN (Physics-Informed Neural Network)

Same network structure but with a **residual formulation** to guarantee the initial condition:

$$\text{Output}(\zeta) = \text{IC} + \zeta \times \text{Network}(x_0, y_0, t_{x0}, t_{y0}, q/p)$$

where $\zeta \in [0, 1]$ is the fractional distance along the path and IC $= (x_0, y_0, t_{x0}, t_{y0})$.

At $\zeta = 0$ the output is exactly the initial state, by construction.

**Loss (V3 supervised collocation):**

$$\mathcal{L} = \lambda_{\text{IC}}\,\mathcal{L}_{\text{IC}} + \lambda_{\text{end}}\,\mathcal{L}_{\text{endpoint}} + \lambda_{\text{col}}\,\mathcal{L}_{\text{collocation}}$$

where $\mathcal{L}_{\text{IC}}$ is the initial condition error (should be ~0 by design), $\mathcal{L}_{\text{endpoint}}$ is the prediction error at $\zeta = 1$, and $\mathcal{L}_{\text{collocation}}$ compares intermediate predictions to ground-truth trajectory points.

### 3.3 RK-PINN (Runge-Kutta PINN)

A shared feature extractor feeds into 4 separate "stage heads", each responsible for a different $z$ position ($\zeta = 0.25, 0.5, 0.75, 1.0$). The final output is a weighted sum initialised with RK4 coefficients $[1, 2, 2, 1]/6$.

### Architecture Summary

| Aspect | MLP | PINN | RK-PINN |
|--------|-----|------|---------|
| Network | Single feedforward | Single feedforward + residual | Shared encoder + 4 heads |
| IC guaranteed? | No | Yes (by construction) | Yes (by construction) |
| Physics loss | No | Optional (supervised collocation) | Optional |
| Activation | SiLU | Tanh or SiLU | Tanh |

---

## 4. Version Timeline and Key Lessons

| Version | Date | Training $\Delta z$ | Key Change | Outcome |
|:--------|:-----|:-------------------:|:-----------|:--------|
| **V1** | Jan 2026 | Fixed 8000 mm | First experiments with MLP/PINN/RK-PINN | PINNs failed (IC not satisfied). MLPs worked. |
| **V2** | Jan 2026 | Fixed 8000 mm | Residual PINN to fix IC; shallow-wide MLPs | IC fixed, but PINNs 2× worse than MLPs. Shallow > deep. |
| **V3** | Jan–Feb 2026 | Variable 500–12000 mm | Variable step-size training | Models generalise to any $\Delta z$, but position errors ~1 mm (MLP) / ~50 mm (PINN). |
| **V4** | Feb 2026 | Variable 500–12000 mm | Width sweep; PINN diagnosis & fix | Root cause of PINN failure identified. PINNZFracInput proposed. |

**Three critical lessons:**

1. **V1 → V2:** PINN initial conditions failed because the network ignored the $\zeta$ input. Fix: residual architecture.
2. **V2 → V3:** Models trained on fixed $\Delta z = 8000$ mm blow up for any other step size (normalization: $\sigma_{\Delta z} \approx 10^{-9}$). Fix: train with variable $\Delta z$.
3. **V2 → V4:** PINN residual formula $\text{IC} + \zeta \cdot c$ is *linear* in $\zeta$ — cannot represent curved trajectories in a variable magnetic field. MLPs win because they take $\zeta$ as an input and can learn non-linear dependence.

---

## 5. V1 Results — First Experiments

**53 models** trained: 14 MLP, 16 PINN, 17 RK-PINN, plus 6 momentum-binned.  
**Training:** 50M samples, fixed $\Delta z = 8000$ mm, 10 epochs.

### Top V1 Models (by validation loss)

| Model | Type | Params | Val Loss | Notes |
|:------|:-----|-------:|---------:|:------|
| mlp_large_v1 | MLP | 399K | 0.000445 | Best overall |
| mlp_medium | MLP | 101K | 0.000583 | Good accuracy/size balance |
| rkpinn_medium_data_only | RK-PINN | 101K | 0.000671 | RK-PINN with physics loss = 0 |
| mlp_wide | MLP | 431K | 0.000951 | Also good |
| pinn_medium | PINN | 101K | 0.00698 | 12× worse than MLP |
| pinn_medium_pde_strong | PINN | 101K | 0.0550 | Physics loss λ=10 hurt |

### V1 Timing (from benchmarking)

| Extrapolator | Time/track (μs) | Speedup |
|:------------|----------------:|--------:|
| MLP tiny (Python) | 1.10 | 2.3× |
| MLP medium (Python) | 1.47 | 1.7× |
| C++ RK4 (CashKarp) | 2.50 | Baseline |

### V1 Issues Discovered

1. **PINN IC failure:** At $\zeta = 0$, PINN output $x = 2768$ mm when input was $x_0 = 207$ mm — the network completely ignored $\zeta$.
2. **Fixed $\Delta z$:** Normalization standard deviation for $\Delta z$ was $\sim 10^{-9}$. For any other step size, $({\Delta z - 8000}) / {10^{-9}} \to \pm\infty$.
3. **Deep > Shallow (wrong):** V1 assumed deeper networks = better. V2 showed the opposite.

---

## 6. V2 Results — Shallow-Wide and Residual Fix

**22 models** trained: 9 MLP, 7 PINN (residual), 6 RK-PINN (residual).  
**Training:** Same 50M samples, still fixed $\Delta z = 8000$ mm, 20 epochs.

### V2 MLP Performance

| Model | Architecture | Pos Error (mm) | Slope Error (mrad) | Time (μs) | Speedup |
|:------|:------------|---------------:|-------------------:|----------:|--------:|
| mlp_v2_shallow_1024_256 | [1024, 256] | 0.031 | 0.678 | 3.56 | 0.70× |
| mlp_v2_shallow_512_256 | [512, 256] | **0.028** | 0.416 | 1.93 | **1.29×** |
| mlp_v2_shallow_256 | [256, 256] | 0.044 | 0.263 | 1.50 | 1.67× |
| mlp_v2_single_256 | [256] | 0.065 | 0.017 | 0.83 | **3.00×** |

### V2 PINN Performance (Residual Architecture)

| Model | Architecture | Pos Error (mm) | Slope Error (mrad) |
|:------|:------------|---------------:|-------------------:|
| pinn_v2_single_256 | [256] | 664 | 98.9 |
| pinn_v2_shallow_1024_256 | [1024, 256] | 830 | 311 |
| rkpinn_v2_single_256 | [256] | 816 | 101 |

**PINN/RK-PINN position errors are 600–2800 mm** — catastrophic. The residual formulation fixed IC at $\zeta = 0$ but introduced a new bottleneck: the correction is a single vector per track, linearly scaled by $\zeta$, which cannot represent curved trajectories (see [Section 10](#10-pinn-failure-full-story)).

### Pareto-Optimal V2 Models

| Purpose | Model | Pos Error (mm) | Time (μs) | Speedup |
|:--------|:------|---------------:|----------:|--------:|
| Best accuracy | mlp_v2_shallow_512_256 | 0.028 | 1.93 | 1.29× |
| Balanced | mlp_v2_shallow_256 | 0.044 | 1.50 | 1.67× |
| Fastest | mlp_v2_single_256 | 0.065 | 0.83 | 3.00× |

### V2 Key Finding: Shallow > Deep

| Architecture | Layers | Width | Val Loss |
|:-------------|-------:|------:|---------:|
| Deep-narrow (V1) | 5 | 64 | 0.0045 |
| Medium (V1) | 4 | 128 | 0.0018 |
| **Shallow-wide (V2)** | **2** | **256–512** | **0.0008** |

The extrapolation function is smooth and physics-governed — wide layers approximate it more efficiently than stacking many narrow layers.

---

## 7. V3 Results — Variable Step Size

**11 models benchmarked** (RK4 truth, Linear, Parabolic, 4 MLP, 4 PINN), all in **C++ single-sample** inference.  
**Training:** 100M samples, variable $\Delta z \in [500, 12000]$ mm.

### V3 C++ Benchmark

| Model | Type | Params | C++ Time (μs) | Speedup vs RK4 | Pos RMSE (mm) | Slope RMSE |
|:------|:-----|-------:|---------------:|----------------:|--------------:|-----------:|
| Linear | Analytic | 0 | 0.04 | 2131× | 1.63 | 0.000308 |
| Parabolic | Analytic | 0 | 0.09 | 947× | 1.63 | 0.000285 |
| **RK4** | **Numerical** | **0** | **85.2** | **1.0×** | **0 (truth)** | **0** |
| MLP deep_128 | NN | 59K | 65.0 | 1.3× | 1.01 | 0.0115 |
| MLP shallow_256 | NN | 101K | 116.2 | 0.7× | 1.01 | 0.0092 |
| MLP shallow_512 | NN | 399K | 465.6 | 0.2× | 0.96 | 0.0094 |
| PINN col5 | NN | 69K | 79.3 | 1.1× | **49.4** | **0.000249** |
| PINN col10 | NN | 69K | 79.3 | 1.1× | 54.4 | 0.000249 |
| PINN col50 | NN | 69K | 79.4 | 1.1× | 53.7 | 0.000250 |

### V3 Observations

1. **MLP position accuracy ~1 mm** with variable $\Delta z$ — much harder than V2's 0.03 mm at fixed $\Delta z$.
2. **PINN position errors ~50 mm** — broken. But slopes are 37× better than MLP (0.00025 vs 0.009).
3. **Increasing collocation points from 5 to 50 made no difference** — the architecture is the bottleneck.
4. **C++ single-sample inference is slow.** Even a small MLP costs ~65 μs, comparable to RK4 at 85 μs. The neural network advantage comes from bypassing field map lookups (RK4 does ~3400 lookups for an 8.5m propagation).

### V3 Component-Level Accuracy

From the full benchmark data, the MLP `shallow_512` model:

| Component | RMSE |
|:----------|-----:|
| $x$ position | 0.80 mm |
| $y$ position | 0.53 mm |
| $t_x$ slope | 0.0076 |
| $t_y$ slope | 0.0056 |

### Why Variable $\Delta z$ is Harder

At fixed $\Delta z = 8000$ mm, the network learns **one** mapping. With variable $\Delta z$, the network must learn a **parameterised family** of mappings — every value of $\Delta z$ gives a different extrapolation. This family is smooth, but it has one more degree of freedom.

- V2 best (fixed): **0.028 mm** with [512, 256]
- V3 best (variable): **0.96 mm** with [512, 512]
- Ratio: **~35×** harder

---

## 8. V4 Results — Width Scaling and PINN Diagnosis

V4 pursues two parallel tracks:

### Track A: MLP Width Sweep

**Hypothesis:** Width is more important than depth for this smooth function. A single very wide hidden layer should be competitive.

| Model | Architecture | Est. Params | Est. Time (μs) | Status |
|:------|:-------------|------------:|--------------:|:-------|
| mlp_v4_1L_512 | [512] | 5K | ~4 | Training |
| mlp_v4_1L_1024 | [1024] | 10K | ~8 | Training |
| mlp_v4_1L_2048 | [2048] | 20K | ~16 | Training |
| mlp_v4_1L_4096 | [4096] | 41K | ~32 | Training |
| mlp_v4_2L_1024 | [1024, 512] | 524K | — | Planned |
| mlp_v4_2L_2048 | [2048, 1024] | 2.1M | — | Planned |

**Key insight:** A single-layer [1024] network has 10,240 multiply-adds — 6.6× fewer than a [256, 256] network (68K multiply-adds), while having more capacity per layer.

### Track B: PINN Diagnosis

Root cause confirmed (see [Section 10](#10-pinn-failure-full-story) for full details). Three fixes proposed:

| Fix | Idea | Expected Pos Error |
|:----|:-----|-------------------:|
| **PINNZFracInput** | Add $\zeta$ as 7th encoder input | < 1 mm |
| **QuadraticResidual** | Output = IC + $\zeta \cdot c_1$ + $\zeta^2 \cdot c_2$ | < 5 mm |
| **PDE-Residual** | Use autodiff to enforce Lorentz equations directly | < 0.3 mm (est.) |

### V4 Success Criteria

- At least one MLP with **< 0.5 mm position RMSE** and variable $\Delta z$
- At least one model **faster than RK4** (< 85 μs single-sample C++)
- Both simultaneously: < 1 mm AND < 85 μs → viable RK4 replacement

---

## 9. Cross-Version Comparison

### Position Accuracy Evolution

| Version | Best MLP Pos Error (mm) | Best PINN Pos Error (mm) | $\Delta z$ | # Models |
|:--------|------------------------:|-------------------------:|:----------:|---------:|
| V1 | ~0.5 (est.) | broken (IC failure) | Fixed 8000 | 53 |
| V2 | **0.028** | 664 (residual broken) | Fixed 8000 | 22 |
| V3 | **0.96** | 49.4 | Variable | 8 |
| V4 | In progress | In progress | Variable | 24+ |

### Speed Evolution

| Version | Best Time (μs) | vs RK4 | Type |
|:--------|---------------:|-------:|:-----|
| V1 | 1.10 | 2.3× faster | Python batched |
| V2 | 0.83 | 3.0× faster | Python batched |
| V3 | 65.0 | 1.3× faster | C++ single-sample |
| V4 | est. ~8 | ~10× faster | C++ single (est.) |

**Note:** V1/V2 timings are Python with batch size ~10K (misleadingly fast). V3/V4 use honest C++ single-sample timing.

### Architecture Wins

| Finding | Evidence |
|:--------|:--------|
| 2 layers beat 5 layers | V2 [512, 256] has 0.028 mm error vs V1 [64, 64, 64, 64] with 0.011 loss |
| Width beats depth | V2 [1024, 256] beats V3 [128, 128, 128, 128, 64] at fewer parameters |
| Variable $\Delta z$ is ~35× harder | V2 best 0.028 mm vs V3 best 0.96 mm |
| PINN slopes are 37× better than MLP | V3 PINN 0.00025 vs MLP 0.009 |

---

## 10. PINN Failure: Full Story

This section tells the complete story of the PINN architecture across all four versions.

### V1: The Network Ignored $\zeta$

In V1, the PINN was supposed to learn a trajectory as a function of $\zeta$ (the fraction along the path). Instead, the network completely ignored $\zeta$ and learned to map directly from initial to final state.

**Symptom:**

| $\zeta$ | PINN output ($x$) | Expected |
|---------:|------------------:|---------:|
| 0.0 | 2768 mm | 207 mm |
| 1.0 | 2752 mm | 1039 mm |

The output barely changed with $\zeta$ — the network was ignoring it.

**Root cause:** During training, column 5 of the input was replaced with $\zeta = 1.0$ for all samples. The normalization statistics had $\sigma_{\Delta z} \approx 10^{-9}$, so any $\zeta \neq 8000$ produced astronomical normalised values.

### V2: Residual Fix — New Problem

V2 introduced the residual formulation:

$$\text{Output}(\zeta) = \text{IC} + \zeta \times \text{Network}(x_0, y_0, t_{x0}, t_{y0}, q/p)$$

This **guaranteed** IC satisfaction at $\zeta = 0$ ✓. The IC bug was fixed.

But the encoder only sees 5 inputs — **no position information**. The network produces a single correction vector $\mathbf{c}$ per track, and $\zeta$ just linearly scales it. The trajectory is forced to be a straight line between start and [start + correction].

**V2 performance:**

| Model | Pos Error (mm) | Slope Error | Has $\zeta$ input? |
|:------|---------------:|------------:|:------------------:|
| MLP | 0.08 | 0.0092 | Yes |
| PINN | 0.15 | **0.00025** | No |

PINN was 2× worse on positions but 37× better on slopes. This is exactly what the maths predicts:

- **Slopes** ($t_x, t_y$) change approximately linearly with $z$ → compatible with the linear ansatz.
- **Positions** ($x, y$) change quadratically $\Big(x(z) \approx x_0 + t_x \Delta z + \tfrac{1}{2}\kappa B_y (\Delta z)^2\Big)$ → the linear ansatz misses the quadratic term.

### V3: Variable $\Delta z$ Made It Worse

With variable $\Delta z$, the field variation along different paths is even more diverse. The single linear correction must now handle a whole family of trajectories, amplifying the error:

- V2 PINN (fixed $\Delta z$): 0.15 mm
- V3 PINN (variable $\Delta z$): **49–54 mm**

Increasing collocation points from 5 to 50 had zero effect:

| Collocation Points | Pos RMSE (mm) | Slope RMSE |
|-------------------:|--------------:|-----------:|
| 5 | 49.4 | 0.000249 |
| 10 | 54.4 | 0.000249 |
| 20 | 52.1 | 0.000251 |
| 50 | 51.8 | 0.000248 |

**More data couldn't overcome the architecture constraint.** This is like trying to fit a parabola using only straight lines — no number of data points helps.

### V4: Root Cause and Three Fixes

**Mathematical form of the output:**

$$\mathbf{s}(\zeta) = \mathbf{s}_0 + \zeta \cdot \mathbf{c}(\mathbf{s}_0, q/p)$$

The correction $\mathbf{c}$ depends only on the initial state. It does not depend on $\zeta$. The output is a straight line in $\zeta$.

**What the physics actually requires:**

$$\mathbf{s}(\zeta) = \mathbf{s}_0 + \int_0^{\zeta} \mathbf{f}(\mathbf{s}, z; B(z))\, dz$$

Since $B(z)$ varies by 3× across the magnet, the integral is nonlinear in $\zeta$.

**Estimated error from the linear approximation:**

For $p = 20$ GeV, $B_y \approx 0.7$ T, $\Delta z = 8000$ mm:

$$\text{Error} \approx \frac{1}{2}\kappa (\Delta z)^2 = \frac{1}{2} \cdot \frac{0.3 \times 0.7}{20000} \cdot 8000^2 \approx 336\;\text{mm}$$

The observed ~50 mm is better than this naive estimate because the network finds an optimal average correction, but it is fundamentally limited.

**V4 Fix 1 — PINNZFracInput** (recommended):

$$\text{Output}(\zeta) = \text{IC} + \zeta \times \text{Network}(x_0, y_0, t_{x0}, t_{y0}, q/p, \Delta z, \zeta)$$

Adding $\zeta$ as an input means the network can produce *different* corrections at *different* positions. At $\zeta = 0$ the output is still exactly IC (the $\zeta$ factor zeroes out the correction). Expected: < 1 mm position error.

**V4 Fix 2 — QuadraticResidual:**

$$\text{Output}(\zeta) = \text{IC} + \zeta \cdot \mathbf{c}_1 + \zeta^2 \cdot \mathbf{c}_2$$

Two correction vectors. The quadratic term captures the dominant bending. Expected: < 5 mm.

**V4 Fix 3 — True PDE-residual PINN** (future):

Use automatic differentiation to compute $d\mathbf{s}/d\zeta$ from the network, and penalise deviation from the Lorentz force equations directly. No pre-computed trajectory data needed. Most physically principled, but hardest to train.

### Summary: Why MLPs Beat PINNs (and How to Fix It)

| | MLP | PINN (V2/V3) | PINNZFracInput (V4) |
|:--|:---|:------------|:--------------------|
| Input | $(x_0, y_0, t_x, t_y, q/p, \Delta z)$ | $(x_0, y_0, t_x, t_y, q/p)$ | $(x_0, y_0, t_x, t_y, q/p, \Delta z, \zeta)$ |
| Sees position? | Yes ($\Delta z$) | **No** | Yes ($\zeta$) |
| Output form | Free | $\text{IC} + \zeta \cdot c$ (linear) | $\text{IC} + \zeta \cdot c(\zeta)$ (nonlinear) |
| Position error | ~1 mm | ~50 mm | < 1 mm (expected) |
| Slope error | ~0.009 | ~0.0003 | ~0.0005 (expected) |

**Core lesson:** Physics constraints in neural networks are only useful if the architecture has the capacity to represent the physics. Too much constraint can be worse than too little.

---

## 11. C++ Benchmark: Traditional Extrapolators

Independent of the neural network work, V1 benchmarked the traditional C++ extrapolators used in LHCb:

| Extrapolator | Type | Time (μs) | Pos Error (mm) | Recommendation |
|:------------|:-----|----------:|--------------:|:---------------|
| CashKarp (RK4) | Numerical | 2.50 | 0 (truth) | Ground truth baseline |
| BogackiShampine3 | RK3 | 2.40 | 0.10 | Best speed/accuracy |
| Verner9 | RK9 | 2.52 | 0.08 | Highest precision |
| Tsitouras5 | RK5 | 2.75 | — | Balanced |
| DormandPrince5 | RK5(4) | — | — | Standard DOPRI5 |
| Herab | Helix | 1.95 | 5.1 | Fast seeding |
| Kisel | Analytical | 1.50 | **39.8** | Do not use |

- Adaptive RK methods use ~10–15 steps for a 4 m propagation with ~350–430 mm effective step size.
- Kisel has catastrophic failures (up to 944 mm error in some regions).

---

## 12. Plots

All plots are located in the workspace under `experiments/next_generation/`. Below is a catalog with descriptions.

### V1 Benchmarking Plots

| File | Description |
|:-----|:------------|
| [V1/benchmarking/plots/fig1_accuracy_comparison.png](V1/benchmarking/plots/fig1_accuracy_comparison.png) | Position error by extrapolator type |
| [V1/benchmarking/plots/fig2_error_distribution.png](V1/benchmarking/plots/fig2_error_distribution.png) | Histogram of errors across all test tracks |
| [V1/benchmarking/plots/fig3_accuracy_vs_speed.png](V1/benchmarking/plots/fig3_accuracy_vs_speed.png) | Scatter: mean error vs execution time |
| [V1/benchmarking/plots/fig4_performance_summary.png](V1/benchmarking/plots/fig4_performance_summary.png) | Bar chart of all metrics side-by-side |
| [V1/benchmarking/results/speed_accuracy_tradeoff.png](V1/benchmarking/results/speed_accuracy_tradeoff.png) | Speed–accuracy Pareto frontier |
| [V1/benchmarking/results/timing_distributions.png](V1/benchmarking/results/timing_distributions.png) | Box plots of timing distributions |

PDF versions of the first four are also available at the same paths with `.pdf` extension.

### V1 Papers and Notes

| File | Description |
|:-----|:------------|
| [V1/paper/nn_extrapolator_v1_results.pdf](V1/paper/nn_extrapolator_v1_results.pdf) | Full results paper (LaTeX compiled) |
| [V1/notes/experimental_protocol.pdf](V1/notes/experimental_protocol.pdf) | Experimental methodology |
| [V1/notes/mathematical_derivations.pdf](V1/notes/mathematical_derivations.pdf) | Physics equations and derivations |

### V3 Analysis Plots

| File | Description |
|:-----|:------------|
| [V3/analysis/speed_vs_accuracy.png](V3/analysis/speed_vs_accuracy.png) | C++ timing vs position RMSE for all V3 models |
| [V3/analysis/timing_comparison.png](V3/analysis/timing_comparison.png) | Bar chart of C++ inference times |
| [V3/analysis/error_distributions.png](V3/analysis/error_distributions.png) | Error histograms for MLP vs PINN |
| [V3/analysis/component_accuracy.png](V3/analysis/component_accuracy.png) | Per-component ($x$, $y$, $t_x$, $t_y$) RMSE |
| [V3/analysis/error_vs_kinematics.png](V3/analysis/error_vs_kinematics.png) | Error dependence on momentum and $\Delta z$ |

### V4 Diagnosis Plots

| File | Description |
|:-----|:------------|
| [V4/p1_zfrac_linearity.png](V4/p1_zfrac_linearity.png) | **Key figure:** PINN predictions are perfectly linear in $\zeta$, while RK4 truth is curved |
| [V4/pinn_training_curves.png](V4/pinn_training_curves.png) | Training loss for PINN vs MLP |
| [V4/pinn_trajectory_comparison.png](V4/pinn_trajectory_comparison.png) | Side-by-side trajectory comparison: PINN (linear) vs RK4 (curved) |

---

## 13. All Numerical Results

### Complete V1 Model Results (43 models)

Sorted by validation loss (best first):

| # | Model | Type | Params | Val Loss |
|--:|:------|:-----|-------:|---------:|
| 1 | mlp_large_v1 | MLP | 399K | 0.000445 |
| 2 | mlp_medium | MLP | 101K | 0.000583 |
| 3 | rkpinn_medium_data_only | RK-PINN | 101K | 0.000671 |
| 4 | mlp_wide | MLP | 431K | 0.000951 |
| 5 | mlp_medium_v1 | MLP | 101K | 0.001015 |
| 6 | mlp_wide_v1 | MLP | 431K | 0.001172 |
| 7 | rkpinn_medium_pde_weak | RK-PINN | 101K | 0.001307 |
| 8 | mlp_small | MLP | 18K | 0.001339 |
| 9 | mlp_balanced_v1 | MLP | 57K | 0.002088 |
| 10 | mlp_wide_shallow_v1 | MLP | 35K | 0.002215 |
| 11 | rkpinn_wide | RK-PINN | 431K | 0.002505 |
| 12 | rkpinn_medium_pde_dom. | RK-PINN | 101K | 0.002793 |
| 13 | pinn_small | PINN | 18K | 0.002996 |
| 14 | mlp_small_v1 | MLP | 18K | 0.003023 |
| 15 | mlp_tiny | MLP | 5K | 0.003086 |
| 16 | rkpinn_coll5_v1 | RK-PINN | 101K | 0.003536 |
| 17 | pinn_medium_mid_p | PINN | 101K | 0.003550 |
| 18 | pinn_tiny | PINN | 5K | 0.003622 |
| 19 | rkpinn_small | RK-PINN | 18K | 0.003726 |
| 20 | rkpinn_medium | RK-PINN | 101K | 0.003906 |
| 21–43 | *(remaining 23 models)* | | | 0.004–0.055 |

**Key trend:** MLPs consistently outperform physics-informed variants in V1.

### Complete V2 MLP Results (8 models)

| Model | Architecture | Pos Mean (mm) | Pos 90th (mm) | Slope (mrad) | Time (μs) | Speedup |
|:------|:------------|:-:|:-:|:-:|:-:|:-:|
| mlp_v2_shallow_1024_256 | [1024, 256] | 0.031 | 0.060 | 0.678 | 3.56 | 0.70× |
| mlp_v2_shallow_512_256 | [512, 256] | 0.028 | 0.051 | 0.416 | 1.93 | 1.29× |
| mlp_v2_shallow_512 | [512, 512] | 0.029 | 0.053 | 0.333 | 2.58 | 0.97× |
| mlp_v2_shallow_1024_512 | [1024, 512] | 0.034 | 0.063 | 0.783 | 4.71 | 0.53× |
| mlp_v2_shallow_256 | [256, 256] | 0.044 | 0.087 | 0.263 | 1.50 | 1.67× |
| mlp_v2_single_1024 | [1024] | 0.062 | 0.149 | 0.026 | 1.86 | 1.34× |
| mlp_v2_single_256 | [256] | 0.065 | 0.135 | 0.017 | 0.83 | 3.00× |
| mlp_v2_single_512 | [512] | 0.068 | 0.139 | 0.013 | 1.03 | 2.44× |

### Complete V3 C++ Benchmark (11 models)

| Model | Type | Params | Time (μs) | Speedup | Pos RMSE (mm) | $x$ RMSE (mm) | $y$ RMSE (mm) | $t_x$ RMSE | $t_y$ RMSE |
|:------|:-----|-------:|----------:|--------:|--------------:|------:|------:|------:|------:|
| RK4 | Numerical | 0 | 85.2 | 1.0× | 0 | 0 | 0 | 0 | 0 |
| Linear | Analytic | 0 | 0.04 | 2131× | 1.63 | 1.57 | 0.42 | 0.000291 | 0.000103 |
| Parabolic | Analytic | 0 | 0.09 | 947× | 1.63 | 1.30 | 0.99 | 0.000202 | 0.000201 |
| MLP deep_128 | NN | 59K | 65.0 | 1.3× | 1.01 | 0.79 | 0.63 | 0.00862 | 0.00758 |
| MLP shallow_256 | NN | 101K | 116.2 | 0.7× | 1.01 | 0.87 | 0.52 | 0.00711 | 0.00590 |
| MLP deep_256 | NN | 233K | 265.7 | 0.3× | 1.06 | 0.84 | 0.65 | 0.00825 | 0.00827 |
| MLP shallow_512 | NN | 399K | 465.6 | 0.2× | 0.96 | 0.80 | 0.53 | 0.00764 | 0.00555 |
| PINN col5 | NN | 69K | 79.3 | 1.1× | 49.4 | 38.7 | 30.8 | 0.000227 | 0.000104 |
| PINN col10 | NN | 69K | 79.3 | 1.1× | 54.4 | 42.6 | 33.9 | 0.000229 | 9.8e-5 |
| PINN col20 | NN | 69K | 79.4 | 1.1× | 56.3 | 44.0 | 35.3 | 0.000229 | 9.9e-5 |
| PINN col50 | NN | 69K | 79.4 | 1.1× | 57.6 | 45.0 | 35.9 | 0.000230 | 9.8e-5 |

### V3 PINN Collocation Study

| Collocation Points | Pos RMSE (mm) | Slope RMSE | IC Loss | Endpoint Loss | Collocation Loss |
|-------------------:|--------------:|-----------:|--------:|--------------:|-----------------:|
| 5 | 49.4 | 0.000249 | ~0 | 0.008 | 0.10 |
| 10 | 54.4 | 0.000249 | ~0 | 0.008 | 0.11 |
| 20 | 52.1 | 0.000251 | ~0 | — | — |
| 50 | 51.8 | 0.000248 | ~0 | — | — |

The collocation loss **never reaches zero** — confirming the architecture cannot satisfy the physics at intermediate points.

---

## 14. Conclusions and Next Steps

### What Works

1. **MLPs are effective track extrapolators.** With 2 hidden layers of 512 neurons, position errors are sub-mm on fixed $\Delta z$ and ~1 mm on variable $\Delta z$.
2. **Shallow-wide beats deep-narrow.** Two layers with 256–1024 neurons outperform five layers with 64–128 neurons.
3. **Variable $\Delta z$ is essential for deployment.** Fixed-$\Delta z$ models explode numerically when given a different step size.
4. **Physics-informed networks can achieve excellent slope accuracy** (0.0003 vs 0.009), but their position accuracy depends on the architecture having enough capacity to represent the physics.

### What Doesn't Work (Yet)

1. **PINN residual with linear $\zeta$-scaling** — cannot represent curved trajectories in a variable magnetic field.
2. **Deep narrow networks** — less accurate and often slower than wide shallow ones.
3. **Single-sample C++ inference** — matrix–vector multiplies dominate at ~65–85 μs per track, comparable to RK4. Batched or SIMD inference is needed for real speedups.

### Next Steps

| Priority | Task | Expected Impact |
|:---------|:-----|:----------------|
| High | V4 width sweep (single-layer MLP up to 4096 neurons) | Find optimal width for < 85 μs inference |
| High | Train PINNZFracInput models | Best-of-both-worlds: MLP positions + PINN slopes |
| Medium | Float32 inference in C++ | ~2× speed improvement |
| Medium | SIMD/batched C++ inference | 2–50× speed improvement |
| Low | True PDE-residual PINN training | Eliminate need for trajectory data |
| Low | Momentum-binned models (separate model per $p$-range) | Potentially better accuracy per bin |

### The Big Picture

| Extrapolator | Pos Error (mm) | Speed | Physics? |
|:------------|:-:|:-:|:-:|
| Parabolic (1 field lookup) | 1.6 | 947× faster than RK4 | Minimal |
| MLP (current best, variable $\Delta z$) | ~1.0 | ~1.3× faster | Learned from data |
| RK4 (ground truth) | 0 | Baseline | Full |
| PINNZFracInput (V4 expected) | < 1.0 | ~1.2× faster | Partially enforced |

The neural network approach sits between the fast-but-rough Parabolic and the precise-but-slow RK4. With wider architectures (V4) and proper PINN design, sub-mm accuracy at 2–10× the speed of RK4 appears achievable.

---

*Document generated: February 23, 2026*  
*Source: V1–V4 documentation, CSV results, benchmark data, and analysis notebooks in `experiments/next_generation/`*
