# Critical Issue: V2 PINN Architecture Limitations

**Date:** February 2026  
**Status:** 🔴 **ROOT CAUSE IDENTIFIED**  
**Impact:** PINN models underperform MLPs by ~2× despite physics-informed training

---

## Executive Summary

The V2 PINN and RK_PINN architectures have a fundamental design flaw that limits their ability to extrapolate particle trajectories in the LHCb variable magnetic field. The networks predict **position-independent corrections** that are linearly scaled by the z position, which cannot capture the **position-dependent curvature** caused by the varying magnetic field.

**Key Finding:** The PINN encoder only processes the initial state `[x0, y0, tx0, ty0, qop]` and produces a single correction vector per track. This correction is then scaled linearly by `z_frac`, forcing a linear interpolation ansatz that is too restrictive for the physics.

---

## The Architecture Flaw

### Current Implementation

```python
# Step 1: Encode ONLY initial state (z_frac not included!)
x0_norm = (x0 - input_mean[:5]) / input_std[:5]
features = encoder(x0_norm)  # Shape: [batch, hidden_dim]

# Step 2: Predict a SINGLE correction vector
corrections = correction_head(features)  # Shape: [batch, 4]

# Step 3: Scale linearly by z_frac
delta = corrections * z_frac  # Linear scaling!

# Step 4: Add to baseline
state(z) = baseline(z) + z_frac * correction
```

**Mathematical Form:**
```
s(ζ) = s₀ + ζ · [t₀ · Δz] + ζ · c(s₀)
```

where:
- `ζ ∈ [0, 1]` is the fractional position
- `c(s₀)` is a **fixed correction** depending only on initial state
- The trajectory is a **linear interpolation** between start and endpoint

### Why This Cannot Work in Variable B Fields

In the LHCb dipole magnet:
- Magnetic field `By(z)` varies from ~1.1 T to ~0.4 T (factor of 3×)
- Trajectory curvature: `κ(z) ∝ (q/p) · By(z)` is position-dependent
- True trajectory is **nonlinear** in z
- Linear corrections can only approximate **constant** curvature

**Physics Equation:**
```
ds/dz = f(s, z; q/p, B(z))
```

The field `B(z)` is position-dependent, but the PINN correction is not!

---

## Evidence

### 1. Performance Data (from v2_model_analysis.ipynb)

| Model Type | Best Model | Position Error (mm) | Parameters |
|:-----------|:-----------|--------------------:|----------:|
| MLP | mlp_v2_shallow_256 | 0.08 | 67k |
| PINN | pinn_v2_single_256 | 0.15 | 67k |
| RK_PINN | rk_pinn_v2_4stage | 0.14 | 71k |

**PINN is 2× worse despite:**
- Similar parameter count
- Physics-informed loss with 10 collocation points
- Lorentz force explicitly enforced during training

### 2. Trajectory Analysis (from physics_exploration.ipynb)

Predictions at intermediate z positions show:
- PINN trajectory is **exactly linear** in z_frac
- RK4 ground truth shows **nonlinear** curvature variation
- Errors accumulate systematically, largest at mid-trajectory
- Increasing collocation points doesn't help (architecture is the bottleneck)

### 3. Field Variation Analysis

```python
z_range: 0 → 8000 mm
By(0):    1.1 T  (strong bending near magnet center)
By(4000): 0.7 T  (moderate bending)
By(8000): 0.4 T  (weak bending at fringe)
```

Curvature changes by **3×** along the trajectory. A linear correction cannot capture this.

---

## Why Collocation Points Don't Help

The physics loss computes at 10 collocation points:
```python
z_fracs = torch.linspace(0.1, 1.0, 10)
for z_frac in z_fracs:
    s_pred = model.forward_at_z(x0, z_frac)
    ds_dz = autograd.grad(s_pred, z_frac)  # Derivative
    residual = ||ds_dz - f(s_pred, z)||²    # Physics violation
```

**The Problem:**
- `ds_dz = c` is **constant** (because s(z) = s₀ + z·c)
- `f(s, z)` **varies** with position (because B(z) varies)
- The network finds a compromise correction that minimizes average violation
- More collocation points → more constraints → same architectural bottleneck

**Analogy:** Trying to fit a cubic polynomial using only linear basis functions. No amount of data points will overcome the model capacity limitation.

---

## Impact of Variable Magnetic Field

### In Uniform Field (hypothetical)
- Trajectory is a perfect helix with **constant** curvature
- Linear corrections might approximate well
- Physics-informed loss would be more effective

### In LHCb Variable Field (actual)
- Curvature `κ(z) ∝ By(z)` varies smoothly
- Trajectory is a **spiral with varying pitch**
- Linear ansatz is fundamentally insufficient
- Position information is **essential**

### Why MLPs Work Better

Standard MLPs take the full 6D input:
```python
mlp_input = [x0, y0, tx0, ty0, qop, dz]  # or z_frac
```

The `dz` or `z_frac` provides **explicit position information**, allowing the network to:
- Learn different corrections for different z positions
- Implicitly capture the varying curvature from data
- Not be constrained by any architectural ansatz

MLPs don't enforce physics explicitly, but their **flexibility** allows them to approximate the true nonlinear relationship.

---

## Recommended Fixes

### Option 1: Add z_frac to Encoder Input ⭐ **Simplest**

**Change:**
```python
# Before: encoder only sees initial state
features = encoder([x0, y0, tx0, ty0, qop])  # 5 inputs

# After: encoder also sees position
features = encoder([x0, y0, tx0, ty0, qop, z_frac])  # 6 inputs
```

**Benefits:**
- Minimal code change
- Network can learn position-dependent corrections
- Collocation points become meaningful
- Still maintains IC satisfaction if corrections→0 at z=0

**Implementation:**
- Change encoder input_dim: 5 → 6
- Remove `* z_frac` scaling in forward_at_z (or keep for IC enforcement)
- Retrain from scratch

---

### Option 2: Polynomial Residual Formulation

**Change:**
```python
# Linear (current)
state(z) = s₀ + z·c₁

# Quadratic (proposed)
state(z) = s₀ + z·c₁ + z²·c₂

# Cubic (even better)
state(z) = s₀ + z·c₁ + z²·c₂ + z³·c₃
```

**Benefits:**
- Can capture acceleration (quadratic) and jerk (cubic)
- No need for explicit z input
- Natural for trajectory physics (Taylor expansion)

**Note:** This is the approach used in V3/V4 models.

---

### Option 3: True PDE-Residual PINN

**Change:**
```python
# Supervised collocation (current)
s_pred = model(x0, z_frac)  # Pre-computed from data
residual = ||s_pred - s_true||²

# PDE residual (proposed)
z_col = torch.linspace(0, 1, n_col, requires_grad=True)
s_pred = model(x0, z_col)
ds_dz = autograd.grad(s_pred, z_col)  # Automatic differentiation
f_physics = lorentz_force(s_pred, z_col, B_field)
residual = ||ds_dz - f_physics||²
```

**Benefits:**
- Network free to choose any s(z) satisfying the ODE
- No architectural constraints
- True physics enforcement, not just supervised matching

**Challenges:**
- Requires differentiable field model (we have InterpolatedFieldTorch ✓)
- More expensive backward pass (second-order derivatives)
- Training can be unstable

---

### Option 4: Compositional/Recurrent Architecture

**Change:**
```python
# Current: single step 0 → 1
s_final = model(s₀, z=1.0)

# Proposed: chain short steps
s₁ = model(s₀, Δz₁)    # 0 → 0.25
s₂ = model(s₁, Δz₂)    # 0.25 → 0.5
s₃ = model(s₂, Δz₃)    # 0.5 → 0.75
s_final = model(s₃, Δz₄)  # 0.75 → 1.0
```

**Benefits:**
- Each step adapts to current state
- Shorter steps → more accurate local approximation
- Error scales as (Δz)³ per step

**Note:** RK_PINN attempted this but kept the same input limitation.

---

## Optimal Collocation Strategy for Variable Fields

Current collocation:
```python
z_fracs = torch.linspace(0.1, 1.0, 10)  # Uniform spacing
```

For variable B fields, sample where field varies most:
```python
# Dense near start (strong field)
# Dense near end (fringe effects)
# Sparse in middle (relatively uniform)
z_fracs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
```

Or adaptive based on field gradient:
```python
dB_dz = compute_field_gradient(z_range)
z_fracs = sample_proportional_to(dB_dz, n_collocation)
```

**However:** This only helps if the architecture can use the information! Fix architecture first, then optimize collocation.

---

## Lessons Learned

### 1. Physics-Informed ≠ Automatic Success

Embedding physics knowledge is powerful, **but only if the architecture has capacity to represent that physics**. 

The V2 PINN:
- ✅ Enforces initial conditions perfectly (by construction)
- ✅ Enforces Lorentz equation at collocation points
- ❌ Cannot represent position-dependent corrections
- ❌ Architectural constraint overrides physics enforcement

### 2. Residual Formulations Need Careful Design

The residual architecture `s(z) = s_baseline(z) + z·c` was designed to:
- Guarantee IC satisfaction at z=0
- Provide a good baseline to correct

But it inadvertently:
- Forces linear interpolation
- Removes position information from encoder
- Creates an over-constrained ansatz

**Better residual:** `s(z) = s_baseline(z) + network(s₀, z)` where network has z as input.

### 3. Variable Fields Require Position Information

In uniform fields:
- Constant curvature
- Simple geometric solutions
- Physics constraints are stronger

In variable fields:
- Position-dependent dynamics
- Must know "where you are" to predict "where you're going"
- Position information (z, dz, or z_frac) is essential

### 4. Expressivity vs Inductive Bias Trade-off

- **High inductive bias** (PINN residual): Guarantees certain properties, limits flexibility
- **Low inductive bias** (MLP): Maximum flexibility, learns from data alone

For complex physics with variable fields:
- Moderate inductive bias is optimal
- E.g., polynomial basis (quadratic/cubic) captures dominant physics
- But allows position-dependent corrections where needed

---

## References

### Primary Analysis Documents

1. **physics_exploration.ipynb** - Section 8: Full mathematical analysis and diagnosis
2. **v2_model_analysis.ipynb** - Section 8: Performance comparison and evidence
3. **This document** - Executive summary and recommendations

### Related Work

- V3/V4 models use polynomial residual formulation (Option 2)
- Addresses this issue by allowing quadratic terms
- Performance improvement confirms diagnosis

---

## Action Items

- [x] Identify root cause of PINN underperformance
- [x] Document mathematical limitation of linear ansatz
- [x] Validate with trajectory predictions and field analysis
- [x] Propose multiple fixes with implementation details
- [ ] Implement and test Option 1 (add z_frac to encoder)
- [ ] Compare fixed PINN with current MLP baseline
- [ ] Analyze whether physics-informed loss helps when architecture is adequate

---

## Conclusion

The V2 PINN architecture failure is **not a training problem** or a **hyperparameter problem**—it is a **fundamental capacity problem**. 

The linear interpolation ansatz cannot represent the nonlinear trajectories in the LHCb variable magnetic field. Adding more collocation points, training longer, or increasing hidden layer sizes won't fix this.

**The architecture must be changed** to allow position-dependent corrections. The fixes above provide multiple paths forward, ranging from simple (add z as input) to sophisticated (true PDE-residual PINN).

Perhaps most importantly: this analysis demonstrates that **physics-informed neural networks require careful co-design of physics constraints and architectural expressivity**. Too much constraint can be worse than too little.

---

*Document prepared by: Analysis of V2 model results and trajectory physics*  
*Last updated: February 20, 2026*
