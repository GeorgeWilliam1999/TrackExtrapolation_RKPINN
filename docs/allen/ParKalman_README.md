# ParKalman — Parametrized Kalman Filter for Allen (LHCb GPU Trigger)

This directory contains the GPU (CUDA) implementation of the **parametrized Kalman filter** used in the LHCb Allen framework for real-time track fitting in the High Level Trigger (HLT1). It is designed to run on GPUs at 30 MHz, fitting tracks that traverse the VELO, UT, and SciFi sub-detectors of the LHCb experiment.

> **ML research** (standalone test harness, MLP extrapolator, notebooks) has moved to
> [`Allen/ML_research/`](../../../ML_research/). This directory contains only the
> production Allen GPU code (`include/` and `src/`).

---

## Table of Contents

1. [Overview & Physics Context](#overview--physics-context)
2. [Directory Structure](#directory-structure)
3. [Detailed File Reference](#detailed-file-reference)
   - [Extrapolators](#extrapolators)
   - [Kalman Filter Core](#kalman-filter-core)
   - [Track Packaging & Particle Creation](#track-packaging--particle-creation)
   - [Lepton Identification](#lepton-identification)
4. [Data Flow & Algorithm Pipeline](#data-flow--algorithm-pipeline)
5. [How to Implement a New Extrapolator](#how-to-implement-a-new-extrapolator)
6. [How to Implement a New Kalman Filter Variant](#how-to-implement-a-new-kalman-filter-variant)

---

## Overview & Physics Context

The Kalman filter reconstructs charged particle trajectories ("tracks") through the LHCb detector by combining hit measurements from three sub-detectors:

- **VELO** (Vertex Locator): Silicon pixel detector around the interaction point. Provides precise position measurements but no momentum information (negligible magnetic field).
- **UT** (Upstream Tracker): Silicon strip detector just upstream of the dipole magnet. Provides a first momentum estimate.
- **SciFi** (Scintillating Fibre Tracker): Located downstream of the dipole magnet across 12 layers. Provides the primary momentum measurement.

The filter performs a **forward pass** (VELO → UT → SciFi) followed by a **backward pass** (back through the VELO), combining predictions and measurements using the standard Kalman formalism to produce fitted track states (position, slopes, momentum, covariance matrix, chi-squared).

The word "parametrized" refers to the fact that the extrapolation between detector planes uses **tuned parametric models** rather than full numerical integration of the equations of motion in the magnetic field. This is the key to achieving the required throughput on GPUs.

---

## Directory Structure

```
ParKalman/
├── include/                          # CUDA header files (.cuh)
│   ├── ButcherTableau.cuh            # Butcher tableaux for Runge-Kutta methods
│   ├── ExtrapolatorCommon.cuh        # Common state definition for extrapolators
│   ├── ParabolicExtrapolator.cuh     # Parabolic (midpoint) extrapolator
│   ├── RungeKuttaExtrapolator.cuh    # Runge-Kutta and Runge-Kutta-Nyström extrapolators
│   ├── ExtrapolateStates.cuh         # Algorithm to extrapolate Kalman states downstream
│   ├── ParKalmanFilter.cuh           # Main Kalman filter algorithm definition
│   ├── ParKalmanMethods.cuh          # Extrapolation functions between detector regions
│   ├── ParKalmanSharedConstants.cuh  # Shared GPU constant memory declarations
│   ├── ParKalmanVeloOnly.cuh         # Simplified VELO-only Kalman filter
│   ├── DownstreamKalmanFilter.cuh    # Kalman filter for downstream tracks (UT+SciFi, no VELO)
│   ├── PackageKalmanTracks.cuh       # Package long tracks into FittedTrack objects
│   ├── PackageMFTracks.cuh           # Package muon-filtered tracks
│   ├── MakeLongTrackParticles.cuh    # Create BasicParticle views from fitted tracks
│   ├── EmptyLeptonID.cuh             # Placeholder lepton ID (all zeros)
│   └── MakeLeptonID.cuh              # Combine muon/electron flags into lepton ID
│
└── src/                              # CUDA source files (.cu)
    ├── ParKalmanFilter.cu            # Main Kalman filter kernel implementation
    ├── DownstreamKalmanFilter.cu     # Downstream-only filter kernel
    ├── ExtrapolateStates.cu          # State extrapolation kernel
    ├── ParKalmanVeloOnly.cu          # VELO-only simplified fit kernel
    ├── ParKalmanSharedConstants.cu   # Shared constant memory definitions & initialization
    ├── KalmanPVIP.cu                 # Primary vertex association & impact parameter
    ├── PackageKalmanTracks.cu        # Long track packaging kernel
    ├── PackageMFTracks.cu            # Muon-filtered track packaging kernel
    ├── MakeLongTrackParticles.cu     # Particle creation kernel with monitoring
    ├── EmptyLeptonID.cu              # Empty lepton ID kernel
    └── MakeLeptonID.cu              # Lepton ID assembly kernel
```

---

## Detailed File Reference

### Extrapolators

These files implement numerical methods for propagating a track state through the magnetic field. They are **independent of the parametrized Kalman filter** and can be used for general-purpose track extrapolation.

#### `ExtrapolatorCommon.cuh` — State Definition

Defines the `Extrapolators::State` struct used by all numerical extrapolators:

```
State = { x, y, z, tx, ty, qop }
```

Where `tx = dx/dz`, `ty = dy/dz`, and `qop = q/p` (charge over momentum, converted with `c_light * eplus`).

Also defines:
- `State::Derivative` — the derivatives `{dx, dy, dz, dtx, dty}` with respect to the step variable.
- `State::Error` — error estimates for adaptive methods.
- `derivative(State, B)` — computes the derivative of the state given a magnetic field vector `B`. This encodes the Lorentz force equation: the slopes `tx, ty` change according to `qop * (cross product terms involving B)`.

#### `ButcherTableau.cuh` — Runge-Kutta Coefficient Tables

Defines Butcher tableaux as compile-time structs for use with the generic `RungeKuttaExtrapolator`. Available methods:

| Struct | Stages | Order | Adaptive? | Description |
|--------|--------|-------|-----------|-------------|
| `Euler` | 1 | 1 | No | Forward Euler (for testing) |
| `HeunEuler` | 2 | 2/1 | Yes | Heun's method with embedded error estimate |
| `RK4` | 4 | 4 | No | Classical 4th-order Runge-Kutta |
| `CashKarp` | 6 | 5/4 | Yes | Cash-Karp method (default in Allen) |

The `a_table`, `b_table`, and `b_star_table` macros work around CUDA's restrictions on `static constexpr` device data members by encoding Butcher table coefficients as constexpr device functions.

#### `ParabolicExtrapolator.cuh` — Parabolic (Midpoint) Extrapolator

A simple, fast extrapolator that:
1. Evaluates the magnetic field at the **midpoint** of the step.
2. Computes the derivative of the state at the current position.
3. Applies a **parabolic** (second-order) update: position gets a quadratic correction from the slope change due to the field.

This is the LHCb `TrackParabolicExtrapolator` ported to CUDA. It is cheap (one field lookup per step) but lower accuracy than Runge-Kutta.

**Interface:**
```cuda
ParabolicExtrapolator<float>::propagate(State& state, float dz, const Magfield& field);
```

#### `RungeKuttaExtrapolator.cuh` — Runge-Kutta Extrapolators

Contains two extrapolator classes:

**1. `RungeKuttaExtrapolator<ftype, Table>`** — Generic RK extrapolator templated on a Butcher tableau. Performs a single step of size `dz` with error estimation:

```cuda
RungeKuttaExtrapolator<float, CashKarp<float>>::propagate(State&, Error&, dz, field);
```

Evaluates the field `N_stages` times per step (6 for Cash-Karp). Returns an error estimate via the embedded method (`b - b_star`).

**2. `RungeKuttaNystromExtrapolator`** — Specialized 4th-order Runge-Kutta-Nyström method optimized for second-order ODEs (which is what the equations of motion are). Key advantages:
- Evaluates the field **only once** at the midpoint (`make_fast_step`), trading accuracy for speed.
- Can also compute the **transport Jacobian** analytically (`make_fast_step_and_evaluate_jacobian`), which is needed for covariance propagation in a Kalman filter.
- Has a `propagate()` method that takes a target z and applies multiple fixed-size steps to reach it.

#### `ExtrapolateStates.cuh / .cu` — State Extrapolation Algorithm

A standalone Allen algorithm (`extrapolate_states_t`) that takes Kalman-fitted track states and propagates them downstream using one of the numerical extrapolators. Configurable properties:
- `step_dz` (default 100 mm): Step size in z per iteration.
- `n_steps` (default 100): Number of steps.

Currently uses `RungeKuttaExtrapolator` with `CashKarp` tableau. The commented-out line shows the alternative `ParabolicExtrapolator`.

> **Note:** The `qop` must be converted with `c_light * eplus` before calling extrapolators, as they use natural units internally.

---

### Kalman Filter Core

#### `ParKalmanSharedConstants.cuh / .cu` — Shared GPU Constant Memory

Declares and defines arrays in CUDA `__constant__` memory for the parametrized extrapolation coefficients. These are loaded once at initialization from host-side `Constants` and shared between the full Kalman filter and the downstream filter:

| Array | Description |
|-------|-------------|
| `dev_V_pars` | VELO internal extrapolation parameters |
| `dev_VUT_pars` | VELO → UT transition parameters |
| `dev_UT_pars` | UT inter-layer extrapolation parameters |
| `dev_UTTF_pars` | UT → SciFi (T-station) transition parameters |
| `dev_T_pars` | SciFi inter-layer extrapolation parameters |
| `dev_TFT_pars` | SciFi first-to-last-layer transition parameters |
| `dev_UT_lay` | UT layer z-positions |
| `dev_T_lay` | SciFi layer z-positions and geometry |
| `dev_UTT_META` | Metadata for the UT→T parametrized extrapolation grid |

The `update_shared_constants()` function copies these from host to device constant memory.

#### `ParKalmanMethods.cuh` — Parametrized Extrapolation Functions

This is the largest and most complex file. It contains the `__device__` functions that perform **parametrized extrapolation** between detector layers. Unlike the numerical extrapolators (which integrate the equations of motion), these use **pre-tuned polynomial parametrizations** of the transport.

Key functions:

| Function | From → To | Description |
|----------|-----------|-------------|
| `ExtrapolateInV()` | VELO layer → VELO layer | Linear transport with momentum-dependent kick, noise model |
| `ExtrapolateVUT()` | Last VELO → First UT | Complex parametrized extrapolation through the fringe field region |
| `ExtrapolateInUT()` | UT layer → UT layer | Inter-layer transport within the UT with field correction |
| `extrapUTT()` | UT → SciFi | Grid-based parametrized extrapolation using pre-computed polynomial coefficients (`KalmanParametrizations`) |
| `ExtrapolateUTT()` | UT → SciFi (wrapper) | Calls `extrapUTT` and builds Jacobian & noise matrices |
| `ExtrapolateInT()` | SciFi layer → SciFi layer | Inter-layer transport within the SciFi tracker |
| `ExtrapolateTFT()` | SciFi → SciFi (long range) | Extrapolation over larger z ranges within SciFi |

Each function:
1. Predicts the state vector `x = (x, y, tx, ty, qop)` at the target z.
2. Fills the **Jacobian matrix** `F` (5×5 transport matrix) for covariance propagation.
3. Fills the **noise matrix** `Q` (4×4 process noise / multiple scattering).

The parametrizations depend on:
- The magnet polarity (`tI.m_polarity`).
- The current state (position, slopes, momentum).
- Tuned coefficients stored in the constant memory arrays.

#### `ParKalmanFilter.cuh / .cu` — Main Kalman Filter

The primary algorithm (`kalman_filter_t`) that fits **long tracks** (VELO + UT + SciFi). 

**Fit procedure (in `ParKalmanFilter::fit()`):**

1. **Seed state**: Initialize at the first VELO hit using positions from the first and last hits.
2. **Forward VELO pass**: Loop over VELO hits from outside-in, alternating `PredictStateV` → `UpdateStateV`.
3. **VELO → UT transition**: `PredictStateVUT` extrapolates to the first UT layer.
4. **UT layers**: Loop over 4 UT layers using the hit map, call `PredictStateUT` → `UpdateStateUT`.
5. **UT → SciFi transition**: `PredictStateUTT` performs the long-range parametrized extrapolation.
6. **SciFi layers**: Loop over 12 SciFi layers using hit maps, call `PredictStateT` → `UpdateStateT`.
7. **Backward VELO pass**: Transport the covariance back to the VELO and refit the VELO hits in the opposite direction.
8. **Beamline propagation**: Straight-line extrapolation to the point of closest approach to the beamline.

**Hit maps**: Hits are encoded in bitmaps (`make_ut_hitmap`, `make_scifi_hitmaps`) using 4 bits per layer. Value `0xf` means no hit in that layer.

**Outputs**: `FittedTrack` containing the fitted state, covariance, chi-squared contributions from each sub-detector, and number of hits.

#### `DownstreamKalmanFilter.cuh / .cu` — Downstream Track Filter

Similar to the main filter but for **downstream tracks** that have no VELO segment (only UT + SciFi). Used for long-lived particles that decay after the VELO.

Differences from the main filter:
- No VELO seed — initializes from the UT state with large covariance.
- Forward fit: UT → SciFi.
- Backward fit: through UT only (no VELO).
- Outputs include state at mid-UT position for particle-making.

#### `ParKalmanVeloOnly.cuh / .cu` — VELO-Only Simplified Filter

A lightweight **1D Kalman filter** applied independently in x and y for the VELO hits only. Used when the full parametrized filter is not needed (e.g., for fast preliminary fitting).

The `simplified_step` function implements a single Kalman predict+update cycle in 1D, including noise from multiple scattering. The `simplified_fit` function loops over all VELO hits.

This is also where the `kalman_velo_only_t` algorithm is defined, which packages VELO-only results and performs PV association.

#### `KalmanPVIP.cu` — Primary Vertex Association & Impact Parameter

Contains the `kalman_pv_ip` kernels for both the full and VELO-only filters. For each fitted track:
1. Computes the **impact parameter** (IP) to every reconstructed primary vertex.
2. Associates the track with the PV giving the smallest IP.
3. Computes `ipChi2` — the IP chi-squared including the full covariance (track + vertex).

The `Distance::kalman_ipchi2` function builds the 2×2 IP covariance matrix including contributions from the track extrapolation and the PV position uncertainty, then inverts it analytically.

---

### Track Packaging & Particle Creation

#### `PackageKalmanTracks.cuh / .cu`

Packages long tracks into `FittedTrack` objects **without** running the full Kalman filter — uses the VELO Kalman beamline state and the tracking `qop` estimate directly. Used in lightweight trigger lines.

#### `PackageMFTracks.cuh / .cu`

Packages **muon-filtered (MF) tracks** (VELO+UT tracks matched to muon stations) into `FittedTrack` objects. Similar to `PackageKalmanTracks` but operates on the subset of tracks that pass the muon filter.

#### `MakeLongTrackParticles.cuh / .cu`

Creates `BasicParticle` views from Kalman-fitted long tracks by combining:
- The fitted track state (from Kalman filter output).
- The associated primary vertex.
- Lepton identification flags.

Also fills monitoring histograms for track multiplicity, eta, phi, q/p, pT, tx, and ty.

---

### Lepton Identification

#### `EmptyLeptonID.cuh / .cu`

A no-op algorithm that initializes all lepton ID flags to zero. Used as a placeholder when lepton identification is not configured.

#### `MakeLeptonID.cuh / .cu`

Combines per-track muon and electron boolean flags into a single `uint8_t` lepton ID:
- Bit 0: is muon
- Bit 1: is electron

Works with both long tracks and downstream tracks via template dispatch.

---

## Data Flow & Algorithm Pipeline

The typical execution order in the Allen trigger sequence:

```
Track Reconstruction (VELO → UT → SciFi matching)
        │
        ▼
┌─────────────────────────┐
│   empty_lepton_id_t     │  (or make_lepton_id_t if muon/electron ID available)
│   Initialize lepton IDs │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   kalman_filter_t        │  (or kalman_velo_only_t, or downstream_kalman_filter_t)
│   Fit tracks, compute    │
│   states & covariances   │
│   + PV IP association    │
└───────────┬─────────────┘
            │
            ▼
┌──────────────────────────┐
│ make_long_track_particles│
│ Create BasicParticle     │
│ views + monitoring       │
└──────────────────────────┘
            │
            ▼
      Selection lines
      (trigger decisions)
```

---

## How to Implement a New Extrapolator

If you want to add a new numerical extrapolation method (e.g., a neural-network-based extrapolator or a different integration scheme):

### Step 1: Understand the Interface

All extrapolators operate on `Extrapolators::State` (defined in `ExtrapolatorCommon.cuh`):

```cuda
struct State {
    float x, y, z, tx, ty, qop;
};
```

An extrapolator must provide a static `__device__` method that modifies the state in-place:

```cuda
__device__ static void propagate(State& state, float dz, const MagneticField::Magfield& field);
```

The `field` object provides `field.fieldVectorLinearInterpolation(float3 pos)` which returns a `float3` with the B-field components at any point.

### Step 2: Create Your Extrapolator Header

Create a new file `include/MyExtrapolator.cuh`:

```cuda
#pragma once
#include <ExtrapolatorCommon.cuh>
#include <MagneticField.cuh>

namespace Extrapolators {
  struct MyExtrapolator {
    
    __device__ static void propagate(
      State& state, 
      float dz, 
      const MagneticField::Magfield& field)
    {
      // 1. Evaluate the magnetic field (possibly at multiple points)
      float3 B = field.fieldVectorLinearInterpolation(
          make_float3(state.x, state.y, state.z));
      
      // 2. Compute the state derivative using the provided function
      auto d = derivative(state, B);
      
      // 3. Update the state
      state.x += dz * d.dx;
      state.y += dz * d.dy;
      state.z += dz * d.dz;
      state.tx += dz * d.dtx;
      state.ty += dz * d.dty;
      // qop is constant (no energy loss modeled here)
    }
  };
} // namespace Extrapolators
```

### Step 3 (Optional): Add a Butcher Tableau

If your method is a Runge-Kutta variant, you can just add a new struct to `ButcherTableau.cuh` and use the existing `RungeKuttaExtrapolator`:

```cuda
template<typename ftype = float>
struct MyMethod {
    static constexpr int N_stages = 3;
    a_table(/* your a coefficients */)
    b_table(/* your b coefficients */)
    b_star_table(/* your b* coefficients, if adaptive */)
};
```

Then use: `RungeKuttaExtrapolator<float, ButcherTableau::MyMethod<float>>`.

### Step 4: Integrate with `ExtrapolateStates`

To use your extrapolator in the Allen pipeline, modify `ExtrapolateStates.cu` to call your method:

```cuda
#include "MyExtrapolator.cuh"

// In the kernel:
Extrapolators::MyExtrapolator::propagate(output, dz, field);
```

### Step 5 (Advanced): Provide a Jacobian

If your extrapolator should be used inside the Kalman filter for covariance transport, you need to compute the **transport Jacobian** (5×5 matrix of partial derivatives of the output state with respect to the input state). See `RungeKuttaNystromExtrapolator::make_fast_step_and_evaluate_jacobian` for an example using the Nyström method.

The Jacobian interface uses a template parameter `Matrix_t` that must support `operator()(i, j)`:

```cuda
template<typename Matrix_t>
__device__ static void propagate_with_jacobian(
    State& state, Matrix_t& jacobian, float dz, const Magfield& field);
```

---

## How to Implement a New Kalman Filter Variant

### Step 1: Understand the Existing Architecture

The Kalman filter works with:
- **State vector**: `Vector5 x = {x, y, tx, ty, qop}`
- **Covariance matrix**: `SymMatrix5x5 C`
- **Transport Jacobian**: `Matrix5x5 F`
- **Process noise**: `SymMatrix4x4 Q` (4×4 because qop noise is not modeled independently)
- **Track info**: `trackInfo tI` — carries accumulated chi-squared, reference states, polarity, etc.

Each "Predict" step does: `x' = f(x)`, `C' = F * C * F^T + Q`.  
Each "Update" step does the standard Kalman gain calculation with the hit measurement.

### Step 2: Create New Algorithm Files

For a new filter variant (e.g., for a different track type or a different fitting strategy):

1. **Header** (`include/MyKalmanFilter.cuh`): Define a `Parameters` struct with Allen `HOST_INPUT`, `DEVICE_INPUT`, `DEVICE_OUTPUT` macros, and an algorithm struct inheriting from `DeviceAlgorithm`.

2. **Source** (`src/MyKalmanFilter.cu`): Implement:
   - `set_arguments_size()` — allocate output buffers.
   - `operator()()` — launch the CUDA kernel.
   - The actual kernel function containing the fit loop.

### Step 3: Choose or Create Extrapolation Functions

You can:
- **Reuse** the existing parametrized functions from `ParKalmanMethods.cuh` (e.g., `ExtrapolateInV`, `ExtrapolateVUT`, etc.) — these are already tuned for the LHCb geometry.
- **Replace** specific transitions with numerical extrapolators. For example, to replace the UT→SciFi parametrized extrapolation with Runge-Kutta-Nyström:

```cuda
// Instead of:
ExtrapolateUTT(dev_pars, dev_UTT_META, kalman_params, x, F, Q, tI);

// Use:
Extrapolators::State extrap_state {x[0], x[1], tI.m_Lastz, x[2], x[3], x[4] * c_light};
Matrix5x5 jacobian;
RungeKuttaNystromExtrapolator::make_fast_step_and_evaluate_jacobian(
    extrap_state, jacobian, target_z - tI.m_Lastz, field);
// Then update x, F from extrap_state and jacobian
```

### Step 4: Register with Allen

After creating your files, you need to:
1. Add them to the appropriate `CMakeLists.txt` in the Allen build system.
2. Register the algorithm with `INSTANTIATE_ALGORITHM(my_namespace::my_algorithm_t)` in the `.cu` file.
3. Add the algorithm to the desired trigger sequence configuration.

### Key Design Patterns to Follow

- **One thread per track**: Each GPU thread processes one track independently.
- **Constant memory for parameters**: Use `__constant__` arrays for tuned coefficients (see `ParKalmanSharedConstants`).
- **Hit maps for sparse layers**: Encode which layers have hits using bitmaps to avoid branching.
- **Forward + backward fit**: The standard approach is forward through all sub-detectors, then backward through the VELO for the best vertex-region state.

---

## External Dependencies (Not in This Directory)

The following headers are referenced but defined elsewhere in Allen:

| Header | Contents |
|--------|----------|
| `KalmanParametrizations.cuh` | Grid-based parametrization coefficients for UT→T extrapolation |
| `ParKalmanDefinitions.cuh` | Constants (`nSetsV`, `nParsV`, `Approx_dy`, etc.) |
| `ParKalmanMath.cuh` | Matrix/vector types (`Vector5`, `SymMatrix5x5`, `Matrix5x5`) and operations |
| `ParKalmanFittedTrack.cuh` | `FittedTrack` struct definition |
| `MagneticField.cuh` | `Magfield` class with `fieldVectorLinearInterpolation()` |
| `States.cuh` | `MiniState`, `KalmanVeloState`, `Velo::Consolidated::States` |
| `*Consolidated.cuh` | Consolidated hit/track views for VELO, UT, SciFi |
| `AlgorithmTypes.cuh` | `DeviceAlgorithm` base class, `HOST_INPUT`/`DEVICE_OUTPUT` macros |
| `PV_Definitions.cuh` | `PV::Vertex` struct |
| `BeamlinePVConstants.cuh` | Beamline position and crossing angle constants |
| `FloatOperations.cuh` | Floating-point utility functions |
| `BackendCommon.h` | Backend abstraction (`UNROLL`, etc.) |
