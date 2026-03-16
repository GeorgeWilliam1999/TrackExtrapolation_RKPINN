# C++ Source Code — TrackExtrapolators

This directory contains the C++ production code for the `Tr/TrackExtrapolators` package in the LHCb framework. All components are built via the top-level [CMakeLists.txt](../CMakeLists.txt) as the `TrackExtrapolators` Gaudi module.

---

## Extrapolators

Track extrapolators propagate charged particle states `[x, y, tx, ty, q/p]` from one z-position to another through the LHCb magnetic field.

| File | Class | Method | Lines | Description |
|------|-------|--------|-------|-------------|
| `TrackRungeKuttaExtrapolator.cpp` | `TrackRungeKuttaExtrapolator` | Adaptive RK4 | 1444 | Primary production extrapolator. Supports CashKarp, BogackiShampine, Verner, and Tsitouras Butcher tableaux. Adaptive step-size control with error estimation. |
| `TrackSTEPExtrapolator.cpp` | `TrackSTEPExtrapolator` | RKN (ATLAS STEP) | 677 | Runge-Kutta-Nyström algorithm from ATLAS (Lund, Bugge, Gavrilenko, Strandlie). Exploits 2nd-order ODE structure for efficiency. |
| `TrackKiselExtrapolator.cpp` | `TrackKiselExtrapolator` | Kisel polynomial | 457 | Originally developed by I. Kisel for CBM. Fast polynomial integration, no multiple scattering. |
| `TrackHerabExtrapolator.cpp` | `TrackHerabExtrapolator` | Hera-B RK5 | 1220 | 5th-order Runge-Kutta from the Hera-B experiment. No multiple scattering. |
| `TrackLinearExtrapolator.cpp` | `TrackLinearExtrapolator` | Straight-line | 96 | Simplest extrapolator — straight-line propagation ignoring the magnetic field. Useful reference and for field-free regions. |
| `TrackParabolicExtrapolator.cpp` | `TrackParabolicExtrapolator` | Parabolic | 212 | 2nd-order parabolic trajectory approximation. Fast but limited accuracy in non-uniform fields. |
| `TrackParametrizedExtrapolator.cpp` | `TrackParametrizedExtrapolator` | Basis functions | 452 | Analytical parametrization using Eigen-based basis function products (`BasisFunctions.h`). Reads coefficients from DetDesc conditions. |

### Neural Network Extrapolator

| File | Class | Lines | Description |
|------|-------|-------|-------------|
| `../ml_models/src/TrackMLPExtrapolator.cpp` | `TrackMLPExtrapolator` | 571 | Eigen-based neural network inference. Loads binary `.bin` model files, supports ReLU/Tanh/SiLU/Sigmoid activations, optional PINN residual mode. See [ml_models/README.md](../ml_models/README.md). |

---

## Orchestration & Selection

| File | Class | Lines | Description |
|------|-------|-------|-------------|
| `TrackMasterExtrapolator.cpp` | `TrackMasterExtrapolator` | 247 | Delegates to other extrapolators based on configuration. The main entry point used by reconstruction algorithms. |
| `TrackDistanceExtraSelector.cpp` | `TrackDistanceExtraSelector` | 45 | Selects which extrapolator to use based on propagation distance. |
| `TrackSimpleExtraSelector.cpp` | `TrackSimpleExtraSelector` | 34 | Simple extrapolator selection strategy (single extrapolator for all cases). |

---

## Material Effects

| File | Class | Lines | Description |
|------|-------|-------|-------------|
| `MaterialLocatorBase.h` / `.cpp` | `MaterialLocatorBase` | 83 / 254 | Base class for material locators implementing the `IMaterialLocator` interface. |
| `DetailedMaterialLocator.cpp` | `DetailedMaterialLocator` | 92 | Uses the `TransportSvc` to find materials along a trajectory for detailed energy-loss and multiple-scattering corrections. |
| `SimplifiedMaterialLocator.cpp` | `SimplifiedMaterialLocator` | 184 | Simplified material model using DetDesc geometry. Only available with DetDesc backend (`#ifndef USE_DD4HEP`). |

---

## Base Classes

| File | Class | Lines | Description |
|------|-------|-------|-------------|
| `TrackExtrapolator.h` / `.cpp` | `TrackExtrapolator` | 105 | Base class implementing the `ITrackExtrapolator` interface. Provides common `propagate()` method signatures. |
| `TrackFieldExtrapolatorBase.h` / `.cpp` | `TrackFieldExtrapolatorBase` | 79 | Base for extrapolators needing magnetic field access. Handles `DeMagnet` / `ILHCbMagnetSvc` service lookup and grid interpolation. |
| `BasisFunctions.h` | — | 148 | Eigen-based template functors for creating product bases of functions (used by `TrackParametrizedExtrapolator`). |

---

## State Management

| File | Class | Lines | Description |
|------|-------|-------|-------------|
| `TrackStateProvider.cpp` | `TrackStateProvider` | 430 | Provides and caches track states at requested z-positions. Used by downstream algorithms to avoid redundant extrapolation. |

---

## Testing

| File | Class | Lines | Description |
|------|-------|-------|-------------|
| `ExtrapolatorTester.cpp` | `ExtrapolatorTester` | ~105 | Original Gaudi algorithm for testing extrapolators. |
| `TrackExtrapolatorTesterSOA.cpp` | `TrackExtrapolatorTesterSOA` | 252 | SOA (Structure of Arrays) benchmark tester with timing measurements. Used to establish the C++ RK4 baseline (2.50 μs/track). Test grid: 11×11 = 121 track states at various momenta and angles. |

### Test Configuration Files

Located in `../tests/options/`:

| File | Purpose |
|------|---------|
| `test_extrapolators.py` | Standard extrapolator test configuration |
| `benchmark_extrapolators.py` | Benchmark all extrapolators with timing |
| `benchmark_extrapolators_v2.py` | Updated benchmark configuration |
| `benchmark_many_events.py` | Extended benchmark with many events |
| `benchmark_minimal.py` | Minimal benchmark for quick checks |
| `benchmark_with_root.py` | Benchmark with ROOT output |

---

## Build

All sources are compiled as a single Gaudi module via `CMakeLists.txt`:

```cmake
gaudi_add_module(TrackExtrapolators
    SOURCES src/*.cpp ml_models/src/TrackMLPExtrapolator.cpp
    LINK Eigen3::Eigen Gaudi::GaudiAlgLib LHCb::MagnetLib ...
)
```

Key dependencies: Gaudi, Eigen3, GSL, LHCb kernel/magnet/track libraries, Boost headers.

---

*Last Updated: March 2026*
