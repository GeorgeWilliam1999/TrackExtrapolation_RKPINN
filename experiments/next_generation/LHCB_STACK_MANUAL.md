# LHCb Stack Instruction Manual

## Overview

This project uses the **LHCb software stack** to test neural network track extrapolators against 
traditional C++ implementations. This manual explains how the stack works and why our V1/V2/V3 
models do or don't work with it.

---

## Table of Contents

1. [Stack Architecture](#stack-architecture)
2. [Environment Setup](#environment-setup)
3. [Building and Running](#building-and-running)
4. [Test Framework (QMTest)](#test-framework-qmtest)
5. [Adding Custom Extrapolators](#adding-custom-extrapolators)
6. [Model Integration Workflow](#model-integration-workflow)
7. [Why V1/V2 Models Don't Work](#why-v1v2-models-dont-work)
8. [V3 Solution](#v3-solution)
9. [Troubleshooting](#troubleshooting)

---

## Stack Architecture

### Directory Structure

```
/data/bfys/gscriven/TE_stack/
├── build.x86_64_v2-el9-gcc13+detdesc-opt/   # Build directory
├── Rec/                                       # Reconstruction project
│   └── Tr/                                    # Tracking subproject
│       └── TrackExtrapolators/                # THIS REPOSITORY
│           ├── src/                           # C++ extrapolator implementations
│           ├── ml_models/                     # ML model integration
│           │   ├── src/                       # TrackMLPExtrapolator.cpp
│           │   └── models/                    # Binary model files (.bin)
│           ├── tests/                         # QMTest framework
│           │   ├── options/                   # Gaudi Python configs
│           │   └── qmtest/                    # Test descriptors
│           └── experiments/next_generation/   # ML development (V1/V2/V3)
└── LHCb/, Gaudi/, Det/, ...                   # Other LHCb projects
```

### Key Components

| Component | Purpose |
|-----------|---------|
| **Gaudi** | Core framework for algorithms, services, tools |
| **LHCb** | LHCb-specific event model, conditions, geometry |
| **Rec** | Reconstruction algorithms including tracking |
| **DetDesc** | Detector description (geometry + conditions) |
| **CVMFS** | Distributed software repository |

---

## Environment Setup

### Prerequisites

1. **CVMFS access**: Required for LHCb software
2. **Platform**: `x86_64_v2-el9-gcc13+detdesc-opt`
3. **Cluster**: Nikhef STBC (stbc-i1, stbc-i2, stbc-i3)

### Initialize Environment

```bash
# 1. SSH to cluster
ssh gscriven@stbc-i2.nikhef.nl

# 2. Set up LHCb environment (ONCE per session)
source /cvmfs/lhcb.cern.ch/lib/LbEnv

# 3. Navigate to stack
cd /data/bfys/gscriven/TE_stack

# 4. Enter build environment
lb-run --nightly lhcb-head/latest Rec/HEAD bash
# OR for the pre-built environment:
source build.x86_64_v2-el9-gcc13+detdesc-opt/run

# 5. For Python ML work (separate from LHCb stack)
conda activate TE
```

### Environment Variables

After `lb-run` or sourcing `run`:
```bash
# These are set automatically:
$LHCb_release    # LHCb version
$LHCBDIR         # LHCb root directory
$GAUDIROOT       # Gaudi installation
$MAGNETROOT      # Magnetic field service
```

---

## Building and Running

### Build the Project

```bash
cd /data/bfys/gscriven/TE_stack

# Full build (after changes)
make
# OR faster rebuild
make fast/Rec

# Build only TrackExtrapolators
make fast/Rec/Tr/TrackExtrapolators
```

### Running Tests

```bash
# Run specific QMTest
make fast/Rec/test ARGS='-R extrapolators -V'

# Or directly via gaudirun.py
Rec/run gaudirun.py Rec/Tr/TrackExtrapolators/tests/options/test_extrapolators.py

# Run benchmark (with timing)
Rec/run gaudirun.py Rec/Tr/TrackExtrapolators/tests/options/benchmark_extrapolators.py
```

---

## Test Framework (QMTest)

### How Tests Work

1. **Test Descriptor** (`tests/qmtest/test_extrapolators.qmt`):
   ```xml
   <extension class="GaudiTest.GaudiExeTest">
     <argument name="program">gaudirun.py</argument>
     <argument name="args">test_extrapolators.py</argument>
   </extension>
   ```

2. **Python Config** (`tests/options/test_extrapolators.py`):
   ```python
   from Configurables import TrackRungeKuttaExtrapolator
   from TrackExtrapolators.TrackExtrapolatorsConf import TrackMLPExtrapolator
   
   extrapolators = [
       TrackRungeKuttaExtrapolator("Reference"),
       TrackMLPExtrapolator("MLP", ModelPath="...", Activation="silu"),
   ]
   ```

3. **ExtrapolatorTester Algorithm**:
   - Loads 1210 track states from test data
   - Extrapolates each with all configured extrapolators
   - Compares results to reference (TrackRungeKuttaExtrapolator)
   - Reports errors in x, y, tx, ty

### Test Output Format

```
ExtrapolatorTester  INFO Reference           |  -4144.31    -2209.06   -1.42821  -0.485459 |
ExtrapolatorTester  INFO MLP_v2_shallow_256  |  -4123.89    -2198.45   -1.41234  -0.478234 |
                                               ^x(mm)       ^y(mm)     ^tx       ^ty
```

### Reference Files

Located in `tests/refs/`:
- `test_extrapolators.ref` - Expected output for standard test
- `test_extrapolators.ref.x86_64+avx2+fma-opt` - Platform-specific

---

## Adding Custom Extrapolators

### Step 1: Create C++ Implementation

File: `ml_models/src/TrackMLPExtrapolator.cpp`

```cpp
#include "TrackFieldExtrapolatorBase.h"

class TrackMLPExtrapolator : public TrackFieldExtrapolatorBase {
public:
    // Gaudi component declaration
    DECLARE_COMPONENT(TrackMLPExtrapolator)
    
    // Constructor with properties
    TrackMLPExtrapolator(const std::string& name, ISvcLocator* pSvc)
        : TrackFieldExtrapolatorBase(name, pSvc) {
        declareProperty("ModelPath", m_modelPath);
        declareProperty("Activation", m_activation = "silu");
    }
    
    StatusCode initialize() override;
    StatusCode propagate(LHCb::State& state, double zNew) const override;
    
private:
    std::string m_modelPath;
    std::string m_activation;
    SimpleNN m_model;  // Neural network
};
```

### Step 2: Add to CMakeLists.txt

```cmake
gaudi_add_module(TrackExtrapolators
    SOURCES
        ...
        ml_models/src/TrackMLPExtrapolator.cpp  # ADD THIS
    LINK
        Eigen3::Eigen  # REQUIRED for neural network
        ...
)
```

### Step 3: Rebuild and Register

```bash
make fast/Rec/Tr/TrackExtrapolators

# Component is auto-registered via DECLARE_COMPONENT macro
# Check registration:
grep TrackMLPExtrapolator build.*/confdb/*.json
```

### Step 4: Add to Test Config

```python
from TrackExtrapolators.TrackExtrapolatorsConf import TrackMLPExtrapolator

extrapolators += [
    TrackMLPExtrapolator("MLP_v2_shallow_256",
        ModelPath="/path/to/model.bin",
        Activation="silu"),
]
```

---

## Model Integration Workflow

### Binary Model Format

The C++ extrapolator loads models from binary `.bin` files:

```
[int32]  n_layers
For each layer:
  [int32]  rows, cols
  [float64[rows*cols]] weights (row-major)
  [float64[rows]] biases
[int32]  input_size (6)
[float64[6]] input_mean
[float64[6]] input_std
[int32]  output_size (4)
[float64[4]] output_mean
[float64[4]] output_std
[int32]  activation_len
[char[]]  activation ("silu", "tanh", "relu")
```

### Export from PyTorch

```bash
cd experiments/next_generation
python deployment/export_to_cpp.py \
    trained_models/mlp_v2_shallow_256 \
    ../../ml_models/models/mlp_v2_shallow_256.bin
```

### Forward Pass (C++)

```cpp
StatusCode TrackMLPExtrapolator::propagate(State& state, double zNew) const {
    // 1. Build input vector
    Eigen::VectorXd input(6);
    input << state.x(), state.y(), state.tx(), state.ty(), 
             state.qOverP(), (zNew - state.z());  // dz!
    
    // 2. Neural network forward pass
    Eigen::VectorXd output = m_model.forward(input);
    
    // 3. Update state
    state.setX(output(0));
    state.setY(output(1));
    state.setTx(output(2));
    state.setTy(output(3));
    state.setZ(zNew);
    
    return StatusCode::SUCCESS;
}
```

---

## Why V1/V2 Models Don't Work

### The dz Mismatch Problem

**V1/V2 Training:**
```python
# From V1/data_generation/generate_data.py
--z-start 4000.0   # default
--z-end   12000.0  # default
# dz = 8000 mm (FIXED for all 50M samples)
```

**QMTest (ExtrapolatorTester.cpp):**
```cpp
const double z1 = 3000.;
const double z2 = 7000.;
// dz = 4000 mm (DIFFERENT from training!)
```

The test doesn't use "variable" dz - it uses a **different fixed dz** than what the models were trained on.

### Why This Causes Numerical Explosion

Because dz was **constant** during training, the normalization statistics become:
```
input_mean[5] = 8000.0    # dz mean
input_std[5]  = 1e-9      # dz std ≈ 0 (all values identical!)
```

When C++ normalizes any different dz:
```cpp
// Normalize: (x - mean) / std
dz_normalized = (4000 - 8000) / 1e-9 = -4e12  // EXPLOSION!
```

Even if we patch std to avoid division-by-zero, the model still fails because it only learned the physics for dz=8000.

### "Why Not Just Train on dz=4000 Then?"

You could match the test... but that's not realistic for production.

In real LHCb reconstruction, extrapolators are called between **arbitrary detector elements**:

| From → To | Typical z range | dz |
|-----------|-----------------|-----|
| VELO layers | 0 - 800 mm | ~50 mm |
| VELO → UT | 800 → 2500 mm | ~1700 mm |
| UT → SciFi T1 | 2500 → 7800 mm | ~5300 mm |
| SciFi T1 → T2 | 7800 → 8500 mm | ~700 mm |
| SciFi T2 → T3 | 8500 → 9400 mm | ~900 mm |
| SciFi → Muon M2 | 9400 → 16000 mm | ~6600 mm |
| Full detector | 0 → 20000 mm | ~20000 mm |

A **production-ready extrapolator must handle any dz** in the range ~50 - 20000 mm.

Training on a single fixed dz (whether 4000 or 8000) creates a model that can only work for that exact distance.

### PINN IC Failure (V1 only)

Separate issue specific to V1 PINNs - they ignored the `z_frac` time input:

```
At z_frac=0 (should = initial state):  Output = [2768, ...]
At z_frac=1 (final state):             Output = [2752, ...]  # Almost same!
```

The network learned a constant mapping, not a trajectory.

**Fixed in V2** with residual architecture: `Output = IC + z_frac × Correction`

---

## V3 Solution

### Variable dz Training

V3 generates data with random dz:
```python
dz = np.random.uniform(500, 12000)  # Variable per sample
```

This creates healthy normalization:
```
input_mean[5] ≈ 6250   # dz mean (center of range)
input_std[5]  ≈ 3300   # dz std (proper spread!)
```

### V3 Workflow

```bash
# 1. Generate variable dz data (100M samples)
cd experiments/next_generation
condor_submit V3/cluster/submit_datagen_v3.sub

# 2. Train V3 models
condor_submit V3/cluster/submit_v3_training.sub

# 3. Export to C++ binary
python deployment/export_to_cpp.py \
    V3/trained_models/mlp_v3_shallow_256 \
    ../../ml_models/models/mlp_v3_shallow_256.bin

# 4. Update test config
# Edit tests/options/test_extrapolators.py to use mlp_v3_*

# 5. Rebuild and test
make fast/Rec/Tr/TrackExtrapolators
make fast/Rec/test ARGS='-R extrapolators -V'
```

### Expected V3 Results

With proper variable dz training:
```
ExtrapolatorTester  INFO Reference            |  -4144.31    -2209.06   -1.42821  -0.485459 |
ExtrapolatorTester  INFO MLP_v3_shallow_256   |  -4141.23    -2207.89   -1.42756  -0.484892 |
                                                 ^^^^^^^ CLOSE MATCH (working!) ^^^^^^^
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `cannot find module TrackMLPExtrapolator` | Not registered in confdb | Rebuild: `make fast/Rec/Tr/TrackExtrapolators` |
| `Model file not found` | Wrong path in Python config | Use absolute path |
| `NaN/Inf output` | dz normalization explosion | Use V3 models with variable dz |
| `Wrong results (but not NaN)` | Model doesn't generalize | Retrain with variable dz (V3) |
| `Symbol not found: Eigen` | Missing Eigen3 link | Check CMakeLists.txt has `Eigen3::Eigen` |
| `CVMFS not mounted` | Network/cluster issue | Check `/cvmfs/lhcb.cern.ch` exists |

### Debugging Commands

```bash
# Check component registration
grep -r "TrackMLPExtrapolator" build.*/confdb/

# Run with verbose output
Rec/run gaudirun.py --option "from Gaudi.Configuration import *; MessageSvc().OutputLevel=DEBUG" test_extrapolators.py

# Check model file
python -c "import struct; f=open('model.bin','rb'); print(struct.unpack('i', f.read(4)))"

# Validate normalization stats
python deployment/export_to_cpp.py --info trained_models/mlp_v3_shallow_256
```

### Getting Help

- LHCb software: https://lhcb.web.cern.ch/lhcb/
- Gaudi documentation: https://gaudi.web.cern.ch/gaudi/
- Internal notes: `V1/notes/`, `V2/analysis/`, `V3/README.md`

---

## Summary

| Version | Status | dz Training | Works with QMTest? |
|---------|--------|-------------|-------------------|
| **V1** | ❌ Deprecated | Fixed 8000mm | ❌ No (explodes) |
| **V2** | ❌ Deprecated | Fixed 8000mm | ❌ No (wrong results) |
| **V3** | ✅ Active | Variable 500-12000mm | ✅ Expected to work |

**Key lesson**: Training data must cover the full operational range of inputs.
V1/V2 trained on fixed dz=8000mm but the LHCb stack tests with variable dz (1000-10000mm typical).
