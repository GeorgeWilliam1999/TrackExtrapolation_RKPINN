# Training Data Generation

Generate high-quality training data for ML track extrapolators using either:
1. **LHCb C++ Extrapolators** (recommended for ground truth)
2. **Python RK4 with Interpolated Field** (standalone alternative)

---

## Table of Contents

1. [LHCb Magnetic Field Map](#lhcb-magnetic-field-map)
2. [C++ Extrapolators (Ground Truth)](#c-extrapolators-ground-truth)
3. [Python Data Generation](#python-data-generation)
4. [Quick Start](#quick-start)
5. [Data Format](#data-format)
6. [HTCondor Scaling](#htcondor-scaling)

---

## LHCb Magnetic Field Map

### Source File

The real LHCb dipole field map is stored at:
```
/data/bfys/gscriven/TE_stack/Rec/Tr/TrackExtrapolators/field_maps/twodip.rtf
```

### Field Map Specifications

| Property | Value | Notes |
|----------|-------|-------|
| **Format** | `x y z Bx By Bz` | Space-separated ASCII |
| **Units** | mm, Tesla | Verified against LHCb TDR |
| **Grid Size** | 81 × 81 × 146 | 957,906 total points |
| **X range** | -4000 to +4000 mm | 100 mm step |
| **Y range** | -4000 to +4000 mm | 100 mm step |
| **Z range** | -500 to +14000 mm | Variable step |

### Field Characteristics

| Property | Value | Notes |
|----------|-------|-------|
| **Peak \|By\|** | **1.03 Tesla** | At z ≈ 5007 mm |
| **∫By·dz** | **4.44 T·m** | Along beam axis |
| **Polarity** | Field points **down** | By is negative at peak |
| **Dominant component** | By (vertical) | Causes horizontal bending |

### Gaussian Approximation

For physics losses in PINN training, the field can be approximated:

```python
By(z) = B0 × exp(-0.5 × ((z - z_center) / z_width)²)
```

**Fitted Parameters (from real field map):**

| Parameter | Value | Old (wrong) | Description |
|-----------|-------|-------------|-------------|
| B0 | **-1.0182 T** | 1.0 T | Peak field (negative = downward) |
| z_center | **5007 mm** | 5250 mm | Center of dipole |
| z_width | **1744 mm** | 2500 mm | Gaussian width |

**Fit Quality:** RMS error = 0.013 T (~1.3% of peak)

### Artifact Warning

⚠️ About 3% of grid points at edges (|x|, |y| > 3500 mm) have non-physical 
|By| > 2T due to extrapolation artifacts. **Mask these regions** for training.

### Pre-computed Data Files

For convenience, the following are saved in `../models/data/`:

| File | Contents |
|------|----------|
| `field_By_vs_z.npy` | 1D profile at x=0, y=0: shape (146, 2) = [z, By] |
| `gaussian_field_params.json` | Fitted Gaussian parameters |
| `field_map_downsampled.npz` | Downsampled 3D grid (~240k points) |

---

## C++ Extrapolators (Ground Truth)

### Available Extrapolators

The LHCb framework provides several C++ extrapolators in `src/`:

| Extrapolator | Method | Field Access | Accuracy | Speed |
|--------------|--------|--------------|----------|-------|
| `TrackRungeKuttaExtrapolator` | Adaptive RK | `MagneticFieldGrid` | <0.01 mm | Medium |
| `TrackSTEPExtrapolator` | STEP method | `MagneticFieldGrid` | ~0.01 mm | Medium |
| `TrackParametrizedExtrapolator` | Parametrized | `MagneticFieldGrid` | ~0.1 mm | Fast |
| `TrackLinearExtrapolator` | Straight line | None | N/A | Fastest |

**Recommendation:** Use `TrackRungeKuttaExtrapolator` for ground truth data.

### How C++ Extrapolators Access the Field

All field-based extrapolators inherit from `TrackFieldExtrapolatorBase`:

```cpp
// From src/TrackFieldExtrapolatorBase.h

class TrackFieldExtrapolatorBase : public TrackExtrapolator {
public:
    /// Get field vector at any 3D point
    FieldVector fieldVector(
        const LHCb::Magnet::MagneticFieldGrid* grid,
        const Gaudi::XYZPoint& position
    ) const {
        return m_fieldFunction(grid ? grid : currentGrid(), position);
    }
    
private:
    /// Field grid loaded from conditions database
    ConditionAccessor<DeMagnet> m_magnet;
    
    /// Interpolation: linear (default) or nearest point
    Gaudi::Property<bool> m_useGridInterpolation{true};
};
```

The `MagneticFieldGrid` provides:
- `fieldVectorLinearInterpolation(point)` - Trilinear interpolation
- `fieldVectorClosestPoint(point)` - Nearest grid point

### RK4 Integration (C++)

From `src/TrackRungeKuttaExtrapolator.cpp`:

```cpp
// Lorentz force equations (same as our Python implementation)
auto B = this->fieldVector(grid, position);  // Get field at current point

auto kappa = qop * Gaudi::Units::c_light;  // c_light = 2.99792458e11 mm/s
auto sqrt_term = std::sqrt(1 + tx*tx + ty*ty);

// Slope derivatives
dtx_dz = kappa * sqrt_term * (tx*ty*B.x() - (1+tx*tx)*B.y() + ty*B.z());
dty_dz = kappa * sqrt_term * ((1+ty*ty)*B.x() - tx*ty*B.y() - tx*B.z());
```

---

## Python Data Generation

### Using Interpolated Field (Standalone)

For generating data without the LHCb framework, use Python RK4 with the real field:

```python
# In utils/archived/rk4_propagator.py

from rk4_propagator import RK4Integrator
from models.architectures import InterpolatedMagneticField

# Load real field map (cubic spline interpolation)
field = InterpolatedMagneticField()

# Create integrator
integrator = RK4Integrator(field, step_size=5.0)  # 5mm steps

# Propagate a track
state_initial = np.array([x, y, tx, ty, qop])  # at z_start
state_final = integrator.propagate(state_initial, z_start=3000, z_end=9000)
```

### Field Model Options

```python
# Option 1: Gaussian approximation (fast, for physics losses)
from models.architectures import MagneticField
field = MagneticField()  # Uses fitted params: B0=-1.02, z_center=5007, z_width=1744

# Option 2: Interpolated from real map (accurate, for data generation)
from models.architectures import InterpolatedMagneticField
field = InterpolatedMagneticField()  # Cubic spline from twodip.rtf
```

---

## Quick Start

### 1. Generate Test Data (Python)

```bash
cd data_generation

# Small test dataset
python generate_data.py \
    --n-tracks 10000 \
    --z-start 3000 \
    --z-end 9000 \
    --p-min 2.0 \
    --p-max 100.0 \
    --output test_data/test_10k.npz \
    --use-real-field  # Use InterpolatedMagneticField
```

### 2. Generate Production Dataset

```bash
# 1M tracks for training
python generate_data.py \
    --n-tracks 1000000 \
    --z-start 2000 \
    --z-end 12000 \
    --p-min 0.5 \
    --p-max 100.0 \
    --output data/training_1M.npz \
    --use-real-field \
    --n-workers 16
```

### 3. Submit to HTCondor (Massive Scale)

```bash
# Edit submit_condor.sub to configure
condor_submit submit_condor.sub

# Monitor
condor_q

# Merge batches after completion
python merge_batches.py --input "data/batch_*.npz" --output data/training_10M.npz
```

---

## Data Format

### NPZ File Contents

```python
import numpy as np

data = np.load('training_1M.npz')

X = data['X']  # Shape: (N, 6) - Inputs [x, y, tx, ty, q/p, dz]
Y = data['Y']  # Shape: (N, 4) - Outputs [x, y, tx, ty] at z_end
P = data['P']  # Shape: (N,) - Momentum in GeV
Z = data['Z']  # Shape: (N, 2) - [z_start, z_end] positions
```

### Input Features (6D)

| Index | Symbol | Description | Units | Typical Range |
|-------|--------|-------------|-------|---------------|
| 0 | x | Initial x position | mm | ±3000 |
| 1 | y | Initial y position | mm | ±2500 |
| 2 | tx | Initial x-slope | - | ±0.3 |
| 3 | ty | Initial y-slope | - | ±0.25 |
| 4 | q/p | Charge/momentum | 1/MeV | ±0.001 |
| 5 | dz | Propagation distance | mm | 1000-10000 |

### Output Features (4D)

| Index | Symbol | Description | Units |
|-------|--------|-------------|-------|
| 0 | x_f | Final x position | mm |
| 1 | y_f | Final y position | mm |
| 2 | tx_f | Final x-slope | - |
| 3 | ty_f | Final y-slope | - |

**Note:** q/p is conserved and not included in output.

---

## HTCondor Scaling

### Configuration (submit_condor.sub)

```
executable = run_data_gen.sh
arguments = $(Process) 10000 data/batch_$(Process).npz
output = logs/job_$(Process).out
error = logs/job_$(Process).err
log = logs/condor.log

request_memory = 4GB
request_cpus = 1

queue 1000  # 1000 jobs × 10k tracks = 10M total
```

### Scaling Guidelines

| Total Tracks | Jobs | Tracks/Job | Time (est.) |
|--------------|------|------------|-------------|
| 1M | 100 | 10,000 | ~1 hour |
| 10M | 1,000 | 10,000 | ~2 hours |
| 50M | 5,000 | 10,000 | ~4 hours |

---

## Verification

```python
import numpy as np

data = np.load('training_1M.npz')
X, Y, P = data['X'], data['Y'], data['P']

# Data integrity
assert np.all(np.isfinite(X)), "NaN/Inf in inputs"
assert np.all(np.isfinite(Y)), "NaN/Inf in outputs"
assert np.all(P > 0), "Negative momentum"

# Physics sanity checks
assert np.abs(Y[:, 0]).max() < 10000, "x position unreasonable"
assert np.abs(Y[:, 2]).max() < 1.0, "tx slope too large"

# Bending check: opposite charges should bend opposite ways
pos_charge = X[:, 4] > 0
neg_charge = X[:, 4] < 0
dx_pos = (Y[pos_charge, 0] - X[pos_charge, 0]).mean()
dx_neg = (Y[neg_charge, 0] - X[neg_charge, 0]).mean()
assert dx_pos * dx_neg < 0, "Charges not bending opposite!"

print("✓ Dataset verified!")
```

---

**Author:** G. Scriven  
**Updated:** 2026-01-19  
**Status:** Field map verified, ready for data generation
