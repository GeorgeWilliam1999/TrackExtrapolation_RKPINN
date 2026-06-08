# Gen-3 Data Generation Specification

**Author**: G. Scriven
**Date**: 2026-04-23
**Scope**: detailed spec for the `train_50M_gen3.npz` dataset used by all Gen-3 models.

Cross-reference:
- Protocol: `docs/reports/gen3_protocol.tex` (§3 and §5)
- Gen-2 results that motivate each change: `docs/reports/gen2_new_data_results.tex` §5 and §4.5.1
- Allen requirements: `Allen/ML_research/README.md` sections "Units", "qop convention", "dz must be bidirectional", "Input-position coverage", "Export / binary format"

This document is the source of truth. Any deviation in the implementation
is a bug.

---

## 0. Executive summary

| Aspect                 | Gen-2                           | **Gen-3**                                              |
|------------------------|---------------------------------|--------------------------------------------------------|
| `qop` convention       | `q/p_MeV`                       | `c_light * q / p_MeV` (Allen)                          |
| `c_light`              | —                               | `299.792458` (Gaudi units mm·ns⁻¹·eplus)               |
| `dz` sign              | `[+25, +10000]` mm              | `[-10000, -25] ∪ [+25, +10000]` mm                     |
| `dz` sampling          | uniform in `dz`                 | log-uniform in `|dz|`, balanced sign                   |
| `x_0` range            | `[-300, +300]` mm               | `[-3500, +3500]` mm                                    |
| `y_0` range            | `[-250, +250]` mm               | `[-2500, +2500]` mm                                    |
| `tx_0` range           | `[-0.30, +0.30]`                | `[-0.40, +0.40]`                                       |
| `ty_0` range           | `[-0.25, +0.25]`                | `[-0.35, +0.35]`                                       |
| Rejection slope        | `|t| < 2.0`                     | `|t| < 0.5`                                            |
| Input columns          | 6: `[x,y,tx,ty,qop,dz]`         | **7**: `[x,y,tx,ty,qop,z_start,dz]`                    |
| Output columns         | 4: `[x_f,y_f,tx_f,ty_f]`        | **5**: `[x_f,y_f,tx_f,ty_f,qop_f]`                     |
| Material effects       | none                            | none (Gen-3.1)                                         |
| Polarity               | MagDown only                    | MagDown only (Gen-3.1 MagUp)                           |
| Total tracks           | 50 M                            | **10 M** (smaller; real field is ~10× slower than Gen-2's Gaussian fallback) |
| Condor batches         | 2 000 × 25 k                    | **5 000 × 2 k** (fine splitting, each batch ~12 min)   |
| Metadata file          | —                               | `train_50M_gen3.meta.json`                             |

Target turn-around on Nikhef HTCondor. Empirical rate with the **real**
interpolated field map: **~3 tracks/s** per 2-worker job (Gen-2's quoted
rate of 2 k tracks/s was achieved with the Gaussian fallback — see §0.1).

* 2 000 tracks / 3 tracks/s ≈ **11 minutes compute + ~5 s startup** per batch.
* 5 000 batches × ~200 simultaneous slots → **≈ 5 hours** total wall clock.

### 0.1 Retroactive note on the Gen-2 field map

While implementing Gen-3 the agent discovered that the default field-map
path in `experiments/gen_2/utils/magnetic_field.py` resolves to
`TrackExtrapolation/field_maps/` (one directory up from the actual
location `experiments/field_maps/`). Gen-2 therefore silently fell back
to the Gaussian approximation (`~1.3 %` RMS field error, warned loudly
but un-asserted). Gen-3 fixes the path in the local `utils/` copy and
the generator hard-fails if the interpolated field is not loaded. This
is a Gen-3 bugfix **and** a retroactive concern for any result that
used Gen-2 data — flagged separately in the review log.

---

## 1. Coordinate system and units

Identical to Allen / Gaudi (mixed-unit system):

| Quantity | Unit                  | Storage dtype |
|----------|-----------------------|---------------|
| `x, y`   | mm                    | float32       |
| `z`, `dz`| mm (signed for `dz`)  | float32       |
| `tx, ty` | dimensionless         | float32       |
| `qop`    | `(mm·ns⁻¹·eplus) / MeV` (Allen) | float32 |
| `p`      | GeV/c (stored separately in `P`) | float32 |
| `q`      | ±1 (`eplus`)          | int8 (stored in new `Q` column) |

`c_light = 299.792458` (mm·ns⁻¹·eplus). Verified against three
independent Allen sources:

- `Allen/ML_research/README.md` Units table.
- `Allen/device/kalman/ParKalman/include/ExtrapolatorCommon.cuh` line 16.
- `Allen/ML_research/standalone/main.cpp` line 478.

**qop at 10 GeV**:
```
qop = 299.792458 / 10 000 ≈ 2.998e-2
```
matching the README worked example.

### 1.1 Integrator constant

The Gen-2 integrator used `kappa = C_LIGHT * (q/p_MeV)` with
`C_LIGHT = 2.99792458e-4`, converting `(q/p) [1/MeV] × B [T]` to
curvature in `1/mm`.

Under the Allen convention `qop` already contains `c_light`, so the
integrator must **not** multiply it in again:
```
kappa_allen = 1e-6 * qop_allen
            = 1e-6 * 299.792458 * (q/p_MeV)
            = 2.99792458e-4 * (q/p_MeV)
            = kappa_gen2  ✓
```
Gen-3 therefore sets `C_LIGHT_KAPPA = 1.0e-6` in the integrator when the
dataset is generated with Allen-convention `qop`. A `qop_convention`
argument toggles between old and new; the default is `"allen"`.

---

## 2. Input vector `X ∈ ℝ^{N × 7}`

Order (Allen loader V3 contract, matches `A5` in protocol):

```
X[:, 0] = x_0       mm
X[:, 1] = y_0       mm
X[:, 2] = tx_0      dim.less
X[:, 3] = ty_0      dim.less
X[:, 4] = qop_0     Allen units
X[:, 5] = z_start   mm   ← new in Gen-3 (Fix H)
X[:, 6] = dz        mm   signed (Fix C2)
```

### 2.1 Sampling distributions

| Variable | Distribution                                                            | Rationale                                   |
|----------|-------------------------------------------------------------------------|---------------------------------------------|
| `z_start`| `U(0, 14000 - |dz|)` mm (clamped to keep `z_end ∈ [0, 14000]`)          | keeps integration inside field-map support  |
| `|dz|`   | `10^{U(log10 25, log10 10000)}` mm                                      | over-represents short VELO steps            |
| `sign(dz)`| `Bernoulli(0.5) → {-1, +1}`                                            | balanced forward/backward                   |
| `p`      | `exp(U(log 1, log 200))` GeV/c                                          | log-uniform; covers Allen p range           |
| `q`      | `{-1, +1}` uniform                                                      | both charges equally represented            |
| `x_0`    | `U(-3500, +3500)` mm                                                    | Allen C3: inputs reach ±3200 mm             |
| `y_0`    | `U(-2500, +2500)` mm                                                    | SciFi y-acceptance ≈ ±2500 mm               |
| `tx_0`   | `U(-0.40, +0.40)`                                                       | > LHCb acceptance (~0.30) with 33% margin   |
| `ty_0`   | `U(-0.35, +0.35)`                                                       | as above, scaled for y-acceptance           |

### 2.2 Backward-pass compatibility

When `dz < 0` the integrator steps with negative `h`. The field map is
time-reversal-symmetric (`B` depends only on position), so no changes
to `magnetic_field.py` are required. `z_end = z_start + dz` may lie at
or above 0; we clamp `z_start` such that `z_end ∈ [0, 14000]` for both
signs:

```
if dz > 0:
    z_start ∈ [0, 14000 - dz]
else:
    z_start ∈ [-dz, 14000]     # guarantees z_end = z_start + dz ≥ 0
```

### 2.3 Post-integration rejection

A track is rejected and resampled if any of the following after
propagation:

- `|x_f| > 5000` mm  or  `|y_f| > 5000` mm (diverged)
- `|tx_f| > 0.5`  or  `|ty_f| > 0.5` (tightened from Gen-2's 2.0)
- any component non-finite

Expected rejection rate: **< 0.2 %** (Gen-2 rate on the larger
envelope was ~0.05 %; the wider slope range in Gen-3 may double this).

---

## 3. Output vector `Y ∈ ℝ^{N × 5}`

```
Y[:, 0] = x_f       mm
Y[:, 1] = y_f       mm
Y[:, 2] = tx_f      dim.less
Y[:, 3] = ty_f      dim.less
Y[:, 4] = qop_f     Allen units
```

In vacuum propagation `qop_f ≡ qop_0`; storing the column explicitly
honours the Allen binary V3 contract (A5) without loader synthesis and
leaves the data pipeline ready for Gen-3.1 Bethe–Bloch.

---

## 4. Auxiliary arrays

| Array | Shape     | Dtype   | Contents                                                                  |
|-------|-----------|---------|---------------------------------------------------------------------------|
| `P`   | `(N,)`    | float32 | true momentum in GeV/c (pre-sampling, before `qop` calc)                  |
| `Q`   | `(N,)`    | int8    | charge ±1 (`eplus`)                                                       |
| `Z`   | `(N, 2)`  | float32 | `[z_start, z_end]` (redundant with `X[:,5]`, `X[:,5]+X[:,6]`; kept for cross-check) |

---

## 5. Metadata file `train_50M_gen3.meta.json`

Written alongside the merged `.npz` by `merge_batches.py`.

```jsonc
{
  "dataset_name": "train_50M_gen3",
  "n_tracks": 50000000,
  "n_batches": 5000,
  "tracks_per_batch": 10000,
  "created_utc": "2026-04-23T...Z",

  "qop_convention": "allen_v1",
  "c_light_value": 299.792458,
  "dz_signed": true,
  "x_range_mm": [-3500, 3500],
  "y_range_mm": [-2500, 2500],
  "tx_range": [-0.40, 0.40],
  "ty_range": [-0.35, 0.35],
  "p_range_GeV": [1.0, 200.0],
  "dz_abs_range_mm": [25.0, 10000.0],
  "dz_distribution": "log_uniform_abs_balanced_sign",
  "polarity": "MagDown",

  "feature_order": ["x", "y", "tx", "ty", "qop", "z_start", "dz"],
  "output_order": ["x_f", "y_f", "tx_f", "ty_f", "qop_f"],

  "rk4_step_size_mm": 5.0,
  "field_map": "field_maps/twodip.rtf",
  "field_map_sha256": "<computed>",

  "batch_seed_formula": "42 + 7919 * batch_id",
  "generator_git_hash": "<computed>",
  "data_sha256": "<computed over concatenated X,Y,P,Q,Z>"
}
```

The Allen binary export (A6) reads these fields and stamps them into
the `.bin` header; a mismatch between binary metadata and Allen Kalman
runtime is a hard error.

---

## 6. Closed-form regression tests (pre-generation gate)

Run `data_generation/closed_form_test.py` **before** submitting any
Condor batch. Both tests must pass.

### 6.1 Uniform `B_y = -1 T` arc

- Field: replace `InterpolatedFieldNumpy` with a constant
  `(B_x, B_y, B_z) = (0, -1, 0)` T.
- Track: `q = +1`, `p = 10` GeV/c → `qop_allen = 2.998e-2`;
  `x_0 = y_0 = tx_0 = ty_0 = 0`; `dz = 1000` mm.
- Analytic expectation: circular arc in the x–z plane,
  `R = p / (c · q · B) = 1e4 / (2.998e-4 × 1 × 1) × 1e3 = 3.3356e4 mm`.
  Small-angle: `x_f ≈ dz² / (2R) = 14.990 μm` to 5 sig-figs.
- Pass: `|x_f_rk4 - 14.990e-3| < 1e-3` mm (1 μm) on 10 k tracks.

### 6.2 Uniform `B_x = +1 T` helix

- Field: `(B_x, B_y, B_z) = (+1, 0, 0)` T.
- Same initial state and `dz`.
- Analytic expectation: helix in the y–z plane,
  `y_f ≈ -dz² / (2R) = -14.990 μm` (negative because sign of
  `dtx/dz` flips).
- Pass: `|y_f_rk4 + 14.990e-3| < 1e-3` mm on 10 k tracks.

This second test catches regressions in the `B_x` branch of the
Lorentz RHS (Fix G depends on it) — the first test alone would miss
them.

---

## 7. Condor job configuration

### 7.1 Splitting strategy

- **5 000 jobs × 2 000 tracks = 10 M tracks**
  (Gen-2 was 2 000 × 25 k = 50 M with Gaussian field; Gen-3 real field is
  ~10× slower per track, so halve both axes.)
- 2 CPUs, 2 GB RAM, 500 MB disk per job (same as Gen-2).
- `+JobCategory = "medium"` (each job ~12 min).
- Seed `s_i = 42 + 7919 * batch_id` (primes, unchanged from Gen-2).
- Output: `data/batches/batch_{0000..4999}.npz`.

### 7.2 Per-batch timing budget

| Stage                              | Time (s) |
|------------------------------------|----------|
| Conda env activation               | ~3       |
| Field-map load + interpolator init | ~2       |
| 2 k tracks × ~330 ms each @2 CPU   | ~660     |
| npz compressed save                | ~1       |
| **Total per batch**                | **~670 s (~11 min)** |

With ~200 free slots on `taai-007`: `5000 / 200 × 670 s ≈ 4.6 hours of
compute`; practical wall time including queue waits: ~5–6 hours.

### 7.3 Merge step

After all batches complete:

```bash
cd experiments/gen_3/data_generation
python merge_batches.py \
    --input "../data/batches/batch_*.npz" \
    --output ../data/train_10M_gen3.npz \
    --write-metadata \
    --verify
```

`merge_batches.py` writes `train_10M_gen3.meta.json` alongside and
computes `data_sha256` over the concatenated arrays.

---

## 8. Validation after generation (smoke checks on merged dataset)

Run `notebooks/gen3_data_inspection.ipynb` with the merged file.
Gates (all must pass before training):

1. `X.shape == (10e6, 7)`, `Y.shape == (10e6, 5)`, dtype `float32`.
2. No NaN / Inf anywhere.
3. `qop` at 10 GeV: `|qop| ≈ 3e-2` (not 3e-5 or 3e-4).
4. `dz` histogram: 50/50 sign balance ±0.5 %, log-flat in `|dz|`.
5. `x_0, y_0` distributions fill the full `[-3500, 3500] × [-2500, 2500]` box.
6. Spot-check 100 random tracks against a direct RK4 re-propagation:
   `max|X_pred - Y| < 1e-5` mm (round-trip sanity).
7. The two closed-form arc/helix tests still pass on a fresh seed.

If any gate fails, regeneration is cheaper than debugging: 30 min wall.

---

## 9. File manifest (Gen-3 generation deliverables)

```
experiments/gen_3/
├── GENERATION_SPEC.md                  (this file)
├── utils/
│   ├── __init__.py
│   ├── magnetic_field.py               (copy from gen_2, adds C_LIGHT_KAPPA)
│   └── rk4_propagator.py               (qop_convention flag; default "allen")
├── data_generation/
│   ├── generate_data.py                (7-in/5-out, signed dz, Allen qop)
│   ├── merge_batches.py                (writes .meta.json)
│   ├── closed_form_test.py             (§6 gates)
│   └── inspect_dataset.py              (CLI one-shot of §8 gates)
├── data/
│   ├── batches/                        (5 000 × npz filled by Condor)
│   ├── train_10M_gen3.npz              (merged)
│   └── train_10M_gen3.meta.json        (provenance)
├── condor/
│   ├── submit_datagen.sub
│   ├── run_datagen.sh
│   └── logs/
└── notebooks/
    └── gen3_data_inspection.ipynb      (§8 visual checks)
```
