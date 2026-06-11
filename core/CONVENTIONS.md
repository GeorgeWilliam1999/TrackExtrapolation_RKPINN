# Core conventions (locked — every gate, corpus, and export assumes these)

## kappa (curvature prefactor)

```
kappa = 1e-3 * qop
qop   = 0.299792458 * q/p[1/GeV]     # = Allen's c * q/p (q/p in 1/MeV internally)
```

`_ALLEN_KAPPA_PREFACTOR = 1e-3` lives in `models/architectures.py` and
`core/rk4_propagator.py`. The P0.0 finding: an earlier corpus used kappa
x1000 too large; everything gen-3+ uses the physical value above.

## Magnetic field

- **Canonical:** `core/field_v8r1.py` — LHCb FieldMap **v8r1 down**, read from
  the same CVMFS `field.v8r1.down.bin` that Allen consumes as `magfield.bin`.
  **Raw sign convention:** MagDown has `By < 0`. No sign flips anywhere.
- **Legacy:** `core/magnetic_field.py` — twodip.rtf loader, kept only for
  gen-1/gen-2 reproducibility. Do not use for new corpora or gates.

## Production baseline (extrapUTT)

The Allen production extrapolator uses the `extrapUTT` polynomial **pairs**
with `m_polarity = -1` (MagDown). Baseline comparisons
(`gates/baseline/`) evaluate both polarity tags and the production pairing.

## Corpus contract (gen-3/gen-4, see datagen/generate_data_v2.py)

- Signed `dz` in **[-10000, +10000] mm** (|dz| log-uniform [25, 10000], sign 50/50)
- `z0` uniform in **[0, 14000] mm**; `zf` clipped to [-400, 13900]
- Population: **70% PV-pointing** (|z_pv| < 50 mm, +-0.5 mm transverse —
  the production population), **30% broad non-pointing** (x in +-1000,
  y in +-800 mm — Kalman intermediate-state cover)
- p log-uniform [1, 200] GeV, both charges; truth = vectorised RK4, 5 mm step
- `X[N,7] = (x, y, tx, ty, qop, z0, dz)`, `Y[N,5] = (x, y, tx, ty, qop)_zf`

## Data location

Code lives in this repo. Big artifacts (corpora `.npz`, `trained_models/`,
`mlruns/`) live in the lab; scripts resolve it via
`TE_LAB` (default `/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3`).
