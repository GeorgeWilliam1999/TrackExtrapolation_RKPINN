# PINN_v2 UT→T neural extrapolator — Allen integration

**Branch:** `gscriven/nrk-extrapolator-exercise` · **Updated:** 2026-06-08 · **Status:** built & pushed; R6 gates pending

This is the Allen-side (Repo B) record of the neural drop-in for the UT→T track
extrapolation inside the GPU ParKalman filter. The model, training, export, and the
project-wide status live in the companion repo **`track-extrapolation-pinn`** (Repo A) and
on the Notion page *Track Extrapolation*, which is the single source of truth.

## What this is

The classical analytic polynomial `extrapUTT` (UT→T, Δz ≈ 5213 mm) is augmented so that,
under a runtime toggle, the **new state** is produced by a small physics-informed neural
network (`pinn_v2_ALLEN_v1`, 10,372 params, fp32 weights baked into a generated header).
This is a **true replacement** for the *state* propagation: no field map and no RK loop at
inference.

## What changed (files)

| File | Change |
|---|---|
| `include/PINN_V2_UTT.cuh` | **Generated** read-only header: fp32 weights + `pinn_v2_utt_state(...)` forward pass. Do not edit by hand. |
| `include/ParKalmanMethods.cuh` | `#include "PINN_V2_UTT.cuh"` (l.17); in `extrapUTT` the NN supplies `x'` (l.533 `pinn_v2_utt_state`) while the polynomial still runs (l.525) to supply the Jacobian. |
| `include/ParKalmanFilter.cuh` | Property `m_use_nn_utt` (`"use_nn_utt"`, default `false`). |
| `src/ParKalmanFilter.cu` | Plumbs the toggle into the kernel. |
| `configuration/python/AllenConf/…` | `make_kalman_long` exposes the toggle. |

## Hybrid Jacobian rule (important)

- The **NN supplies the new state** `x' = (x, y, tx, ty)`.
- The transport **Jacobian `F` is kept from the analytic polynomial** `extrapUTT` — the
  network is *not* auto-differentiated on device and has no Jacobian head.
- The noise matrix **`Q` is unchanged** (physics-derived multiple scattering).
- Consequence: the Kalman gain uses a linearisation consistent with the polynomial, not the
  NN. This is acceptable iff residual χ² and long-track ghost rate do not degrade in Moore
  (the R6 physics gate). If they do, escalate to Jacobian co-supervision (Repo A
  `docs/plans/EXECUTION_PLAN.md` §8).

## Provenance (blob → header)

- Source blob: `track-extrapolation-pinn/For_Allen/artifacts/blobs/v3/pinn_v2_ALLEN_v1.bin`
- SHA256 `c66576709288f046d399b4578353c81549df930a4e4617ed5545dc649c87e52c`, CRC32 `0x1a139335`
- Spec: `For_Allen/pins/loader_v3_spec.md` (v3, magic `NRKv3`)
- Regenerate the header: `python For_Allen/scripts/emit_cuda_header.py` (Repo A)
- Allen baseline commit pinned: `12f26514959d`

## How to run

```bash
# Enable the neural UT->T extrapolator (default is off → bit-identical to master)
#   AllenConf make_kalman_long(..., use_nn_utt=True)
# Standalone CPU parity harness (no GPU needed):
cd ML_research/standalone && make && ./run_extrapolators   # untracked dev harness
```

## Current accuracy & remaining R6 gates

- UT→T single-step accuracy (latest eval, Repo A `results/R7_utt_eval_2026-05-22.json`):
  **median ‖Δx‖ ≈ 293 µm**, p95 ≈ 1894 µm on n=50 tracks. This is the production-relevant
  number (the full-distribution 11.7 µm median is dominated by easier short steps).
- [ ] CUDA↔Python bit-bound parity (max |Δy| < 1 ULP on 200 reference tracks).
- [ ] `allen_throughput`: per-track GPU cost ≤ classical RKN4 (baseline cached in Repo A
      `For_Allen/pins/baseline_throughput.txt`). Earlier MR `!2497` showed a −10% A5000
      regression to root-cause.
- [ ] Moore `HltEfficiencyChecker`: no track-quality line degrades > 0.5 % absolute vs RK4
      — the decisive test of the 293 µm UT→T accuracy vs the polynomial baseline.

> **Note:** this file and the untracked `ML_research/` harness are left out of the commit
> history deliberately — fold them into the MR (or not) as you prefer. The production
> integration (header + Kalman wiring + AllenConf) is already committed and pushed.
