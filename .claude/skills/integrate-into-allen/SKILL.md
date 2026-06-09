---
name: integrate-into-allen
description: Export a locked extrapolator checkpoint to a V3 blob + generated CUDA header and integrate it into the Allen GPU Kalman filter (UT->T step). Use when asked to deploy a model into Allen, export the blob, regenerate the CUDA header, or run the Allen/Moore integration gates.
---

# Integrate the extrapolator into Allen

Two repos: the model side (`/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3`, and the deliverable `track-extrapolation-pinn`) and the Allen clone (`/data/bfys/gscriven/Allen`, branch `gscriven/nrk-extrapolator-exercise`). Env: TE conda.

## 1. Export blob + emit CUDA header

```bash
cd .../experiments/gen_3
# V3 blob (byte-locked): For_Allen/src/for_allen/export/blob_writer.py  (write_v3_blob)
# Round-trip + parity tests: cd For_Allen && pytest tests/test_blob_roundtrip.py
# Generate the read-only CUDA header from the blob:
python For_Allen/scripts/emit_cuda_header.py   # --help for args
```
This produces `PINN_V2_UTT.cuh` (weights + `pinn_v2_utt_state()` forward pass). The blob is pinned by SHA256 + CRC32 (currently `0x1a139335`); changing weights changes both, detected at load.

## 2. Exactly what to replace in Allen (do NOT replace everything)

All under `device/kalman/ParKalman/`. Only the **dipole crossing** is worth replacing.

- **`ExtrapolateUTT`** (`include/ParKalmanMethods.cuh:486`) — UT (z=2642.5) -> T (z=7855), dz~5213mm. **This is the target.** Pattern: run the polynomial `extrapUTT` (fills state + derivative buffers), then if `use_nn_utt` overwrite the **state** `x[0..3]` with `pinn_v2_utt_state(...)` while keeping the polynomial **Jacobian** `F` (hybrid-Jacobian rule; avoids on-device autodiff). qop invariant.
- Toggle: `m_use_nn_utt` (`include/ParKalmanFilter.cuh:189`), plumbed in `src/ParKalmanFilter.cu`, exposed in AllenConf `make_kalman_long`. Default `false` -> bytecode-identical to master.
- Phase-2 target: `src/ExtrapolateStates.cu::extrapolate_states_kernel` (standalone long-state RK loop, no Jacobian needed) -> a full "pure replacement" kernel.
- **KEEP** (out of dipole / geometric, already near-exact): `ExtrapolateInUT`, `ExtrapolateInT`, `ExtrapolateTFT`/`Def`, `ExtrapolateToVertex`.

`PINN_V2_UTT.cuh` is a GENERATED file — never hand-edit; regenerate via emit_cuda_header.py.

## 3. The R6 gates (in order)

1. **CUDA<->Python bit-bound parity (A5):** `pinn_v2_utt_state()` reproduces the Python reference to < 1 ULP on 200 reference tracks. Reduction order in the header must match the numpy reference forward (scalar fmaf order).
2. **Throughput (A6):** `allen_throughput` with `hlt1_pp_default`, `m_use_nn_utt` on; per-track GPU cost <= classical RKN4. Baseline cached in `For_Allen/pins/baseline_throughput.txt` (A5000 74.2 kHz). Root-cause any regression on the **classical** path vs the NN path.
3. **Moore physics (decisive):** `HltEfficiencyChecker` on standard MC; no track-quality line (VELO-UT match, SciFi seeding, long-track ghost rate) degrades > 0.5% absolute vs RK4. This is the real arbiter of whether the UT->T accuracy is sufficient.

## 4. Build / standalone harness

`Allen/ML_research/standalone/` is a CPU-compiled harness using the real Allen headers (via cuda_compat shims) for NN-vs-polynomial-vs-RK comparisons without a GPU (`make && ./run_extrapolators`). Full GPU build per Allen `CONTRIBUTING.md` (lb-stack / cmake), target a V100/A5000.

## 5. Provenance to keep in sync

Blob SHA/CRC, `For_Allen/pins/loader_v3_spec.md`, the Allen commit pin, and `device/kalman/ParKalman/PINN_V2_UTT_INTEGRATION.md` (the Allen-side integration note). If you re-lock a model, re-export the blob, regenerate the header, and bump these together.
