---
name: generate-corpus
description: Generate the RK4 ground-truth training corpus for the LHCb neural track extrapolator (signed dz, z_start, Allen qop). Use when asked to generate/regenerate training data, make a new dataset, or change the data contract.
---

# Generate the training corpus

The corpus is the ground truth the network imitates: fine-step classical RK4 propagation through the measured dipole field. It is **deterministic** and regenerable; it stays local (gitignored), never committed.

Working dir: `/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3`. Spec: `docs/plans/GENERATION_SPEC.md` (in the deliverable repo) / `GENERATION_SPEC.md`.

## Contract (gen-3)

- Input `X[N,7] = (x, y, tx, ty, q/p, z_start, dz)`; output `Y[N,5] = (x_f, y_f, tx_f, ty_f, qop_f)`.
- **Signed** `dz ∈ [-10000, +10000]` mm (Kalman backward pass); `z_start ∈ [0, 14000]` mm; `p ∈ [1, 200]` GeV/c log-uniform.
- `qop` = Allen `c·q/p` convention (C1). Polarity −1 (MagDown).
- Ground truth: `utils/rk4_propagator.py` (5 mm fine step) using `utils/magnetic_field.py` (trilinear interp of `twodip.rtf`, the 81×81×146 field grid). **The field map is required for data-gen** (and for PINN training) but NOT at inference.
- Determinism: per-batch `seed = 42 + batch_id * 7919`; 2000 batches × 25k tracks = 50M (or 10M for gen-3).

## Run

Via HTCondor (parallel batches):
```bash
condor_submit condor/submit_datagen.sub      # calls condor/run_datagen.sh per batch
```
Each batch writes a shard; merge with the data/merge script into `data/train_10M_gen3.npz` (keys `X`, `Y`, and metadata). Confirm the output shapes and the `data_sha256` (train.py stamps it into each run for provenance).

## Gotchas

- The loader in `train.py` loads the **full** npz into RAM then slices `max_samples` — size `request_memory` for the full file (~1 GB for 10M), not the subsample.
- `rk4_propagator.py` is the reference generator, **not** a deployment model — never confuse it with the trained candidates.
- If you change the contract (e.g. add a feature), update: the generator, `architectures.py` input_dim, `train.py` loaders, the V3 blob `feature_order`, and the Allen loader manifest — they must all agree.
