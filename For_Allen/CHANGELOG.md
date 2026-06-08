# Changelog

All changes to this workspace are logged here in addition to git
history. Entries are reverse-chronological. Format: `YYYY-MM-DD —
phase — short description. (commit-sha-or-link)`.

## 2026-05-20 — Phase R5 (export pipeline, Python half)
* Locked the V3 blob byte layout in
  [`pins/loader_v3_spec.md`](pins/loader_v3_spec.md): magic `NRKv3`,
  16-B-aligned weight payload, row-major `nn.Linear` storage, CRC32
  trailer. Only `arch_id = 1 = PINN_V2_DIPOLE` and
  `activation_id = 1 = TANH` are valid in v3; the format intentionally
  does **not** carry an MLP or NRK4 arch tag (narrower than the
  re-anchor note in `PLAN.md` suggested — see review below).
* Added `for_allen.export` package with the writer / reader /
  numpy-reference-forward implementation
  ([`src/for_allen/export/blob_writer.py`](src/for_allen/export/blob_writer.py)).
* Added round-trip + CRC + size-budget tests
  ([`tests/test_blob_roundtrip.py`](tests/test_blob_roundtrip.py));
  `pytest tests/test_blob_roundtrip.py` → **6 / 6 PASS**.
* Froze the locked artefact
  [`artifacts/blobs/v3/pinn_v2_ALLEN_v1.bin`](artifacts/blobs/v3/pinn_v2_ALLEN_v1.bin)
  (41 604 B, CRC32 `0x1a139335`, SHA256
  `c66576709288f046d399b4578353c81549df930a4e4617ed5545dc649c87e52c`);
  provenance in
  [`artifacts/blobs/v3/TAG_INFO.json`](artifacts/blobs/v3/TAG_INFO.json).
* Independent reviewer audit (2026-05-21) recomputed CRC32, parsed the
  header from scratch, verified `n_layers = 3`, `total_params = 10 372`,
  shapes `[(96,6),(96,96),(4,96)]`, and confirmed the normalisation
  values are physically plausible (x,y ~ O(1 mm), tx,ty ~ O(1e-3),
  qop ~ O(1e-4) MeV⁻¹). **Verdict: PASS** for the Python half. The
  CUDA-vs-Python bit-bound parity gate (spec §4, 1 ULP on 200 A4
  tracks) is the R6 entry test and remains open.

## 2026-05-12 — Phase 1b (Allen wiring, part 1)
* Extended the `NRKExtrapolator` socket on the
  `gscriven/nrk-extrapolator-exercise` Allen branch with the
  Magfield-aware `make_step(state, dz, field)` overload and a fixed-size
  `propagate(state, target_z, field, step_size, max_steps)` entry whose
  signature matches `RungeKuttaNystromExtrapolator::propagate`. Defaults
  encode the Phase 1a winner: `default_step_size = 500 mm`,
  `default_max_steps = 100`, giving `n_rk_steps = 2` over a typical 1 m
  Kalman step.
* Added a `use_nrk` boolean property (default `false`) to
  `extrapolate_states_t`. With the default it is byte-for-byte
  equivalent to the prior commit; with `true` it dispatches to the new
  NRK propagator. This is the smallest pipeline test surface and the
  template is MR
  [!2407](https://gitlab.cern.ch/lhcb/Allen/-/merge_requests/2407)
  (Hoffmann, Adaptive RKN).
* Extended `TestNRKExtrapolator.cu` with two multi-step tests
  (zero-field N-step consistency, qop linearity across chained
  sub-steps).
* New ADR
  [`docs/decisions/0008-allen-wiring-plan.md`](docs/decisions/0008-allen-wiring-plan.md)
  records the wiring strategy and acceptance gates, and frames the
  PrKalmanFilter substitution as the next MR.

## 2026-05-12 — Phase 1a
* Phase 1a no-training sweep executed on the frozen
  `nrk4_tiny_1step_v1` checkpoint.
  Grid: `n_rk_steps ∈ {1, 2, 4, 8, 16}`, corrector OFF (ADR 0002 frozen),
  A4 on 200 random fwd/bwd-balanced tracks, stage-1 on 5 000 tracks
  from the frozen test set.
* **Winner: `n_rk_steps = 2`, corrector OFF.** VELO ⟨\|Δx\|⟩ = 8.85 µm,
  UT ⟨\|Δx\|⟩ = 9.10 µm, A4 Frobenius mean = 5.1e-4. Already inside the
  Phase 2b production gates (12 µm / 50 µm) on the *frozen* M1 weights
  before any retraining.
* Methodological correction: deep-dive §22 A4 failure was an fp32-FD
  numerical artefact, not a structural property of the integrator.
  Recorded in [`docs/decisions/0007-phase1a-winner.md`](docs/decisions/0007-phase1a-winner.md).
* ADR 0003 (`n_rk_steps ≥ 8`) updated to `superseded-in-part by 0007`.
* `pins/n_rk_steps_prod.txt` created = `2`.
* `configs/README.md` example updated from `n_rk_steps: 8` to
  `n_rk_steps: 2`.
* `PLAN.md` Phase 1a output section and decision-D-2 row updated to
  point at ADR 0007.
* Source-level change in `models/architectures.py`: `NeuralRK4` gains a
  `disable_correction: bool = False` flag — the Action-N path is now
  expressed in source, not by monkey-patching.
* New artefacts under `artifacts/phase1a/`:
  `ablation.csv`, `summary.txt`, `sweep.log`, `X_a4.npy`,
  `J_rk4_reference.npy`, and `J_model_n??_corr_off.npy` for the five
  cells.
* New driver: `scripts/phase1a_arch_ablation.py`.

## 2026-05-08 — Phase 0
* Initial repo skeleton created.
* `experiment-reviewer` audit recorded as ADR 0001.
* PLAN.md, ACCEPTANCE.md, ENVIRONMENT.md, README.md committed.
* Allen commit pinned: `12f26514959d`.
