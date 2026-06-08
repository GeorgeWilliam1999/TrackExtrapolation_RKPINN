# `For_Allen` — Plan to deploy a neural RK replacement into full LHCb Allen + Moore

**Owner:** G. Scriven · **Started:** 2026-05-08 · **Status:** Phase 0 (bootstrap) — re-anchored 2026-05-19

> **RE-ANCHOR NOTICE (2026-05-19).** This plan was originally drafted around
> the hybrid `nrk4_tiny_1step_v1` candidate. Following [ADR 0009](docs/decisions/0009-replacement-goal-restated.md)
> and [`../REPLACEMENT_PLAN.md`](../REPLACEMENT_PLAN.md), the deployment slot
> moves to a **full neural replacement** of the RK extrapolator (MLP /
> PINN_v2 family). Until the per-phase rewrite lands, read the following
> substitutions throughout this document:
>
> | Old reference (hybrid) | New reference (replacement) |
> |---|---|
> | `nrk4_tiny_1step_v1` / "the M1 candidate" | the §R4 PINN_v2 winner from `REPLACEMENT_PLAN.md` |
> | "the trained corrector" / "the corrector" | "the learned model" (the network *is* the propagator, not a correction on top of one) |
> | `n_rk_steps`, `corrector_enabled` (per-arch metadata) | `hidden_dims`, `activation`, `input_dim` |
> | Phase 1a "no-training `n_rk_steps` sweep" | Phase R4 "PINN_v2 width × depth × λ_pde sweep" |
> | Phase 1b V3 loader magic `NRK4` | generic `NN_EXTRAP` magic with an `arch_tag` field (`MLP` / `PINN_V2` / `NRK4`) |
> | "hybrid RK4 + residual surface" | "drop-in full-NN surface; classical RKN4 is the safety-net fallback only" |
>
> The CUDA surface in Allen MR `!2497` (`NRKExtrapolator.cuh`) is retained as
> the correct abstraction layer; a follow-up MR replaces its inner RKN4 loop
> with a generic `forward_pass(weights, state)` call. See
> [`../CLEANUP_LIST.md`](../CLEANUP_LIST.md) §1.3.

This document is the *only* source of truth for what we are doing, in
what order, and what each phase has to deliver before the next can
start. It is written against the audit `docs/reports/allen_conditions_audit.tex`
and the deep-dive notebook (archived 2026-05-19) `experiments/gen_3/notebooks/_archive/nrk4_tiny_deep_dive_HYBRID.ipynb`,
and was reviewed by the `experiment-reviewer` subagent on 2026-05-08
(record in `docs/decisions/0001-experiment-reviewer-audit.md`).

## Goal

Take the gen-3 `nrk4` candidate model from "structurally Allen-ABI compliant
but failing physics + A4" (M1 outcome) to **passing the standalone Allen
Kalman smoke test, all kernel-level Allen unit tests with a V3 loader,
the `allen_throughput` regression, and Moore `HltEfficiencyChecker`
within 0.5 % per-line absolute efficiency loss.**

Anything outside that scope is out of scope until those gates are
green — no architecture exploration beyond the family decided in
Phase 1a, no new dataset generation beyond the Gen-3 corpus, no Gen-4
work.

## State at start of Phase 0

| Item | Value | Source |
|------|-------|--------|
| Allen commit | `12f26514959d` (master) | `pins/allen_commit.txt` |
| Allen checkout | sparse: `device/kalman/` + `ML_research/` only | `/data/bfys/gscriven/Allen` |
| Candidate | `nrk4_tiny_1step_v1` | `experiments/gen_3/trained_models/` |
| Stage-1 status | VELO 113 µm (gate 12 µm) FAIL; UT 116 µm (gate 50 µm) FAIL | deep-dive §6 |
| A4 status | FAIL — 16 of 25 off-diagonals at ~100 % rel err. | deep-dive §21 |
| §22 finding | 1-step RK4 is the cause of A4 failure, not the corrector | deep-dive §22 |
| §12 finding | Trained corrector is **anti-helpful** (113 µm → 35 µm when ablated) | deep-dive §12 |
| Loader gap | V2 in-tree loader hard-codes 6→4; V3 patch (M0.5) not yet upstream | audit §A7 |
| Training data so far | 200 k subset of 10 M corpus (2 %) | M1 results |

## Phase order — short version

```
0  Bootstrap  →  1a Arch ablation (no training)  →  1b V3 spec freeze
                                                         ↓
                                             2a 1 M calibration
                                                         ↓
                                             2b Full-10 M retrain (3 seeds)
                                                         ↓
                                             3  Standalone load test (CPU)
                                                         ↓
                                             4  G1–G3 standalone gates
                                                         ↓
                                             5  Full Allen GPU build + unit tests
                                                         ↓
                                             6  allen_throughput
                                                         ↓
                                             7  Moore HltEfficiencyChecker
                                                         ↓
                                             8  nsight-compute, fp16 deployment
```

## Decisions already taken (do not re-litigate)

| ID | Decision | Source | ADR |
|----|----------|--------|-----|
| D-1 | **Action N: corrector is removed.** §12 ablation drops mean \|Δx\| from 113 µm to 35 µm with corrector off; §22 confirms the corrector contributes nothing to the Jacobian. The corrector is a defect, not a feature. | deep-dive §§12, 22 | `docs/decisions/0002-action-N-corrector-removed.md` |
| D-2 | **Fix L (multi-step RK4) is mandatory, not optional.** Originally motivated by deep-dive §22's claim that A4 fails structurally at `n_rk_steps=1`; that claim was withdrawn on 2026-05-12 after the Phase 1a sweep showed it was an fp32-FD artefact. The *direction* (multi-step is needed) stands, driven by endpoint accuracy: **Phase 1a winner is `n_rk_steps = 2`, corrector OFF** ([ADR 0007](docs/decisions/0007-phase1a-winner.md)). | deep-dive §22 (historical) + Phase 1a sweep 2026-05-12 | `docs/decisions/0003-multistep-rk4-mandatory.md` (superseded-in-part) + `docs/decisions/0007-phase1a-winner.md` |
| D-3 | **`qop` convention = Allen `c·q/p`.** Not negotiable per `Allen/ML_research/README.md`. Embedded in the loader manifest and asserted at load time. | gen3_protocol.tex §C1 | — |
| D-4 | **Splits are event-grouped.** Track-level splits leak B-field/material correlations across train/val/test. The M1 `test_indices.npy` is rebuilt on this basis (Phase 0). | reviewer audit (b)(2) | `docs/decisions/0004-event-grouped-splits.md` |
| D-5 | **The M1 test set is frozen as `test_v1_frozen.npy`** (renamed, hashed, read-only). New event-grouped splits are used for Phase 2 development; the frozen set is reported only at gate decisions. | reviewer audit (b)(3) | `docs/decisions/0005-frozen-test-set.md` |

---

# Phase 0 — Bootstrap

**Owner:** GS · **Estimated effort:** 1–2 days · **Kill criteria:** none (this phase has to succeed)

## Outputs

* This `For_Allen/` repo, committed clean.
* `pins/allen_commit.txt`, `pins/protocol_sha.txt`, `pins/host.txt`,
  `pins/data_manifests/{train_10M,val,test_v1_frozen}.sha256`.
* `environment-lock.yml`, `pip-freeze.txt`.
* `MLflow` server reachable; `tracking/check_tags.py` refuses to start
  any run without the mandatory tag set.
* The 6 cheap sanity tests in `src/for_allen/sanity/` runnable from
  `pytest` against a 1 k-param toy model.
* `docs/decisions/0001`–`0005` written.

## Tasks

1. Initialise git, commit skeleton, set up pre-commit (black, ruff, no-large-files).
2. `conda env export --no-builds > environment-lock.yml`.
3. Compute SHA-256 of:
   * `experiments/gen_3/data/train_10M_gen3.npz` (and any field-map referenced in its metadata),
   * the existing `test_indices.npy` → after re-splitting, also stamp `test_v1_frozen.npy`.
4. Write the V3-loader manifest schema (`pins/loader_v3_spec.md`). Until
   this exists, no training; until this is reviewed, no Allen-side patch.
5. Write the 5 ADRs above.
6. Stand up MLflow on `mlruns/` (local file backend is fine for now).

## Gate to Phase 1

* `pytest tests/` green.
* `python scripts/sanity_check.py --toy` green.
* `git status` clean on `main`; pins all populated.

---

# Phase 1a — Architectural ablation, **no training**

**Owner:** GS · **Estimated effort:** 1 day (one notebook) · **Depends on:** Phase 0

## Why this phase exists

The reviewer's biggest single recommendation: before retraining, measure
how much of stage-1 and A4 is fixed by the *integrator* alone, with the
existing weights frozen, by sweeping `n_rk_steps`. This decides whether
Phase 2 is "loss redesign" (small intervention) or "full architecture
redesign" (big intervention).

## Sweep grid

| Axis | Values |
|------|--------|
| `n_rk_steps` | 1 (baseline), 2, 4, 8, 16, adaptive (target step ≤ 50 mm) |
| corrector | on (M1 default), **off** (per D-1; this is the primary axis) |
| precision | fp32 (sweep), fp16 (sanity at the chosen point) |

Cartesian product: 12 cells × 2 precisions = 24 evaluations. Each cell
runs the full deep-dive evaluation suite on `test_v1_frozen.npy`:
stratified 3 × 3 (\|q/p\|, \|dz\|) stage-1 grid, A4 (Frobenius +
max-off-diagonal), bwd/fwd, fp16 round-trip, edge probes.

## Acceptance

A cell is the **Phase 1a winner** if it satisfies, on the frozen test
set, with bootstrap 95 % CIs:

* VELO ⟨\|Δx\|⟩ < 24 µm in **every** (\|q/p\|, \|dz\|) cell (i.e. ≤ 2 × gate, headroom for retraining), and
* UT  ⟨\|Δx\|⟩ < 100 µm in every cell (≤ 2 × gate), and
* A4 Frobenius rel-err < 0.10 (relaxed from the eventual gate of 0.05), and
* bwd/fwd ratio ∈ [0.80, 1.25].

If **no** cell satisfies the above, see kill criteria below — Phase 2
will not save us.

## Kill criteria (project-level)

* No (n_steps, corrector) cell brings VELO under 24 µm or A4 under 0.10 → the tiny RHS MLP is too weak; widen RHS hidden dims (escalate to a Phase 1c) or fall back to "RKN with a learned residual" (this is a different family, escalate).

## Output

* `notebooks/01_arch_ablation.ipynb`, `configs/phase1a_arch_ablation/*.yaml`.
* `artifacts/phase1a/ablation.csv` — one row per (n_steps, corrector) cell, with all metrics + CIs.
* `docs/decisions/0007-phase1a-winner.md` — picks the cell, justifies, sets `N_RK_STEPS_PROD` constant for everything downstream.
  Result (2026-05-12): **winner is `n_rk_steps = 2`, corrector OFF**.
  Pinned in [`pins/n_rk_steps_prod.txt`](pins/n_rk_steps_prod.txt). [ADR 0003](docs/decisions/0003-multistep-rk4-mandatory.md) (`n_rk_steps ≥ 8`) is superseded-in-part by [ADR 0007](docs/decisions/0007-phase1a-winner.md).

---

# Phase 1b — V3 loader spec freeze, **no training, no Allen build**

**Owner:** GS · **Estimated effort:** 1 day · **Depends on:** Phase 0 (parallel with Phase 1a)

> **Status update (2026-05-12):** Phase 1a closed with corrector OFF and
> `n_rk_steps = 2` ([ADR 0007](docs/decisions/0007-phase1a-winner.md)),
> which means the "model" body has **no learned weights** — the V3
> loader is therefore no longer on the critical path for the first
> pipeline test. An interim Allen-side wiring step has been done in its
> place: the `NRKExtrapolator` socket grew a complete `propagate(...)`
> surface and the `extrapolate_states_t` algorithm now exposes a
> `use_nrk` toggle, mirroring the surface choices of
> MR [!2407](https://gitlab.cern.ch/lhcb/Allen/-/merge_requests/2407).
> See [ADR 0008](docs/decisions/0008-allen-wiring-plan.md). The V3
> loader work below is preserved for the day a learned RHS comes back
> on the table.

## Outputs

* `pins/loader_v3_spec.md` — schema for the `.bin` manifest, complete and signed off.
* `src/for_allen/export/manifest.py` — Python writer + reader matching the spec.
* `src/for_allen/export/bin_v3.py` — Python writer for the `.bin` blob (header + weights + manifest).
* `tests/test_manifest_roundtrip.py` — round-trip a 1 k-param toy model through writer → bytes → reader; assert all fields recoverable, weights bit-exact (fp32) or within 1 ULP (fp16).
* C++ stub of the V3 loader (`Allen/ML_research/standalone/MLPExtrapolatorV3.cuh.draft`) compiles standalone; **not yet integrated into Allen**.

## Mandatory manifest fields (the loader rejects any `.bin` missing any one)

```
loader_version_required        int (= 3)
input_layout_version           int (= 3)
output_layout_version          int (= 3)
qop_convention                 string (asserted == "allen_v1" at load)
c_light_value                  double (asserted within 1e-9 of 299.792458)
dz_signed                      bool (asserted true)
feature_order                  list<string>  (must equal ["x","y","tx","ty","qop","z_start","dz"])
output_order                   list<string>  (must equal ["x_f","y_f","tx_f","ty_f","qop_f"])
n_rk_steps                     int
adaptive_rk                    bool
input_mean                     float[7]
input_std                      float[7]
output_mean                    float[5]
output_std                     float[5]
input_mean_sha                 hex32  (SHA-256 of input_mean)
weights_sha                    hex32  (SHA-256 of the weight blob)
data_train_sha                 hex32
training_seed                  int
mlflow_run_id                  string
git_sha                        hex40
allen_commit                   hex40
protocol_sha                   hex40
torch_version                  string
exported_utc                   ISO-8601 string
```

This list is locked at the end of Phase 1b. Any change after that is a
re-export, not a re-train, but the schema itself does not change.

## Acceptance

* Round-trip test green.
* `pins/loader_v3_spec.md` reviewed (one human reviewer).
* C++ draft compiles in the existing standalone build (`make` in `ML_research/standalone/`) — **not** wired into `main.cpp` yet.

---

# Phase 2a — 1 M calibration run

**Owner:** GS · **Estimated effort:** 2–3 days · **Depends on:** 1a, 1b

## Why this phase exists

A 200 k → 10 M scale jump (50 ×) without an intermediate is the most
likely silent failure mode (reviewer audit b.1). 1 M lets us tune LR
schedule, batch size, weight decay, and epoch count cheaply.

## Inputs

* `n_rk_steps` and corrector setting from Phase 1a winner.
* `Fix I` — detector-σ-weighted endpoint loss:
  σ = (12 µm, 12 µm, σ_tx, σ_ty, σ_qop) at VELO; layer-dependent at UT/SciFi.
  Implementation: σ-weighted MSE per output, weights set per target z_f bin.
* `Fix J` — Jacobian regularisation: on a per-batch subset of 64 states,
  `torch.func.jacfwd` against a cached RK45-reference Jacobian (see Phase 0).
  Loss term: `λ_J * ||J_model − J_RK45||_F^2 / ||J_RK45||_F^2`.
  λ_J swept ∈ {0, 1e-3, 1e-2, 1e-1, 1}.

## Acceptance

* Val loss decreases by at least √50 ≈ 7 × the 200 k val loss within
  2 epochs of the 1 M run (training-mechanics sanity).
* The 6 per-checkpoint sanity tests (see §“Per-checkpoint smoke battery”)
  pass on every saved ckpt.
* On the frozen test set, the best 1 M model achieves stage-1 within
  2 × gate (i.e. VELO < 24 µm, UT < 100 µm) — same threshold as Phase
  1a, since the goal is *progress*, not yet PASS.

## Kill

* Best 1 M model worse than the Phase 1a winner on the frozen test set →
  the loss redesign is regressing what the integrator already gives us.
  Stop, debug.

## Output

* MLflow runs tagged `phase=2a`.
* `configs/phase2a_calibration_1M/*.yaml`, one per LR/λ_J combination.
* `docs/decisions/0007-phase2a-recipe.md` — picks the LR/wd/epoch/λ_J recipe for Phase 2b.

---

# Phase 2b — Full 10 M retrain, 3 seeds

**Owner:** GS · **Estimated effort:** 5–7 days (compute bound) · **Depends on:** 2a

## Recipe

* Architecture: Phase 1a winner.
* Loss: σ-weighted endpoint MSE + λ_J Jacobian-regularisation (Fix J at
  the Phase 2a-chosen λ_J).
* Train data: full 10 M, event-grouped train/val split (D-4).
* Seeds: 3 independent (numpy, torch, cuda, dataloader, PYTHONHASHSEED).
* Precision: fp32 training; fp16 evaluated only at the export step.

## Acceptance — the **production gate**

On the frozen `test_v1_frozen.npy`, in *every cell* of a 3 × 3
(\|q/p\|, \|dz\|) grid, with 1000-sample BCa bootstrap CIs:

| Metric | Gate | Stretch |
|--------|------|---------|
| VELO ⟨\|Δx\|⟩ | < 12 µm AND < 1.2 × RKN AND 95 % bootstrap upper bound < 15 µm | < 8 µm |
| UT ⟨\|Δx\|⟩ | < 50 µm AND < 1.2 × RKN | < 30 µm |
| SciFi-T3 \|ρ(Δx, q/p)\| | < 0.10 | < 0.05 |
| bwd/fwd ratio | ∈ [0.80, 1.25] | ∈ [0.90, 1.10] |
| A4 ‖J − J_RK45‖_F / ‖J_RK45‖_F | < 0.05 AND max off-diagonal rel-err < 0.20 | F-rel-err < 0.02 |
| fp16 max position shift | < 1 µm | < 0.5 µm |
| Determinism (per (e)) | bit-identical fp32 | — |

All three seeds must pass; the bootstrap CIs must agree across seeds
within ± 1 σ of the gate.

## Kill

* After Phase 2b, with all 3 seeds: VELO mean still > 24 µm on the
  frozen set across all seeds → architecture-level kill, escalate.
* A4 Frobenius rel-err still > 0.10 with the chosen `n_rk_steps` → the
  RHS MLP is too weak even with regularisation; escalate.

## Output

* 3 final checkpoints in MLflow, each with the 6-test sanity battery
  green and the per-cell stratified gate table green.
* The seed-mean checkpoint is selected as the Phase 2b winner.
* `docs/decisions/0008-phase2b-winner.md`.

---

# Phase 3 — V3 export and CPU-build standalone load test

**Owner:** GS · **Estimated effort:** 2 days · **Depends on:** 1b, 2b

## Tasks

1. Run `scripts/export_bin.py` on the Phase 2b winner → `nrk4.fp32.bin`, `nrk4.fp16.bin`, manifest, SHA-256s.
2. Apply the V3 loader patch from Phase 1b draft into
   `Allen/ML_research/standalone/MLPExtrapolator.cuh` (or a new
   `MLPExtrapolatorV3.cuh`); rebuild `run_extrapolators` via `make` in
   the existing standalone Makefile (CPU-only, no CUDA). Add a
   `--loader-version` flag.
3. Add a unit test in `Allen/ML_research/standalone/test_v3_loader.cpp`
   that asserts:
   * `nrk4.fp32.bin` loads,
   * RKN baseline (no `--model`) still passes G1–G3 with reference numbers,
   * `mlp_medium.bin` (gen-2 V2 control) **still fails** G1–G3,
   * a deliberately-corrupted manifest (`qop_convention="legacy"`) is **rejected**.

## Acceptance

* All four assertions PASS.
* No silent acceptance: a manifest with a missing required field returns a non-zero exit code.

## Kill

* V2 control no longer fails after the patch → loader is silently
  accepting wrong layouts. Hard stop, bug hunt before any GPU work.

## Output

* `Allen/ML_research/standalone/MLPExtrapolatorV3.cuh` (final, header-only, ready to upstream).
* CPU-build smoke test green.
* PR-ready patch against Allen master tagged `m0.5/v3-loader`.

---

# Phase 4 — G1–G3 on the candidate

**Owner:** GS · **Estimated effort:** 0.5 day · **Depends on:** 3

## Tasks

```bash
cd Allen/ML_research/standalone
./run_extrapolators --model nrk4.fp32.bin
python <<'EOF'
import pandas as pd
r = pd.read_csv('kalman_results_mlp.csv').dropna(subset=['chi2'])
assert r.chi2_ndof.mean() < 2.0
assert abs((r.p_fit/r.p_true - 1).mean()) < 0.01
assert r.pull_x.std() < 1.5
assert abs(r.pull_x.mean()) < 0.1
EOF
```

Then repeat for `nrk4.fp16.bin`.

## Acceptance

* All four assertions PASS for both fp32 and fp16, with bootstrap 95 %
  upper bound on χ²/ndof < 2.2 and CI on \|⟨pull_x⟩\| < 0.13.

## Kill

* G1–G3 individually green but χ²/ndof > 2 with bootstrap → covariance
  plumbing wrong; halt before GPU work.

## Output

* `notebooks/04_standalone_g123.ipynb` reproducing the assertions with CIs.
* `artifacts/phase4/g123_summary.json`.

---

# Phase 5.0 — Allen exercise merge request (skeleton, no model)

**Owner:** GS · **Estimated effort:** done (2026-05-10) · **Depends on:** none

Supervised exercise to learn the Allen contribution workflow before the
real V3 MR. Lands a header-only `Extrapolators::NRKExtrapolator` with a
passthrough body and a 5-case Catch2 test under `[NRKExtrapolator]`. No
algorithm wrapper, no sequence wiring, no weights — algorithm is
unreachable from any current Allen sequence. See
`docs/decisions/0006-allen-exercise-pr.md` for the full record. Cheap
CI gates (copyright, formatting) verified locally; full build + ctest
deferred to push. **Not pushed; awaiting review and supervisor sign-off.**

---

# Phase 5 — Full Allen GPU build + kernel unit tests

**Owner:** GS · **Estimated effort:** 2–4 days (most of it env setup) · **Depends on:** 4, 5.0

## Tasks

1. Densify Allen checkout (`git sparse-checkout disable`); pin same commit `12f26514959d`.
2. Apply the V3 loader patch from Phase 3.
3. Build with the project's standard `lb-stack-setup` / `cmake` recipe (per `Allen/CONTRIBUTING.md`); target an A5000 (or equivalent A100/T4) compute capability.
4. Run the full Allen unit-test suite under `Allen/test/` (specifically the kalman/extrapolator unit tests). Both with the model and without.

## Acceptance

* All Allen unit tests PASS with the patched loader (with and without `--model`).
* No new failures vs. the unpatched master — i.e. the patch has zero
  regressions on tests not involving the V3 loader.

## Kill

* Kernel unit tests fail under cuDNN nondeterminism only → document,
  constrain to deterministic algos for validation, do not paper over.

## Output

* `docs/reports/phase5_unit_tests.md` with the full test matrix.
* GPU-build artifacts (gitignored, but their SHAs in `pins/`).

---

# Phase 6 — `allen_throughput` regression

**Owner:** GS · **Estimated effort:** 1–2 days · **Depends on:** 5

## Tasks

1. Run `allen_throughput` with the standard MC input file used by Allen CI.
2. Configure two sequences:
   * `hlt1_pp_default` (RKN extrapolator),
   * `hlt1_pp_default_nrk4` (the same sequence with our extrapolator swapped in),
   and measure events/s, p99 latency, GPU memory, and constant-mem use for both.
3. Repeat 5 times per configuration, report mean + SE.

## Acceptance

| Metric | Gate | Stretch |
|--------|------|---------|
| Throughput vs RKN | ≥ 0.90 × | ≥ 0.95 × |
| p99 latency | ≤ 1.10 × RKN | ≤ 1.05 × |
| Constant-mem use | ≤ 32 kB (50 % of budget) | ≤ 16 kB |

## Kill

* Throughput < 0.5 × RKN with no clear optimisation path → deployment
  blocked, escalate (fp16-only? smaller model? Phase 8 first?).

## Output

* `docs/reports/phase6_throughput.md`.
* `artifacts/phase6/allen_throughput_log/`.

---

# Phase 7 — Moore `HltEfficiencyChecker`

**Owner:** GS · **Estimated effort:** 2–3 days (Moore env setup) · **Depends on:** 6

## Tasks

1. Build Moore against the patched Allen (project standard `lb-dev` workflow).
2. Run `HltEfficiencyChecker` on standard MC samples for the lines that
   depend on track quality:
   * `Hlt1TrackMVA`,
   * `Hlt1TwoTrackMVA`,
   * `Hlt1MaterialVertexSeeds`,
   * any line tagged `requires:long_tracks` in the TCK.
3. Report per-line absolute and relative efficiency vs RKN baseline.

## Acceptance

* Per-line absolute efficiency loss < 0.5 % (gate); average across
  lines < 0.2 % (gate).
* Stretch: per-line < 0.2 %, average < 0.1 %.
* Stat-only error bars reported alongside.

## Kill

* Per-line regression > 0.5 % on any line and the mechanism is not
  understood within one phase of work → escalate.

## Output

* `docs/reports/phase7_hlt_efficiencies.md`.

---

# Phase 8 — Profile and fp16 deployment decision

**Owner:** GS · **Estimated effort:** 1 day · **Depends on:** 7

## Tasks

1. `nsight-compute` on the patched extrapolator kernel.
2. Report occupancy, register pressure, shared-mem use, memory throughput, instruction throughput.
3. Rerun Phase 6 with fp16 weights; compare throughput and accuracy.
4. Final recommendation: fp32 or fp16, with hard numbers.

## Output

* `docs/reports/phase8_profile.md`, the **deployment recipe**.
* `docs/decisions/0009-deployment-precision.md`.

---

# Per-checkpoint smoke battery (mandatory at every save)

These run automatically in `scripts/sanity_check.py` and are wired into
the training loop via a callback. A checkpoint that fails any one is
moved to `artifacts/scratch/` and **never** promoted.

1. **Determinism**: same 8 inputs forwarded twice → bit-identical fp32 / max ULP diff ≤ 2 fp16.
2. **Zero-dz identity**: dz = 0 inputs → outputs match inputs to < 1e-5 (fp32).
3. **A4-lite**: 16 random states, autograd Jacobian vs. 5-point finite-diff Jacobian, Frobenius rel-err < 0.05.
4. **Forward/backward closure**: 64 tracks, propagate forward then backward, ⟨‖x_round_trip − x_0‖⟩ < 50 µm.
5. **100-track stage-1**: subset of `test_v1_frozen.npy` reporting VELO ⟨\|Δx\|⟩ and bwd/fwd. Soft-warn outside gate, hard-fail at > 3 × gate.
6. **Manifest round-trip**: write `.bin`, read back via Python V3 loader mirror, assert all metadata recoverable and outputs match the `.pt` to < 1e-4 (fp32) / < 1e-2 (fp16).

Total wall time: < 30 s on a single GPU. No exceptions.

---

# Project-level kill criteria

Any one of these triggers a Phase-0 architecture review (not a tweak):

* After Phase 1a + 2b: VELO mean > 24 µm on the frozen test set across all 3 seeds.
* After Phase 1a: A4 Frobenius rel-err > 0.10 at adaptive RK with > 32 effective steps.
* After Phase 6: best achievable throughput < 0.7 × RKN with no clear path to fix.
* After Phase 7: no recipe achieves < 0.5 % per-line efficiency loss.

## Fall-back architectures (in order of preference)

1. **Wider RHS MLP** (hidden dims 64 → 128 or 256) inside the same NeuralRK4 family.
2. **RKN-residual**: take Allen's production RKN as a fixed front-end, learn only a small additive residual on top. Smaller failure surface.
3. **Neural-ODE** with adaptive solver. Higher cost, deferred.
4. **Learned correction on RK45**. Last resort.

The decision tree is in `docs/decisions/0001-experiment-reviewer-audit.md`.

---

# Cross-references

* Audit document: [docs/reports/allen_conditions_audit.tex](../../docs/reports/allen_conditions_audit.tex)
* Deep-dive notebook: [experiments/gen_3/notebooks/nrk4_tiny_deep_dive.ipynb](../notebooks/nrk4_tiny_deep_dive.ipynb)
* Gen-3 protocol: [docs/reports/gen3_protocol.tex](../../docs/reports/gen3_protocol.tex)
* Generation spec: [experiments/gen_3/GENERATION_SPEC.md](../GENERATION_SPEC.md)
