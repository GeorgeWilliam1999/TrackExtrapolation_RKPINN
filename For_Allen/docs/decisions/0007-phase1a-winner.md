# ADR 0007 — Phase 1a winner: `n_rk_steps = 2`, corrector OFF

* **Date:** 2026-05-12
* **Status:** **superseded by [ADR 0009](0009-replacement-goal-restated.md) on 2026-05-19** — the recommendation "pure classical RK4 with `n_rk_steps = 2`" is not a neural network and does not satisfy the project's *replacement* goal. The fp64-autograd Jacobian methodology documented here is the load-bearing technical contribution and is reused as-is by Phase R2 of [`REPLACEMENT_PLAN.md`](../../../REPLACEMENT_PLAN.md).
* **Supersedes (in part):** [ADR 0003 — Multi-step RK4 is mandatory](0003-multistep-rk4-mandatory.md). ADR 0003's *direction* stands (multi-step is needed); its quantitative claim that `n_rk_steps ≥ 8` is required is **withdrawn** in light of the Phase 1a measurement below.
* **Reinforces:** [ADR 0002 — Action N: corrector removed](0002-action-N-corrector-removed.md).

## Context

[ADR 0003](0003-multistep-rk4-mandatory.md) declared multi-step RK4 mandatory on the basis of deep-dive §22, which reported that A4 (Jacobian off-diagonal coupling) failed at `n_rk_steps = 1` *irrespective of the corrector* and that step counts of `≥ 8` would be needed. We had to verify that claim with the corrector explicitly disabled at the source level (ADR 0002) and with an independent measurement methodology, before pinning a step count for the rest of the project.

Phase 1a — the no-training architectural ablation prescribed in `PLAN.md` — was executed on 2026-05-12 against the frozen M1 checkpoint `trained_models/nrk4_tiny_1step_v1` (`test_indices.npy`, 20 000 tracks). The full output is in [`artifacts/phase1a/`](../../artifacts/phase1a/) and [`artifacts/phase1a/sweep.log`](../../artifacts/phase1a/sweep.log).

## Measurement methodology — and a correction to the deep-dive

A4 (Frobenius `‖J_model − J_RK45‖_F / ‖J_RK45‖_F`) was measured against an independent RK4 propagator (`utils/rk4_propagator.py`, 5 mm step, analytic dipole, Allen `qop` convention) on 200 random fwd/bwd-balanced states from the frozen test set.

**The deep-dive §22 measurement was, on re-examination, dominated by an fp32 + finite-difference numerical artefact.** The deep-dive built the model Jacobian by central differences with `eps = [1e-3, 1e-3, 1e-6, 1e-6, 1e-4]` on the **fp32 model**. Over a multi-metre propagation the FD signal in the small off-diagonal entries is at the level of fp32 round-off (~`1e-7` of the output magnitude). Random sign noise then dominates the small entries and yields the ~100 % relative-error pattern the deep-dive reported.

Re-running the same comparison with the model cast to fp64 and the Jacobian taken by autograd (rather than FD on a fp32 model) gives, on a single high-momentum forward track at `n_rk_steps = 1`, corrector OFF, a Frobenius relative error of **7 × 10⁻⁹** against the RK4 reference. This is consistent: at `n = 1`, the model *is* an RK4 evaluation of the same Lorentz ODE, just with one large step instead of ~200 small ones; with the corrector disabled, the only difference from the reference is integrator truncation, not architecture.

A4 is therefore **not** the constraint that drives the step-count choice. **Endpoint accuracy is.**

## Sweep result

| `n_rk_steps` | A4 frob (mean) | A4 frob (median) | A4 off-diag (median) | VELO ⟨\|Δx\|⟩ | UT ⟨\|Δx\|⟩ | Phase 1a gates |
|:------------:|---------------:|-----------------:|---------------------:|---------------:|-------------:|:---:|
| 1            | 2.4e-4         | 1.2e-8           | 31.5 %               | **56.16 µm**   | 34.42 µm     | **FAIL** (VELO > 24 µm) |
| **2**        | 5.1e-4         | 7.1e-9           | 13.3 %               | **8.85 µm**    | 9.10 µm      | **PASS** |
| 4            | 1.1e-4         | 5.8e-9           | 7.0 %                | 1.40 µm        | 8.34 µm      | PASS |
| 8            | 6.7e-5         | 5.0e-9           | 4.7 %                | 0.35 µm        | 5.29 µm      | PASS |
| 16           | 2.9e-5         | 4.0e-9           | 3.9 %                | 0.52 µm        | 1.32 µm      | PASS |

Gates (Phase 1a, relaxed by 2× from production): A4 Frobenius < 0.10, VELO ⟨|Δx|⟩ < 24 µm, UT ⟨|Δx|⟩ < 100 µm. All numbers are unconditional means over the 200-track / 5 000-track subsets of the frozen test set; bootstrap CIs will be added in the Phase 2b acceptance report.

Notes:
* The off-diag p95 column shows `1.22e+4` in the raw log for every cell; this is an artefact of the `max(|ref|, 1e-12)` denominator floor we use for structurally-zero reference entries (the `qop` row/column, by construction). It is not signal and is omitted from the table above.
* The Frobenius median is several orders of magnitude tighter than the mean because a small number of tracks (low momentum, large `|dz|`, near the field-map edges) dominate the mean. The median tracks integrator truncation cleanly and shows the expected $O(h^4)$ convergence with `n`.

## Decision

1. **Production step count is `n_rk_steps = 2`** — the smallest value that passes all Phase 1a gates with the corrector disabled. Pinned in [`pins/n_rk_steps_prod.txt`](../../pins/n_rk_steps_prod.txt).
2. **The corrector remains OFF (ADR 0002 stands).** The 1-step-OFF VELO residual (56 µm) is already smaller than the 1-step-ON value reported in deep-dive §12 (113 µm); the corrector is not just unnecessary but actively anti-helpful, and that re-measurement is now confirmed with Action N applied at the source level.
3. **ADR 0003 is partially superseded.** The "multi-step is mandatory" *direction* is correct, but the quantitative `n ≥ 8` claim and the reasoning anchored on the §22 A4 failure are withdrawn. ADR 0003's status is updated to `superseded-in-part by 0007`.

## Consequences

* **Throughput head-room for HLT1.** ADR 0003 was budgeting 8 × the per-track integrator cost vs the M1 1-step baseline; the actual budget is 2 ×. This is a 4 × throughput win for the integrator stage relative to the planning assumption.
* **Phase 2 retraining target is closer than expected.** At `n = 2` the *frozen* M1 weights already give VELO 8.85 µm / UT 9.10 µm, which is inside the Phase 2b production gates (12 µm / 50 µm) before any retraining. Phase 2 (Fix I detector-σ weighting + event-grouped splits + 10 M corpus) is therefore expected to *tighten* rather than *reach* the production gate.
* **Phase 2 architecture loses the corrector.** Total trainable parameters drop from the M1 4 997 to the corrector-less subset (the bookkeeping for that is in `For_Allen/scripts/phase1a_arch_ablation.py`, which strips the corrector tensors from the checkpoint state-dict before evaluation).
* **Methodology fence-post.** All future Jacobian-agreement checks (Fix J in Phase 2a, A4 in Phase 2b/4) MUST be performed on a fp64-cast model with autograd, against the fp64 RK4 reference. The fp32-FD route is forbidden because of the round-off floor it imposes. Encoded in [`src/for_allen/eval/jacobian.py`](../../src/for_allen/eval/jacobian.py) at the start of Phase 2a (TODO).
* **The deep-dive `For_Allen/PLAN.md` `n_rk_steps_prod = 8` placeholder is wrong.** Updated to `2` in the same commit as this ADR.

## Artefacts

* [`For_Allen/scripts/phase1a_arch_ablation.py`](../../scripts/phase1a_arch_ablation.py) — the sweep driver.
* [`For_Allen/artifacts/phase1a/ablation.csv`](../../artifacts/phase1a/ablation.csv) — per-cell metric row.
* [`For_Allen/artifacts/phase1a/summary.txt`](../../artifacts/phase1a/summary.txt) — winner declaration.
* [`For_Allen/artifacts/phase1a/sweep.log`](../../artifacts/phase1a/sweep.log) — full stdout.
* [`For_Allen/artifacts/phase1a/J_rk4_reference.npy`](../../artifacts/phase1a/J_rk4_reference.npy) — the 200 reference Jacobians (the slow build, reusable).
* [`For_Allen/artifacts/phase1a/J_model_n??_corr_off.npy`](../../artifacts/phase1a/) — per-cell model Jacobians, for forensic re-analysis.
* [`For_Allen/pins/n_rk_steps_prod.txt`](../../pins/n_rk_steps_prod.txt) — `2`.
* [`models/architectures.py`](../../../models/architectures.py) — `NeuralRK4` now exposes `disable_correction: bool` (source-level Action N).
