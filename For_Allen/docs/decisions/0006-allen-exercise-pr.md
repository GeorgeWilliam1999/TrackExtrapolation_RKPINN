# ADR 0006 — Allen exercise merge request: NRKExtrapolator skeleton

**Status:** Proposed (exercise — not pushed)
**Date:** 2026-05-10
**Authors:** G. Scriven, supervised exercise per supervisor request
**Allen base commit:** `12f26514959d` (master, 2026-05-09)
**Branch (local only):** `gscriven/nrk-extrapolator-exercise`

## Context

Phase 5 of `PLAN.md` ends with a real merge request landing the V3 NRK4
extrapolator into Allen `device/kalman/ParKalman/`. That MR cannot be
written today: the V3 model has not been trained, and the deep-dive
notebook §21 + §22 demonstrated that no 1-step RK4 model can satisfy the
A4 Jacobian fingerprint. The supervisor nevertheless asked us to draft
the MR scaffolding now as a contributing exercise.

## Decision

Land an Allen MR that introduces a header-only `Extrapolators::NRKExtrapolator`
struct in `device/kalman/ParKalman/include/`, with the same surface API as
the existing `RungeKuttaNystromExtrapolator` but taking the magnetic
field vector directly (`float3 B`) rather than a `MagneticField::Magfield`
grid. The body is a faithful restatement of the in-tree RKN
`make_fast_step` recipe — numerically equivalent to RKN under identical
inputs.

The MR ships **no** algorithm wrapper, **no** sequence wiring, **no**
property hooks, and **no** trained weights. The header is unreachable
from any current Allen sequence.

## Consequences

### Why this is safe to propose

- **Unreachable code path.** Throughput Δ ≈ 0 (dead code), physics Δ = 0
  exactly. Allen `run` and `test` CI stages should be neutral.
- **Strictly precedented shape.** Mirrors `RungeKuttaNystromExtrapolator`,
  which is itself a header-only struct in the same namespace.
- **Stable API for V3 follow-up.** When V3 weights and architecture are
  ready, only the body of `make_step` changes. API and unit tests are
  preserved.

### Why this is *not* a real algorithm yet

- `make_step` is currently a numerical clone of `RKN::make_fast_step`.
  No machine-learning component is present.
- The Jacobian-emitting overload (`make_fast_step_and_evaluate_jacobian`)
  is **not** included. Any future Kalman-filter integration will require
  re-implementing the analytic Jacobian; we explicitly defer that to the
  V3 PR and call it out in the MR description.

### CI gates expected to pass on push

| Gate | Expected | Verified locally |
|---|---|---|
| `check-env` | Pass — branch lives in `lhcb/Allen` | n/a |
| `check-copyright` | Pass — Apache-2.0 header verbatim on all new files | ✅ exit 0 via `lb-check-copyright master` |
| `check-formatting` | Pass — `lb-format` produces empty patch | ✅ `lb-format -n` clean |
| `build` (CPU + CUDA) | Should pass — header-only addition compiles in Catch2 unit, no new warnings expected | ⏸ deferred (full LCG build out of scope for exercise) |
| `check-warnings` | Should pass | ⏸ deferred |
| `run` (throughput, physics) | Pass — algorithm is unreachable | ⏸ deferred |
| `test` (CTest) | Pass — five new Catch2 cases register via `catch_discover_tests` | ⏸ deferred (depends on full Allen build) |

The four "deferred" gates are exactly the ones that require a full Allen
LCG/CUDA build. We accept that CI on push is the first place those
actually run.

## Files (4)

| File | Status | Lines |
|---|---|---|
| `device/kalman/ParKalman/include/NRKExtrapolator.cuh` | NEW | 99 |
| `device/kalman/ParKalman/include/NRKExtrapolatorConstants.cuh` | NEW | 23 |
| `test/unit_tests/CMakeLists.txt` | EDIT | +1/-1 (link `WrapperInterface`) |
| `test/unit_tests/generic/src/TestNRKExtrapolator.cu` | NEW | 149 |

Total: +272 / −1.

## Unit tests

`TestNRKExtrapolator.cu` registers five Catch2 v3 cases under tag
`[NRKExtrapolator]`:

1. `unit_tests.nrk_extrapolator.zero_field_straight_line` (A2)
2. `unit_tests.nrk_extrapolator.sign_symmetry_zero_field` (A4)
3. `unit_tests.nrk_extrapolator.qop_invariant` (A1)
4. `unit_tests.nrk_extrapolator.qop_linearity_small_step` (A3)
5. `unit_tests.nrk_extrapolator.zero_step_is_identity` (smoke)

These are deliberately analytic invariants that the passthrough body
satisfies exactly (or to FP32 round-off). They will continue to hold
when the body becomes a learnable extrapolator only if the model
respects them — i.e. they form a regression net for the V3 follow-up.

## Out of scope for this MR (explicit list for follow-up)

- V3 weights binary
- Algorithm wrapper class (`DeviceAlgorithm + Parameters`)
- Sequence wiring in `configuration/python/AllenConf/`
- Jacobian-emitting overload
- Throughput benchmark on reference GPU
- Physics-efficiency reference diff

## Verification log

- 2026-05-10 — `lb-check-copyright master` → exit 0.
- 2026-05-10 — `lb-format -n` on staged files → clean.
- 2026-05-10 — Branch `gscriven/nrk-extrapolator-exercise` created off
  `master@12f26514959d`. Not pushed.
