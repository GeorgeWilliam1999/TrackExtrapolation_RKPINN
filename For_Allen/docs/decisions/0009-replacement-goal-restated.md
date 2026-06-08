# ADR 0009 — Replacement goal restated; hybrid demoted

* **Date:** 2026-05-19
* **Status:** Accepted
* **Supersedes:** [0002](0002-action-N-corrector-removed.md), [0003](0003-multistep-rk4-mandatory.md), [0007](0007-phase1a-winner.md)
* **References:** [`../../REPLACEMENT_PLAN.md`](../../../REPLACEMENT_PLAN.md), [`../../CLEANUP_LIST.md`](../../../CLEANUP_LIST.md)

## Context

ADRs 0002, 0003 and 0007 progressively walked the Allen deployment path
from "hybrid NeuralRK4 with a small learned corrector" towards "pure
classical RK4 with `n_rk_steps = 2`, corrector OFF". Neither end-point
satisfies the canonical project goal stated in `PROJECT_CONTEXT.md`:

> *Replace the C++ adaptive Runge-Kutta track extrapolator with a faster
> neural network that maintains high accuracy.*

A *hybrid* evaluates the analytic Lorentz right-hand side and the magnetic
field map at inference; a *tuned classical RK4* is not a neural network at
all. Both fail the goal.

## Decision

1. The deployment candidate is a **full neural replacement** — at
   inference time the model is the entire function
   `(x, y, tx, ty, q/p, z_start, dz) → (x_f, y_f, tx_f, ty_f, q/p_f)`
   with no field-map lookup and no analytic Lorentz step.
2. The two architecture families that satisfy this criterion are
   `MLP` and `PINN_v2` (both already present in
   `models/architectures.py`).
3. `NeuralRK4` is demoted from "M1 candidate" to "research baseline /
   F2-F3 audit reference"; its checkpoints move to
   `trained_models/_archive_hybrid/`.
4. The Allen-side surface (`NRKExtrapolator.cuh`, MR `!2497`) is **kept**
   as the correct abstraction layer, but in a follow-up MR its inner
   loop is replaced with a generic `forward_pass(weights, state)` call
   that loads PINN_v2 weights instead of classical RKN4.
5. The roadmap to deployment is the six-phase plan in
   `REPLACEMENT_PLAN.md` §6 (R1 loss reform → R2 Jacobian gate
   re-measurement → R3 MLP modernisation → R4 PINN_v2 scaling →
   R5 bin export + Allen wiring → R6 throughput).

## Consequences

* ADRs 0002 / 0003 / 0007 are marked superseded; their experimental
  measurements (fp64-autograd Jacobian methodology, heavy-tail residual
  finding) remain valid and are referenced from `REPLACEMENT_PLAN.md`.
* Allen MR `!2497` lands as-is (pure classical RKN4 behind a default-OFF
  property switch) because it is bytecode-identical to master and
  provides the deployment surface. A follow-up MR will repurpose the
  inner loop.
* The < 0.10 mm position-error gate from `PROJECT_CONTEXT.md` becomes
  the hard exit criterion of Phase R4, against PINN_v2; the < 0.05 mm
  "ideal" target moves to a stretch goal for a follow-up campaign.
* No code is deleted. All hybrid artefacts are reversibly archived
  under `_archive_hybrid/` and can be reactivated by reverting the
  rename commits.

## Reversibility

This ADR re-anchors the project direction; it does not delete code or
data. If a future measurement shows that no point-prediction neural
network can pass the Allen Kalman-Jacobian constraint A4 (see
`REPLACEMENT_PLAN.md` §5), the contingency path is documented in
`REPLACEMENT_PLAN.md` §6 Phase R-X (Jacobian co-supervision, or
straight-line-residual output parametrisation). Reverting to a hybrid
candidate is a last resort and would require a new ADR.
