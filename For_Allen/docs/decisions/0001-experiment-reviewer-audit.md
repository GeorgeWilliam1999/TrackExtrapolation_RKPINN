# ADR 0001 — `experiment-reviewer` audit of the integration plan

* **Date:** 2026-05-08
* **Status:** accepted
* **Context:** Before committing the Phase-0–8 plan in `PLAN.md`, the `experiment-reviewer` subagent was asked to critique the proposed phase ordering, acceptance gates, and reproducibility policy.
* **Decision:** Adopt the reviewer's seven-point critique in full. Specifically:
  1. **Insert Phase 1a** (architectural ablation, no training) before any retrain. The §22 Jacobian failure and the §12 corrector-anti-helpfulness are two views of the same architectural defect; we cannot attribute Phase 2's improvements without first measuring what the integrator alone gives us with frozen weights.
  2. **Freeze the V3 loader manifest schema in Phase 1b** before any `.bin` is written.
  3. **Settle action N (corrector removal) now** from §12 evidence — do not re-litigate as a Phase 2 experiment. (See ADR 0002.)
  4. **Event-grouped train/val/test splits** (see ADR 0004) and **frozen M1 test set** (see ADR 0005); bootstrap CIs and per-cell stratified gates on every PASS report.
  5. **Add a 1 M intermediate calibration phase** between the 200 k M1 run and the 10 M production retrain.
  6. Tighten **A4 to a Frobenius-norm criterion** with a stated denominator.
  7. **Six-test per-checkpoint smoke battery** runs at every save; failing checkpoints never get promoted.
* **Consequences:** Phases renumbered: 0, 1a, 1b, 2a, 2b, 3, 4, 5, 6, 7, 8. Acceptance criteria tightened across the board (see [`../../ACCEPTANCE.md`](../../ACCEPTANCE.md)). Reviewer's full output is preserved at the project session-resources path and summarised in `PLAN.md`.
* **Fall-back architecture order** (used if any of the project-level kill criteria fire):
  1. Wider RHS MLP within the same NeuralRK4 family.
  2. RKN-residual: production RKN + small additive learned residual.
  3. Neural-ODE with adaptive solver.
  4. Learned correction on RK45.
