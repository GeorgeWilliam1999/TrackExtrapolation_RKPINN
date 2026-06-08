# ADR 0008 — Allen pipeline wiring: NRK propagate surface + ExtrapolateStates toggle

* **Date:** 2026-05-12
* **Status:** accepted
* **Context:** Phase 1a closed (ADR 0007) with the model collapsing — under
  Action N + winner `n_rk_steps = 2` — to a deterministic multi-step pure
  Lorentz Runge-Kutta-Nyström body. There are no learned weights to load
  and the V3 loader (PLAN.md Phase 3) is not needed for the *first*
  pipeline test. What remains is to make the `NRKExtrapolator` socket
  callable from an Allen kernel that runs over real states, so we can
  start producing throughput and physics-validation numbers against the
  in-tree RKN/CashKarp baseline.

  MR
  [!2407 (Hoffmann, AdaptiveRKNystromExtrapolator)](https://gitlab.cern.ch/lhcb/Allen/-/merge_requests/2407)
  is used as the surface template: it places an adaptive-step RKN
  extrapolator in the same `Extrapolators::` namespace, exposes a
  `propagate(state, jacobian, z_in, z_out, field)` entry point, and wires
  it in at multiple call sites (`ExtrapolateStates`, `PrKalmanFilter`,
  `KalmanPVIP`, validators).

* **Decision:**
  1. **Extend the `NRKExtrapolator` socket** with two additions:
     * a Magfield-aware single-step overload
       `make_step(state, dz, field)` that evaluates the field at the
       linear midpoint and delegates to the existing explicit-B form;
     * a fixed-size multi-step entry
       `propagate(state, target_z, field, step_size, max_steps)` whose
       signature matches `RungeKuttaNystromExtrapolator::propagate`. Its
       default `step_size = 500 mm` is the Phase 1a winner: for a typical
       Kalman propagation length of ~1 m it produces 2 sub-steps, i.e.
       `n_rk_steps_prod = 2` per [`pins/n_rk_steps_prod.txt`](../../pins/n_rk_steps_prod.txt).
  2. **Add a `use_nrk` boolean property to `extrapolate_states_t`** that
     dispatches between the existing CashKarp Runge-Kutta path and the
     new NRK propagate. Default is `false`, preserving bit-for-bit the
     existing `ExtrapolateStates` output. `extrapolate_states_t` is the
     smallest Allen surface that already runs a propagator over a real
     state container with the real Magfield — it is the right first
     pipeline test, even though it does not sit on the HLT1 critical path.
  3. **Defer the Kalman-filter substitution** (the bigger half of
     MR !2407 — `PrKalmanFilter`/`ParKalmanFilter` swap) to a separate
     follow-up MR after the `use_nrk = true` configuration of
     `ExtrapolateStates` has passed:
     - bit-for-bit identity smoke test under the same step size,
     - throughput regression measurement on the same input MDF,
     - physics-validation via `KalmanChecker` (eta/qop residuals against
       MC truth) on the same sample.
  4. **No new pin file** is created. The Phase 1a winner is encoded in
     `NRKExtrapolatorConstants::default_step_size = 500 mm` /
     `default_max_steps = 100`, with a comment pointing back to
     `pins/n_rk_steps_prod.txt` and ADR 0007. The defaults are *soft*:
     downstream call sites are free to override per-context, the pin only
     anchors the Phase-1a-validated configuration.

* **Consequences:**
  * The MR !2491 socket now has a complete, callable propagate API. The
    extension is additive: every existing entry point is unchanged.
  * `ExtrapolateStates` becomes the first end-to-end harness for NRK on
    real Allen data. Toggling `use_nrk: true` in
    `configuration/python/AllenConf/algorithms.py` (or wherever
    `extrapolate_states_t` is instantiated for the targeted sequence) is
    the single switch that turns on the new path. With `use_nrk: false`
    the binary is byte-for-byte equivalent to the prior commit.
  * Throughput regression is measurable without changing any sequence
    file: the same throughput harness that wraps
    `ExtrapolateStates` runs both configurations.
  * Physics validation against MC truth, in the style of MR !2407's
    `KalmanChecker` extensions (eta, mcp_pid, mcp_fromSignal branches),
    is *not* triggered by this change because `ExtrapolateStates` is not
    part of the Kalman-filter chain. That check moves with the
    PrKalmanFilter substitution and is in scope for the next MR, tracked
    by the follow-up ADR.

* **Acceptance for this commit (pipeline-readiness):**
  | Gate | Threshold | Source |
  |------|-----------|--------|
  | Catch2 `unit_tests.nrk_extrapolator.*` all pass | exit 0 | `test/unit_tests/generic/src/TestNRKExtrapolator.cu` |
  | `clang-format` clean on changed files | CI green | Allen CI |
  | Default `extrapolate_states_t` byte-for-byte identical to prior commit | identical SHA of `dev_states_t` for the same input | Allen `test_throughput` config with `use_nrk: false` |
  | With `use_nrk: true`, `extrapolate_states_t` produces finite output and matches in-tree RKN/CashKarp to within RK4 truncation error on a zero-field input | A2/A4 unit tests + downstream smoke | Allen unit test + small data sample |

* **Artefacts touched:**
  * `Allen/device/kalman/ParKalman/include/NRKExtrapolator.cuh`
  * `Allen/device/kalman/ParKalman/include/NRKExtrapolatorConstants.cuh`
  * `Allen/device/kalman/ParKalman/include/ExtrapolateStates.cuh`
  * `Allen/device/kalman/ParKalman/src/ExtrapolateStates.cu`
  * `Allen/test/unit_tests/generic/src/TestNRKExtrapolator.cu`

* **Follow-up (next ADR, draft):** PrKalmanFilter NRK substitution +
  `KalmanChecker` MC-truth branches, mirroring MR !2407's diff outside
  `RungeKuttaExtrapolator.cuh`.
