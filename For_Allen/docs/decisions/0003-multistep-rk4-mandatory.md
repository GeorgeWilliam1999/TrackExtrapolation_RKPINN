# ADR 0003 — Multi-step RK4 (Fix L) is mandatory, not optional

* **Date:** 2026-05-08
* **Status:** **superseded by [ADR 0009](0009-replacement-goal-restated.md) on 2026-05-19** (previously superseded-in-part by [ADR 0007](0007-phase1a-winner.md) on 2026-05-12). The `n_rk_steps` hyperparameter belongs to the deprecated hybrid path; the replacement candidates have no such knob.
* **Superseding addendum (2026-05-12):** The *direction* of this ADR — that multi-step RK4 is required — is correct and stands. The *quantitative claim* that `n_rk_steps ≥ 8` is needed, and the supporting argument that A4 fails *structurally* at `n = 1`, are **withdrawn**. The Phase 1a sweep run on 2026-05-12 against the frozen M1 checkpoint with the corrector disabled at the source level shows that:
  * the deep-dive §22 A4 failure was an fp32 + finite-difference numerical artefact, not a structural property of the integrator; on a fp64-cast model with autograd, A4 Frobenius rel-err is at the `1e-4`–`1e-8` level for every `n_rk_steps ∈ {1, 2, 4, 8, 16}` against the RK4 reference;
  * the Phase 1a step-count choice is therefore driven by *endpoint accuracy*, not Jacobian agreement, and the winner is **`n_rk_steps = 2`**;
  * `n_rk_steps_prod` is pinned to 2 in [`pins/n_rk_steps_prod.txt`](../../pins/n_rk_steps_prod.txt). See [ADR 0007](0007-phase1a-winner.md) for the full re-analysis.

  The rest of this ADR is preserved verbatim below as the historical record of the May-8 reasoning.
* **Context:** M1 `nrk4_tiny_1step_v1` uses `n_rk_steps=1` — one Butcher tableau evaluation across the entire (potentially 9 m) propagation. Deep-dive §21 + §22 prove this is incompatible with passing A4: a 1-step RK4 collapses cross-coupling (the off-diagonal Jacobian entries) into a single evaluation, while the FD reference accumulates curvature cross-terms over ~200 sub-steps. The 16 off-diagonal A4 failures all occur where \|J_RK4\| is large and J_θ ≈ 0 — exactly the structural pattern.
* **Decision:** Adopt **multi-step RK4 with `n_rk_steps ≥ 2`** as a hard requirement. Phase 1a sweeps `n_rk_steps ∈ {1, 2, 4, 8, 16, adaptive}` with frozen weights and picks the smallest step count that satisfies:
  * A4 Frobenius rel-err < 0.10 (Phase 1a gate; production gate < 0.05),
  * VELO ⟨\|Δx\|⟩ < 24 µm per (\|q/p\|, \|dz\|) cell (i.e. ≤ 2 × production gate, headroom for retraining),
  * throughput regression ≤ 50 % vs the 1-step baseline (we will tighten this in Phase 6).
* **Consequences:**
  * The chosen step count is pinned in `pins/n_rk_steps_prod.txt` at the end of Phase 1a and embedded in every subsequent `.bin` manifest.
  * If the adaptive variant wins, the loader's manifest carries `adaptive_rk: true` and the per-step tolerance.
  * Throughput budget for Phase 6 must accommodate the chosen step count; if Phase 1a picks 16 steps and Phase 6 fails the throughput gate, fall-back is "wider RHS MLP at fewer steps" (per ADR 0001).
