# ADR 0002 — Action N: the trained corrector is removed

* **Date:** 2026-05-08
* **Status:** **superseded by [ADR 0009](0009-replacement-goal-restated.md) on 2026-05-19** — the entire deployment slot moves from the hybrid `NeuralRK4` family (corrector ON or OFF) to a full neural replacement (`MLP` / `PINN_v2`). The fact recorded below (corrector is anti-helpful) remains valid for the hybrid baseline but is no longer load-bearing for production.
* **Context:** The M1 model `nrk4_tiny_1step_v1` includes a 4-layer trained "corrector" applied after the 1-step RK4 base prediction. Two pieces of independent evidence settle whether to keep it:
  * deep-dive §12 (corrector-ablation): turning the corrector off drops mean \|Δx\| at the VELO-end residual from 113 µm to 35 µm and improves the bwd/fwd ratio. The corrector is *anti-helpful* on stage-1 endpoint accuracy.
  * deep-dive §22 (Jacobian re-test on the corrector-ablated model): the 100 % off-diagonal A4 failure pattern is *unchanged*. The corrector contributes nothing to the Jacobian; the 1-step RK4 itself is the cause of A4's failure.
* **Decision:** **The corrector is removed.** Phase 1a sweeps confirm this on multi-step variants; if confirmed (which §12 + §22 already strongly imply), Phase 2 retrains the corrector-less architecture. The corrector is not re-litigated.
* **Consequences:**
  * Phase 2's architecture has fewer parameters; expect smaller weight blob (≈ 4.4 kB fp32 vs the M1 4.9 kB).
  * The export pipeline still emits a corrector-less `.bin` cleanly — the V3 loader has a `corrector_enabled: bool` field set false.
  * If, later, a *correctly trained* corrector is needed (e.g. an FSAL accelerator), it is a new ADR.
