# Analytic Flattening — Phase-Space Charts & Measures for Track Extrapolation

**Owner:** G. Scriven · **Started:** 2026-06-10 · **Status:** F0–F3 complete (2026-06-11); F4 (Allen) is next
**Notion project:** "Analytic Flattening — Phase-Space Charts" (canonical tracker; this file mirrors it)
**Parent project:** Track Extrapolation (gen-1→3, PINN_v2 / `pinn_v2_ALLEN_v1`)

---

## TL;DR (state as of 2026-06-11)

The **zero-parameter analytic chart** (`charts/chart.py`) is the result: **12.1 µm median /
267 µm p95 / 884 µm p99** on the UT→T pool — **25× better than the locked NN** (293 µm) and
within ~2× of the true-field model floor (5.7 µm). It carries the dipole kick from 1-D
transverse-profile tables `c_ab(z)` (28 even terms, 63.5 kB) + a fixed chord quadrature; it
never touches the field map at inference. Three independent accuracy levers were tested and
**closed**: a learned residual (F2 — not learnable from the endpoint state), a Maxwell field
expansion (F3.2 — FD-Laplacian noise), and rung-2 path iteration (F3.3 — negligible; proved
the 5.7 µm floor is *kick-model* error, not the straight chord). The chart is at the practical
floor of the first-order analytic-kick approach. **Next: F4 — get it past the Allen pipeline.**

## 0. The idea (one paragraph)

The RK extrapolator is expensive because the dipole field makes phase-space
trajectories curved. Instead of learning the curved endpoint map outright
(gen-1→3), we **analytically flatten** as much of the curvature as quadrature
allows, and let a network learn only the *non-integrable residual*. For the
idealised on-axis field B = B_y(z)·ŷ the flattening is **exact and universal**:
the canonical momentum P_x = p_x + q·F(z) (with F(z)=∫B_y(0,0,z')dz') is
conserved, and the field-weighted measure τ(z) ∝ F(z) makes the leading
dynamics constant-coefficient. The real field's transverse dependence
(forced by Maxwell: ∇×B=0 ⇒ fringe B_x,B_z off-axis) breaks integrability and
is the irreducible learning target. The residual after the order-1 chart
scales as **O(κ²)** (κ = c·q/p) — surgically aimed at the high-|q/p| UT→T tail
(1127 µm at 10M) that data alone provably does not fix.

## 1. Data inventory — what we have and where it lives

| Asset | Path | Format / contents | Status |
|---|---|---|---|
| Field map | `experiments/field_maps/twodip.rtf` | ASCII 6-col x y z Bx By Bz; 81×81×146 grid; x,y∈[−4000,4000] (100mm), z∈[−500,14000] mm; peak \|B_y\|≈1.03 T @ z≈5007 | ✅ have |
| Field loader | `experiments/gen_3/utils/magnetic_field.py` | `get_field_numpy(polarity=-1)` → callable (x,y,z)→(Bx,By,Bz); C_LIGHT=2.99792458e-4 | ✅ have |
| 10M gen-3 corpus | `experiments/gen_3/data/train_10M_gen3.npz` | X[N,7]=(x,y,tx,ty,qop,z_start,dz) signed dz, Allen qop (≈1/p[GeV]); Y[N,5] = **RK truth** (5mm step, polarity −1) | ✅ have |
| UT→T pool | mask on corpus: z0∈[2300,3000], z0+dz∈[7600,9500], dz>0 | ≈23k tracks ("Split B") | ✅ derivable |
| NN baselines | `experiments/gen_3/trained_models/{pinn_v2_small_v1, pinn_v2_kick_*}` + `results/R7_utt_eval_2026-06-10.json` | UT→T medians: 293 / 148 / 153 µm; high-\|q/p\| quartile 487 / 1228 / 1127 µm | ✅ have |
| RK45 Jacobian ref | `experiments/gen_3/For_Allen/artifacts/phase1a/{J_rk4_reference,X_a4}.npy` | 200 tracks fp64 — for A4 gates of chart-based models | ✅ have |
| extrapUTT polynomial | Allen `device/kalman/ParKalman/include/ParKalmanMethods.cuh:287` + KalmanParametrizations param files | the production analytic map we ultimately compare against | ⚠️ needs the standalone CPU harness (`Allen/ML_research/standalone/`) to score — Phase F4 |
| F/G integral tables | `experiments/flattening/charts/field_integrals.npz` | F(z)=∫B_y(0,0), G(z)=∫F + κ₀ calibration: κ₀=1.0117e-6, I₁=4.444 T·m, R²=0.9992 | ✅ built 06-10 (F0) |
| Even-multipole chart tables | `experiments/flattening/charts/chart_tables.npz` | `c_ab(z)` 28 even terms (O12, X_N=4500/Y_N=2800, σ_w=3000, clamp), 63.5 kB — the deployment chart | ✅ built 06-11 (F3.1) |
| Residual corpus (closed) | `experiments/flattening/data/residual_2M.npz`, `residual_fwd.npz` | Y−chart baseline; used by F2 (verdict: residual not learnable) | ✅ built 06-11 |

No new training data is needed for F0–F2: the analytic rungs are evaluated
directly against the corpus RK truth; chart-based NNs retrain on the existing
10M corpus.

## 2. Phases

### F0 — Bootstrap: field-integral tables + κ calibration ✅ DONE 06-10
- `charts/build_field_integrals.py` → `field_integrals.npz`.
- **Outcome:** κ₀=1.0117e-6 (κ=κ₀·qop), I₁=4.444 T·m, regression R²=0.9992 (Δtx_true
  vs −qop·ΔF on high-p near-axis tracks). Slope ≈ 1.0 — calibration validated.

### F1 — The ladder benchmark (no training) ✅ DONE 06-10
- `benchmarks/ladder_utt.py`, `ladder_decompose.py`.
- **Outcome:** order-1 kick chart (rung-1.5, true field) = **5.7 µm median**, beating all
  NNs **25–51×**; the residual is ∝ κ² as predicted. F3a proved geometry is NOT the win
  (the transverse field is) → the multipole chart (F3) was promoted to the critical path.

### F2 — Residual network on the chart ✅ DONE 06-11 (verdict: CLOSED)
- Realised as `chart_predict(X) + MLP(X)` (cleaner than the baked-IC PINN). Corpus
  `make_residual_corpus.py`; runs `configs/residual_mlp_{2M,fwd}.yaml`; eval `eval_chart_nn.py`.
- **Verdict:** the chart↔RK residual is a **path functional, not learnable from the 7-D
  endpoint state**. Both runs (full-pool [96,96]; focused [128,128,64] on 163k long-fwd
  steps) best-stop ~epoch 5 and move UT→T median only 11.9→10.2 µm with the tail flat;
  both even worsen the easy Q1. The chart ALONE already clears G-F2 (25× the locked NN),
  so the NN is off the critical path. See `results/F2_residual_note_2026-06-11.md`.

### F3 — Even-multipole chart + refinements ✅ DONE 06-10/11
- **F3.1 (the chart, `charts/chart.py`):** even-multipole `c_ab(z)` tables, O12, anisotropic
  window (X_N=4500 / Y_N=2800), σ_w=3000, clamp. **CANONICAL = 12.1 / 267 / 884 µm**
  (median/p95/p99) on UT→T, 0 params. The σ_w=3000 weight (vs 1000) halved the p99 tail
  (1489→884) at +0.2 µm median — the old weight starved the large-|x| paths (`sweep_weight.py`).
- **F3.2 Maxwell / generalized-gradient expansion — CLOSED (dead end):** FD-Laplacian powers
  are 30× noisier than the direct fit (46 mT @ n≤1 vs 1.57 mT; n=2 diverges). `chart_maxwell.py`.
- **F3.3 rung-2 path iteration — CLOSED (negligible):** re-integrating along the bent path
  changes nothing (5.7→5.7) — proving the 5.7 µm floor is **kick-model** error (dropped
  Bₓ,B_z cross-terms + frozen geometry), not the straight chord. `rung2.py`, `diag_truncation.py`.
- **F3.4 higher-order kick model — PAUSED (low priority):** the only remaining median lever;
  evolve tₓ(z) through 𝒢 and add tₓtᵧBₓ+tᵧB_z. Deferred — 12 µm is almost certainly inside
  the deployment tolerance.

### F4 — extrapUTT bake-off + ADR 0011
- Score the production polynomial on the same pool via the standalone harness.
- **ADR 0011:** are 1-D field-integral tables (~kB, frozen constants) admissible
  under the ADR 0009 replacement criterion? (Argument: they are weights by
  another name; the ban's intent is the 3-D map + RK loop.) Decision required
  before any Allen deployment of chart-based models.
- Exit: written ruling + the full comparison table (straight / chart / extrapUTT / NN / chart+NN).

### F5 — Deployment path (conditional on F2 + ADR 0011)
- Extend the V3/V4 blob to carry the F,G tables; regenerate the CUDA header
  (`pinn_v2_utt_state` gains the kick baseline — a few fmaf lines + 2 table reads).
- Re-run A5 parity, throughput, Moore gates (the existing R6 machinery).

### F6 — Research track (gen-4): learned flattening
- Flow-conjugation / Koopman architecture: invertible z-conditioned chart ψ_θ,
  linear latent dynamics; exact composition (any Δz free), near-free symplectic
  Jacobian. Documented as the principled successor; prototype only after F2 verdict.

## 3. Decision gates

- **G-F1 ✅ TRIPPED (hard):** the order-1 chart alone reaches **12 µm** median (≪100 µm) —
  the NN's role collapsed entirely, not just to tail-cleanup. F2 re-scoped to a residual net,
  then closed as unlearnable.
- **G-F2 ✅ PASSED by the chart alone:** the 0-param chart beats the locked candidate 25× on
  the Split-B median and ~6× on the high-|q/p| quartile (82.9 vs 487 µm). A4 Jacobian gate
  still to be run under F4. The residual NN is not required.
- **G-ADR:** if 1-D tables are ruled inadmissible, F5 is blocked; fall back to
  baking the chart into the network inputs at *training* time only (engineered
  features), losing the exactness but keeping the criterion.

## 4. Risks

- Unit/sign conventions of qop vs κ — mitigated by the F0 empirical calibration.
- The corpus Y is RK truth at polarity −1; tables must use the same polarity.
- Backward propagation (dz<0): F,G are antiderivatives, so signs flow through
  automatically — but must be covered in the F1 ladder explicitly.
- extrapUTT scoring needs the C++ harness — isolated to F4, doesn't block F1/F2.
