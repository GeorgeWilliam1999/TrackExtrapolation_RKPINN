# Gen-3 Replacement Plan — Neural Drop-in for the LHCb RK Extrapolator

* **Date:** 2026-05-19
* **Status:** active — supersedes the working assumption that `NeuralRK4` (RK4 + learned residual) is the deployment candidate.
* **Authority:** [`PROJECT_CONTEXT.md`](../../PROJECT_CONTEXT.md), [`README.md`](../../README.md).

---

## 1. The project goal, restated

> **Replace the C++ adaptive Runge-Kutta track extrapolator with a faster neural network that maintains high accuracy.** ([PROJECT_CONTEXT.md](../../PROJECT_CONTEXT.md))

"Replace" means the neural model is the **entire** function from input state to output state:

```
        ┌──────────────────────────────┐
   (x, y, tx, ty, q/p, z_start, dz) ─► │   neural network (only)   │ ─► (x_f, y_f, tx_f, ty_f, q/p)
        └──────────────────────────────┘
```

It does **not** mean:

* a small neural correction to an RK4 base prediction (residual / "tacked on"),
* a learned RHS evaluated inside a classical RK4 loop (NeuralODE-style hybrid),
* a classical RK4 with one trainable hyperparameter.

If a candidate model evaluates the analytic Lorentz RHS (or any closed-form ODE step) **at inference time**, it is **not** a replacement. It may still be useful as a scaffold or fallback, but it does not satisfy the project goal and must not occupy the production slot.

This document re-anchors gen-3 to the replacement goal and inventories the candidates we have that meet it.

---

## 2. Architecture taxonomy (what counts as a replacement)

Of the five families in [`PROJECT_CONTEXT.md`](../../PROJECT_CONTEXT.md) §"Architecture Approaches", only two families produce a true replacement at inference time:

| Family | Inference path | Is a replacement? |
|---|---|:---:|
| **MLP** | direct map `(x, y, tx, ty, q/p, z_start, dz) → state_out` | ✅ yes |
| **PINN_v2** | encoder of `[x₀, ζ]` predicts a non-affine residual, evaluated at `ζ=1`; **no field map at inference**, the field is consumed only in the training PDE residual | ✅ yes |
| ~~PINN (legacy)~~ | superseded by PINN_v2 ([ISSUES_WITH_PINNS_AND_RK_PINNS.md](../gen_2/ISSUES_WITH_PINNS_AND_RK_PINNS.md) §3) | ❌ structural pathology |
| ~~RK_PINN (legacy)~~ | softmax-mixture of state heads at fixed ζ values | ❌ pathological + not a replacement |
| **NeuralRK4** | classical RK4 of analytic Lorentz RHS + optional learned residual | ❌ **not a replacement** (calls field map and analytic RHS at inference) |

**NeuralRK4 is hereby reclassified as scaffolding / fallback, not as a deployment candidate.** It is retained because (a) its `disable_correction=True` mode gave us a sanity baseline, and (b) its export pipeline and Allen `NRKExtrapolator.cuh` surface are reusable for the actual replacement candidates. See [`CLEANUP_LIST.md`](CLEANUP_LIST.md) for the consequences.

---

## 3. Best replacement candidates across generations

All numbers are mean ‖Δposition‖ on the test split of each generation's dataset. **Bold** = true replacement. ~~Struck~~ = hybrid / non-replacement.

### Gen-1 (fixed `dz = 8000 mm`, no `z_start`, no signed `dz`)

Source: [`experiments/gen_1/archive/analysis_v2/evaluation_report.md`](../gen_1/archive/analysis_v2/evaluation_report.md).

| Model | Params | Pos error (μm) | 95% (μm) | Notes |
|---|---:|---:|---:|---|
| **`mlp_tiny_v1`** | 2 660 | **22.8** | 59.6 | Best gen-1 true replacement, smallest model |
| **`mlp_xlarge_v1`** | 430 980 | **27.4** | 62.2 | Larger but no clear gain |
| ~~`rkpinn_wide_v1`~~ | 533 012 | 9.2 | 21.6 | **Excluded**: gen-1 RK-PINN architecture is the pathological softmax-mixture one (gen-2 audit §5). The 9.2 µm is on fixed `dz` only and the model does not extrapolate to variable / signed `dz`. |

**Lesson from gen-1:** MLPs can hit ~20 µm on a fixed `dz` slice with very few parameters. The challenge has always been generalising to variable `dz` and to the Allen call pattern.

### Gen-2 (variable `dz ∈ [25, 10000] mm`, forward only)

Source: [`experiments/gen_2/README.md`](../gen_2/README.md).

| Model | Params | Pos error (mm) | Notes |
|---|---:|---:|---|
| `mlp_tiny`      | 4 868   | 0.387 | |
| `mlp_medium`    | 102 788 | 0.305 | Best gen-2 MLP |
| `mlp_large`     | 401 412 | 0.380 | Diminishing returns |
| **`pinn_v2_medium_lam0.1`** | 68 612 | **0.235** | **Best gen-2 true replacement** |
| `pinn_v2_medium_lam1.0`     | 68 612 | 0.322 | Higher physics weight → worse |
| ~~`neural_rk4_small_1step`~~ | 17 925 | 0.125 | Hybrid (RK4 + residual). Excluded from production. |

**Lesson from gen-2:** PINN_v2 (a true replacement with a physics-informed *training* loss but no physics at *inference*) closes most of the gap to NeuralRK4 with comparable parameter counts. The 0.235 mm vs 0.125 mm gap is the residual headroom we need to close in gen-3.

### Gen-3 (signed `dz ∈ [-10000, 10000] mm`, 10 M corpus, Allen `qop`)

Source: [`README.md`](README.md) §M1 results.

| Model | Params | Pos error (mm) | Status |
|---|---:|---:|---|
| `mlp_small_v1`     | 18 076 | 18.41   | **COLLAPSED** — see §4.A |
| `mlp_medium_v1`    | 51 076 |  5.57   | **COLLAPSED** — same |
| **`pinn_v2_small_v1`** | 10 372 |  **0.117** | **Best gen-3 true replacement** — just above the 0.1 mm target |
| ~~`nrk4_tiny_1step_v1`~~  |  4 997 |  0.113 | Hybrid, **NEEDS INVESTIGATION** (F3 destabilisation) |
| ~~`nrk4_small_1step_v1`~~ | 18 181 |  0.210 | Hybrid, NEEDS INVESTIGATION |

**Gen-3 truth:** the best true replacement (PINN_v2 small, 10 k params, 117 µm) **already matches the best hybrid** (NRK4 tiny, 5 k params, 113 µm) to within statistical noise. The hybrid's apparent advantage in gen-2 has evaporated under the gen-3 data contract.

This is good news for the project: **we do not need the hybrid.** A true replacement is competitive, and the only remaining task is to close the last ~20% gap to the 0.1 mm target — and then attack the 0.05 mm "ideal" target.

---

## 4. Overcoming the current limitations

Three concrete blockers stand between us and Stage-1 deployment.

### 4.A  Gen-3 MLPs collapsed from 0.3 mm → 5–18 mm

**Diagnosis.** The gen-3 MLP architecture was inherited verbatim from gen-2 with a 6-dim input `[x, y, tx, ty, qop, dz]`, with no awareness of:

1. **Signed `dz`** — the gen-2 MLP only ever saw `dz > 0`. With `dz ∈ [-10k, +10k]`, the symmetric input layout has no way to distinguish forward from backward propagation. The network averages forward + backward residuals and produces neither.
2. **`z_start`** — the gen-2 MLP only ever saw extrapolations starting at fixed depth. With variable `z_start ∈ [0, 14000]`, the same `(x, y, tx, ty, qop, dz)` corresponds to wildly different field histories.

NeuralRK4 quietly absorbed both of these via Fix C2 (signed dz) and Fix H (`z_start` in input), but the MLP recipe was never updated. That is the entire collapse.

**Fix M1 (MLP modernisation), required before anything else:**

* **Input layout change**: 6-dim → **7-dim** `[x, y, tx, ty, qop, z_start, dz]` (matches NeuralRK4 and PINN_v2 gen-3).
* **Engineered features**: append `log10(|dz| / 100)` and `sign(dz)` as in NeuralRK4's correction net (Fix K). This gives the MLP a basis adapted to signed, scale-varying `dz`.
* **Loss function**: switch the model-selection criterion from `mean squared normalised residual` to **`median |Δx|`** or **log-cosh loss** (mitigates F2 heavy-tail). The MSE loss is dominated by < 1 % of low-momentum / large-`|dz|` outliers and is **not** a valid quantitative selection metric (gen-3 §F2).
* **Training**: 10 M corpus, batch 4096, lr 5e-4 with cosine restart, ≥ 60 epochs. The 200 k subsets used in M1 are not enough for this regime.

**Expected outcome:** gen-3 MLP back to gen-2-like 0.3 mm at the minimum, ideally 0.15 mm with the wider 7-dim input and 10 M data.

### 4.B  Gen-3 PINN_v2 is at 0.117 mm but the target is < 0.1 mm

**Diagnosis.** PINN_v2 in gen-3 is parameter-starved at 10 k params and was run on the same 200 k subset as the NRK4 ablations. The PINN_v2 family scales with parameters in gen-2 only weakly (0.235 → 0.322 mm when λ_pde was raised), but it has never been scaled in the *width × dataset* axis.

**Fix P1 (PINN_v2 scaling):**

* **Width sweep**: `hidden_dims ∈ {[256,256], [256,256,128], [512,256,256]}` at λ_pde = 0.1 (gen-2 winner). Largest variant ~ 200 k params.
* **Full-corpus training**: 10 M corpus (not the 200 k subset). PINN_v2's stochastic Monte-Carlo collocation (`n_c = 2`) gives an effective 4× sample multiplier in the physics term, so this is comparable to a 40 M-effective-sample MLP run.
* **Loss**: log-cosh + median-|Δx| selection (same as MLP fix above).
* **Physics curriculum**: extend `physics_warmup_epochs` from 5 to 10 and **add a late-stage λ_pde decay** to 0.01 over the last 20 % of training. Rationale: the physics term is most useful as a regulariser at the start (anchors the trajectory to Lorentz); reducing it at the end lets the data term polish endpoint accuracy.

**Expected outcome:** **0.08–0.10 mm** at 100 k–200 k parameters. This would pass the < 0.1 mm "minimum" target from [PROJECT_CONTEXT.md](../../PROJECT_CONTEXT.md) and put the 0.05 mm "ideal" target in striking distance for a follow-up campaign.

### 4.C  Loss metric is heavy-tail dominated (gen-3 §F2)

**This is a project-wide bug, not a model-specific one.** Quoted from gen-3 `README.md`:

> `val_loss = 1.426e-05`, `test_loss = 1.918e-07`, `train_loss = 2.162e-04` — **same model, same data pipeline**, varies by 1100× because < 1 % of tracks contribute > 90 % of the MSE.

**Fix L1 (loss metric reform), required for all families:**

1. **Selection metric**: switch from squared-normalised-MSE to **median |Δx|** (or detector-σ-normalised median residual) for model selection and reporting. This is robust to the heavy-tail.
2. **Training loss**: replace MSE with **log-cosh** loss. Log-cosh is quadratic near zero (preserves precision on typical tracks) and linear in the tails (caps the gradient contribution of outliers).
3. **Headline number**: report mean, median, 68 %, 95 % quantiles of `|Δx|` and `|Δslope|`. Single-number summaries are misleading.

This is **non-negotiable**: without it we cannot tell whether scaling experiments are improving the model or just shifting the heavy-tail population. Fix L1 should land before any new training runs.

---

## 5. Allen integration constraints

The replacement model must satisfy all of these to deploy. Numbers are inherited from the existing Allen audit ([`For_Allen/docs/decisions/`](For_Allen/docs/decisions/)), restated here in plain language.

| ID | Constraint | Hard limit | Origin |
|---|---|---|---|
| **A1** | Constant memory at inference — no per-event allocation | strict | Allen design rule |
| **A2** | Weights must export to fp32 and fp16 with deterministic round-trip | bit-exact reload | export pipeline gate |
| **A3** | Total trainable weight size ≤ 64 kB | hard | Allen GPU shared-mem budget |
| **A4** | Jacobian ‖J_model − J_RK45‖_F / ‖J_RK45‖_F < 0.05, max off-diag rel-err < 0.20 | hard | Kalman filter usage |
| **A5** | fp32 inference must reproduce Python output to ≤ 1e-4 relative | hard | bit-bound smoke test |
| **A6** | Per-track GPU inference cost must be ≤ classical RK4 cost on the same hardware | hard | "must be faster" |
| **A7** | Drop-in via `ITrackExtrapolator::propagate` signature, no API change to Moore/Allen | hard | non-disruptive deploy |

**How each replacement family satisfies these**:

| Constraint | MLP | PINN_v2 |
|---|---|---|
| A1 constant-memory | ✅ no recurrence, no field map at inference | ✅ same |
| A2 fp32/fp16 export | ⚠️ exporter stub at `For_Allen/src/for_allen/export/` — implementation pending in Phase R5 (see V4 schema in `docs/reports/gen3_allen_integration_2026-05-19.tex` §3) | ⚠️ same |
| A3 ≤ 64 kB | ✅ at width ≤ [256, 256] | ⚠️  Fix P1's `[512,256,256]` variant is ~150 kB — need to test [256,256] precision first |
| A4 Jacobian gate | ❓ **untested for both** — see §6 | ❓ **untested** — see §6 |
| A5 bit-bound | ✅ proven for MLP via `TrackMLPExtrapolator.cpp`; PINN_v2 has same forward arithmetic | ✅ same |
| A6 GPU cost | ✅ inherently — single matrix multiply chain | ✅ same |
| A7 API | ✅ `TrackMLPExtrapolator` is already wired into Gaudi for gen-1 MLPs | ✅ trivial extension |

**A4 is the open question.** A4 ("Jacobian agreement") is the constraint that motivated the move toward NeuralRK4 originally, on the (now-withdrawn — see ADR 0007) belief that a full-NN replacement could not pass it. The gen-3 §F2/F3 audit and the [ADR 0007 re-measurement](For_Allen/docs/decisions/0007-phase1a-winner.md) showed that the original A4 failure was **an fp32 + finite-difference artefact**, not a structural problem with the model class. We have not yet re-measured A4 on the *replacement* candidates (MLP, PINN_v2) with the corrected fp64-autograd methodology. **This must be the first thing we do in §6.A.**

---

## 6. Roadmap to Stage-1 deployment

Phases are sequential. Each one must report a written verdict before the next begins.

### Phase R1 — Loss metric reform  (Fix L1)

* **Code**: change `models/train.py` to log-cosh training loss; change `models/eval.py` to report median + quantiles of `|Δx|`, `|Δslope|`; change all configs to select on `median_dx_mm` not `val_loss`.
* **Re-evaluate**: rerun evaluation on the existing M1 checkpoints (no retraining) with the new metric. We need to know what the current models *actually* are.
* **Exit**: published median/95 %/99 % numbers for `mlp_*_v1`, `pinn_v2_small_v1`, `nrk4_*_v1` on the frozen test set.

### Phase R2 — A4 Jacobian re-measurement on replacement candidates

* **Code**: lift [`For_Allen/scripts/phase1a_arch_ablation.py`](For_Allen/scripts/phase1a_arch_ablation.py)'s fp64-autograd Jacobian routine into a generic `evaluate_a4(model, test_states)` utility.
* **Measure**: A4 on `mlp_medium_v1` (current, broken), `pinn_v2_small_v1` (current), and against the fp64 RK45 reference Jacobian cached in [`For_Allen/artifacts/phase1a/J_rk4_reference.npy`](For_Allen/artifacts/phase1a/J_rk4_reference.npy).
* **Exit**: if either MLP or PINN_v2 passes A4 (Frobenius < 0.05, off-diag < 0.20), we have proof-of-concept that a true replacement is feasible. If neither does, escalate to §6.X.

### Phase R3 — MLP modernisation  (Fix M1)

* **Code**: change `models/architectures.py::MLP` input to 7-dim `[x, y, tx, ty, qop, z_start, dz]` + 2-dim engineered features `[log10(|dz|/100), sign(dz)]`. Add `disable_engineered_features=False` config knob.
* **Train**: `mlp_{small,medium,large}_v2` on the 10 M corpus, 60 epochs, log-cosh loss.
* **Exit**: at least one MLP at < 0.5 mm median |Δx| with A3-compliant parameter budget. (If the best MLP cannot beat 0.5 mm even after this, MLP is structurally retired and the project rests on PINN_v2.)

### Phase R4 — PINN_v2 scaling  (Fix P1)

* **Code**: existing PINN_v2 in `models/architectures.py`. Add cosine-decayed-λ_pde schedule.
* **Train**: 3-way width sweep × 10 M corpus, 60 epochs, log-cosh loss.
* **Exit**: at least one PINN_v2 at < 0.10 mm median |Δx|, passing A4. **This is the candidate that goes into Allen.**

### Phase R5 — Bin export + smoke-test bridge

Existing [`For_Allen/`](For_Allen/) infrastructure. Three changes:

1. **Re-anchor [`For_Allen/PLAN.md`](For_Allen/PLAN.md)** so that the "deployment candidate" is the §R4 PINN_v2 winner, not `nrk4_tiny_1step_v1`.
2. **Re-purpose `NRKExtrapolator.cuh`** in Allen MR `!2497`: keep the property switch `m_use_nrk` and the header surface, but make it call a generic `MLForwardPass(weights, state)` instead of an RK4 loop. The classical RK4 path becomes the *fallback*, not the production path.
3. **Re-run the bit-bound smoke test** (`test_nrk4_v3.cpp`, `dump_smoke_tracks.py`) on the new candidate.

### Phase R6 — Allen MR + Moore integration

Standard from existing [`For_Allen/PLAN.md`](For_Allen/PLAN.md) Phases 5–8. Throughput in `hlt1_pp_default` must improve (A6), accuracy gates from Moore physics tests must hold.

### Phase R-X (contingency) — if A4 still fails

If both MLP and PINN_v2 fail A4 after Fix P1 widening, the diagnosis is that *no point-prediction* model can match RK4's Jacobian. Two escalation routes:

1. **Jacobian co-supervision** — add `λ_J · ‖J_model − J_RK45‖_F²` to the training loss. The Jacobian is cheap to evaluate on the fly via `torch.func.jacfwd`. This is the PLAN.md Phase 2a "Fix J".
2. **Output structure** — re-parametrise the MLP output as `[x_lin + Δx, y_lin + Δy, tx + Δtx, ty + Δty]` where `(x_lin, y_lin) = (x₀ + tx·dz, y₀ + ty·dz)` is the **straight-line baseline**. This is *not* a hybrid (no RK4, no field map at inference) — it is a re-parametrisation that gives the network a physics-respecting initial guess. The straight-line Jacobian is trivial and known; the network only has to learn the deviation.

Both routes preserve the true-replacement property.

---

## 7. Production budget at exit of Phase R6

| Item | Target | Stretch |
|---|---|---|
| Median ‖Δposition‖ | < 0.10 mm | < 0.05 mm |
| 95 % quantile ‖Δposition‖ | < 0.50 mm | < 0.20 mm |
| Jacobian A4 Frobenius rel-err | < 0.05 | < 0.02 |
| Per-track GPU cost | ≤ classical RKN5 | ≤ 0.5 × RKN5 |
| Weight blob | ≤ 64 kB fp32 | ≤ 32 kB fp16 |
| Forward direction support | full ✓ | + Kalman backward propagation |

---

## 8. Open questions

1. **Should we retain a small NeuralRK4 fallback in Allen?** Pro: zero-risk safety net behind a property switch. Con: maintenance burden, confuses the deployment story. Recommendation: **keep the C++ surface, remove the trained Python NRK4 from the production candidate list.**
2. **Is the 10 M corpus sufficient at PINN_v2 [512,256,256]?** Unknown — gen-3 has never trained that wide. If validation curves are still descending at epoch 60, generate 20 M.
3. **Do we need a `z_end ∈ {VELO, UT, SciFi}` family of specialised models, or one universal model?** Universal is cleaner; the 200 µm gen-3 result on universal `dz ∈ [-10k, 10k]` is the most stringent test. Default to universal until evidence forces a split.

---

## 9. References

* Project-level goal: [`PROJECT_CONTEXT.md`](../../PROJECT_CONTEXT.md), [`README.md`](../../README.md)
* Gen-2 final results: [`experiments/gen_2/README.md`](../gen_2/README.md) §"v2 sweep"
* Gen-2 PINN audit: [`experiments/gen_2/ISSUES_WITH_PINNS_AND_RK_PINNS.md`](../gen_2/ISSUES_WITH_PINNS_AND_RK_PINNS.md)
* Gen-3 M1 audit: [`experiments/gen_3/README.md`](README.md) §F1/F2/F3
* Allen deployment infrastructure: [`For_Allen/PLAN.md`](For_Allen/PLAN.md) — **to be re-anchored, see [`CLEANUP_LIST.md`](CLEANUP_LIST.md)**
* Cleanup of off-goal work: [`CLEANUP_LIST.md`](CLEANUP_LIST.md)
