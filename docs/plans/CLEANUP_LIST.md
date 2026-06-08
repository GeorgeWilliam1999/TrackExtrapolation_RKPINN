# Gen-3 Cleanup List — items off the "replace the RK extrapolator" goal

* **Date:** 2026-05-19
* **Companion to:** [`REPLACEMENT_PLAN.md`](REPLACEMENT_PLAN.md)
* **Purpose:** itemise everything in the repo that drifted from the *replacement* goal into a *hybrid* / *augmentation* goal, what to do with each item, and what to keep.

The criterion for "off-goal" is simple: **at inference time, does the model evaluate the analytic Lorentz RHS or the field map?** If yes, it is an augmentation of RK4, not a replacement.

Items are grouped by severity: **DELETE / RETIRE / RE-ANCHOR / KEEP**.

---

## Severity legend

| Tag | Meaning |
|---|---|
| 🔴 **RE-ANCHOR** | Code/doc that drives a production decision toward the hybrid. Must be re-pointed at the replacement candidates before any further deployment work. |
| 🟠 **RETIRE** | Specific trained checkpoints, configs, or scripts whose results are no longer the project's headline. Move out of the candidate list; keep in archive for forensic reference. |
| 🟡 **REWORD** | Documentation that frames the hybrid as the goal. Edit to clarify scope. |
| 🟢 **KEEP** | Infrastructure that is reusable by the replacement candidates. Do not touch other than to redirect inputs. |
| ⚪ **MERGE** | Duplicated or stale parallel artefacts to consolidate. |

---

## 1. Production decision artefacts (highest priority)

### 1.1 🔴 [`For_Allen/PLAN.md`](For_Allen/PLAN.md)

Currently anchored on `nrk4_tiny_1step_v1` as the "M1 candidate" and on hybrid-NRK4 as the deployment surface. Eight phases reference NRK4 by name.

**Action.** Edit each phase to refer to "the §R4 PINN_v2 winner" instead of `nrk4_tiny_1step_v1`. The phase *structure* (1a arch sweep → 1b weight loader → 2 retrain → 3/4 CPU gates → 5/6/7/8 GPU integration & throughput) is reusable — only the candidate name and the family-specific bookkeeping changes.

* The hybrid §6.D-2 / [ADR 0003](For_Allen/docs/decisions/0003-multistep-rk4-mandatory.md) / [ADR 0007](For_Allen/docs/decisions/0007-phase1a-winner.md) discussion of `n_rk_steps` becomes irrelevant: a true replacement has no `n_rk_steps`.
* Phase 1a (no-training architectural ablation) re-purposes to "PINN_v2 width × depth sweep at fixed corpus" and is essentially Fix P1 from `REPLACEMENT_PLAN.md` §4.B.
* Phase 1b (V3 weight loader spec) is mostly format-compatible; the per-architecture metadata field changes from `n_rk_steps, corrector_enabled, …` to `hidden_dims, activation, …`.

### 1.2 🔴 [`For_Allen/docs/decisions/0002-action-N-corrector-removed.md`](For_Allen/docs/decisions/0002-action-N-corrector-removed.md), [`0003-multistep-rk4-mandatory.md`](For_Allen/docs/decisions/0003-multistep-rk4-mandatory.md), [`0007-phase1a-winner.md`](For_Allen/docs/decisions/0007-phase1a-winner.md)

ADR 0002 + 0007 endorse "pure classical RK4 with `n_rk_steps = 2`, corrector OFF" as the deployment architecture. **That is not a neural network at all** — it is a classical RK4 with one tuned hyperparameter, and it does not satisfy the project goal.

**Action.** Mark all three ADRs as **`Status: superseded by REPLACEMENT_PLAN.md (2026-05-19)`**. Do not delete (they document important measurements — in particular the fp64-autograd Jacobian methodology and the heavy-tail residual finding). Write a new ADR `0009-replacement-goal-restated.md` that references this cleanup list and the new plan.

### 1.3 🔴 Allen MR `!2497`  ([`Allen/device/kalman/ParKalman/include/NRKExtrapolator.cuh`](../../../../TE_stack/Allen/device/kalman/ParKalman/include/NRKExtrapolator.cuh))

The CUDA surface currently calls pure classical RKN4 with the neural component absent. The property switch `m_use_nrk = false` makes it bytecode-identical to master.

**Action.** **Keep the surface — it is the right abstraction layer** — but rename and re-purpose it:

* Rename file in a follow-up MR: `NRKExtrapolator.cuh` → `NeuralTrackExtrapolator.cuh` (or keep the name if rename churn is undesirable; the contents matter more than the filename).
* Replace the RKN4 inner loop with a `forward_pass(weights, state) → state` function. The weights blob is loaded once at job start (constant memory) and passed in. The first concrete implementation can call PINN_v2 weights from the §R4 winner.
* Keep the classical RKN4 path as `if (!m_weights_loaded) fallback_to_classical()`. **Document explicitly that the classical path is a safety net, not the goal.**

### 1.4 🔴 `experiments/gen_3/notebooks/allen_mr_validation_report.ipynb` (markdown framing)

Line 20 currently says:

> "This MR introduces a **drop-in replacement surface** — `NRKExtrapolator` — whose long-term goal is to absorb a small learned corrector (the `nrk4_tiny` family) without changing the call sites in `ExtrapolateStates`."

**Action.** Edit the framing: the surface is a drop-in replacement, but the long-term goal is to absorb the **§R4 PINN_v2 (or MLP) winner — a full neural replacement — not a corrector**. The word "corrector" should not appear in the deployment story.

---

## 2. Trained-model checkpoints

### 2.1 🟠 `experiments/gen_3/trained_models/nrk4_tiny_1step_v1/`, `nrk4_small_1step_v1/`

**Status.** These were the closest to the gate on the wrong metric. They are not deployment candidates; they remain useful as a hybrid baseline.

**Action.**

* Move from `trained_models/` to `trained_models/_archive_hybrid/`. Update [`README.md`](README.md) §"Headline numbers" table to mark them as archived hybrids.
* Re-evaluate with Fix L1 (log-cosh / median |Δx|) once for the record, then stop running ablations on them.
* The checkpoints remain on disk and are referenced by any historical analysis; do not delete.

### 2.2 🟢 `experiments/gen_3/trained_models/pinn_v2_small_v1/`

**Status.** **Current best true-replacement candidate.** Keep, but it is not yet at the < 0.1 mm gate.

**Action.** Re-evaluate under Fix L1. Use as the starting checkpoint (warm-start) for the wider PINN_v2 variants in Phase R4.

### 2.3 🟠 `experiments/gen_3/trained_models/mlp_{small,medium}_v1/`

**Status.** Collapsed (5–18 mm). Trained with the 6-dim input that ignored signed `dz` and `z_start`.

**Action.** Mark as `_v1_broken` in [`README.md`](README.md). Retrain as `mlp_*_v2` per §R3 (7-dim input + engineered features). The `_v1_broken` checkpoints stay on disk as the negative-result evidence for the gen-3 inheritance audit.

### 2.4 ⚪ `experiments/gen_2/trained_models_v2_fixes/neural_rk4_*` and `pinn_v2_*`

**Status.** Reference checkpoints, well-documented in [`gen_2/README.md`](../gen_2/README.md). Frozen.

**Action.** No change. They are gen-2 archival. Cite them from `REPLACEMENT_PLAN.md` §3.

---

## 3. Code / architecture

### 3.1 🟠 `experiments/gen_3/models/architectures.py::NeuralRK4`

A 130-line class implementing the hybrid forward pass + correction net.

**Action.** **Keep the class** (it is the only working integrator-of-Lorentz reference inside the codebase, used by the F2/F3 audit). Add a class-level docstring:

```python
"""DEPRECATED for deployment as of 2026-05-19 (REPLACEMENT_PLAN.md §2).

NeuralRK4 is a *hybrid* (classical RK4 + learned RHS residual) and does not
satisfy the project goal of *replacing* the RK extrapolator. It is retained
as a research baseline and as the F2/F3 audit reference. Production
candidates are MLP and PINN_v2.
"""
```

Do **not** add new configs / training runs for NeuralRK4. Future research on it (if any) goes into a separate `experiments/gen_3/research_hybrid/` subtree.

### 3.2 🟠 `experiments/gen_3/configs/nrk4_*.json` (training configs)

**Action.** Move to `experiments/gen_3/configs/_archive_hybrid/`. They reference the deprecated class; they should not be re-run by the standard reproduction script.

### 3.3 🟢 `experiments/gen_3/models/architectures.py::MLP`, `::PINN_v2`

**Action.** Keep, modify per `REPLACEMENT_PLAN.md` §4.A (MLP: 7-dim input + Fix K engineered features) and §4.B (PINN_v2: width scaling + λ_pde decay).

### 3.4 🟢 `experiments/gen_3/utils/rk4_propagator.py`

NumPy fine-step RK4 reference. **Not a model, not a candidate — purely the ground-truth generator for training data and the Jacobian reference.**

**Action.** Keep as-is. Add a top-of-file comment clarifying its role ("reference propagator — *not* a deployment candidate").

### 3.5 🟢 `For_Allen/scripts/phase1a_arch_ablation.py`

**Action.** Generalise to `phase_R2_jacobian_a4.py` per `REPLACEMENT_PLAN.md` §6.R2. The fp64-autograd Jacobian routine is the reusable bit; the NRK4-specific sweep is not.

### 3.6 🟢 `experiments/gen_3/For_Allen/src/for_allen/export/` (V3 weight-blob exporter — currently a stub)

**Status.** The `For_Allen/src/for_allen/export/` package is present but only contains `__init__.py`; no implemented exporter has been written yet. The `For_Allen/PLAN.md` Phase 1b inline field list is the *only* schema documentation; no `pins/loader_v3_spec.md` exists. (Earlier drafts of this list and of `REPLACEMENT_PLAN.md` referenced `For_Allen/scripts/export_bin.py` — that file does **not** exist; the canonical location is `src/for_allen/export/`.)

**Action.** When the exporter is first implemented (Phase R5), generalise it to support per-architecture metadata from the outset:

* Generic magic `NN_EXTRAP` (0x4E4E4558) instead of an `NRK4`-specific magic.
* Header field `arch_tag : char[8]` selecting `MLP` / `PINN_V2` / `NRK4`.
* Variable-length `hidden_dims : uint32[n_layers]` so width sweeps don't require a new schema.
* Existing fp32/fp16 export logic (once written) is reusable verbatim across architectures.

The full V4 schema is defined in [`/data/bfys/gscriven/TrackExtrapolation/docs/reports/gen3_allen_integration_2026-05-19.tex`](../../../docs/reports/gen3_allen_integration_2026-05-19.tex) §3. Land `pins/loader_v3_spec.md` as part of the same MR.

---

## 4. Documentation

### 4.1 🟡 [`experiments/gen_3/README.md`](README.md) §"M1 results"

Currently presents `nrk4_tiny_1step_v1` and `pinn_v2_small_v1` side-by-side without making clear that one is a hybrid and the other is a true replacement.

**Action.** Edit the table to add a "Replacement?" column with ✅ / ❌. Add a paragraph note above the table referencing this cleanup list and `REPLACEMENT_PLAN.md`.

### 4.2 🟡 [`experiments/gen_3/README.md`](README.md) §"Likely thesis-section structure"

The C.* item says:

> "Whether Fix G fixes the PINN PDE residual or whether PINN_v2 is structurally retired in favour of NRK4."

That direction is now **reversed**.

**Action.** Edit to:

> "Whether scaled PINN_v2 (Fix P1) closes the < 0.1 mm gate, and how NRK4 was correctly identified as a hybrid and demoted from the deployment slot."

### 4.3 🟡 `experiments/gen_3/For_Allen/PLAN.md` lines referring to "the corrector" or "the trained corrector"

There are at least 30 such references. They were written when the corrector / NRK4 was the candidate.

**Action.** Bulk-edit: "trained corrector" → "learned model" / "trained network". The semantic change is that the model **is** the propagator, not a correction on top of one.

### 4.4 🟡 `models/architectures.py` module docstring

Currently says:

> "NeuralRK4 ... is the M1 candidate."

**Action.** Edit to: "NeuralRK4 is a research-baseline hybrid. M1 candidates are MLP and PINN_v2 (see REPLACEMENT_PLAN.md)."

### 4.5 ⚪ Duplicate references to "the candidate model"

`PROJECT_CONTEXT.md`, gen_3 `README.md`, `For_Allen/PLAN.md`, multiple notebooks all refer informally to "the candidate" without naming the architecture.

**Action.** After Phase R4 picks a winner, search-and-replace "the candidate"/"the M1 candidate" with the actual checkpoint name (`pinn_v2_medium_v2`, or similar).

---

## 5. Notebooks & analysis

### 5.1 🟠 `experiments/gen_3/notebooks/nrk4_tiny_deep_dive.ipynb`

**Action.** Rename to `_archive/nrk4_tiny_deep_dive_HYBRID.ipynb`. Add a top-cell markdown banner:

> "**Archival.** This notebook analyses the hybrid `NeuralRK4` family. As of 2026-05-19 (REPLACEMENT_PLAN.md) the hybrid is no longer the deployment candidate. The §22 / §12 findings remain technically correct but are not part of the production story."

### 5.2 🟡 `experiments/gen_3/notebooks/allen_mr_validation_report.ipynb`

**Action.** Edit per item 1.4 above.

### 5.3 🟢 `experiments/gen_3/gen2_results_review.ipynb` (and similar) — purely descriptive, no deployment claims

**Action.** No change.

---

## 6. C++ infrastructure (LHCb framework side)

### 6.1 🟢 `ml_models/src/TrackMLPExtrapolator.cpp`

571-line Eigen-based MLP inference. This is **the right surface** for an MLP replacement and was the original gen-1 deployment target.

**Action.** Keep. After Phase R3, deploy the modernised MLP `_v2` weights into this class. The weight-blob loader may need a 7-dim input vs the gen-1 6-dim — small change.

### 6.2 🟢 `tests/options/test_extrapolators.py`

**Action.** Keep. After Phase R5, add a `test_pinn_v2_extrapolator.py` companion that exercises the new replacement candidate against `TrackRungeKuttaExtrapolator` reference output.

---

## 7. Items to **NOT** clean up

The following are correct and should be left alone:

* The 50 M / 10 M training corpora. They are produced by fine-step classical RK4 (`utils/rk4_propagator.py`) and are the ground-truth labels. The fact that training data comes from classical RK4 does *not* make the trained model a hybrid — the model learns the input→output map.
* The field map (`twodip.rtf`) and `InterpolatedFieldTorch`. Used in PINN_v2 *training* (PDE residual term) but **not at inference**. Inference-only field-map usage would make a model a hybrid; training-loss-only usage does not.
* MLflow tracking infrastructure — orthogonal to the architecture choice.
* The HTCondor submission scripts — orthogonal.

---

## 8. Execution checklist

In rough recommended order, lowest-risk first:

- [ ] Add this file and `REPLACEMENT_PLAN.md` to git, commit message: `docs: re-anchor gen-3 onto RK-replacement goal`.
- [ ] Item 4.1 / 4.2 / 4.4 — small README and docstring edits.
- [ ] Item 2.1 / 2.3 — move directories, update tables.
- [ ] Item 1.2 — ADR status edits + new ADR 0009.
- [ ] Item 1.4 / 5.1 / 5.2 — notebook re-framing.
- [ ] Item 1.1 / 4.3 — `For_Allen/PLAN.md` rewrite.
- [ ] Phase R1 (loss metric reform) — code change.
- [ ] Phase R2 (A4 re-measurement on replacement candidates) — code + measurement.
- [ ] Phase R3 / R4 — retraining campaigns.
- [ ] Item 1.3 — Allen MR follow-up to re-purpose `NRKExtrapolator.cuh`.

Items 1–6 in this list are **all reversible**: nothing is deleted, only re-tagged and moved into `_archive_*` subtrees. The plan can be rolled back by reverting the rename commits.
