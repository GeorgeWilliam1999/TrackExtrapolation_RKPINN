# track-extrapolation-pinn

**A neural drop-in replacement for the LHCb Runge–Kutta track extrapolator.**

This is the *deliverable* repository: a curated, reproducible extract of the gen-3
research (`TrackExtrapolation/experiments/gen_3`) containing exactly what is needed to
**train → evaluate → export → deploy** a physics-informed neural network (PINN) that
replaces the adaptive RK extrapolator inside the LHCb Allen GPU Kalman filter.

> **Single source of truth for project status:** [`STATUS.md`](STATUS.md) and the
> Notion page *Track Extrapolation*. This README is the orientation map; `STATUS.md`
> holds the live numbers and the roadmap.

---

## The one-paragraph summary

Charged tracks in LHCb are propagated through the dipole field by an adaptive
Runge–Kutta integrator that is called O(10⁶)×/event and reads a 957k-point field
map. We replace that function — `(x, y, tx, ty, q/p, z_start, dz) → state_out` — with a
**single small neural network that uses no field map at inference**. The locked
candidate, `pinn_v2_ALLEN_v1`, is a 10,372-parameter PINN_v2 (40.5 kB fp32, fits the
64 kB Allen constant-memory budget). It reaches **11.7 µm median ‖Δx‖** on the full
signed-Δz test distribution and **passes the Kalman Jacobian (A4) gate with ~100×
margin**. It has been exported to a byte-locked V3 blob, baked into a generated CUDA
header (`PINN_V2_UTT.cuh`), and wired into the Allen UT→T Kalman step. The remaining
work is the Allen integration gates (R6: parity, throughput, Moore physics) and closing
the accuracy on the hardest single step (UT→T, currently ~293 µm median — see
[`STATUS.md`](STATUS.md)).

---

## Repository map

| Path | What it is |
|---|---|
| [`STATUS.md`](STATUS.md) | **Live status, headline numbers, roadmap. Read this first.** |
| [`candidate/pinn_v2_ALLEN_v1/`](candidate/pinn_v2_ALLEN_v1) | The locked deployment candidate: checkpoint, config, normalisation, `TAG_INFO.json`, and the generated `PINN_V2_UTT.cuh`. |
| [`models/`](models) | Model definitions + training/eval (`architectures.py`, `train.py`, `eval.py`, `detector_sigma.py`). |
| [`utils/`](utils) | Reference physics: `rk4_propagator.py` (ground-truth generator — *not* a deployment model) and `magnetic_field.py`. |
| [`For_Allen/`](For_Allen) | Deployment workspace: V3 blob writer/loader (`src/for_allen/export/`), A4 Jacobian gate (`src/for_allen/eval/jacobian.py`), CUDA header emitter (`scripts/emit_cuda_header.py`), the locked blob (`artifacts/blobs/v3/`), pins, tests, and ADRs (`docs/decisions/`). |
| [`docs/plans/`](docs/plans) | `REPLACEMENT_PLAN.md` (strategy), `EXECUTION_PLAN.md` (live ops checklist), `CLEANUP_LIST.md`, `GENERATION_SPEC.md` (corpus regeneration). |
| [`docs/reports/`](docs/reports) | Written reports (`.tex` + `.pdf`): theory, results, Allen integration, audit. |
| [`docs/figures/`](docs/figures) | Plots used in the reports and the Notion page. |
| [`results/`](results) | Phase exit one-pagers (R1, R2, R4, R7). |

## What is deliberately *not* here (stays local)

The 1.2 GB training corpus, the field map (`twodip.rtf`), MLflow runs, and the 132 MB of
historical/negative-result checkpoints are **gitignored** — they are local-only and
regenerable. The corpus is deterministic: see
[`docs/plans/GENERATION_SPEC.md`](docs/plans/GENERATION_SPEC.md) (RK4 ground truth, seed
`42 + i·7919`). Only the single locked candidate checkpoint is tracked.

## The two-repo split

This project ships as **two** repositories:

1. **`track-extrapolation-pinn`** (this repo) — the reproducible model pipeline. Hosted on
   GitHub at **`GeorgeWilliam1999/TrackExtrapolation_RKPINN`** on **`main`**. The earlier
   gen-1/gen-2 research history is preserved on the `archive/research-history` branch.
2. **The Allen merge request** (GitLab, `gitlab.cern.ch/lhcb/Allen`) — the literal C++/CUDA
   drop-in on branch `gscriven/nrk-extrapolator-exercise`: the generated `PINN_V2_UTT.cuh`,
   the AllenConf wiring for the UT→T Kalman step, and the standalone parity harness
   (`ML_research/standalone/`). It consumes the blob produced here.

## Quick start

```bash
# Python env (PyTorch 2.9, mlflow) — conda env name: TE
conda env create -f For_Allen/environment-lock.yml   # or: pip install -e For_Allen

# Deployment-side tests (blob round-trip, CUDA-header parity, tag guard)
cd For_Allen && pytest tests/

# Re-export the candidate to a V3 blob + CUDA header
python For_Allen/scripts/emit_cuda_header.py --help

# Training (requires the local corpus — see docs/plans/GENERATION_SPEC.md)
python models/train.py --help
```

## Provenance

Curated from `/.../TrackExtrapolation/experiments/gen_3` on 2026-06-08. The research repo
(`TrackExtrapolation`) remains the full lab notebook (gen_1 → gen_2 → gen_3, legacy, all
notebooks and negative results); this repo is the product extracted from it.
