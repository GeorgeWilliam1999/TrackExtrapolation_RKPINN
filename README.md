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
map. We replace that function — `(x, y, tx, ty, q/p, z_start, dz) → state_out` — with
compact surrogates (neural networks and analytic chart methods) that use **no field
map at inference**.

> **⚠️ 2026-06-11 κ correction.** A bake-off against the production `extrapUTT`
> polynomial exposed a ×1000-weak magnetic coupling (κ = 1e-6 instead of 1e-3) present
> in all corpora since gen-1, plus a sign-flipped polarity in the legacy field loader.
> Every accuracy number predating 2026-06-11 (the 11.7 µm / 293 µm headlines, the
> flattening 5.7–12 µm chart results) describes a quasi-field-free toy. Conventions are
> now locked in [`core/CONVENTIONS.md`](core/CONVENTIONS.md); the externally calibrated
> stack reproduces production to **15 µm median** (see `gates/baseline/`). The gen-4
> physical corpus (FieldMap v8r1, κ = 1e-3, 70% PV-pointing) is being generated; all
> models and charts are being re-baselined against the incumbent's true profile
> (15 µm median / 748 µm low-p tail / 2.2 mm p95).

---

## Repository map

| Path | What it is |
|---|---|
| [`STATUS.md`](STATUS.md) | **Live status, headline numbers, roadmap. Read this first.** |
| [`core/`](core) | Shared physics: `field_v8r1.py` (**canonical** FieldMap v8r1 down loader), `rk4_propagator.py` (ground-truth generator — *not* a deployment model), `magnetic_field.py` (legacy twodip loader), and [`CONVENTIONS.md`](core/CONVENTIONS.md) (kappa, field sign, corpus contract — locked). |
| [`models/`](models) | Model definitions + training/eval (`architectures.py`, `train.py`, `eval.py`, `detector_sigma.py`). |
| [`datagen/`](datagen) | Corpus generation: `generate_data_v2.py` + HTCondor wrappers (`datagen_v2.sub`, `run_datagen_v2.sh`). |
| [`gates/`](gates) | Acceptance gates: `run_r2_jacobian.py` (A4), `run_r7_utt_eval.py` (UT→T), and `gates/baseline/` (extrapUTT production-baseline comparison, former `paper_p0/`). |
| [`charts/`](charts) | Field-flattening / chart-coordinates research line (former `flattening/`): `PLAN.md`, chart builders, benchmarks, results. |
| [`candidate/pinn_v2_ALLEN_v1/`](candidate/pinn_v2_ALLEN_v1) | The locked deployment candidate: checkpoint, config, normalisation, `TAG_INFO.json`, and the generated `PINN_V2_UTT.cuh`. |
| [`For_Allen/`](For_Allen) | Deployment workspace: V3 blob writer/loader (`src/for_allen/export/`), A4 Jacobian gate (`src/for_allen/eval/jacobian.py`), CUDA header emitter (`scripts/emit_cuda_header.py`), the locked blob (`artifacts/blobs/v3/`), pins, tests, and ADRs (`docs/decisions/`). |
| [`allen_bridge/`](allen_bridge) | Standalone CUDA-compat harness + extrapUTT baseline build. |
| [`configs/`](configs), [`condor/`](condor) | Training configs and HTCondor submit files. |
| [`docs/plans/`](docs/plans) | `REPLACEMENT_PLAN.md` (strategy), `EXECUTION_PLAN.md` (live ops checklist), `CLEANUP_LIST.md`, `GENERATION_SPEC.md` (corpus regeneration). |
| [`docs/reports/`](docs/reports) | Written reports (`.tex` + `.pdf`): theory, results, Allen integration, audit. |
| [`docs/figures/`](docs/figures) | Plots used in the reports and the Notion page. |
| [`results/`](results) | Phase exit one-pagers (R1, R2, R4, R7). |

Big artifacts (corpora `.npz`, `trained_models/`, `mlruns/`) stay in the lab; scripts
resolve the lab via the `TE_LAB` env var
(default `/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3`).

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
