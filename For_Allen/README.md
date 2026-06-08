# `For_Allen/`

Workspace for taking the gen-3 Neural-RK4 extrapolator from
"Allen-ABI compliant but failing physics" (M1 outcome) to **passing
the standalone Allen Kalman smoke test, the kernel-level Allen unit
tests with a V3 loader, the `allen_throughput` regression, and Moore
`HltEfficiencyChecker`** within 0.5 % per-line absolute efficiency loss.

## Read these in order

1. [PLAN.md](PLAN.md) — phase plan, decisions, kill criteria.
2. [ACCEPTANCE.md](ACCEPTANCE.md) — quantitative acceptance matrix.
3. [ENVIRONMENT.md](ENVIRONMENT.md) — pins, hosts, MLflow policy.
4. [docs/decisions/](docs/decisions/) — ADRs (one per non-trivial choice).

## Status

> **This 0–8 table is historical (the original NRK4-hybrid plan).** The project
> re-anchored onto a true PINN_v2 replacement on 2026-05-19 (ADR 0009) and now tracks the
> **R-phases**. For live status see the top-level [`../STATUS.md`](../STATUS.md) and
> [`../docs/plans/EXECUTION_PLAN.md`](../docs/plans/EXECUTION_PLAN.md).

**Live (R-phase) status, 2026-06-08:** R1–R5 complete; candidate `pinn_v2_ALLEN_v1` locked
(11.7 µm median, A4 PASS, 40.5 kB fp32); V3 blob frozen; CUDA header generated and wired into
the Allen UT→T Kalman step; **R6 (CUDA parity / throughput / Moore physics) in progress.**

<details><summary>Original NRK4-hybrid phase table (historical)</summary>

| Phase | Name | Maps to |
|------:|:-----|:-------|
| 0  | Bootstrap | superseded |
| 1a | Architectural ablation (no training) | superseded |
| 1b | V3 loader spec freeze | done as R5 |
| 2a/2b | calibration + 10 M retrain | superseded by R3/R4 |
| 3  | V3 export + standalone CPU load test | done as R5 |
| 4–8 | standalone gates → GPU → throughput → Moore → profile | folded into R6 |

</details>

## Project rules (short)

* **No retrain before Phase 1a.** Measure the integrator alone with frozen weights first.
* **No `.bin` export before the V3 loader spec is frozen** (Phase 1b).
* **No GPU work before Phase 4** is green on CPU.
* **Every checkpoint** runs the 6-test smoke battery (`scripts/sanity_check.py`); failures go to `artifacts/scratch/` and are never promoted.
* **Every PASS report** includes a 95 % BCa bootstrap CI, a per-cell stratified breakdown, and an OOD line. Naked numbers are not acceptance.
* **Decisions are ADRs.** Anything not trivially reversible has a numbered file in `docs/decisions/`. The ADR is referenced from the commit message and from any MLflow run that depended on it.

## Cross-references outside this directory

* Live status: [`../STATUS.md`](../STATUS.md)
* Audit: [`../docs/reports/allen_conditions_audit.tex`](../docs/reports/allen_conditions_audit.tex)
* Gen-3 protocol: [`../docs/reports/gen3_protocol.tex`](../docs/reports/gen3_protocol.tex)
* Allen integration design: [`../docs/reports/gen3_allen_integration_2026-05-19.tex`](../docs/reports/gen3_allen_integration_2026-05-19.tex)
* Allen MR branch: `gscriven/nrk-extrapolator-exercise` on `/data/bfys/gscriven/Allen` (commit pinned in `pins/allen_commit.txt`)
