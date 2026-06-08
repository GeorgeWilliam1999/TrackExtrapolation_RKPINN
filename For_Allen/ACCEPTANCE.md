# Acceptance criteria — quick-reference matrix

This file is the single-source list of every metric × gate × phase used
by the project. The PLAN narrates the *why*; this file is the
*specification* the scripts under `scripts/` and `src/for_allen/eval/`
test against. **If a metric is reported anywhere, it must appear here
with a gate and a CI policy.**

All "PASS" claims require:
* the **frozen test set** `test_v1_frozen.npy`, hash pinned in `pins/data_manifests/test_v1_frozen.sha256`,
* a 1000-sample BCa bootstrap CI (RNG seed = 20260508, logged),
* stratification by 3 × 3 (\|q/p\|, \|dz\|) cells unless explicitly marked aggregate,
* MLflow tags including `phase`, `git_sha`, `data_test_sha`, `seed_*`, `n_rk_steps`, `precision`.

## Per-phase gates

| Phase | Metric | Gate | Stretch | Notes |
|------:|:-------|:-----|:--------|:------|
| 1a | VELO ⟨\|Δx\|⟩ per cell | < 24 µm | < 18 µm | Phase 1a is "before retrain"; gate is 2 × the production gate, headroom for retraining. |
| 1a | UT ⟨\|Δx\|⟩ per cell | < 100 µm | < 75 µm | Same — 2 × production. |
| 1a | A4 ‖J − J_RK45‖_F / ‖J_RK45‖_F | < 0.10 | < 0.05 | Frobenius, on 200 random states. |
| 1a | bwd/fwd ratio | ∈ [0.80, 1.25] | ∈ [0.90, 1.10] | Asymmetry baseline is 1.44; reviewer recommended this window. |
| 1b | Manifest round-trip | bit-exact fp32 / ≤ 1 ULP fp16 | — | All schema fields recoverable. |
| 2a | val-loss reduction at 2 epochs | ≥ 7 × the 200 k val loss | ≥ 10 × | Sanity check for the scale jump. |
| 2a | per-checkpoint smoke battery | all 6 PASS | — | Hard gate: failing checkpoints go to `artifacts/scratch/`. |
| 2a | VELO ⟨\|Δx\|⟩ per cell | < 24 µm | — | Same threshold as 1a; the goal here is *progress*, not yet PASS. |
| 2b | VELO ⟨\|Δx\|⟩ per cell | < 12 µm AND < 1.2 × RKN AND 95 % BCa upper bound < 15 µm | < 8 µm AND < 1.0 × RKN | Production gate. |
| 2b | UT ⟨\|Δx\|⟩ per cell | < 50 µm AND < 1.2 × RKN | < 30 µm | Production gate. |
| 2b | SciFi-T3 \|ρ(Δx, q/p)\| | < 0.10 | < 0.05 | Same correlation gate as M1, tightened. |
| 2b | bwd/fwd ratio | ∈ [0.80, 1.25] | ∈ [0.90, 1.10] | Asymmetry, per cell. |
| 2b | A4 Frobenius rel-err | < 0.05 AND max off-diagonal rel-err < 0.20 | < 0.02 | Production gate. |
| 2b | fp16 max position shift | < 1 µm | < 0.5 µm | vs fp32 reference, per element. |
| 2b | determinism | bit-identical fp32 forward | — | Mandatory. |
| 2b | seed agreement | all 3 seeds inside ± 1 σ of the gate | — | Stability check. |
| 3 | RKN baseline G1–G3 | unchanged from pre-patch reference | — | Loader patch must not regress baseline. |
| 3 | V2 control (`mlp_medium`) | still **fails** G1–G3 | — | Loader rejecting wrong layouts. |
| 3 | Corrupted manifest test | non-zero exit code | — | Loader rejecting bad metadata. |
| 4 | χ²/ndof | mean < 2.0 AND 95 % BCa upper bound < 2.2 | mean < 1.5 | fp32 and fp16. |
| 4 | \|⟨dp/p⟩\| per \|q/p\| cell | < 1 % | < 0.5 % | Stratified. |
| 4 | pull_x.std | < 1.5 with CI; report skew, kurtosis | < 1.2 | Tail check. |
| 4 | \|⟨pull_x⟩\| per cell | < 0.1 | < 0.05 | Sign-bias detection. |
| 5 | Allen kernel unit tests | 100 % PASS, no new failures vs unpatched master | — | Hard. |
| 6 | Throughput vs RKN | ≥ 0.90 × | ≥ 0.95 × (or 1.5 × stretch on faster GPUs) | Mean of 5 runs. |
| 6 | p99 latency vs RKN | ≤ 1.10 × | ≤ 1.05 × | Same input file. |
| 6 | Constant-mem use | ≤ 32 kB | ≤ 16 kB | Half / quarter of the 64 kB budget. |
| 7 | per-line abs efficiency loss | < 0.5 % per line, < 0.2 % averaged | < 0.2 % per line, < 0.1 % averaged | All long-track HLT1 lines. |
| 7 | stat-only error bar | reported alongside | — | No naked numbers. |
| 8 | nsight-compute occupancy | ≥ 0.5 | ≥ 0.7 | Documentation only, not a gate. |

## Bootstrap protocol (mandatory)

* `n_boot = 1000`,
* BCa intervals (95 %),
* RNG: `numpy.random.default_rng(20260508)`,
* Minimum sample size: 5000 tracks for stage-1 metrics, 500 events for HLT efficiency, 200 random states for A4.
* The bootstrap script is `src/for_allen/eval/bootstrap.py` and its git SHA is logged into MLflow per run.

## Stratification grid

\|q/p\| bins (in `1/MeV` after the convention reconciliation in §C1 of the protocol):
`[0, q33, q67, q100]` where the q's are the 33rd, 67th, 100th percentiles of the **training** data \|q/p\| distribution (computed once and pinned in `pins/data_manifests/qop_bins.txt`).

\|dz\| bins (mm):
`[0, 500, 2000, 9000]` covering VELO-internal, VELO→UT, UT→SciFi.

3 × 3 = 9 cells. **Every** stage-1 gate must PASS in **every** cell.

## OOD slice (monitored, not gated)

* \|q/p\| above the 99th training percentile,
* \|dz\| beyond the training maximum.

Reported in every Phase-2b/4/6/7 summary; a > 5 × gate failure is a
warning that triggers a deployment-decision review but does not block.

## What a "PASS" report must contain

Every PASS claim in the project is reported as:

```
metric_name = central_value [ci_low, ci_high]   gate=…   PASS/FAIL
  per-cell: { (qop_lo, dz_lo): …, …, (qop_hi, dz_hi): … }
  per-seed: { 0: …, 1: …, 2: … }      # Phase 2b only
  ood:      central_value [ci_low, ci_high]      # monitored, not gated
```

A summary missing the CI, the per-cell breakdown, or the OOD line is
*not* a valid Phase-2b/4/6/7 acceptance report.
