# Project Status — Neural RK Extrapolator

**Updated:** 2026-06-08 · **Owner:** G. Scriven · **Phase:** R5 complete → R6 (Allen integration) in progress

This file is the **single source of truth** for where the project actually is. It is
mirrored on the Notion page *Track Extrapolation*. If a generation README, a `.tex`
report, or a notebook disagrees with this file, this file (and `docs/plans/EXECUTION_PLAN.md`)
win.

---

## 1. Headline

The locked deployment candidate is **`pinn_v2_ALLEN_v1`** — a PINN_v2 with hidden dims
`[96, 96]`, **10,372 parameters**, exported to a **40.5 kB fp32** blob (fits the 64 kB
Allen constant-memory budget at both fp32 and fp16).

| Gate | Target | Candidate | Verdict |
|---|---|---|---|
| Accuracy — median ‖Δx‖ (full signed-Δz test set) | < 100 µm | **11.7 µm** | ✅ 8× under |
| Jacobian agreement (A4) — Frobenius rel-err | < 0.05 | **9.0e-4** | ✅ ~100× margin |
| Weight blob size | ≤ 64 kB | **40.5 kB** (fp32) | ✅ |
| Drop-in API (`ITrackExtrapolator`) | no API change | ✅ surface exists | ✅ |

**The science is done and successful: a true neural *replacement* (no field map at
inference) meets every CPU-side gate.** What remains is Allen integration verification
(R6) and the one honest accuracy caveat below.

## 2. The honest caveat — UT→T single step

The 11.7 µm headline is the median over the **full** training distribution
(signed Δz ∈ [−10, +10] m), which is dominated by short, easy steps. On the **specific
UT→T Kalman step** (Δz ≈ 5213 mm — the hardest single extrapolation and the actual first
integration target), the latest eval (R7, 2026-05-22, n=50) gives:

| Metric | `pinn_v2_ALLEN_v1` (UT→T) |
|---|---|
| median ‖Δx‖ | **293 µm** |
| p95 ‖Δx‖ | 1894 µm |
| median ‖Δtx‖ | 31 µrad |

This is **above** the aspirational per-step UT gate (< 50 µm). The open question for R6/R7
is whether 293 µm median on UT→T degrades Moore physics relative to the **polynomial
`extrapUTT` baseline it replaces** (the right comparison is NN-vs-polynomial on this step,
not NN-vs-target). If it does, the contingency routes (Jacobian co-supervision; straight-line
output head — `docs/plans/EXECUTION_PLAN.md` §8) are pre-planned.

## 3. Architecture (locked)

```
inputs (x, y, tx, ty, qop) -- z-score normalised, z_frac = 1.0 appended internally
   └─ encoder: Linear(6, 96) → tanh → Linear(96, 96) → tanh
   └─ head:    Linear(96, 4)            (no activation)
outputs (Δtx, Δty, x_corr/Δz, y_corr/Δz)  wrapped by a physics envelope; qop passes through
```

- `qop` convention = Allen `c·q/p` (asserted at blob load).
- No field map, no analytic Lorentz RHS, no RK loop at inference → a **true replacement**,
  not a hybrid. (The `NeuralRK4` hybrid was demoted 2026-05-19, ADR 0009.)
- Blob: `For_Allen/artifacts/blobs/v3/pinn_v2_ALLEN_v1.bin`, 41,604 B,
  CRC32 `0x1a139335`, SHA256 `c665767092…87e52c`.
- Generated CUDA header: `candidate/pinn_v2_ALLEN_v1/PINN_V2_UTT.cuh` (emitted 2026-05-21).

## 4. Pipeline progress (R-phases)

| Phase | What | Status |
|---|---|---|
| R1 | Loss-metric reform (log-cosh + median selection) | ✅ done 2026-05-20 |
| R2 | A4 Jacobian re-measurement on replacement candidates | ✅ done — PINN_v2 PASS |
| R3 | MLP modernisation (7-dim + engineered features) | ✅ done — **MLP arm retired** (best 0.97 mm, fails 0.5 mm gate) |
| R4 | PINN_v2 scaling sweep | ✅ done — **scaling did not help** (10k→69k regressed median 2.2×); small model locked |
| R5 | V3 blob export + round-trip parity (Python) | ✅ done — 6/6 tests green; blob locked |
| R6 | Allen MR: CUDA↔Python parity, throughput (A6), Moore physics | 🟡 **in progress** — see §5 |

**Negative results that are settled (do not re-litigate):** the MLP family cannot match
PINN_v2 on signed Δz even with engineered features; widening PINN_v2 beyond 10k params on
the 2 M corpus regresses accuracy (corpus saturation + late-epoch instability).

## 5. Allen integration (R6) — current state & what's left

- **Built:** `PINN_V2_UTT.cuh` generated from the locked blob and wired into the UT→T
  Kalman step (`ParKalmanMethods.cuh::ExtrapolateUTT`); AllenConf toggle enabled in
  `make_kalman_long`. Branch: `gscriven/nrk-extrapolator-exercise` on the
  `/data/bfys/gscriven/Allen` clone. (The earlier `ExtrapolateStates`/NRK surface on the
  `TE_stack/Allen` clone is **historical / phase-2**.)
- **Hybrid Jacobian rule:** the NN supplies the new state `x'`; the transport `F` is kept
  from the existing analytic polynomial; noise `Q` unchanged. Avoids on-device autodiff.
- **Throughput baseline cached** (master @ bfac9073): A5000 74.2 kHz, 2080Ti 53.4 kHz,
  3090 89.8 kHz (`For_Allen/pins/baseline_throughput.txt`).

**Remaining R6 gates (the actual unfinished work):**

- [ ] CUDA↔Python bit-bound parity for `pinn_v2_ALLEN_v1` (max |Δy| < 1 ULP on 200 tracks).
- [ ] `allen_throughput` A6: per-track GPU cost ≤ classical RKN4 on the same hardware.
      (An earlier MR `!2497` branch showed a −10% A5000 regression to root-cause first.)
- [ ] Moore `HltEfficiencyChecker`: no track-quality line degrades > 0.5 % absolute vs RK4.
- [ ] Resolve the UT→T 293 µm question (§2) against the polynomial baseline.

## 6. Roadmap

```
R6.1 CUDA↔Python parity  →  R6.2 throughput regression  →  R6.3 Moore physics
                                                                  ↓
                                        (if UT→T physics regresses) contingency:
                                        Jacobian co-supervision / straight-line head
                                                                  ↓
                              R6.4 finalise Allen MR + Moore companion MR
```

## 7. Key references

- Strategy: [`docs/plans/REPLACEMENT_PLAN.md`](docs/plans/REPLACEMENT_PLAN.md)
- Live ops checklist + decision log: [`docs/plans/EXECUTION_PLAN.md`](docs/plans/EXECUTION_PLAN.md)
- Allen integration design: [`docs/reports/gen3_allen_integration_2026-05-19.tex`](docs/reports/gen3_allen_integration_2026-05-19.tex)
- V3 status (latest report): [`docs/reports/gen3_v3_status_2026-05-22.tex`](docs/reports/gen3_v3_status_2026-05-22.tex)
- ADRs: [`For_Allen/docs/decisions/`](For_Allen/docs/decisions) (0001–0009; ADR 0009 = replacement re-anchor)
- Candidate provenance: [`candidate/pinn_v2_ALLEN_v1/TAG_INFO.json`](candidate/pinn_v2_ALLEN_v1/TAG_INFO.json)
