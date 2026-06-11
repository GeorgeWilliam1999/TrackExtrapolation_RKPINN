# F2 — Residual network on the analytic chart (verdict: marginal; redirect to the chart)

Date: 2026-06-11. Pool: UT→T (z0∈[2300,3000], zf∈[7600,9500], Δz>0), N=23,107.
Eval: `benchmarks/eval_chart_nn.py` → `results/F2_chartnn_2026-06-11.json`.

## Setup
- Chart = F3.1 winner (`charts/chart.py`, O12 even-multipole, anisotropic window
  XN_x=4500 / XN_y=2800, clamp). Deployment-legal: 1-D c_ab(z) tables + fixed quadrature.
- Residual corpus (`charts/make_residual_corpus.py` → `data/residual_2M.npz`):
  Y_res = Y_RK − chart_predict(X), stratified 2M subset of the 10M corpus.
  Residual is TINY in the bulk (median |x| 0.1 µm, p95 33 µm) — the corpus is
  dominated by short steps the chart already nails. Large residuals live only in
  long forward steps (3000<Δz<8000 ≈ UT→T): median 4.6 µm, p95 199 µm, p99 771 µm.
- Run A (full-pool): MLP [96,96] SiLU, log_cosh, 2M, `configs/residual_mlp_2M.yaml`.
- Run B (focused): MLP [128,128,64], 163k long-forward-step subset
  (`data/residual_fwd.npz`), `configs/residual_mlp_fwd.yaml`.

## Results (UT→T median |Δx| µm; byQ = median by |q/p| quartile low→high)

| config                       | median | p95  | p99  | Q1  | Q2  | Q3   | Q4(low-p) | params |
|------------------------------|-------:|-----:|-----:|----:|----:|-----:|----------:|-------:|
| chart alone (0 params)       |   11.9 |  371 | 1489 | 1.6 | 6.0 | 22.0 |      82.9 |      0 |
| chart + full-pool MLP        |   10.4 |  363 | 1480 | 3.2 | 6.1 | 18.9 |      72.3 |  ~11k  |
| chart + focused MLP          |   10.2 |  361 | 1486 | 3.5 | 6.1 | 18.7 |      72.0 |  ~40k  |
| REF rung-1.5 (true 3-D field)|    5.7 |  159 |  480 |  —  |  —  |   —  |        —  |      0 |
| NN pinn_v2_small_v1 (locked) |    293 | 1894 |   —  | 175 | 219 |  383 |       487 |  ~50k  |
| NN pinn_v2_kick_10M          |    153 | 2827 |   —  |  24 |  94 |  343 |      1127 |  ~50k  |

## Findings
1. **The chart is the win.** 0 params, 11.9 µm median, 25× better than the locked
   NN on median and ~6× on the low-momentum quartile (82.9 vs 487 µm).
2. **The residual net is a marginal polish and cannot reach 5.7 µm.** Both runs
   best-stop at epoch 4–5 (immediate validation plateau), shave only ~1.5 µm off the
   median, and leave the tail (p95/p99) statistically unchanged. Both even *worsen*
   the easiest quartile (Q1 1.6→3.5 µm) — injecting ~3 µm noise on tracks the chart
   already nails: the fingerprint of a model fitting variance it cannot resolve.
3. **Distribution focusing did not help** (10.4 vs 10.2 µm) — so it is not an
   undertraining/imbalance problem; the residual is genuinely hard.
4. **Why:** the dominant residual term is the multipole-table truncation,
   ~sqrt(11.9²−5.7²)≈10.4 µm, a *path functional* of the table fit-error that is
   high-frequency in the 7-D endpoint inputs. The integrable (smooth, removable)
   part of the dynamics was already spent on the chart; the remainder is, by
   construction, not compressible into a small smooth network.

## Verdict / redirect
- G-F2: chart ALONE already clears the "beat the locked candidate" bar 25×; the
  residual NN is **off the critical path** for accuracy. Demote to optional sub-µm
  cosmetic.
- Accuracy lever is the **chart's field representation**, not a learned residual:
  - F3.2: Maxwell-consistent transverse expansion B_y(x,y,z) from the midplane
    trace g(x,z)=B_y(x,0,z) via B_y = Σ_n (−1)^n y^{2n}/(2n)! (∂_x²+∂_z²)^n g
    (generalized-gradient / Venturini–Dragt) — respects ∇·B=∇×B=0 exactly, attacks
    the 10.4 µm truncation term, stays deployment-legal (1-D tables).
  - F3.3: one rung-2 (Magnus-2) path iteration to remove the 5.7 µm straight-chord
    floor — re-integrate along the once-bent trajectory.
  - Budget: 11.9² ≈ 10.4²(trunc) + 5.7²(chord); neither lever alone reaches single-µm,
    both are needed for that target.
