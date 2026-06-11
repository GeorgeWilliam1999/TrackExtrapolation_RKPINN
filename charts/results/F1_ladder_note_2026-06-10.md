# F1 ladder — first results (2026-06-10)

UT->T pool = 23,107 tracks (full corpus, z0∈[2300,3000], zf∈[7600,9500]).
Median |dx| (µm), p95, median |dtx| (µrad), median-|dx| by |q/p| quartile:

| rung | med | p95 | |dtx| | byQ (low→high |q/p|) |
|---|---|---|---|---|
| 0 straight | 141.3 | 2826 | 41.1 | [23.5, 83.5, 315.7, 1202.3] |
| 1 on-axis kick (F,G tables) | 115.3 | 2184 | 34.8 | [19.8, 72.3, 277.7, 1025.7] |
| **1.5 path-integrated kick** | **5.7** | **159** | **1.8** | **[0.8, 3.0, 11.4, 42.6]** |
| NN pinn_v2_small_v1 (locked) | 293.0 | 1894 | ~31 | [174.6, 219.2, 383.0, 486.6] |
| NN pinn_v2_kick_10M | 152.6 | 2827 | ~44 | [23.5, 94.2, 342.5, 1126.6] |

## Findings
1. **The κ² prediction is confirmed** — rung 1.5's quartile profile [0.8→42.6] is the
   κ²-envelope shape; the hardest (low-p) quartile improves 11–28× vs every NN.
2. **The on-axis profile B(0,0,z) is NOT the dominant chart term** at UT->T: tracks
   fan off-axis (|y| up to ~1.5 m in the magnet) where By differs strongly; the
   on-axis kick removes only ~15% of the slope error. The field ALONG THE PATH is
   what matters.
3. Rung 1.5 = single fixed 120-sample quadrature of the 3-D map along the
   STRAIGHT-LINE path (no ODE, no adaptivity, no iteration). Zero trained parameters.
   It beats the locked NN by 51× on median and is 25× better on the worst quartile.

## Consequence for the plan
- **F3 is promoted to the critical path**: expand By(x,y,z) in transverse multipoles
  b_n(z); the path integral then factorises into ~10 precomputed 1-D tables contracted
  with (x0, y0, tx, ty) — deployment-legal (no 3-D map at inference), capturing rung-1.5
  analytically. NN (F2) then learns only the few-µm remainder — or may be unnecessary.
- ADR 0011 question sharpens: 1-D multipole tables ≈ weights (admissible?); the raw
  rung-1.5 per-track 3-D quadrature is NOT deployment-legal under ADR 0009 as written,
  but is now the reference target (and possibly a fine HLT2 CPU method on its own).
