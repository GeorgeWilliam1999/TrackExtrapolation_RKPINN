# Throughput bench — Allen RK vs extrapUTT vs PINN_V2 (Tier-1 + Tier-2)

Measures **per-track extrapolation throughput** of the current Allen system relative to
the NN surrogate. Pure relative throughput under extreme scrutiny (not accuracy). Plan:
Notion "PLAN — Throughput Benchmark".

## What is timed
| method | what | footprint |
|---|---|---|
| `rk_field` | Allen `RungeKuttaExtrapolator<float,CashKarp>` + **v8r1 field-map texture** lookups (6×`tex3D`/step), crossing each track's `dz` in 100 mm steps | ~11.5 MB texture |
| `extraputt` | production 19-param chart polynomial `extrapUTT` (UT→T) | ~937 KB chart |
| `pinn_v2_utt` | locked `PINN_V2_UTT` forward pass (6→96→96→4) | ~40 KB |
| analytic chart | *not yet implemented* (`charts/` is Python-only) | — |

All Allen device code is `#include`d **verbatim** from the read-only Allen checkout via
`generate_snippets.sh`; only NVRTC-safe shims (`shims/`) stand in for host-heavy headers.
fp32 throughout (`KalmanFloat=float`).

## Toolchain (no external nvcc needed)
There is no GPU and no CUDA toolkit on the login node; the built Allen is a **CPU** target.
Tier-1 compiles the kernels at runtime with **NVRTC** (bundled in the conda env) driven by
**cupy** (installed into `_pyenv/`), and runs on a GPU condor slot. Tier-2 is a g++ CPU build.

## Reproduce
```bash
P=/data/bfys/gscriven/conda/envs/TE/bin/python
cd allen_bridge/bench

# 0. extract verbatim device snippets from read-only Allen
bash generate_snippets.sh

# 1. host: chart tables + 19-elem META (CPU g++)
bash build_bench_host.sh
./dump_utt_params /data/bfys/gscriven/TE_stack/PARAM/ParamFiles/data artifacts

# 2. stage tracks (real gen-4 population) + field map (CPU)
$P prepare_inputs.py --n 1000000

# 3. Tier-1: GPU micro-bench (condor slot, request_gpus=1 cpus=1 mem=8GB)
condor_submit microbench.sub          # -> results/tier1_microbench.json

# 4. Tier-2: CPU in-situ reality check
bash build_insitu.sh
./insitu_parkalman /data/bfys/gscriven/TE_stack/PARAM/ParamFiles/data \
    artifacts/tracks_f32.bin artifacts/utt_meta.bin 200 30 results/tier2_insitu.json

# 5. assemble the deliverable + 12-point checklist + validity gates
$P assemble_throughput.py             # -> ../../gates/baseline/throughput.json
```

## Confound controls (plan §2) — see `throughput.json` for recorded values
CUDA events (#3); ≥200 warm-up + ≥30 repeats, median+IQR (#2); real gen-4 (z0,dz,p) (#4);
field-map **texture** lookups kept (#5); fp32 (#6); Allen block=256 + single-warp latency
(#7); kernel-only vs end-to-end (#8); per-track variable RK steps → real divergence (#9);
recorded NVRTC/arch flags (#10); explicit unit (#11); Tier-1↔Tier-2 cross-check (#12).

## Files
`bench_kernels.cu` (3 kernels) · `bench_rk.cuh`/`bench_extraputt.cuh` (verbatim wrappers) ·
`shims/` (NVRTC stand-ins) · `microbench.py` (cupy harness) · `insitu_parkalman.cpp`+`.md`
(Tier-2) · `assemble_throughput.py` · `explore_throughput.ipynb`.
