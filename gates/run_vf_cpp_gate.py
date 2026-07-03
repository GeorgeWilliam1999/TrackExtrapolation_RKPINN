#!/usr/bin/env python3
"""VF deployment gate — C++ kernel parity and single-call speed.

Answers the two integration questions for the Rec/HLT2 vertex-fit slot:
  1. PARITY  — does deploy/vf_kernel.cpp reproduce the torch model? Forward on
     200k held-out test rows, Jacobian on 20k, against the checkpoint AND the
     exact J labels (so the C++ J's physics fidelity is shown directly).
  2. SPEED   — single-call latency (state + 5x5 Jacobian) measured inside the
     shared library on real test rows, single thread, vs the production
     TrackMasterExtrapolator reference of 8.35 us/call/core (March 2026
     profiling) and the torch b=1 dispatch floor (~70 us).

Usage: run_vf_cpp_gate.py [experiment]      (default vf_zfeat_h96)
Writes experiments/vertexfit/results/VF_cpp_gate_{date}_{exp}.json
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LAB = Path(os.environ.get("VF_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/vertexfit"))
sys.path.insert(0, str(REPO / "models"))
from train import _build_model, _model_state_jacobian  # noqa: E402

EXP = sys.argv[1] if len(sys.argv) > 1 else "vf_zfeat_h96"
EXP_DIR = LAB / "trained_models" / EXP
N_FWD, N_JAC, N_BENCH = 200_000, 20_000, 200_000
REF_RK_US = 8.35          # TrackMasterExtrapolator, us/call/core (Mar 2026 PDF)

# ---------------------------------------------------------------- build
blob = LAB / "results" / f"vf_kernel_{EXP}.blob"
lib_path = LAB / "results" / "libvfkernel.so"
subprocess.run([sys.executable, str(REPO / "deploy" / "vf_export_weights.py"),
                str(EXP_DIR), str(blob)], check=True)
subprocess.run(["g++", "-O3", "-march=native", "-ffast-math", "-shared",
                "-fPIC", "-o", str(lib_path),
                str(REPO / "deploy" / "vf_kernel.cpp")], check=True)
lib = ctypes.CDLL(str(lib_path))
lib.vf_load.restype = ctypes.c_int
lib.vf_load.argtypes = [ctypes.c_char_p]
lib.vf_propagate_batch.argtypes = [ctypes.c_long] + [
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")] * 3
lib.vf_propagate_batch_mode.argtypes = [ctypes.c_long] + [
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")] * 3 + [ctypes.c_int]
lib.vf_bench.restype = ctypes.c_double
lib.vf_bench.argtypes = [ctypes.c_long,
                         np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                         ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
assert lib.vf_load(str(blob).encode()) == 0, "blob load failed"
print(f"kernel built and loaded (blob {blob.name})")

# ---------------------------------------------------------------- data
with np.load(LAB / "data" / "vf_corpus_10M.npz") as d:
    X, Y, LEG = d["X"].astype(np.float32), d["Y"].astype(np.float32), d["LEG"]
idx = np.load(EXP_DIR / "test_indices.npy")
rng = np.random.default_rng(11)
sel = rng.choice(len(idx), N_FWD, replace=False)
Xs, Ys, Ls = X[idx[sel]], Y[idx[sel]], LEG[idx[sel]]

ckpt = torch.load(EXP_DIR / "best_model.pt", weights_only=False, map_location="cpu")
model = _build_model(ckpt["config"])
model.load_normalization(str(EXP_DIR / "normalization.json"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# ---------------------------------------------------------------- parity: fwd
Xd = Xs.astype(np.float64)
out_cpp = np.empty((N_FWD, 5))
Jbuf = np.empty((N_FWD, 25))
lib.vf_propagate_batch(N_FWD, np.ascontiguousarray(Xd), out_cpp, Jbuf)
with torch.no_grad():
    out_t = model(torch.from_numpy(Xs)).numpy().astype(np.float64)
d_pos = np.abs(out_cpp[:, :2] - out_t[:, :2])
d_slope = np.abs(out_cpp[:, 2:4] - out_t[:, 2:4])
fwd = {
    "n": N_FWD,
    "pos_mm": {"median": float(np.median(d_pos)), "p99": float(np.quantile(d_pos, 0.99)),
               "max": float(d_pos.max())},
    "slope": {"median": float(np.median(d_slope)), "p99": float(np.quantile(d_slope, 0.99)),
              "max": float(d_slope.max())},
}
print(f"forward parity vs torch: |dpos| med {fwd['pos_mm']['median']:.2e} mm "
      f"p99 {fwd['pos_mm']['p99']:.2e} max {fwd['pos_mm']['max']:.2e} | "
      f"|dslope| p99 {fwd['slope']['p99']:.2e}")

# model-vs-truth context (the kernel must not add error visible next to this)
err_cpp = np.abs(out_cpp[:, 0] - Ys[:, 0].astype(np.float64))
err_t = np.abs(out_t[:, 0] - Ys[:, 0].astype(np.float64))
fwd["model_med_dx_um_cpp"] = float(np.median(err_cpp) * 1e3)
fwd["model_med_dx_um_torch"] = float(np.median(err_t) * 1e3)

# ---------------------------------------------------------------- parity: J
Jl = np.load(LAB / "data" / "vf_corpus_10M_J.npy", mmap_mode="r")
Jlab = np.asarray(Jl[idx[sel[:N_JAC]], :4, :], dtype=np.float64)
J_cpp = Jbuf[:N_JAC].reshape(N_JAC, 5, 5)[:, :4, :]
Xj = Xs[:N_JAC]
J_t = np.concatenate([
    _model_state_jacobian(model, torch.from_numpy(Xj[i:i + 8192])).detach().numpy()
    for i in range(0, N_JAC, 8192)]).astype(np.float64)
den_t = np.linalg.norm(J_t.reshape(N_JAC, -1), axis=1)
relf_ct = np.linalg.norm((J_cpp - J_t).reshape(N_JAC, -1), axis=1) / den_t
den_l = np.linalg.norm(Jlab.reshape(N_JAC, -1), axis=1)
relf_cl = np.linalg.norm((J_cpp - Jlab).reshape(N_JAC, -1), axis=1) / den_l
relf_tl = np.linalg.norm((J_t - Jlab).reshape(N_JAC, -1), axis=1) / den_l
jac = {
    "n": N_JAC,
    "cpp_vs_torch_relF": {"median": float(np.median(relf_ct)),
                          "p99": float(np.quantile(relf_ct, 0.99))},
    "cpp_vs_labels_relF_median": float(np.median(relf_cl)),
    "torch_vs_labels_relF_median": float(np.median(relf_tl)),
}
print(f"J parity: cpp-vs-torch relF med {jac['cpp_vs_torch_relF']['median']:.2e} "
      f"p99 {jac['cpp_vs_torch_relF']['p99']:.2e} | vs exact labels: "
      f"cpp {jac['cpp_vs_labels_relF_median']:.2e} torch {jac['torch_vs_labels_relF_median']:.2e}")

# ---------------------------------------------------------------- speed
bench_rows = np.ascontiguousarray(Xd[:N_BENCH])
cs = ctypes.c_double()
speed = {}
for tag, jmode in [("state_only", 0), ("state_plus_exactJ", 1),
                   ("state_plus_headJ", 2)]:
    lib.vf_bench(2000, bench_rows, jmode, ctypes.byref(cs))          # warm-up
    runs = [lib.vf_bench(N_BENCH, bench_rows, jmode, ctypes.byref(cs)) / N_BENCH
            for _ in range(5)]
    speed[tag] = {"ns_per_call_best": float(min(runs)),
                  "ns_per_call_median": float(sorted(runs)[2])}
    print(f"speed {tag}: best {min(runs):.0f} ns/call  "
          f"(median-of-5 {sorted(runs)[2]:.0f} ns)")

# ------------------------------------------- same-machine RK cost reference
# Production-shaped incumbent: Cash-Karp, fixed 100 mm steps, 6 real v8r1
# trilinear lookups per step (mirrors allen_bridge bench rk_kernel). Timing
# it here, on the same rows and the same (loaded) machine, makes the ratio
# meaningful where the absolute 8.35 us production figure is not.
fblob = LAB / "results" / "vf_field_v8r1.blob"
if not fblob.exists():
    sys.path.insert(0, str(REPO / "core"))
    from field_v8r1 import FieldV8R1
    fmap = FieldV8R1()
    with open(fblob, "wb") as fh:
        fh.write(b"VFF1")
        fh.write(np.asarray(fmap.N, np.int32).tobytes())      # Nx, Ny, Nz
        fh.write(np.asarray(fmap.min, np.float32).tobytes())
        fh.write(np.asarray(fmap.invD, np.float32).tobytes())
        for G in (fmap.Bx, fmap.By, fmap.Bz):
            fh.write((G.astype(np.float32) * np.float32(fmap.scale)).tobytes())
    print(f"wrote field blob {fblob.name}")
rk_lib_path = LAB / "results" / "librkref.so"
subprocess.run(["g++", "-O3", "-march=native", "-ffast-math", "-shared",
                "-fPIC", "-o", str(rk_lib_path),
                str(REPO / "deploy" / "vf_rk_reference.cpp")], check=True)
rk = ctypes.CDLL(str(rk_lib_path))
rk.rkref_load.restype = ctypes.c_int
rk.rkref_load.argtypes = [ctypes.c_char_p]
rk.rkref_propagate.argtypes = [
    np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")] * 2
rk.rkref_bench.restype = ctypes.c_double
rk.rkref_bench.argtypes = [ctypes.c_long,
                           np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                           ctypes.POINTER(ctypes.c_double)]
assert rk.rkref_load(str(fblob).encode()) == 0, "field blob load failed"
# physics sanity: fp32 CK@100mm vs exact labels on 2000 rows (production-like
# truncation, mm-scale on the long legs, is expected and fine for a cost ref)
o4 = np.empty(4)
errs = []
for i in range(2000):
    rk.rkref_propagate(np.ascontiguousarray(Xd[i]), o4)
    errs.append(abs(o4[0] - float(Ys[i, 0])))
rk_sanity_mm = float(np.median(errs))
print(f"RK reference physics sanity: median |dx| vs labels {rk_sanity_mm:.3f} mm")
N_RK = 20_000
rk.rkref_bench(500, bench_rows, ctypes.byref(cs))                    # warm-up
rk_runs = [rk.rkref_bench(N_RK, bench_rows, ctypes.byref(cs)) / N_RK
           for _ in range(5)]
speed["rk_reference"] = {"ns_per_call_best": float(min(rk_runs)),
                         "ns_per_call_median": float(sorted(rk_runs)[2]),
                         "median_dx_vs_labels_mm": rk_sanity_mm}
print(f"speed RK reference (state only): best {min(rk_runs):.0f} ns/call "
      f"(median-of-5 {sorted(rk_runs)[2]:.0f} ns)")
ratio_exact = min(rk_runs) / speed["state_plus_exactJ"]["ns_per_call_best"]
ratio_head = min(rk_runs) / speed["state_plus_headJ"]["ns_per_call_best"]
speed["ref_trackmaster_us_production_hw"] = REF_RK_US
speed["nn_exactJ_speedup_vs_rk_same_machine"] = float(ratio_exact)
speed["nn_headJ_speedup_vs_rk_same_machine"] = float(ratio_head)
print(f"vs RK reference (state only), same machine: "
      f"NN+exactJ {ratio_exact:.2f}x | NN+headJ {ratio_head:.2f}x")

# head-only J physics: relF vs exact labels (deployable-J fidelity)
Jh_buf = np.empty((N_JAC, 25))
outh = np.empty((N_JAC, 5))
lib.vf_propagate_batch_mode(N_JAC, np.ascontiguousarray(Xd[:N_JAC]), outh,
                            Jh_buf, 2)
Jh = Jh_buf.reshape(N_JAC, 5, 5)[:, :4, :]
relf_hl = np.linalg.norm((Jh - Jlab).reshape(N_JAC, -1), axis=1) / den_l
jac["headJ_vs_labels_relF_median"] = float(np.median(relf_hl))
print(f"head-only J vs exact labels: relF med {jac['headJ_vs_labels_relF_median']:.2e}")

# ---------------------------------------------------------------- verdict
gate = {
    "fwd_pos_p99_below_1um": fwd["pos_mm"]["p99"] < 1e-3,
    "fwd_slope_p99_below_1e-5": fwd["slope"]["p99"] < 1e-5,
    "jac_cpp_torch_p99_below_1e-3": jac["cpp_vs_torch_relF"]["p99"] < 1e-3,
    "headJ_relF_below_bar_0.01": jac["headJ_vs_labels_relF_median"] < 0.01,
    "deployable_faster_than_rk_same_machine": ratio_head > 1.0,
}
out = {"experiment": EXP, "date": f"{datetime.now():%Y-%m-%d %H:%M}",
       "forward_parity": fwd, "jacobian_parity": jac, "speed": speed,
       "gate": gate, "all_pass": all(gate.values())}
p = LAB / "results" / f"VF_cpp_gate_{datetime.now():%Y%m%d}_{EXP}.json"
json.dump(out, open(p, "w"), indent=1)
print(f"\nGATE {'PASS' if out['all_pass'] else 'FAIL'} -> {p}")
