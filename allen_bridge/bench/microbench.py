#!/usr/bin/env python3
"""microbench.py — Tier-1 throughput micro-bench (runs on a GPU condor slot).

Times three __device__ paths (RK+field, extrapUTT, PINN_V2_UTT) over the SAME N
tracks, with CUDA-event timing, >=200 warm-up iters discarded and >=30 timed
repeats, reporting median + IQR. Kernels are compiled at runtime with NVRTC
(bundled with the conda env; no external nvcc/CUDA toolkit needed) from the
verbatim Allen device headers. Writes a Tier-1 results JSON.

Confounds controlled (see plan §2): CUDA events (#3), warm-up+repeats+median/IQR
(#2), real gen-4 population (#4), field-map texture lookups (#5), fp32 (#6),
Allen block size 256 + single-warp latency (#7), kernel-only vs end-to-end (#8),
real per-track step divergence (#9), recorded toolchain/flags (#10), explicit unit
(#11). Hardware (#1) and external-validity (#12) handled by the runner/JSON.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ALLEN = os.environ.get("ALLEN_DIR", "/data/bfys/gscriven/Allen")


def gpu_info():
    info = {}
    try:
        q = ("name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,"
             "memory.total,power.limit,persistence_mode,compute_mode")
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], text=True).strip()
        fields = [s.strip() for s in out.splitlines()[0].split(",")]
        keys = q.split(",")
        info = dict(zip(keys, fields))
    except Exception as e:
        info["nvidia_smi_error"] = str(e)
    return info


def percentiles(a):
    a = np.asarray(a, dtype=np.float64)
    p = np.percentile(a, [0, 25, 50, 75, 100])
    med = float(p[2])
    iqr = float(p[3] - p[1])
    return {
        "min": float(p[0]), "p25": float(p[1]), "median": med, "p75": float(p[3]),
        "max": float(p[4]), "iqr": iqr,
        "rel_iqr": float(iqr / med) if med else None,
        "rel_range": float((p[4] - p[0]) / med) if med else None,
        "cv": float(np.std(a) / np.mean(a)) if np.mean(a) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=os.path.join(HERE, "artifacts"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "tier1_microbench.json"))
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--block", type=int, default=256)  # Allen extrapolate_states block_dim
    ap.add_argument("--pyenv", default=os.path.join(HERE, "_pyenv"))
    args = ap.parse_args()

    if args.pyenv and os.path.isdir(args.pyenv):
        sys.path.insert(0, args.pyenv)
    os.environ.setdefault("CUPY_CACHE_DIR", os.path.join(HERE, ".cupy_cache"))
    os.makedirs(os.environ["CUPY_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import cupy as cp
    from cupy.cuda import texture, runtime

    # ----- load inputs -----
    inp = np.load(os.path.join(args.art, "bench_inputs.npz"))
    h = {k: np.ascontiguousarray(inp[k], dtype=np.float32) for k in
         ["x", "y", "tx", "ty", "qop", "z0", "dz"]}
    N = h["x"].shape[0]
    fld = np.load(os.path.join(args.art, "field_v8r1_down.npz"))
    Bx, By, Bz = fld["Bx"], fld["By"], fld["Bz"]
    invD, mn, Ngrid = fld["invD"], fld["minXYZ"], fld["N"]
    nx, ny, nz = (int(v) for v in Ngrid)
    struct_bytes = np.fromfile(os.path.join(args.art, "utt_struct.bin"), dtype=np.uint8)
    meta = np.fromfile(os.path.join(args.art, "utt_meta.bin"), dtype=np.float32)
    with open(os.path.join(args.art, "inputs_meta.json")) as f:
        inputs_meta = json.load(f)

    dev = cp.cuda.Device()
    cc = dev.compute_capability  # e.g. "70"

    # ----- compile kernels (NVRTC) from verbatim Allen device headers -----
    with open(os.path.join(HERE, "bench_kernels.cu")) as f:
        src = f.read()
    incs = [os.path.join(HERE, "shims"), HERE,
            os.path.join(ALLEN, "device", "kalman", "ParKalman", "include"),
            os.path.join(ALLEN, "device", "event_model", "common", "include")]
    options = tuple(["--std=c++20", "-DMAGFIELD_USE_TEXTURE"] + [f"-I{d}" for d in incs])
    t0 = time.time()
    module = cp.RawModule(code=src, backend="nvrtc", options=options)
    rk = module.get_function("rk_kernel")
    eutt = module.get_function("extraputt_kernel")
    pinn = module.get_function("pinn_kernel")
    read_kp = module.get_function("read_kp_scalars")
    compile_s = time.time() - t0

    # ----- device buffers -----
    d = {k: cp.asarray(v) for k, v in h.items()}
    ox, oy, otx, oty = (cp.empty(N, np.float32) for _ in range(4))
    d_meta = cp.asarray(meta)
    d_struct = cp.asarray(struct_bytes)  # raw chart struct bytes on device

    # field textures (one per component), unnormalised coords + linear filtering
    ch = texture.ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
    texobjs, cuarrs = [], []
    for comp in (Bx, By, Bz):
        arr = texture.CUDAarray(ch, nx, ny, nz)
        arr.copy_from(cp.asarray(comp))  # shape (nz,ny,nx) == (depth,height,width)
        res = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)
        td = texture.TextureDescriptor(
            (runtime.cudaAddressModeClamp,) * 3, runtime.cudaFilterModeLinear,
            runtime.cudaReadModeElementType, normalizedCoords=0)
        texobjs.append(texture.TextureObject(res, td))
        cuarrs.append(arr)
    tBx, tBy, tBz = texobjs
    f32 = np.float32
    minX, minY, minZ = (f32(v) for v in mn)
    iDx, iDy, iDz = (f32(v) for v in invD)
    STEP_DZ, MAX_STEPS, POL = f32(100.0), np.int32(200), f32(-1.0)

    def args_rk(n, ox_, oy_, otx_, oty_):
        return (d["x"], d["y"], d["tx"], d["ty"], d["qop"], d["z0"], d["dz"], np.int32(n),
                STEP_DZ, MAX_STEPS, tBx, tBy, tBz, minX, minY, minZ, iDx, iDy, iDz,
                ox_, oy_, otx_, oty_)

    def args_eutt(n, ox_, oy_, otx_, oty_):
        return (d["x"], d["y"], d["tx"], d["ty"], d["qop"], np.int32(n),
                d_struct, d_meta, POL, ox_, oy_, otx_, oty_)

    def args_pinn(n, ox_, oy_, otx_, oty_):
        return (d["x"], d["y"], d["tx"], d["ty"], d["qop"], d["dz"], np.int32(n),
                ox_, oy_, otx_, oty_)

    KERNELS = {
        "rk_field": (rk, args_rk),
        "extraputt": (eutt, args_eutt),
        "pinn_v2_utt": (pinn, args_pinn),
    }

    # ----- ABI self-test: device reads chart scalars from the raw upload -----
    probe = cp.empty(6, np.float32)
    read_kp((1,), (32,), (d_struct, probe))
    dev.synchronize()
    p = cp.asnumpy(probe)
    abi_ok = bool(abs(p[0] - 2665.0) < 1 and abs(p[1] - 7826.0) < 1 and abs(p[2] - 60) < 0.5)
    abi = {"ZINI": float(p[0]), "ZFIN": float(p[1]), "Nbinx": float(p[2]),
           "Nbiny": float(p[3]), "PMIN": float(p[4]), "DEGX2": float(p[5]), "ok": abi_ok}
    if not abi_ok:
        print("WARNING: chart struct ABI self-test FAILED:", abi)

    def grid_for(n):
        return ((n + args.block - 1) // args.block,)

    def time_kernel_only(kern, mkargs, n, block, warmup, repeats):
        grid = ((n + block - 1) // block,)
        a = mkargs(n, ox, oy, otx, oty)
        for _ in range(warmup):
            kern(grid, (block,), a)
        dev.synchronize()
        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ts = []
        for _ in range(repeats):
            ev0.record(); kern(grid, (block,), a); ev1.record(); ev1.synchronize()
            ts.append(cp.cuda.get_elapsed_time(ev0, ev1))  # ms
        return ts

    def time_end_to_end(kern, key, n, block, warmup, repeats):
        # alloc once (NOT timed); time H2D(inputs)+kernel+D2H(outputs) per repeat.
        grid = ((n + block - 1) // block,)
        dbuf = {k: cp.empty(n, np.float32) for k in ["x", "y", "tx", "ty", "qop", "z0", "dz"]}
        obuf = [cp.empty(n, np.float32) for _ in range(4)]
        hin = {k: np.ascontiguousarray(h[k][:n]) for k in dbuf}
        hout = np.empty(n, np.float32)

        def mk():
            if key == "rk_field":
                return (dbuf["x"], dbuf["y"], dbuf["tx"], dbuf["ty"], dbuf["qop"], dbuf["z0"],
                        dbuf["dz"], np.int32(n), STEP_DZ, MAX_STEPS, tBx, tBy, tBz, minX, minY,
                        minZ, iDx, iDy, iDz, *obuf)
            if key == "extraputt":
                return (dbuf["x"], dbuf["y"], dbuf["tx"], dbuf["ty"], dbuf["qop"], np.int32(n),
                        d_struct, d_meta, POL, *obuf)
            return (dbuf["x"], dbuf["y"], dbuf["tx"], dbuf["ty"], dbuf["qop"], dbuf["dz"],
                    np.int32(n), *obuf)

        def one():
            for k in dbuf:
                dbuf[k].set(hin[k])
            kern(grid, (block,), mk())
            obuf[0].get(out=hout)
        for _ in range(max(20, warmup // 5)):
            one()
        dev.synchronize()
        ev0, ev1 = cp.cuda.Event(), cp.cuda.Event()
        ts = []
        for _ in range(repeats):
            ev0.record(); one(); ev1.record(); ev1.synchronize()
            ts.append(cp.cuda.get_elapsed_time(ev0, ev1))
        return ts

    def validity_probe(kern, mkargs):
        # run once, check outputs are finite and actually changed vs input
        ox[:] = 0; oy[:] = 0; otx[:] = 0; oty[:] = 0
        kern(grid_for(N), (args.block,), mkargs(N, ox, oy, otx, oty))
        dev.synchronize()
        s = slice(0, min(N, 100000))
        oxn = cp.asnumpy(ox[s])
        finite = float(np.mean(np.isfinite(oxn)))
        moved = float(np.mean(np.abs(oxn - h["x"][s]) > 1e-4))
        return {"finite_frac_x": finite, "moved_frac_x": moved}

    results = {}
    for key, (kern, mkargs) in KERNELS.items():
        vp = validity_probe(kern, mkargs)
        if vp["finite_frac_x"] < 0.999:
            print(f"WARNING: {key} produced non-finite outputs: {vp}")
        ko = time_kernel_only(kern, mkargs, N, args.block, args.warmup, args.repeats)
        e2e = time_end_to_end(kern, key, N, args.block, args.warmup, args.repeats)
        warp = time_kernel_only(kern, mkargs, 32, 32, args.warmup, args.repeats)  # single warp
        ko_p, e2e_p, warp_p = percentiles(ko), percentiles(e2e), percentiles(warp)
        med_ms = ko_p["median"]
        results[key] = {
            "kernel_only_ms": ko_p,
            "end_to_end_ms": e2e_p,
            "single_warp_us": {k: (v * 1000 if isinstance(v, float) else v)
                               for k, v in warp_p.items()},
            "us_per_track": med_ms * 1e3 / N,
            "tracks_per_s": N * 1e3 / med_ms,
            "us_per_track_end_to_end": e2e_p["median"] * 1e3 / N,
            "tracks_per_s_end_to_end": N * 1e3 / e2e_p["median"],
            "validity": vp,
        }
        print(f"{key:14s} kernel-only median={med_ms:.4f} ms  "
              f"{results[key]['us_per_track']*1e3:.4f} ns/track  "
              f"{results[key]['tracks_per_s']:.3e} tracks/s  rel_iqr={ko_p['rel_iqr']*100:.2f}%")

    # ----- speedup ratios (kernel-only median us/track) -----
    upt = {k: results[k]["us_per_track"] for k in results}
    ratios = {
        "RK_div_NN": upt["rk_field"] / upt["pinn_v2_utt"],
        "extrapUTT_div_NN": upt["extraputt"] / upt["pinn_v2_utt"],
        "RK_div_chart": None,  # analytic chart kernel not yet implemented
        "RK_div_extrapUTT": upt["rk_field"] / upt["extraputt"],
    }

    out = {
        "tier": 1,
        "description": "Isolated CUDA micro-bench: RK+field vs extrapUTT vs PINN_V2_UTT over the "
                       "same gen-4 population. NVRTC-compiled verbatim Allen device code.",
        "n_tracks": int(N),
        "block_size": args.block,
        "warmup_iters": args.warmup,
        "timed_repeats": args.repeats,
        "dtype": "fp32 (KalmanFloat=float, KALMAN_DOUBLE_PRECISION=OFF)",
        "timing": "CUDA events (cudaEventElapsedTime), device-synchronised per repeat",
        "unit": "one track, one extrapolation across its given dz (state->state)",
        "rk_step_dz_mm": 100.0,
        "rk_stages_cashkarp": 6,
        "toolchain": {
            "compiler": "NVRTC", "nvrtc_version": list(cp.cuda.nvrtc.getVersion()),
            "cupy_version": cp.__version__, "target_compute_capability": cc,
            "nvrtc_options": list(options), "compile_seconds": round(compile_s, 3),
            "note": "production Allen GPU build uses nvcc -O3 -arch=sm_70/80 with the SAME "
                    "verbatim device code; NVRTC targets sm_%s here." % cc,
        },
        "gpu": gpu_info(),
        "chart_struct_abi_selftest": abi,
        "population": inputs_meta["population"],
        "footprints_bytes": inputs_meta["footprints_bytes"],
        "field": inputs_meta["field"],
        "methods": results,
        "speedup_ratios_kernel_only": ratios,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRK/NN = {ratios['RK_div_NN']:.1f}x   extrapUTT/NN = {ratios['extrapUTT_div_NN']:.2f}x"
          f"   RK/extrapUTT = {ratios['RK_div_extrapUTT']:.1f}x")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
