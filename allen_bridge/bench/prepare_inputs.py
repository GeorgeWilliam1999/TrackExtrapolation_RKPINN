#!/usr/bin/env python3
"""prepare_inputs.py — stage Tier-1 micro-bench inputs on the shared filesystem.

Runs on the login node (no GPU). Produces, under <out>/:
  bench_inputs.npz       : N tracks (x,y,tx,ty,qop,z0,dz) sampled from the real
                           gen-4 (z0,dz,p) population + the population stats that
                           drive RK step counts (confound #4).
  field_v8r1_down.npz    : Bx,By,Bz grids (nz,ny,nx) + affine params, extracted
                           from the v8r1.down field map so the GPU worker never
                           needs /cvmfs (and the texture footprint is recorded).
  inputs_meta.json       : provenance + footprints + population summary.

The extrapUTT chart (utt_struct.bin / utt_meta.bin) is produced separately by
dump_utt_params (build_bench_host.sh).
"""
from __future__ import annotations
import argparse, json, hashlib, os
import numpy as np

GEN4 = "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3/data/train_10M_gen4.npz"
FIELD = "/cvmfs/lhcb.cern.ch/lib/lhcb/DBASE/FieldMap/v8r1/cdf/field.v8r1.down.bin"


def sha256(path, nbytes=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "artifacts"))
    ap.add_argument("--n", type=int, default=1_000_000, help="number of tracks")
    ap.add_argument("--seed", type=int, default=20260614)
    ap.add_argument("--gen4", default=GEN4)
    ap.add_argument("--field", default=FIELD)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # --- tracks: sample the real gen-4 (state, z0, dz) joint population ---
    d = np.load(args.gen4)
    X = d["X"]  # (M,7): x,y,tx,ty,qop,z0,dz
    M = X.shape[0]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(M, size=min(args.n, M), replace=False)
    idx.sort()
    S = X[idx].astype(np.float32)
    cols = dict(x=S[:, 0], y=S[:, 1], tx=S[:, 2], ty=S[:, 3], qop=S[:, 4], z0=S[:, 5], dz=S[:, 6])
    np.savez(os.path.join(args.out, "bench_inputs.npz"), **cols, index=idx.astype(np.int64))
    # flat N x 7 float32 (x,y,tx,ty,qop,z0,dz) for the C++ in-situ harness (Tier-2)
    np.ascontiguousarray(S[:, :7], dtype=np.float32).tofile(os.path.join(args.out, "tracks_f32.bin"))

    dz = cols["dz"]
    steps100 = np.ceil(np.abs(dz) / 100.0)
    pop = {
        "n_tracks": int(S.shape[0]),
        "source": args.gen4,
        "source_n": int(M),
        "seed": args.seed,
        "dz_abs_mm": {q: float(np.percentile(np.abs(dz), p)) for q, p in
                      [("p1", 1), ("p25", 25), ("median", 50), ("p75", 75), ("p99", 99)]},
        "p_GeV": {q: float(np.percentile(d["P"][idx], p)) for q, p in
                  [("p1", 1), ("median", 50), ("p99", 99)]},
        "rk_steps_at_100mm": {
            "mean": float(steps100.mean()), "median": float(np.median(steps100)),
            "p99": float(np.percentile(steps100, 99)), "max": float(steps100.max()),
        },
        "rk_field_lookups_per_track_mean": float(6.0 * steps100.mean()),  # CashKarp = 6 stages
        "frac_full_UTT_like_dz_gt_3000": float(np.mean(np.abs(dz) > 3000)),
    }

    # --- field map: parse raw bin (4f invD | 4i N | 4f min | N*4 floats) ---
    raw = np.fromfile(args.field, dtype=np.float32)
    invD = raw[0:3].astype(np.float32)
    N = raw[4:8].view(np.int32)[0:3]
    mn = raw[8:11].astype(np.float32)
    nx, ny, nz = int(N[0]), int(N[1]), int(N[2])
    B = raw[12:12 + nx * ny * nz * 4].reshape(-1, 4)
    Bx = np.ascontiguousarray(B[:, 0].reshape(nz, ny, nx).astype(np.float32))
    By = np.ascontiguousarray(B[:, 1].reshape(nz, ny, nx).astype(np.float32))
    Bz = np.ascontiguousarray(B[:, 2].reshape(nz, ny, nx).astype(np.float32))
    np.savez(os.path.join(args.out, "field_v8r1_down.npz"),
             Bx=Bx, By=By, Bz=Bz, invD=invD, minXYZ=mn, N=np.array([nx, ny, nz], np.int32))

    field_footprint = 3 * nx * ny * nz * 4
    meta = {
        "population": pop,
        "field": {
            "path": args.field, "sha16": sha256(args.field),
            "N": [nx, ny, nz], "voxel_mm": [float(1.0 / v) for v in invD],
            "min_mm": [float(v) for v in mn],
            "texture_footprint_bytes": int(field_footprint),
            "texture_footprint_MB": round(field_footprint / 1e6, 3),
            "note": "raw Gaudi-unit values (the implicit 1e-3*qop scale); loaded verbatim, as production.",
        },
        "footprints_bytes": {
            "field_map_texture": int(field_footprint),
            "extraputt_chart": 960092,  # sizeof(KalmanParametrizations), from dump_utt_params
            "pinn_v2_weights": 10382 * 4,  # W0+b0+W1+b1+W2+b2+mean+std floats
            "analytic_chart_kernel": None,  # not yet implemented (charts/ is Python-only)
        },
    }
    with open(os.path.join(args.out, "inputs_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"tracks: {pop['n_tracks']:,}  mean RK steps@100mm={pop['rk_steps_at_100mm']['mean']:.2f} "
          f"(~{pop['rk_field_lookups_per_track_mean']:.0f} field lookups/track)")
    print(f"field : {nx}x{ny}x{nz}  texture {meta['field']['texture_footprint_MB']} MB")
    print(f"wrote -> {args.out}/{{bench_inputs.npz, field_v8r1_down.npz, inputs_meta.json}}")


if __name__ == "__main__":
    main()
