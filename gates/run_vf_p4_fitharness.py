#!/usr/bin/env python3
"""VF P4 — fit-level evaluation (PLAN 3915d544-b9d9-81b7 §7 phase 1, tier 3).

Emulates the production vertex-fit structure on toy two-track downstream
("K_S-like") candidates and compares three arms that differ ONLY in how the
field-aware state fetch (TrackStateProvider's job) is performed:

    RK   — CppEngine RK4 + exact discrete-tangent Jacobian  (the reference arm)
    NN   — the neural surrogate + its own forward-mode Jacobian
    LIN  — straight-line transport + straight-line Jacobian  (no field at all)

Fit structure mirrors the documented production maths:
  * ParticleVertexFitter Newton loop (ParticleVertexFitter.cpp:339-371,
    815-848): per track ONE field-aware fetch at the current fetch z, then
    straight-line projection inside <=5 Newton iterations on v=(x,y,z),
    stop at |dchi2| < 0.01. Residual r_i = xy(track at v_z) - xy(v), weighted
    by the transported position covariance at v_z.
  * DecayTreeFitter outer loop (RecoTrack.cpp:119: ztolerance; Fitter.h:69):
    re-fetch the states THROUGH THE FIELD when the vertex z has moved more
    than 10 mm from the current fetch z; <=10 outer iterations.

Toys: true vertex z_v ~ U[300, 2000] mm; two opposite-charge tracks with the
corpus kinematics, truth-propagated (RK) to UT-band reference planes; measured
states = truth + Gaussian smearing with a diagonal downstream-track covariance
C0 = diag(0.3 mm, 0.3 mm, 5e-4, 5e-4, 1% rel qop)^2. Identical toys, seeds and
smearing in all arms — the only difference is the fetch.

Tier-3 bar (spec §6): |vertex bias(NN) - bias(RK)| < 10 % of the RK vertex
resolution, per coordinate. Also reported: robust resolutions (1.4826*MAD),
z-pull widths, chi2 medians, outer-iteration counts, convergence rates, and
the per-toy NN-RK vertex distance.

Outputs: experiments/vertexfit/results/VF_p4_fit_{date}.json + stdout.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LAB = Path(os.environ.get("VF_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/vertexfit"))
sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "core"))
sys.path.insert(0, str(REPO / "datagen"))

from train import _build_model  # noqa: E402
from generate_vertexfit_v1 import CppEngine, _sample_kinematics  # noqa: E402

# the generator module sets the global default dtype to fp64 for its torch
# reference engine; the NN checkpoints are fp32 — reset before building models
torch.set_default_dtype(torch.float32)

N_TOYS = 2000
SEED = 20260703
SIG0 = np.array([0.3, 0.3, 5e-4, 5e-4])          # x,y [mm], tx,ty; qop 1% rel
Z_TOL = 10.0                                      # DTF ztolerance [mm]
MAX_OUTER, MAX_NEWTON = 10, 5
DCHI2 = 0.01


# ----------------------------------------------------------------------------
# fetch arms: state5 + cov5 at z_fetch, given measured (state5, C0) at z_ref
# ----------------------------------------------------------------------------
class RKArm:
    name = "RK"

    def __init__(self, eng: CppEngine):
        self.eng = eng

    def fetch(self, S, z_ref, z_fetch):
        Y, J = self.eng.propagate(S[None, :], np.array([z_ref]), np.array([z_fetch]))
        return Y[0], J[0]


class LinArm:
    name = "LIN"

    def fetch(self, S, z_ref, z_fetch):
        dz = z_fetch - z_ref
        Y = S.copy()
        Y[0] += S[2] * dz
        Y[1] += S[3] * dz
        J = np.eye(5)
        J[0, 2] = dz
        J[1, 3] = dz
        return Y, J


class NNArm:
    name = "NN"

    def __init__(self, exp_dir: Path):
        import torch as t
        ckpt = t.load(exp_dir / "best_model.pt", weights_only=False, map_location="cpu")
        self.model = _build_model(ckpt["config"])
        self.model.load_normalization(str(exp_dir / "normalization.json"))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.exp = exp_dir.name

    def fetch(self, S, z_ref, z_fetch):
        x = torch.tensor(np.concatenate([S, [z_ref, z_fetch - z_ref]]),
                         dtype=torch.float32)[None, :]
        with torch.no_grad():
            Y = self.model(x)[0].numpy().astype(np.float64)
        # forward-mode J of outputs 0..3 wrt state 0..4 (same pattern as tier-2)
        from train import _model_state_jacobian
        Jm = _model_state_jacobian(self.model, x)[0].detach().numpy().astype(np.float64)
        J = np.zeros((5, 5))
        J[:4, :] = Jm
        J[4, 4] = 1.0
        return Y, J


# ----------------------------------------------------------------------------
# the emulated fit
# ----------------------------------------------------------------------------
def fit_vertex(arm, tracks, v_seed):
    """tracks: list of (S_meas[5], C0[5,5], z_ref). Returns dict."""
    v = v_seed.copy()
    n_fetch = 0
    states = []          # (S_f, C_f, z_f) per track
    z_f = [None] * len(tracks)

    def refetch(i):
        nonlocal n_fetch
        S, C0, z_ref = tracks[i]
        Sf, J = arm.fetch(S, z_ref, v[2])
        Cf = J @ C0 @ J.T
        if len(states) <= i:
            states.append((Sf, Cf, v[2]))
        else:
            states[i] = (Sf, Cf, v[2])
        z_f[i] = v[2]
        n_fetch += 1

    for i in range(len(tracks)):
        refetch(i)

    chi2_prev, converged = None, False
    for outer in range(MAX_OUTER):
        # PVF Newton loop on v = (x, y, z), straight-line inside
        for _ in range(MAX_NEWTON):
            H = np.zeros((3, 3))
            g = np.zeros(3)
            chi2 = 0.0
            for (Sf, Cf, zf) in states:
                dz = v[2] - zf
                Hp = np.zeros((2, 5))       # xy at v_z from the fetched state
                Hp[0, 0] = 1.0; Hp[0, 2] = dz
                Hp[1, 1] = 1.0; Hp[1, 3] = dz
                r = np.array([Sf[0] + Sf[2] * dz - v[0],
                              Sf[1] + Sf[3] * dz - v[1]])
                W = np.linalg.inv(Hp @ Cf @ Hp.T)
                A = np.array([[-1.0, 0.0, Sf[2]],
                              [0.0, -1.0, Sf[3]]])   # dr/dv
                H += A.T @ W @ A
                g += A.T @ W @ r
                chi2 += float(r @ W @ r)
            dv = np.linalg.solve(H, -g)
            v = v + dv
            if chi2_prev is not None and abs(chi2_prev - chi2) < DCHI2:
                converged = True
            chi2_prev = chi2
        # DTF outer: refetch through the field if the vertex left the tolerance
        moved = [abs(v[2] - zf_i) > Z_TOL for zf_i in z_f]
        if not any(moved):
            break
        for i, m in enumerate(moved):
            if m:
                refetch(i)
    cov_v = np.linalg.inv(H)
    return {"v": v, "chi2": chi2, "cov": cov_v, "n_fetch": n_fetch,
            "n_outer": outer + 1, "converged": converged}


def robust_sigma(a):
    return float(1.4826 * np.median(np.abs(a - np.median(a))))


def main():
    rng = np.random.default_rng(SEED)
    eng = CppEngine()
    arms = [RKArm(eng), NNArm(LAB / "trained_models" / sys.argv[1] if len(sys.argv) > 1
                              else LAB / "trained_models" / "vf_jac_h96"), LinArm()]
    print(f"arms: RK | NN={arms[1].exp} | LIN;  {N_TOYS} toys, seed {SEED}")

    toys = []
    for _ in range(N_TOYS):
        z_v = rng.uniform(300.0, 2000.0)
        sig = 0.5 + 2.0 * z_v / 2300.0
        v_true = np.array([rng.normal(0, sig), rng.normal(0, sig), z_v])
        tracks = []
        p, qop, tx, ty = _sample_kinematics(rng, 2)
        qop[1] = -abs(qop[1]) if qop[0] > 0 else abs(qop[1])   # opposite charges
        for j in range(2):
            Sv = np.array([v_true[0], v_true[1], tx[j], ty[j], qop[j]])
            z_ref = rng.uniform(2300.0, 2700.0)
            S_true, _ = eng.propagate(Sv[None, :], np.array([z_v]), np.array([z_ref]),
                                      want_J=False)
            sm = np.concatenate([SIG0, [0.01 * abs(qop[j])]])
            S_meas = S_true[0] + rng.normal(0, 1, 5) * sm
            C0 = np.diag(sm ** 2)
            tracks.append((S_meas, C0, z_ref))
        v_seed = v_true + np.array([rng.normal(0, 1), rng.normal(0, 1),
                                    rng.normal(0, 30.0)])
        toys.append((v_true, tracks, v_seed))

    res = {a.name: [] for a in arms}
    for (v_true, tracks, v_seed) in toys:
        for a in arms:
            f = fit_vertex(a, tracks, v_seed)
            res[a.name].append({"dv": (f["v"] - v_true), "chi2": f["chi2"],
                                "sz": float(np.sqrt(f["cov"][2, 2])),
                                "n_outer": f["n_outer"], "n_fetch": f["n_fetch"],
                                "conv": f["converged"]})

    out = {"n_toys": N_TOYS, "seed": SEED, "nn_experiment": arms[1].exp,
           "sig0": SIG0.tolist(), "arms": {}}
    ref = None
    for a in arms:
        dv = np.array([r["dv"] for r in res[a.name]])
        sz = np.array([r["sz"] for r in res[a.name]])
        stats = {
            "bias_mm": [float(np.median(dv[:, k])) for k in range(3)],
            "res_mm": [robust_sigma(dv[:, k]) for k in range(3)],
            "pull_z_width": robust_sigma(dv[:, 2] / sz),
            "chi2_median": float(np.median([r["chi2"] for r in res[a.name]])),
            "outer_mean": float(np.mean([r["n_outer"] for r in res[a.name]])),
            "fetches_mean": float(np.mean([r["n_fetch"] for r in res[a.name]])),
            "conv_rate": float(np.mean([r["conv"] for r in res[a.name]])),
        }
        out["arms"][a.name] = stats
        if a.name == "RK":
            ref = stats
        print(f"\n[{a.name}] bias (x,y,z) = "
              + ", ".join(f"{b*1e3:+.1f}" for b in stats["bias_mm"]) + " um"
              f"\n     resolution   = " + ", ".join(f"{r*1e3:.1f}" for r in stats["res_mm"]) + " um"
              f"\n     z-pull width = {stats['pull_z_width']:.3f}  chi2_med = {stats['chi2_median']:.2f}"
              f"  outer = {stats['outer_mean']:.2f}  fetches = {stats['fetches_mean']:.1f}"
              f"  conv = {stats['conv_rate']*100:.1f}%")

    # per-toy NN-RK deltas + tier-3 bar
    dnn = np.array([r["dv"] for r in res["NN"]])
    drk = np.array([r["dv"] for r in res["RK"]])
    d = dnn - drk
    out["nn_minus_rk"] = {"median_mm": [float(np.median(d[:, k])) for k in range(3)],
                          "spread_mm": [robust_sigma(d[:, k]) for k in range(3)]}
    bars = []
    for k, cname in enumerate("xyz"):
        db = abs(out["arms"]["NN"]["bias_mm"][k] - out["arms"]["RK"]["bias_mm"][k])
        bar = 0.1 * out["arms"]["RK"]["res_mm"][k]
        bars.append({"coord": cname, "delta_bias_mm": db, "bar_mm": bar,
                     "pass": db <= bar})
    out["tier3_bars"] = bars
    print("\ntier-3 (|bias_NN - bias_RK| < 10% of RK resolution):")
    for b in bars:
        print(f"  {b['coord']}: {b['delta_bias_mm']*1e3:.1f} um vs bar "
              f"{b['bar_mm']*1e3:.1f} um -> {'PASS' if b['pass'] else 'FAIL'}")
    print("\nper-toy NN-RK vertex delta (median):",
          ", ".join(f"{m*1e3:+.1f}" for m in out["nn_minus_rk"]["median_mm"]), "um")

    (LAB / "results").mkdir(exist_ok=True)
    p = LAB / "results" / f"VF_p4_fit_{datetime.now():%Y%m%d}.json"
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
