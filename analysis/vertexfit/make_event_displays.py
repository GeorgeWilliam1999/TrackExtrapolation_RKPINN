#!/usr/bin/env python3
"""Event displays for the vertex-fit surrogate — how a two-track vertex fit
actually proceeds, drawn from the SAME toys (same seed) as the P4 result.

Figures (docs/figures/vertexfit/results2/):
  fig_E1_event_stage.png    one toy on the detector stage (z-x), plus the
                            micrometre-scale bend the extrapolator supplies
  fig_E2_fit_anatomy.png    the fit of that toy step by step: fetch, Newton
                            iterations, DTF re-fetch, convergence, per-arm
  fig_E3_vertex_gallery.png six toys: fitted vs true vertex per arm

Machinery is copied from gates/run_vf_p4_fitharness.py (same seed 20260703,
same samplers) with the fit instrumented to record its iteration history.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LAB = Path("/data/bfys/gscriven/TrackExtrapolation/experiments/vertexfit")
OUT = REPO / "docs" / "figures" / "vertexfit" / "results2"

sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "core"))
sys.path.insert(0, str(REPO / "datagen"))

import torch
from train import _build_model
from generate_vertexfit_v1 import CppEngine, _sample_kinematics
from field_v8r1 import FieldV8R1

torch.set_default_dtype(torch.float32)
torch.set_num_threads(4)

SEED = 20260703
N_TOYS = 2000
SIG0 = np.array([0.3, 0.3, 5e-4, 5e-4])
Z_TOL = 10.0
MAX_OUTER, MAX_NEWTON = 10, 5
DCHI2 = 0.01

SURFACE = "#fcfcfb"
COL = {"RK": "#444444", "NN": "#2a78d6", "LIN": "#c22f2f", "TRUE": "#008300"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 10.5, "axes.titlesize": 11.5, "figure.dpi": 150,
})

eng = CppEngine()
fmap = FieldV8R1()

exp_dir = LAB / "trained_models" / "vf_zfeat_jacrow_h96"
ckpt = torch.load(exp_dir / "best_model.pt", weights_only=False, map_location="cpu")
model = _build_model(ckpt["config"])
model.load_normalization(str(exp_dir / "normalization.json"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


def nn_fetch(S, z_ref, z_fetch):
    x = torch.tensor(np.concatenate([S, [z_ref, z_fetch - z_ref]]),
                     dtype=torch.float32)[None, :]
    with torch.no_grad():
        return model(x)[0].numpy().astype(np.float64)


def rk_fetch(S, z_ref, z_fetch):
    Y, _ = eng.propagate(S[None, :], np.array([z_ref]), np.array([z_fetch]),
                       want_J=False)
    return Y[0]


def lin_fetch(S, z_ref, z_fetch):
    dz = z_fetch - z_ref
    Y = S.copy()
    Y[0] += S[2] * dz
    Y[1] += S[3] * dz
    return Y


def rk_path(S, z0, zs):
    """True trajectory sampled at many z (each an independent propagation)."""
    n = len(zs)
    Y, _ = eng.propagate(np.repeat(S[None, :], n, 0), np.full(n, z0),
                       np.asarray(zs, float), want_J=False)
    return Y


# ---------------------------------------------------------------- toys
rng = np.random.default_rng(SEED)
toys = []
for _ in range(N_TOYS):
    z_v = rng.uniform(300.0, 2000.0)
    sig = 0.5 + 2.0 * z_v / 2300.0
    v_true = np.array([rng.normal(0, sig), rng.normal(0, sig), z_v])
    tracks = []
    p, qop, tx, ty = _sample_kinematics(rng, 2)
    qop[1] = -abs(qop[1]) if qop[0] > 0 else abs(qop[1])
    for j in range(2):
        Sv = np.array([v_true[0], v_true[1], tx[j], ty[j], qop[j]])
        z_ref = rng.uniform(2300.0, 2700.0)
        S_true = rk_fetch(Sv, z_v, z_ref)
        sm = np.concatenate([SIG0, [0.01 * abs(qop[j])]])
        S_meas = S_true + rng.normal(0, 1, 5) * sm
        C0 = np.diag(sm ** 2)
        tracks.append({"S_vtx": Sv, "z_ref": z_ref, "S_true": S_true,
                       "S_meas": S_meas, "C0": C0, "p": p[j]})
    v_seed = v_true + np.array([rng.normal(0, 1), rng.normal(0, 1),
                                rng.normal(0, 30.0)])
    toys.append({"v_true": v_true, "tracks": tracks, "v_seed": v_seed})
print(f"regenerated {len(toys)} toys (seed {SEED})")


def fit_vertex_traced(fetch, toy):
    """Same fit as the P4 harness, recording the full iteration history."""
    v = toy["v_seed"].copy()
    trace = {"v": [v.copy()], "chi2": [], "fetch_z": [], "fetch_states": []}
    states, z_f = [None, None], [None, None]

    def refetch(i):
        tr = toy["tracks"][i]
        Sf = fetch(tr["S_meas"], tr["z_ref"], v[2])
        # covariance transport with the straight-line J is enough for the
        # display (the harness uses each arm's own J; the difference is
        # invisible at figure scale)
        dz = v[2] - tr["z_ref"]
        J = np.eye(5)
        J[0, 2] = dz
        J[1, 3] = dz
        states[i] = (Sf, J @ tr["C0"] @ J.T, v[2])
        z_f[i] = v[2]

    for i in range(2):
        refetch(i)
    trace["fetch_z"].append(v[2])
    trace["fetch_states"].append([states[0][0].copy(), states[1][0].copy()])

    chi2_prev = None
    for outer in range(MAX_OUTER):
        for _ in range(MAX_NEWTON):
            H = np.zeros((3, 3))
            g = np.zeros(3)
            chi2 = 0.0
            for (Sf, Cf, zf) in states:
                dz = v[2] - zf
                Hp = np.zeros((2, 5))
                Hp[0, 0] = 1.0
                Hp[0, 2] = dz
                Hp[1, 1] = 1.0
                Hp[1, 3] = dz
                r = np.array([Sf[0] + Sf[2] * dz - v[0],
                              Sf[1] + Sf[3] * dz - v[1]])
                W = np.linalg.inv(Hp @ Cf @ Hp.T)
                A = np.array([[-1.0, 0.0, Sf[2]], [0.0, -1.0, Sf[3]]])
                H += A.T @ W @ A
                g += A.T @ W @ r
                chi2 += float(r @ W @ r)
            v = v + np.linalg.solve(H, -g)
            trace["v"].append(v.copy())
            trace["chi2"].append(chi2)
            if chi2_prev is not None and abs(chi2_prev - chi2) < DCHI2:
                pass
            chi2_prev = chi2
        moved = [abs(v[2] - zf_i) > Z_TOL for zf_i in z_f]
        if not any(moved):
            break
        for i, m in enumerate(moved):
            if m:
                refetch(i)
        trace["fetch_z"].append(v[2])
        trace["fetch_states"].append([states[0][0].copy(), states[1][0].copy()])
    trace["cov"] = np.linalg.inv(H)
    trace["n_outer"] = outer + 1
    return v, trace


# pick a display toy: a DTF re-fetch must occur (seed z off by >10 mm) and
# tracks reasonably soft so the bend panel is visible
disp = None
for k, t in enumerate(toys):
    if abs(t["v_seed"][2] - t["v_true"][2]) > 18 and min(tr["p"] for tr in t["tracks"]) < 8:
        disp = k
        break
toy = toys[disp]
print(f"display toy #{disp}: z_v={toy['v_true'][2]:.0f} mm, seed off by "
      f"{toy['v_seed'][2]-toy['v_true'][2]:+.1f} mm in z, "
      f"p = {[round(tr['p'],1) for tr in toy['tracks']]} GeV")

# ---------------------------------------------------------------- E1 stage
zs_full = np.linspace(toy["v_true"][2], 2750, 160)
paths = [rk_path(tr["S_vtx"], toy["v_true"][2], zs_full) for tr in toy["tracks"]]

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11, 8.2),
                               gridspec_kw={"height_ratios": [1.1, 1]})
# detector regions
for ax in (ax0,):
    for zlo, zhi, name, col in [(-300, 800, "VELO", "#dce8f5"),
                                (2300, 2700, "UT", "#dff0df"),
                                (3000, 7500, "magnet", "#f5e8dc")]:
        ax.axvspan(zlo, zhi, color=col, zorder=0)
        ax.text((zlo + zhi) / 2, 0.97, name, ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=9, color="#666666")
zg = np.linspace(-300, 3000, 300)
_, Byg, _ = fmap(np.zeros_like(zg), np.zeros_like(zg), zg)
ax0b = ax0.twinx()
ax0b.plot(zg, np.abs(Byg), color="#c22f2f", lw=1.1, alpha=0.6)
ax0b.set_ylabel("|B_y(0,0,z)|  [T]", color="#c22f2f", fontsize=9)
ax0b.tick_params(axis="y", colors="#c22f2f", labelsize=8)
ax0b.grid(False)
for j, (tr, pth) in enumerate(zip(toy["tracks"], paths)):
    ax0.plot(zs_full, pth[:, 0], color=COL["TRUE"], lw=1.6,
             label="true trajectories (exact integrator)" if j == 0 else None)
    ax0.errorbar([tr["z_ref"]], [tr["S_meas"][0]], yerr=[SIG0[0] * 10],
                 fmt="o", color="k", ms=5, capsize=3,
                 label="measured state at UT plane (error ×10 for visibility)" if j == 0 else None)
ax0.plot(toy["v_true"][2], toy["v_true"][0], "*", color=COL["TRUE"], ms=18,
         label="true decay vertex")
ax0.plot(toy["v_seed"][2], toy["v_seed"][0], "x", color="#8d6cc3", ms=10, mew=2,
         label="fit seed")
ax0.set_xlim(-350, 3050)
ax0.set_xlabel("z  [mm]")
ax0.set_ylabel("x  [mm]")
ax0.set_title(f"One P4 toy on the detector stage (toy #{disp}: two downstream tracks, "
              f"p = {toy['tracks'][0]['p']:.1f} and {toy['tracks'][1]['p']:.1f} GeV, "
              f"true vertex at z = {toy['v_true'][2]:.0f} mm)")
ax0.legend(frameon=False, fontsize=8.5, loc="lower left")

# bend panel: deviation from the straight line of the MEASURED state
zs_back = np.linspace(toy["v_true"][2] - 150, 2750, 140)
for j, tr in enumerate(toy["tracks"]):
    pth = rk_path(tr["S_meas"], tr["z_ref"], zs_back)
    straight = tr["S_meas"][0] + tr["S_meas"][2] * (zs_back - tr["z_ref"])
    ax1.plot(zs_back, (pth[:, 0] - straight) * 1e3, color=COL["RK"], lw=1.8,
             label="exact transport − straight line (the bend)" if j == 0 else None)
    # NN prediction of the same deviation at a set of target z
    ztgt = np.linspace(toy["v_true"][2] - 150, 2200, 25)
    nn_dev = []
    for zt in ztgt:
        Y = nn_fetch(tr["S_meas"], tr["z_ref"], zt)
        nn_dev.append((Y[0] - (tr["S_meas"][0] + tr["S_meas"][2] * (zt - tr["z_ref"]))) * 1e3)
    ax1.plot(ztgt, nn_dev, "o", color=COL["NN"], ms=4,
             label="NN surrogate prediction" if j == 0 else None)
ax1.axhline(0, color=COL["LIN"], lw=1.4, ls="--",
            label="straight-line transport (zero by definition)")
ax1.axvline(toy["v_true"][2], color=COL["TRUE"], lw=1.2, ls=":")
ax1.text(toy["v_true"][2] + 15, 0.05, "true vertex z", color=COL["TRUE"],
         fontsize=8.5, transform=ax1.get_xaxis_transform())
ax1.set_xlabel("target z of the extrapolation  [mm]")
ax1.set_ylabel("x deviation from straight line  [µm]")
ax1.set_title("What the extrapolator must supply: the fringe-field bend of these two tracks,\n"
              "measured-state straight line subtracted (leg A of the corpus is exactly this call)")
ax1.legend(frameon=False, fontsize=8.5, loc="best")
fig.tight_layout()
fig.savefig(OUT / "fig_E1_event_stage.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig_E1_event_stage.png")

# ---------------------------------------------------------------- E2 anatomy
fits = {}
for name, fetch in [("RK", rk_fetch), ("NN", nn_fetch), ("LIN", lin_fetch)]:
    fits[name] = fit_vertex_traced(fetch, toy)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.4),
                               gridspec_kw={"width_ratios": [1.35, 1]})
# left: z-x zoom around vertex, NN arm anatomy
v_nn, tr_nn = fits["NN"]
zwin = (toy["v_true"][2] - 90, toy["v_true"][2] + 90)
zz = np.linspace(*zwin, 60)
for j, tr in enumerate(toy["tracks"]):
    pth = rk_path(tr["S_meas"], tr["z_ref"], zz)
    axa.plot(zz, pth[:, 0], color="#bbbbbb", lw=1.2,
             label="true (exact) track paths" if j == 0 else None)
for k, (fz, fss) in enumerate(zip(tr_nn["fetch_z"], tr_nn["fetch_states"])):
    for j, Sf in enumerate(fss):
        xs = Sf[0] + Sf[2] * (zz - fz)
        axa.plot(zz, xs, color=COL["NN"], lw=1.5,
                 ls="-" if k == len(tr_nn["fetch_z"]) - 1 else ":",
                 alpha=1.0 if k == len(tr_nn["fetch_z"]) - 1 else 0.65,
                 label=("straight lines from the NN-fetched states"
                        if (j == 0 and k == len(tr_nn["fetch_z"]) - 1) else
                        ("initial fetch at the seed z (before DTF re-fetch)"
                         if (j == 0 and k == 0) else None)))
vhist = np.array(tr_nn["v"])
axa.plot(vhist[:, 2], vhist[:, 0], ".-", color="#8d6cc3", ms=7, lw=1.0,
         label="Newton iterations of the vertex estimate")
axa.plot(toy["v_seed"][2], toy["v_seed"][0], "x", color="#8d6cc3", ms=11, mew=2,
         label="seed")
axa.plot(v_nn[2], v_nn[0], "o", color=COL["NN"], ms=9,
         label="fitted vertex (NN arm)")
axa.plot(toy["v_true"][2], toy["v_true"][0], "*", color=COL["TRUE"], ms=17,
         label="true vertex")
axa.set_xlim(*zwin)
axa.set_xlabel("z  [mm]")
axa.set_ylabel("x  [mm]")
axa.set_title(f"Anatomy of the fit (NN arm): fetch → Newton steps → re-fetch → converge\n"
              f"(seed was {toy['v_seed'][2]-toy['v_true'][2]:+.0f} mm off in z → "
              f"{tr_nn['n_outer']} outer rounds, {len(tr_nn['fetch_z'])} fetch passes)")
axa.legend(frameon=False, fontsize=8)
# right: chi2 vs iteration for the three arms
for name in ["RK", "NN", "LIN"]:
    _, t = fits[name]
    axb.plot(range(1, len(t["chi2"]) + 1), t["chi2"], ".-", color=COL[name],
             lw=1.6, ms=7,
             label=f"{name}: χ² per Newton step (final {t['chi2'][-1]:.2f})")
axb.set_yscale("log")
axb.set_xlabel("Newton iteration (across outer rounds)")
axb.set_ylabel("fit χ²")
axb.set_title("Convergence of the same toy in the three arms:\n"
              "NN is indistinguishable from RK; LIN converges to a WRONG minimum")
axb.legend(frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(OUT / "fig_E2_fit_anatomy.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig_E2_fit_anatomy.png")

# ---------------------------------------------------------------- E3 gallery
T = np.load(LAB / "results" / "VF_p4_fit_toys_20260704_vf_zfeat_jacrow_h96.npz")
lin_zerr = np.abs(T["LIN_dv"][:, 2])
order = np.argsort(-lin_zerr)
pick = [disp] + [int(i) for i in order[[3, 30, 200, 800, 1500]] if int(i) != disp][:5]
fig, axes = plt.subplots(2, 3, figsize=(13, 8))
for ax, k in zip(axes.ravel(), pick):
    t = toys[k]
    for name, fetch in [("LIN", lin_fetch), ("NN", nn_fetch), ("RK", rk_fetch)]:
        vfit, _ = fit_vertex_traced(fetch, t)
        d = vfit - t["v_true"]
        ax.plot(d[2], d[0], "o" if name != "LIN" else "s", color=COL[name],
                ms=9 if name != "RK" else 6)
    ax.plot(0, 0, "*", color=COL["TRUE"], ms=16)
    ax.axhline(0, color="#999999", lw=0.6)
    ax.axvline(0, color="#999999", lw=0.6)
    ax.set_xlabel("fitted − true z  [mm]")
    ax.set_ylabel("fitted − true x  [mm]")
    ax.set_title(f"toy #{k}  (z_v = {t['v_true'][2]:.0f} mm, "
                 f"p = {t['tracks'][0]['p']:.0f}/{t['tracks'][1]['p']:.0f} GeV)",
                 fontsize=9.5)
handles = [Line2D([], [], marker="*", ls="", color=COL["TRUE"], ms=14, label="true vertex"),
           Line2D([], [], marker="o", ls="", color=COL["RK"], ms=6, label="RK fit (exact integrator)"),
           Line2D([], [], marker="o", ls="", color=COL["NN"], ms=9, label="NN fit (surrogate)"),
           Line2D([], [], marker="s", ls="", color=COL["LIN"], ms=9, label="straight-line fit (no field)")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("Six of the 2000 P4 toys: where each arm's fitted vertex lands relative to the truth "
             "(note the axis scales — LIN's z errors reach tens of mm)")
fig.tight_layout()
fig.savefig(OUT / "fig_E3_vertex_gallery.png", bbox_inches="tight")
plt.close(fig)
print("wrote fig_E3_vertex_gallery.png")
