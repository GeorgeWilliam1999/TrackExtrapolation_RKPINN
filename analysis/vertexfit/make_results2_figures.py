#!/usr/bin/env python3
"""Figure set for Results II — the P3b attribution matrix, the z-features
model read-out, the fit-level proof (4-arm P4), the deployment gate (parity +
same-machine speed) and the stack-integration target.

Everything is a DIRECT comparison: each figure carries the new model, its
predecessors, and the relevant incumbent (straight-line transport or the
production-shaped Cash-Karp) on the same axes.

Inputs (read-only): corpus + J labels, all trained_models/, logs/train_*.log,
results/VF_tier1_*.json, VF_p4_fit_toys_*_vf_zfeat_h96.npz, VF_cpp_gate_*.json.
Outputs: REPO/docs/figures/vertexfit/results2/fig_S*.png (150 dpi).
"""
from __future__ import annotations

import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import torch
torch.set_num_threads(6)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LAB = Path("/data/bfys/gscriven/TrackExtrapolation/experiments/vertexfit")
OUT = REPO / "docs" / "figures" / "vertexfit" / "results2"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "models"))
from train import _build_model, _model_state_jacobian  # noqa: E402

SURFACE = "#fcfcfb"
LEG_NAME = {0: "A  UT→vertex", 1: "B  T→vertex", 2: "C  VELO", 3: "D  intra-band"}
# model → (label, colour); ordered oldest → newest
MODELS = [
    ("vf_resid_h96", "P2 baseline (endpoint loss only)", "#9bbfe8"),
    ("vf_jac_h96", "P3 +J supervision (λ=100)", "#eda100"),
    ("vf_zfeat_h96", "P3b z-features", "#008300"),
    ("vf_zfeat_jacrow_h96", "P3b z-features + row-norm J", "#2a78d6"),
]
SL_COL = "#777777"
ARM_COL = {"RK": "#444444", "NN": "#008300", "NNH": "#2a78d6", "LIN": "#c22f2f"}
ARM_LAB = {"RK": "RK reference (exact integrator)",
           "NN": "NN, exact autodiff J",
           "NNH": "NN, zero-cost head-only J (deployable)",
           "LIN": "straight-line (no field)"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 10.5, "axes.titlesize": 11.5, "figure.dpi": 150,
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def cdf(ax, vals, clip=1e-3, **kw):
    v = np.sort(np.maximum(np.abs(vals), clip))
    ax.plot(v, np.arange(1, len(v) + 1) / len(v), **kw)


def profile_median(xv, yv, bins):
    idx = np.digitize(xv, bins) - 1
    cs, ms = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 200:
            cs.append(0.5 * (bins[b] + bins[b + 1]))
            ms.append(np.median(yv[m]))
    return np.array(cs), np.array(ms)


# ---------------------------------------------------------------- load
print("loading corpus + tier-1 JSONs ...")
with np.load(LAB / "data" / "vf_corpus_10M.npz") as d:
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    P = d["P"].astype(np.float32)
    LEG = d["LEG"]
idx = np.load(LAB / "trained_models" / "vf_zfeat_h96" / "test_indices.npy")
Xt, Yt, Pt, Lt = X[idx], Y[idx], P[idx], LEG[idx]
del X, Y, P, LEG
dz = Xt[:, 6]

t1 = {}
for f in sorted(glob.glob(str(LAB / "results" / "VF_tier1_*.json"))):
    for r in json.load(open(f)):
        t1[r["experiment"]] = r          # later files overwrite older entries
print("tier-1 entries:", sorted(t1))

gate = {}
for f in sorted(glob.glob(str(LAB / "results" / "VF_cpp_gate_*.json"))):
    g = json.load(open(f))
    gate[g["experiment"]] = g


def load_model(exp):
    d = LAB / "trained_models" / exp
    ck = torch.load(d / "best_model.pt", weights_only=False, map_location="cpu")
    m = _build_model(ck["config"])
    m.load_normalization(str(d / "normalization.json"))
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


@torch.no_grad()
def predict(m, Xa, batch=65536):
    out = []
    for i in range(0, len(Xa), batch):
        out.append(m(torch.from_numpy(Xa[i:i + batch])).numpy())
    return np.concatenate(out)


print("predicting on test split ...")
ERR = {}
for exp, _, _ in MODELS:
    ERR[exp] = predict(load_model(exp), Xt)[:, :4] - Yt[:, :4]
    print("  ", exp)
err_sl = np.stack([Xt[:, 0] + Xt[:, 2] * dz - Yt[:, 0],
                   Xt[:, 1] + Xt[:, 3] * dz - Yt[:, 1],
                   Xt[:, 2] - Yt[:, 2], Xt[:, 3] - Yt[:, 3]], axis=1)

# ---------------------------------------------------------------- S1 curves
print("S1")
LOGMAP = [
    ("train_h96.log", "P2 baseline", "#9bbfe8"),
    ("train_vf_jac_h96.log", "P3 +J (λ=100 std)", "#eda100"),
    ("train_vf_jac_slope_h96.log", "P3 +J +slope-weight", "#d9c06b"),
    ("train_vf_jacL10_h96.log", "P3b λ_J=10 soften", "#8d6cc3"),
    ("train_vf_jacrow_h96.log", "P3b row-norm J", "#b06a8f"),
    ("train_vf_zfeat_h96.log", "P3b z-features", "#008300"),
    ("train_vf_zfeat_jacrow_h96.log", "P3b z-features + row-norm J", "#2a78d6"),
]
PAT = re.compile(r"\[\s*(\d+)/(\d+)\]\s+tr=([\d.]+)\s+median_dx=([\d.]+)")


def parse_log(fn):
    ep, med = [], []
    for line in open(LAB / "logs" / fn, errors="replace"):
        m = PAT.search(line)
        if m:
            ep.append(int(m.group(1)))
            med.append(float(m.group(4)))
    return np.array(ep), np.array(med)


fig, ax = plt.subplots(figsize=(9.5, 5.2))
for fn, lab, col in LOGMAP:
    ep, med = parse_log(fn)
    lw = 2.2 if "z-features" in lab else 1.4
    ax.plot(ep, med, color=col, lw=lw, label=lab)
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("bulk-validation median |Δx|  [µm]")
ax.set_title("One architecture fix beats every loss-tuning intervention:\n"
             "all λ-rebalancing runs plateau at 230–270 µm; both z-features "
             "runs drop an order of magnitude")
ax.legend(frameon=False, fontsize=9)
save(fig, "fig_S1_training_curves.png")

# ---------------------------------------------------------------- S2 CDFs
print("S2")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for code, ax in zip(range(4), axes.ravel()):
    m = Lt == code
    cdf(ax, err_sl[m, 0] * 1e3, color=SL_COL, lw=1.6, ls="--",
        label="straight-line transport (what the fit uses today between fetches)"
        if code == 0 else None)
    for exp, lab, col in MODELS:
        cdf(ax, ERR[exp][m, 0] * 1e3, color=col,
            lw=2.2 if "zfeat" in exp else 1.4,
            label=lab if code == 0 else None)
    if code == 0:
        ax.axvline(100, color="#c22f2f", lw=1.0, ls=":")
        ax.text(107, 0.05, "spec bar\n100 µm", color="#c22f2f", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-1, 1e6)
    ax.set_ylim(0, 1)
    ax.set_title(f"leg {LEG_NAME[code]}   (n = {m.sum():,})")
    ax.set_xlabel("|Δx| at target plane  [µm]")
    ax.set_ylabel("fraction of tracks")
fig.suptitle("Endpoint accuracy vs the incumbents, per leg "
             "(held-out test split, 996,717 tracks)", y=1.0)
fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout()
save(fig, "fig_S2_endpoint_cdf.png")

# ---------------------------------------------------------------- S3 bars
print("S3")
LEGKEYS = ["A UT->vtx", "B T->vtx", "C VELO", "D intra"]
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
panels = [("med_dx_um", "median |Δx| [µm]", 100.0, "A UT->vtx"),
          ("p95_dx_um", "p95 |Δx| [µm]", 1000.0, "A UT->vtx"),
          ("med_dtx_urad", "median |Δtx| [µrad]", 50.0, "A UT->vtx")]
names = [lab for _, lab, _ in MODELS]
for ax, (stat, ylab, bar, legk) in zip(axes, panels):
    vals = [t1[e]["legs"][legk]["nn"][stat] for e, _, _ in MODELS]
    cols = [c for _, _, c in MODELS]
    ax.bar(range(len(vals)), vals, color=cols)
    ax.axhline(bar, color="#c22f2f", lw=1.4, ls=":")
    ax.text(0.02, bar * 1.15, f"spec bar {bar:.0f}", color="#c22f2f",
            fontsize=9, transform=ax.get_yaxis_transform())
    sl = t1["vf_zfeat_h96"]["legs"][legk]["straight"][stat]
    ax.axhline(sl, color=SL_COL, lw=1.2, ls="--")
    ax.text(0.98, sl * 1.15, "straight-line", color=SL_COL, fontsize=8.5,
            ha="right", transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(["P2", "P3 +J", "z-feat", "z-feat\n+rowJ"], fontsize=9)
    ax.set_ylabel(ylab)
    ax.set_title(f"leg A: {ylab.split('[')[0].strip()}")
fig.suptitle("The three leg-A acceptance bars (dotted red): both z-features "
             "models pass all of them")
fig.tight_layout()
save(fig, "fig_S3_legA_bars.png")

# ---------------------------------------------------------------- S4 vs p
print("S4")
pb = np.geomspace(1.5, 120, 22)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, code in zip(axes, [0, 1]):
    m = Lt == code
    c, v = profile_median(Pt[m], np.abs(err_sl[m, 0]) * 1e3, pb)
    ax.plot(c, v, color=SL_COL, lw=1.6, ls="--", label="straight-line")
    for exp, lab, col in MODELS[1:]:
        c, v = profile_median(Pt[m], np.abs(ERR[exp][m, 0]) * 1e3, pb)
        ax.plot(c, v, color=col, lw=2.0 if "zfeat" in exp else 1.4, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("momentum p  [GeV]")
    ax.set_ylabel("median |Δx|  [µm]")
    ax.set_title(f"leg {LEG_NAME[code]}")
    if code == 0:
        ax.axhline(100, color="#c22f2f", lw=1.0, ls=":")
        ax.legend(frameon=False, fontsize=8.5)
fig.suptitle("Momentum dependence: the P3 models passed the 100 µm bar only "
             "above ~20 GeV; the z-features models pass at every momentum")
fig.tight_layout()
save(fig, "fig_S4_err_vs_p.png")

# ---------------------------------------------------------------- S5 tier-2
print("S5 (Jacobian CDFs, 50k subsample)")
Jmm = np.load(LAB / "data" / "vf_corpus_10M_J.npy", mmap_mode="r")
rng = np.random.default_rng(7)
sel = rng.choice(len(idx), 200_000, replace=False)[:50_000]
Xs, Ls = Xt[sel], Lt[sel]
Jl4 = np.asarray(Jmm[idx[sel], :4, :], dtype=np.float64)
den = np.linalg.norm(Jl4.reshape(len(Jl4), -1), axis=1)


def relf_of(J):
    return np.linalg.norm((J - Jl4).reshape(len(J), -1), axis=1) / den


relf = {}
for exp in ["vf_jac_h96", "vf_zfeat_h96", "vf_zfeat_jacrow_h96"]:
    m = load_model(exp)
    r = np.empty(len(Xs))
    for i in range(0, len(Xs), 8192):
        xb = torch.from_numpy(Xs[i:i + 8192])
        Jm = _model_state_jacobian(m, xb).detach().numpy().astype(np.float64)
        r[i:i + 8192] = relf_of(Jm)[0:0] if False else np.linalg.norm(
            (Jm - Jl4[i:i + 8192]).reshape(len(Jm), -1), axis=1) / den[i:i + 8192]
    relf[exp] = r
    print("  ", exp)
# straight-line J and zero-cost head-only J (from zfeat forward outputs)
Jsl = np.zeros_like(Jl4)
for d4 in range(4):
    Jsl[:, d4, d4] = 1.0
Jsl[:, 0, 2] = Xs[:, 6].astype(np.float64)
Jsl[:, 1, 3] = Xs[:, 6].astype(np.float64)
relf_sl = relf_of(Jsl)
Yz = predict(load_model("vf_zfeat_h96"), Xs).astype(np.float64)
Xd = Xs.astype(np.float64)
Jh = Jsl.copy()
Jh[:, 0, 4] = (Yz[:, 0] - Xd[:, 0] - Xd[:, 2] * Xd[:, 6]) / Xd[:, 4]
Jh[:, 1, 4] = (Yz[:, 1] - Xd[:, 1] - Xd[:, 3] * Xd[:, 6]) / Xd[:, 4]
Jh[:, 2, 4] = (Yz[:, 2] - Xd[:, 2]) / Xd[:, 4]
Jh[:, 3, 4] = (Yz[:, 3] - Xd[:, 3]) / Xd[:, 4]
relf_head = relf_of(Jh)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
CURVES = [
    (relf_sl, "straight-line J (incumbent between fetches)", SL_COL, "--"),
    (relf["vf_jac_h96"], "P3 +J supervision, autodiff J", "#eda100", "-"),
    (relf["vf_zfeat_h96"], "z-features, autodiff J (no J training!)", "#008300", "-"),
    (relf_head, "z-features, zero-cost head-only J (deployable)", "#2a78d6", "-"),
    (relf["vf_zfeat_jacrow_h96"], "z-features + row-norm J, autodiff J", "#b06a8f", "-"),
]
for code, ax in zip(range(4), axes.ravel()):
    m = Ls == code
    for vals, lab, col, ls in CURVES:
        cdf(ax, vals[m], clip=1e-8, color=col, lw=1.7, ls=ls,
            label=lab if code == 0 else None)
    ax.axvline(0.01, color="#c22f2f", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1e-5, 1e1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("relative Frobenius error of J (rows x,y,tx,ty)")
    ax.set_ylabel("fraction of tracks")
    ax.set_title(f"leg {LEG_NAME[code]}   (n = {m.sum():,})")
fig.suptitle("Covariance-transport fidelity: per-track Jacobian error vs the "
             "exact labels (bar 0.01 dotted red)", y=1.0)
fig.legend(loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.07))
fig.tight_layout()
save(fig, "fig_S5_jacobian_cdf.png")

# ---------------------------------------------------------------- S6-S9 P4
print("S6-S9 (P4, 4 arms)")
T = np.load(LAB / "results" / "VF_p4_fit_toys_20260703_vf_zfeat_h96.npz")
p4 = json.load(open(LAB / "results" / "VF_p4_fit_20260703_vf_zfeat_h96.json"))
ARMS = ["LIN", "NN", "NNH", "RK"]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
ranges = [(-8, 8), (-8, 8), (-60, 60)]
for k, (ax, cname) in enumerate(zip(axes, "xyz")):
    for arm in ARMS:
        v = T[f"{arm}_dv"][:, k]
        ax.hist(np.clip(v, *ranges[k]), bins=90, range=ranges[k],
                histtype="step", lw=1.6, color=ARM_COL[arm],
                label=ARM_LAB[arm] if k == 0 else None)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(f"fitted − true vertex {cname}  [mm]")
    ax.set_ylabel("toys / bin")
    ax.set_title(f"vertex {cname} residual")
fig.suptitle("Fit-level proof: vertex residuals on 2000 identical two-track "
             "toys — the NN arms lie on the exact-integrator curve; "
             "the no-field arm does not")
fig.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
save(fig, "fig_S6_p4_residuals.png")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
gx = np.linspace(-6, 6, 300)
gauss = np.exp(-0.5 * gx ** 2) / np.sqrt(2 * np.pi)
for k, (ax, cname, skey) in enumerate(zip(axes, "xyz", ["sx", "sy", "sz"])):
    for arm in ARMS:
        pull = T[f"{arm}_dv"][:, k] / T[f"{arm}_{skey}"]
        ax.hist(np.clip(pull, -6, 6), bins=72, range=(-6, 6), density=True,
                histtype="step", lw=1.6, color=ARM_COL[arm],
                label=ARM_LAB[arm] if k == 0 else None)
    ax.plot(gx, gauss, color="k", lw=1.0, ls=":", label="N(0,1)" if k == 0 else None)
    ax.set_xlabel(f"pull: Δ{cname} / σ_{cname}(fit)")
    ax.set_ylabel("density")
    ax.set_title(f"vertex {cname} pull")
fig.suptitle("The reported covariance is honest: NN pulls sit on the unit "
             "Gaussian (widths 0.98–1.01), like the exact integrator")
fig.legend(loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.tight_layout()
save(fig, "fig_S7_p4_pulls.png")

from scipy import stats as sps
fig, ax = plt.subplots(figsize=(8, 4.6))
for arm in ["RK", "NN", "NNH", "LIN"]:
    cdf(ax, T[f"{arm}_chi2"], clip=1e-4, color=ARM_COL[arm], lw=1.8,
        label=ARM_LAB[arm])
xq = np.geomspace(1e-4, 300, 400)
ax.plot(xq, sps.chi2.cdf(xq, df=1), color="k", ls=":", lw=1.2,
        label="ideal χ²(1 dof)")
ax.set_xscale("log")
ax.set_xlim(1e-3, 300)
ax.set_xlabel("fit χ²  (1 dof)")
ax.set_ylabel("fraction of toys")
ax.set_title("Fit χ²: all three field-aware arms lie on the ideal law; "
             "no field inflates χ² by 18×")
ax.legend(frameon=False, fontsize=9, loc="upper left")
save(fig, "fig_S8_p4_chi2.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ypos = np.arange(3)[::-1]
off = {"RK": 0.27, "NN": 0.09, "NNH": -0.09, "LIN": -0.27}
for arm in ARMS[::-1]:
    b = np.array(p4["arms"][arm]["bias_mm"]) * 1e3
    r = np.array(p4["arms"][arm]["res_mm"]) * 1e3
    axes[0].errorbar(b, ypos + off[arm], xerr=r / np.sqrt(2000) * 1.2533,
                     fmt="o", color=ARM_COL[arm], ms=6, capsize=3,
                     label=ARM_LAB[arm])
    axes[1].barh(ypos + off[arm], r, height=0.16, color=ARM_COL[arm])
rk_b = np.array(p4["arms"]["RK"]["bias_mm"]) * 1e3
bars = np.array([bb["bar_mm"] for bb in p4["tier3_bars"]]) * 1e3
for k in range(3):
    axes[0].barh(ypos[k], 2 * bars[k], left=rk_b[k] - bars[k], height=0.8,
                 color="#2a78d6", alpha=0.12, zorder=0)
axes[0].axvline(0, color="k", lw=0.7)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(["x", "y", "z"])
axes[0].set_xlabel("vertex bias (median residual)  [µm]")
axes[0].set_xlim(-450, 1350)
axes[0].set_title("bias; shaded band = tier-3 acceptance around the RK bias")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
axes[1].set_yticks(ypos)
axes[1].set_yticklabels(["x", "y", "z"])
axes[1].set_xscale("log")
axes[1].set_xlabel("robust resolution (1.4826·MAD)  [µm]")
axes[1].set_title("resolution: NN arms within 1–2 % of the exact integrator")
fig.suptitle("P4 summary, all arms — the deployable head-only-J arm is "
             "indistinguishable from exact-J and from the RK reference")
fig.tight_layout()
save(fig, "fig_S9_p4_summary.png")

# ---------------------------------------------------------------- S10 speed
print("S10 (throughput; live torch timings)")
mz = load_model("vf_zfeat_h96")
x1 = torch.from_numpy(Xt[:1])
with torch.no_grad():
    for _ in range(50):
        mz(x1)
    t0 = time.perf_counter()
    for _ in range(300):
        mz(x1)
    torch_b1_us = (time.perf_counter() - t0) / 300 * 1e6
xb = torch.from_numpy(Xt[:4096])
with torch.no_grad():
    for _ in range(3):
        mz(xb)
    t0 = time.perf_counter()
    for _ in range(20):
        mz(xb)
    torch_b4096_us = (time.perf_counter() - t0) / 20 / 4096 * 1e6
g = gate["vf_zfeat_h96"]["speed"]
entries = [
    ("torch, single call (dispatch-bound)", torch_b1_us, "#cccccc", "state only"),
    ("C++ kernel, exact autodiff J", g["state_plus_exactJ"]["ns_per_call_best"] / 1e3,
     "#9bbfe8", "state + 5×5 J"),
    ("production TrackMaster (8.35 µs, HLT2 hardware)", 8.35, "#999999",
     "state + transport matrix + material"),
    ("Cash-Karp RK reference (same machine)", g["rk_reference"]["ns_per_call_best"] / 1e3,
     "#444444", "state only"),
    ("C++ kernel, head-only J  ← deployable", g["state_plus_headJ"]["ns_per_call_best"] / 1e3,
     "#2a78d6", "state + 5×5 J"),
    ("C++ kernel, state only", g["state_only"]["ns_per_call_best"] / 1e3,
     "#008300", "state only"),
    ("torch, batch 4096 (offline amortised)", torch_b4096_us, "#d9c06b", "state only"),
]
fig, ax = plt.subplots(figsize=(10, 5))
ypos = np.arange(len(entries))[::-1]
for y, (lab, us, col, note) in zip(ypos, entries):
    ax.barh(y, us, color=col, height=0.62)
    ax.text(us * 1.06, y, f"{us:.2f} µs   ({note})", va="center", fontsize=9)
ax.set_yticks(ypos)
ax.set_yticklabels([e[0] for e in entries], fontsize=9.5)
ax.set_xscale("log")
ax.set_xlim(0.05, 300)
ax.set_xlabel("time per extrapolation call  [µs]  (log scale)")
ax.set_title("Single-call latency: NN kernel measured on a loaded 2017 Zen-1 node; "
             "same-machine Cash-Karp reference makes the ratio hardware-honest.\n"
             "Deployable NN (state + Jacobian) beats the RK doing state-only work.")
save(fig, "fig_S10_throughput.png")

# ---------------------------------------------------------------- S11 diagram
print("S11 (integration target)")
fig, ax = plt.subplots(figsize=(11, 6.2))
ax.axis("off")
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)


def box(x, y, w, h, text, fc, fontsize=9.5, mono=False, ec="#888888"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                facecolor=fc, edgecolor=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, family="monospace" if mono else None)


def arrow(x0, y0, x1, y1, text="", col="#555555"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, color=col, lw=1.4))
    if text:
        ax.text((x0 + x1) / 2 + 0.12, (y0 + y1) / 2, text, fontsize=8.5,
                color=col, ha="left")


box(0.4, 8.4, 4.2, 1.2, "DecayTreeFitter\n(re-fetch when |Δz| > 1 cm, ≤10 outer)", "#eef3fb")
box(5.4, 8.4, 4.2, 1.2, "ParticleVertexFitter\n(one fetch per track at seed z)", "#eef3fb")
box(2.9, 6.5, 4.2, 1.1, "TrackStateProvider\n(state cache; one field-aware fetch per (track, z))", "#e8f0e8")
arrow(2.5, 8.4, 4.2, 7.6)
arrow(7.5, 8.4, 5.8, 7.6)
box(2.9, 4.7, 4.2, 1.05, "ITrackExtrapolator\npropagate(stateVec, z0, z1, transMat*)", "#fdf3e0", mono=True)
arrow(5.0, 6.5, 5.0, 5.75, " Rec/Tr/TrackInterfaces")
box(2.9, 2.9, 4.2, 1.05, "TrackMasterExtrapolator\n(material: scattering + dE/dx — unchanged)", "#f5f5f5")
arrow(5.0, 4.7, 5.0, 3.95)
box(0.6, 0.6, 4.0, 1.5, "TODAY: TrackRungeKuttaExtrapolator\nCash-Karp + ~69 field-map lookups\n≈ 8.35 µs/call/core (90 % lookups)", "#f0e2e2")
box(5.4, 0.6, 4.2, 1.5, "PROPOSED: NNVertexFitExtrapolator\ndeploy/vf_kernel.cpp + VFK1 weight blob\nstate + head-only J, ≈ 4 µs on 2017 CPU\n(weights-header pattern: FieldMapNNWeights.h)", "#e2f0e2")
arrow(3.8, 2.9, 2.6, 2.1, " field step (today)")
arrow(6.2, 2.9, 7.5, 2.1, " field step (this work)", col="#008300")
ax.set_title("Integration target in the LHCb Upgrade-II stack (Rec): the NN replaces the "
             "field-propagation engine inside the vertex-fit fetch path.\nEverything above "
             "the bottom row is unchanged; material handling stays with TrackMasterExtrapolator.",
             fontsize=11)
save(fig, "fig_S11_integration_target.png")

# ---------------------------------------------------------------- S12 timeline
print("S12")
prog = [("straight line", t1["vf_zfeat_h96"]["legs"]["A UT->vtx"]["straight"]["med_dx_um"], SL_COL),
        ("P2 baseline", t1["vf_resid_h96"]["legs"]["A UT->vtx"]["nn"]["med_dx_um"], "#9bbfe8"),
        ("P3 +J", t1["vf_jac_h96"]["legs"]["A UT->vtx"]["nn"]["med_dx_um"], "#eda100"),
        ("P3b z-features", t1["vf_zfeat_h96"]["legs"]["A UT->vtx"]["nn"]["med_dx_um"], "#008300"),
        ("P3b z-feat + rowJ", t1["vf_zfeat_jacrow_h96"]["legs"]["A UT->vtx"]["nn"]["med_dx_um"], "#2a78d6")]
fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.bar(range(len(prog)), [p[1] for p in prog], color=[p[2] for p in prog])
for i, (lab, v, _) in enumerate(prog):
    ax.text(i, v * 1.15, f"{v:.0f}" if v > 20 else f"{v:.1f}", ha="center", fontsize=10)
ax.axhline(100, color="#c22f2f", lw=1.4, ls=":")
ax.text(0.02, 115, "acceptance bar 100 µm", color="#c22f2f", fontsize=9,
        transform=ax.get_yaxis_transform())
ax.set_yscale("log")
ax.set_xticks(range(len(prog)))
ax.set_xticklabels([p[0] for p in prog], fontsize=9)
ax.set_ylabel("leg-A median |Δx|  [µm]")
ax.set_title("Leg A (the dominant vertex-fit case), campaign progression")
save(fig, "fig_S12_progression.png")

print("\nall figures written to", OUT)
