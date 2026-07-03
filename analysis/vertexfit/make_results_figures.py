#!/usr/bin/env python3
"""Publication-grade figure set for the vertex-fit surrogate Results write-up.

Covers the four CONVERGED models (P2 residual baselines + P3 J-supervision
variants) at tier 1 (endpoint accuracy), tier 2 (Jacobian fidelity) and
tier 3 (P4 fit-level emulation), plus training dynamics and a clearly-flagged
preview of the in-flight P3b attribution matrix.

Inputs (all read-only):
  LAB/data/vf_corpus_10M.npz            X[N,7], Y[N,5], P[N], LEG[N]
  LAB/data/vf_corpus_10M_J.npy          exact 5x5 Jacobian labels (memmap)
  LAB/trained_models/<exp>/             checkpoints + normalisation + split
  LAB/results/VF_tier1_20260703.json    tier-1/2 summary stats
  LAB/results/VF_p4_fit_toys_20260703.npz  per-toy P4 arrays
  LAB/logs/train_*.log                  training curves

Outputs: REPO/docs/figures/vertexfit/results/fig_R*.png (150 dpi).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
torch.set_num_threads(3)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
LAB = Path("/data/bfys/gscriven/TrackExtrapolation/experiments/vertexfit")
OUT = REPO / "docs" / "figures" / "vertexfit" / "results"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO / "models"))
sys.path.insert(0, str(REPO / "core"))
from train import _build_model, _model_state_jacobian  # noqa: E402

# ---------------------------------------------------------------- style
SURFACE = "#fcfcfb"
LEG_COL = {0: "#2a78d6", 1: "#1baf7a", 2: "#eda100", 3: "#008300"}
LEG_NAME = {0: "A  UT→vertex", 1: "B  T→vertex", 2: "C  VELO",
            3: "D  intra-band"}
MODELS = [
    ("vf_resid_h64", "P2 baseline h64", "#9bbfe8"),
    ("vf_resid_h96", "P2 baseline h96", "#2a78d6"),
    ("vf_jac_h96", "P3 +J supervision", "#eda100"),
    ("vf_jac_slope_h96", "P3 +J +slope-weight", "#1baf7a"),
]
SL_COL = "#777777"
ARM_COL = {"RK": "#444444", "NN": "#2a78d6", "LIN": "#c22f2f"}
ARM_LAB = {"RK": "RK reference (Cash-Karp truth engine)",
           "NN": "NN surrogate (vf_jac_h96)",
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
    """median of yv in bins of xv; returns centers, medians (>=200 entries)."""
    idx = np.digitize(xv, bins) - 1
    cs, ms = [], []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() >= 200:
            cs.append(0.5 * (bins[b] + bins[b + 1]))
            ms.append(np.median(yv[m]))
    return np.array(cs), np.array(ms)


# ---------------------------------------------------------------- load
print("loading corpus ...")
with np.load(LAB / "data" / "vf_corpus_10M.npz") as d:
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    P = d["P"].astype(np.float32)
    LEG = d["LEG"]

idx = np.load(LAB / "trained_models" / "vf_jac_h96" / "test_indices.npy")
Xt, Yt, Pt, Lt = X[idx], Y[idx], P[idx], LEG[idx]
del X, Y, P, LEG
dz = Xt[:, 6]
z0 = Xt[:, 5]
z1 = z0 + dz
print(f"test split: {len(idx):,} rows")

tier1 = json.load(open(LAB / "results" / "VF_tier1_20260703.json"))
t1 = {r["experiment"]: r for r in tier1}


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


print("predicting on test split (4 models) ...")
ERR = {}   # exp -> err[N,4] = pred - truth for x,y,tx,ty
for exp, _, _ in MODELS:
    ERR[exp] = predict(load_model(exp), Xt)[:, :4] - Yt[:, :4]
    print("  ", exp, "done")
err_sl = np.stack([Xt[:, 0] + Xt[:, 2] * dz - Yt[:, 0],
                   Xt[:, 1] + Xt[:, 3] * dz - Yt[:, 1],
                   Xt[:, 2] - Yt[:, 2], Xt[:, 3] - Yt[:, 3]], axis=1)

# ---------------------------------------------------------------- R1 curves
print("R1: training curves")
LOGMAP = {"vf_resid_h64": "train_h64.log", "vf_resid_h96": "train_h96.log",
          "vf_jac_h96": "train_vf_jac_h96.log",
          "vf_jac_slope_h96": "train_vf_jac_slope_h96.log"}
PAT = re.compile(r"\[\s*(\d+)/(\d+)\]\s+tr=([\d.]+)\s+median_dx=([\d.]+)")


def parse_log(fn):
    ep, med = [], []
    for line in open(LAB / "logs" / fn, errors="replace"):
        m = PAT.search(line)
        if m:
            ep.append(int(m.group(1)))
            med.append(float(m.group(4)))
    return np.array(ep), np.array(med)


fig, ax = plt.subplots(figsize=(8.5, 4.6))
for exp, lab, col in MODELS:
    ep, med = parse_log(LOGMAP[exp])
    ax.plot(ep, med, color=col, lw=1.8, label=lab)
    ax.plot(ep[np.argmin(med)], med.min(), "o", color=col, ms=5)
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("bulk-validation median |Δx|  [µm]")
ax.set_title("Training dynamics (dot = best epoch, the saved checkpoint)")
ax.legend(frameon=False)
save(fig, "fig_R1_training_curves.png")

# ---------------------------------------------------------------- R2 CDFs
print("R2: per-leg endpoint CDFs")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for code, ax in zip(range(4), axes.ravel()):
    m = Lt == code
    for exp, lab, col in MODELS:
        cdf(ax, ERR[exp][m, 0] * 1e3, color=col, lw=1.6,
            label=lab if code == 0 else None)
    cdf(ax, err_sl[m, 0] * 1e3, color=SL_COL, lw=1.6, ls="--",
        label="straight-line transport" if code == 0 else None)
    if code == 0:
        ax.axvline(100, color="#c22f2f", lw=1.0, ls=":")
        ax.text(105, 0.06, "spec bar\n100 µm", color="#c22f2f", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-1, 1e6)
    ax.set_ylim(0, 1)
    ax.set_title(f"leg {LEG_NAME[code]}   (n = {m.sum():,})")
    ax.set_xlabel("|Δx| at target plane  [µm]")
    ax.set_ylabel("fraction of tracks")
fig.suptitle("Endpoint accuracy: cumulative |Δx| distributions per leg "
             "(test split, 996,717 tracks)", y=1.0)
fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout()
save(fig, "fig_R2_endpoint_cdf.png")

# ---------------------------------------------------------------- R3 bars
print("R3: per-leg median/p95 bars")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
w = 0.18
xs = np.arange(4)
for ax, (stat, ttl) in zip(axes, [("med_dx_um", "median |Δx|"),
                                  ("p95_dx_um", "95th percentile |Δx|")]):
    for k, (exp, lab, col) in enumerate(MODELS):
        vals = [t1[exp]["legs"][n]["nn"][stat]
                for n in ["A UT->vtx", "B T->vtx", "C VELO", "D intra"]]
        ax.bar(xs + (k - 1.5) * w, vals, w, color=col, label=lab)
    sl = [t1["vf_jac_h96"]["legs"][n]["straight"][stat]
          for n in ["A UT->vtx", "B T->vtx", "C VELO", "D intra"]]
    ax.plot(xs, sl, "v", color=SL_COL, ms=9, ls="none",
            label="straight-line transport")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(["A", "B", "C", "D"])
    ax.set_xlabel("leg")
    ax.set_ylabel(f"{ttl}  [µm]")
    ax.set_title(ttl)
axes[0].axhline(100, color="#c22f2f", lw=1.0, ls=":")
axes[0].text(2.55, 115, "leg-A spec bar 100 µm", color="#c22f2f", fontsize=8.5)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
fig.suptitle("Tier-1 endpoint accuracy by leg and model (log scale)")
fig.tight_layout()
save(fig, "fig_R3_leg_medians.png")

# ---------------------------------------------------------------- R4 vs p
print("R4: error vs momentum")
pb = np.geomspace(1.5, 120, 22)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)
for ax, code in zip(axes, [0, 1]):
    m = Lt == code
    for exp, lab, col in [MODELS[1], MODELS[2]]:
        c, v = profile_median(Pt[m], np.abs(ERR[exp][m, 0]) * 1e3, pb)
        ax.plot(c, v, color=col, lw=1.8, label=lab)
    c, v = profile_median(Pt[m], np.abs(err_sl[m, 0]) * 1e3, pb)
    ax.plot(c, v, color=SL_COL, lw=1.6, ls="--", label="straight-line")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("momentum p  [GeV]")
    ax.set_ylabel("median |Δx|  [µm]")
    ax.set_title(f"leg {LEG_NAME[code]}")
    if code == 0:
        ax.axhline(100, color="#c22f2f", lw=1.0, ls=":")
        ax.legend(frameon=False, fontsize=9)
fig.suptitle("Endpoint error vs momentum: the NN tracks the 1/p bend "
             "where straight-line transport cannot")
fig.tight_layout()
save(fig, "fig_R4_err_vs_p.png")

# ---------------------------------------------------------------- R5 vs |dz|
print("R5: error vs |dz|")
db = np.geomspace(5, 10000, 26)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
for code in range(4):
    m = Lt == code
    c, v = profile_median(np.abs(dz[m]), np.abs(ERR["vf_jac_h96"][m, 0]) * 1e3, db)
    ax.plot(c, v, color=LEG_COL[code], lw=1.8, label=f"leg {LEG_NAME[code]}")
    c, v = profile_median(np.abs(dz[m]), np.abs(err_sl[m, 0]) * 1e3, db)
    ax.plot(c, v, color=LEG_COL[code], lw=1.2, ls="--", alpha=0.55)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("|Δz| of the leg  [mm]")
ax.set_ylabel("median |Δx|  [µm]")
ax.set_title("Endpoint error vs propagation distance, best P3 model "
             "(dashed = straight-line transport)")
ax.legend(frameon=False, fontsize=9)
save(fig, "fig_R5_err_vs_dz.png")

# ---------------------------------------------------------------- R6 slopes
print("R6: the dead slope lever")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
mA = Lt == 0
for exp, lab, col in MODELS:
    cdf(axes[0], ERR[exp][mA, 2] * 1e6, color=col, lw=1.6, label=lab)
cdf(axes[0], err_sl[mA, 2] * 1e6, color=SL_COL, lw=1.6, ls="--",
    label="straight-line (slopes frozen)")
axes[0].axvline(50, color="#c22f2f", lw=1.0, ls=":")
axes[0].text(54, 0.06, "spec bar\n50 µrad", color="#c22f2f", fontsize=8.5)
axes[0].set_xscale("log")
axes[0].set_xlim(1e0, 1e5)
axes[0].set_xlabel("|Δtx| at target plane  [µrad]")
axes[0].set_ylabel("fraction of tracks")
axes[0].set_title("leg A: cumulative |Δtx|")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
vals = [t1[exp]["legs"]["A UT->vtx"]["nn"]["med_dtx_urad"] for exp, _, _ in MODELS]
cols = [c for _, _, c in MODELS]
axes[1].bar([lab for _, lab, _ in MODELS], vals, color=cols)
axes[1].axhline(50, color="#c22f2f", lw=1.2, ls=":")
axes[1].set_ylabel("leg-A median |Δtx|  [µrad]")
axes[1].set_title("median slope error is flat at ~420 µrad across every "
                  "intervention")
axes[1].tick_params(axis="x", rotation=12, labelsize=8.5)
fig.suptitle("The slope lever: all models miss the 50 µrad bar by ~8×, "
             "insensitive to loss weighting")
fig.tight_layout()
save(fig, "fig_R6_slopes.png")

# ---------------------------------------------------------------- R7 fringe
print("R7: z-blindness / fringe scan")
sys.path.insert(0, str(REPO / "datagen"))
from field_v8r1 import FieldV8R1  # noqa: E402
fld = FieldV8R1()
zg = np.linspace(-200, 2900, 400)
_, Byg, _ = fld(np.zeros_like(zg), np.zeros_like(zg), zg)
zb = np.linspace(-200, 2900, 32)
fig, ax = plt.subplots(figsize=(9.5, 4.8))
zmid = z0 + 0.5 * dz
for code in [2, 3]:
    m = Lt == code
    for exp, lab, col, ls in [("vf_resid_h96", "P2 baseline h96", LEG_COL[code], "--"),
                              ("vf_jac_h96", "P3 +J", LEG_COL[code], "-")]:
        c, v = profile_median(zmid[m], np.abs(ERR[exp][m, 0]) * 1e3, zb)
        ax.plot(c, v, color=col, lw=1.8, ls=ls,
                label=f"leg {LEG_NAME[code].split()[0]}  {lab}")
ax.set_yscale("log")
ax.set_xlabel("leg mid-point z  [mm]")
ax.set_ylabel("median |Δx|  [µm]")
ax.set_title("Short-leg (C/D) error vs position along z: the correction network "
             "receives no z₀ or Δz input,\nso it cannot switch off where "
             "the field vanishes nor adapt where the fringe rises")
ax2 = ax.twinx()
ax2.plot(zg, np.abs(Byg), color="#c22f2f", lw=1.3, alpha=0.7)
ax2.set_ylabel("|B_y(0,0,z)|  [T]", color="#c22f2f")
ax2.tick_params(axis="y", colors="#c22f2f")
ax2.grid(False)
ax.legend(frameon=False, fontsize=9, loc="upper left")
save(fig, "fig_R7_fringe_zscan.png")

# ---------------------------------------------------------------- R8 tier-2
print("R8: Jacobian relF CDFs (50k subsample, 2 models) ...")
Jmm = np.load(LAB / "data" / "vf_corpus_10M_J.npy", mmap_mode="r")
rng = np.random.default_rng(7)
sel = rng.choice(len(idx), 200_000, replace=False)[:50_000]
Xs = Xt[sel]
Ls = Lt[sel]
Jl4 = np.asarray(Jmm[idx[sel], :4, :], dtype=np.float64)
den = np.linalg.norm(Jl4.reshape(len(Jl4), -1), axis=1)
relf = {}
for exp in ["vf_resid_h96", "vf_jac_h96"]:
    m = load_model(exp)
    r = np.empty(len(Xs))
    for i in range(0, len(Xs), 8192):
        xb = torch.from_numpy(Xs[i:i + 8192])
        Jm = _model_state_jacobian(m, xb).detach().numpy().astype(np.float64)
        r[i:i + 8192] = np.linalg.norm(
            (Jm - Jl4[i:i + 8192]).reshape(len(Jm), -1), axis=1) / den[i:i + 8192]
    relf[exp] = r
    print("  ", exp, "done")
Jsl = np.zeros_like(Jl4)
for d4 in range(4):
    Jsl[:, d4, d4] = 1.0
Jsl[:, 0, 2] = Xs[:, 6].astype(np.float64)
Jsl[:, 1, 3] = Xs[:, 6].astype(np.float64)
relf_sl = np.linalg.norm((Jsl - Jl4).reshape(len(Jsl), -1), axis=1) / den

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for code, ax in zip(range(4), axes.ravel()):
    m = Ls == code
    cdf(ax, relf["vf_resid_h96"][m], clip=1e-8, color="#2a78d6", lw=1.7,
        label="P2 baseline h96 (no J loss)" if code == 0 else None)
    cdf(ax, relf["vf_jac_h96"][m], clip=1e-8, color="#eda100", lw=1.7,
        label="P3 +J supervision" if code == 0 else None)
    cdf(ax, relf_sl[m], clip=1e-8, color=SL_COL, lw=1.6, ls="--",
        label="straight-line Jacobian" if code == 0 else None)
    ax.axvline(0.01, color="#c22f2f", lw=1.0, ls=":")
    if code == 0:
        ax.text(0.011, 0.06, "spec bar 0.01", color="#c22f2f", fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1e1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("relative Frobenius error of J (rows x,y,tx,ty)")
    ax.set_ylabel("fraction of tracks")
    ax.set_title(f"leg {LEG_NAME[code]}   (n = {m.sum():,})")
fig.suptitle("Tier-2 Jacobian fidelity: per-track "
             "‖J_model − J_exact‖ / ‖J_exact‖ "
             "(50k test subsample)", y=1.0)
fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout()
save(fig, "fig_R8_jac_relf_cdf.png")

# ---------------------------------------------------------------- R9 trade-off
print("R9: leg-B trade-off")
fig, ax = plt.subplots(figsize=(7.5, 5))
for exp, lab, col in MODELS:
    x = t1[exp]["legs"]["B T->vtx"]["nn"]["med_dx_um"] / 1e3
    y = t1[exp]["tier2"]["legs"]["B T->vtx"]["med_relF"]
    ax.plot(x, y, "o", color=col, ms=11)
    ax.annotate(lab, (x, y), textcoords="offset points", xytext=(10, 4),
                fontsize=9)
xsl = t1["vf_jac_h96"]["legs"]["B T->vtx"]["straight"]["med_dx_um"] / 1e3
ysl = t1["vf_jac_h96"]["tier2"]["legs"]["B T->vtx"]["straight_med_relF"]
ax.plot(xsl, ysl, "v", color=SL_COL, ms=11)
ax.annotate("straight-line transport", (xsl, ysl), textcoords="offset points",
            xytext=(-115, 8), fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("leg-B median endpoint |Δx|  [mm]")
ax.set_ylabel("leg-B median Jacobian rel. Frobenius error")
ax.set_title("Leg B (T→vertex, full magnet): J supervision buys Jacobian "
             "fidelity\nat the price of endpoint accuracy — "
             "the std-normalised λ_J=100 trade-off")
save(fig, "fig_R9_legB_tradeoff.png")

# ---------------------------------------------------------------- P4 figures
print("P4 figures ...")
T = np.load(LAB / "results" / "VF_p4_fit_toys_20260703.npz")
p4 = json.load(open(LAB / "results" / "VF_p4_fit_20260703.json"))

# R10 residuals
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
ranges = [(-8, 8), (-8, 8), (-60, 60)]
units = ["mm", "mm", "mm"]
for k, (ax, cname) in enumerate(zip(axes, "xyz")):
    for arm in ["LIN", "NN", "RK"]:
        v = T[f"{arm}_dv"][:, k]
        ax.hist(np.clip(v, *ranges[k]), bins=90, range=ranges[k],
                histtype="step", lw=1.7, color=ARM_COL[arm],
                label=ARM_LAB[arm] if k == 0 else None)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(f"fitted − true vertex {cname}  [{units[k]}]")
    ax.set_ylabel("toys / bin")
    ax.set_title(f"vertex {cname} residual")
fig.suptitle("P4 fit emulation: vertex residuals, 2000 identical two-track toys "
             "per arm (entries outside range piled into edge bins)")
fig.legend(loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))
fig.tight_layout()
save(fig, "fig_R10_p4_residuals.png")

# R11 pulls
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
gx = np.linspace(-6, 6, 300)
gauss = np.exp(-0.5 * gx ** 2) / np.sqrt(2 * np.pi)
for k, (ax, cname, skey) in enumerate(zip(axes, "xyz", ["sx", "sy", "sz"])):
    for arm in ["LIN", "NN", "RK"]:
        pull = T[f"{arm}_dv"][:, k] / T[f"{arm}_{skey}"]
        ax.hist(np.clip(pull, -6, 6), bins=72, range=(-6, 6), density=True,
                histtype="step", lw=1.7, color=ARM_COL[arm],
                label=ARM_LAB[arm] if k == 0 else None)
    ax.plot(gx, gauss, color="k", lw=1.0, ls=":",
            label="N(0,1)" if k == 0 else None)
    ax.set_xlabel(f"pull: Δ{cname} / σ_{cname}(fit)")
    ax.set_ylabel("density")
    ax.set_title(f"vertex {cname} pull")
fig.suptitle("Pull distributions: is the fitted covariance honest? "
             "(RK ≈ unit Gaussian; NN over-narrow — covariance ignores "
             "surrogate error; LIN far off)")
fig.legend(loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.07))
fig.tight_layout()
save(fig, "fig_R11_p4_pulls.png")

# R12 chi2
fig, ax = plt.subplots(figsize=(8, 4.6))
for arm in ["RK", "NN", "LIN"]:
    cdf(ax, T[f"{arm}_chi2"], clip=1e-4, color=ARM_COL[arm], lw=1.8,
        label=ARM_LAB[arm])
from scipy import stats as sps
xq = np.geomspace(1e-4, 300, 400)
ax.plot(xq, sps.chi2.cdf(xq, df=1), color="k", ls=":", lw=1.2,
        label="χ²(1 dof) expectation")
ax.set_xscale("log")
ax.set_xlim(1e-3, 300)
ax.set_xlabel("fit χ²  (2 tracks × 2 residuals − 3 parameters "
              "= 1 dof)")
ax.set_ylabel("fraction of toys")
ax.set_title("Fit χ²: the RK arm matches the ideal χ²(1) law; "
             "the NN arm inflates mildly; no field inflates 18×")
ax.legend(frameon=False, fontsize=9, loc="upper left")
save(fig, "fig_R12_p4_chi2.png")

# R13 summary forest
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
coords = ["x", "y", "z"]
ypos = np.arange(3)[::-1]
off = {"RK": 0.22, "NN": 0.0, "LIN": -0.22}
for arm in ["RK", "NN", "LIN"]:
    b = np.array(p4["arms"][arm]["bias_mm"]) * 1e3
    r = np.array(p4["arms"][arm]["res_mm"]) * 1e3
    axes[0].errorbar(b, ypos + off[arm], xerr=r / np.sqrt(2000) * 1.2533,
                     fmt="o", color=ARM_COL[arm], ms=6, capsize=3,
                     label=ARM_LAB[arm])
    axes[1].barh(ypos + off[arm], r, height=0.2, color=ARM_COL[arm])
rk_b = np.array(p4["arms"]["RK"]["bias_mm"]) * 1e3
bars = np.array([bb["bar_mm"] for bb in p4["tier3_bars"]]) * 1e3
for k in range(3):
    axes[0].barh(ypos[k], 2 * bars[k], left=rk_b[k] - bars[k], height=0.75,
                 color="#2a78d6", alpha=0.12, zorder=0)
axes[0].axvline(0, color="k", lw=0.7)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels(coords)
axes[0].set_xlabel("vertex bias (median residual)  [µm]")
axes[0].set_xlim(-400, 1300)
axes[0].set_title("bias ± statistical error; shaded band = tier-3 "
                  "acceptance around the RK bias")
axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
axes[1].set_yticks(ypos)
axes[1].set_yticklabels(coords)
axes[1].set_xscale("log")
axes[1].set_xlabel("robust resolution (1.4826·MAD)  [µm]")
axes[1].set_title("resolution")
fig.suptitle("P4 fit-level summary: the NN sits inside the tier-3 acceptance on "
             "all three coordinates")
fig.tight_layout()
save(fig, "fig_R13_p4_summary.png")

# R14 paired NN-RK deltas
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
rangesd = [(-3, 3), (-3, 3), (-15, 15)]
for k, (ax, cname) in enumerate(zip(axes, "xyz")):
    dd = T["NN_dv"][:, k] - T["RK_dv"][:, k]
    ax.hist(np.clip(dd, *rangesd[k]), bins=80, range=rangesd[k],
            histtype="stepfilled", color="#2a78d6", alpha=0.65)
    ax.axvline(0, color="k", lw=0.7)
    ax.axvline(np.median(dd), color="#c22f2f", lw=1.4, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel(f"Δ{cname}(NN) − Δ{cname}(RK)  [mm]")
    ax.set_ylabel("toys / bin")
    ax.set_title(f"{cname}: median {np.median(dd)*1e3:+.1f} µm, "
                 f"spread {1.4826*np.median(np.abs(dd-np.median(dd)))*1e3:.0f} "
                 "µm")
fig.suptitle("Per-toy paired difference NN − RK on identical toys "
             "(red dashed = median): the surrogate scatters but does not shift "
             "the vertex")
fig.tight_layout()
save(fig, "fig_R14_p4_nn_rk_delta.png")

# ---------------------------------------------------------------- R15 preview
print("R15: in-flight matrix preview")
INFLIGHT = [
    ("train_vf_jacL10_h96.log", "P3b  J std-norm, λ=10", "#8d6cc3"),
    ("train_vf_jacrow_h96.log", "P3b  J row-norm, λ=40k", "#c22f2f"),
    ("train_vf_zfeat_h96.log", "P3b  z-features, no J", "#008300"),
    ("train_vf_zfeat_jacrow_h96.log", "P3b  z-features + J row-norm", "#eda100"),
]
fig, ax = plt.subplots(figsize=(9, 5))
ep, med = parse_log(LOGMAP["vf_jac_h96"])
ax.plot(ep, med, color="#999999", lw=1.5, label="P3 +J (converged reference)")
for fn, lab, col in INFLIGHT:
    ep, med = parse_log(fn)
    ax.plot(ep, med, color=col, lw=1.9, label=lab + f"  (epoch {ep.max()})")
ax.set_yscale("log")
ax.set_xlabel("epoch")
ax.set_ylabel("bulk-validation median |Δx|  [µm]")
ax.set_title("PRELIMINARY — P3b attribution matrix, still training "
             "(snapshot 2026-07-03 22:00).\nGiving the correction network its "
             "z-coordinates drops the bulk median by an order of magnitude.")
ax.legend(frameon=False, fontsize=9)
ax.text(0.985, 0.03, "IN TRAINING — numbers will change", fontsize=13,
        color="#c22f2f", alpha=0.55, ha="right", transform=ax.transAxes,
        fontweight="bold")
save(fig, "fig_R15_matrix_preview.png")

print("\nall figures written to", OUT)
