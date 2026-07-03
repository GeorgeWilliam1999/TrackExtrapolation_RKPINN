#!/usr/bin/env python3
"""Vertex-fit corpus generator v1 — implements P0 of the PLAN spec
"Vertex-Fit Extrapolation Corpus & Metrics (v1)" (Notion 3915d544-b9d9-81b7).

Contract (track-state -> vertex-z, NOT vertex -> planes):
  X[N,7]   = [x, y, tx, ty, qop, z0, dz]      (Allen/Gaudi units, gen-4 layout)
  Y[N,5]   = [x, y, tx, ty, qop] at z1=z0+dz  (qop passthrough)
  J[N,5,5] = d(out5)/d(in5)                   (covariance transport, C' = J C J^T)
  LEG[N]   = leg code (0=A,1=B,2=C,3=D)

Legs (spec section 4):
  A 55%  downstream->vertex : z0 ~ U[2300,2700] (UT band),  z1 in [0,2300]
  B 15%  T-track->vertex    : z0 ~ U[7500,9410] (T band),   z1 in [0,2300]
  C 15%  VELO/long->vertex  : z0 ~ U[0,770],                z1 in [-200,800]
  D 15%  intra-band hops    : z0 = z_v +- U[10,500],        z1 near z_v in [0,2300]

Physical states: spawn at a vertex-region origin (beam envelope + displaced,
late-decay-weighted for A/B), draw acceptance angles + soft momentum spectrum,
RK-propagate OUTWARD to the z0 plane, use the arrival state as the input row.
Iteration-basin targets: 3 z1 per block — the seed vertex z plus two
Laplace(0,30mm) perturbations clipped to +-200mm (block-shared so propagation
stays vectorised; variety comes from the >1e5 blocks).

Truth engine: the gen-4 fp64 RK4 (5 mm fixed step) with the v8r1.down field and
kappa = 1e-3*qop — deriv/rk4 duplicated verbatim from generate_data_v2.py but
with the field as an argument (G0 asserts bit-level agreement with the
original). Jacobians: fp64 torch port of the SAME integrator + SAME trilinear
interpolation, differentiated with autograd (5 batched backward passes).

Pre-generation gates (--selftest; ALL must pass before any Condor submission):
  G0  numpy-engine equivalence vs generate_data_v2.rk4 (real field)
  G0b torch-vs-numpy endpoint equivalence (real field)
  G1  uniform-By: RK4 endpoint vs EXACT arc closed form (incl. 14.990 um spot)
  G2  uniform-Bx: RK4 endpoint vs EXACT closed form
  G3  Jacobian: autodiff(RK4) vs autodiff(exact closed form), full 5x5,
      rel Frobenius < 1e-6 (uniform By)
  G4  FD cross-check: central finite differences vs autodiff J on real-field
      legs (the spec's 1%-subsample check, wired here at small n)

Closed forms used by G1-G3 (derived for THIS EOM; sign conventions therefore
self-consistent with Allen's gamma(), not with the retired gen-3 note):
  Uniform By, invariant lambda = ty/sqrt(1+tx^2)  (ty = lambda*sqrt(1+tx^2)):
    with s = tx/sqrt(1+tx^2) = sin(theta), kt = kappa*qop*By*sqrt(1+lambda^2):
      s1 = s0 - kt*dz
      x1 = x0 + (cos(th1) - cos(th0)) / kt
      y1 = y0 - lambda*(th1 - th0) / kt
      tx1 = s1/sqrt(1-s1^2),  ty1 = lambda*sqrt(1+tx1^2)
  Uniform Bx is the mirror (mu = tx/sqrt(1+ty^2), kh = kappa*qop*Bx*sqrt(1+mu^2)):
      s1 = s0 + kh*dz  (s = ty/sqrt(1+ty^2) = sin(phi))
      y1 = y0 - (cos(ph1) - cos(ph0)) / kh
      x1 = x0 + mu*(ph1 - ph0) / kh
      ty1 = s1/sqrt(1-s1^2),  tx1 = mu*sqrt(1+ty1^2)

Usage:
  generate_vertexfit_v1.py --selftest
  generate_vertexfit_v1.py <shard_id> <n_tracks> <out_dir>
Seed: 41000 + shard_id*7919.
"""
from __future__ import annotations

import ctypes
import json
import site
import subprocess
import sys
import time
from pathlib import Path

# The user-site sympy under ~/.local is broken (missing sympy.core.sympify)
# and shadows conda's; torch.func lazily imports it via torch._dynamo. Strip
# the user site so this script is self-contained on login nodes AND Condor.
try:
    _usersite = site.getusersitepackages()
    sys.path[:] = [p for p in sys.path if p != _usersite]
except Exception:
    pass

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "core"))
sys.path.insert(0, str(HERE))
from field_v8r1 import FieldV8R1  # noqa: E402

KAPPA = 1.0e-3
C_QP = 0.299792458
STEP = 5.0
BLOCK = 64          # tracks per block (share z_v-plane, z0-plane, 3 z1 targets)
N_TARGETS = 3       # z1 values per block: seed + 2 basin perturbations
MIN_LEG_MM = 5.0    # below this the consumer transports linearly anyway
LEG_CODE = {"A": 0, "B": 1, "C": 2, "D": 3}
LEG_FRAC = {"A": 0.55, "B": 0.15, "C": 0.15, "D": 0.15}

torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------
# numpy truth engine — verbatim gen-4 EOM/RK4, field passed as argument
# ----------------------------------------------------------------------------
def deriv_np(S, z, field):
    x, y, tx, ty, qop = S.T
    Bx, By, Bz = field(x, y, np.full_like(x, z))
    k = KAPPA * qop
    N = np.sqrt(1 + tx * tx + ty * ty)
    return np.stack([tx, ty,
                     k * N * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz),
                     k * N * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz),
                     np.zeros_like(x)], axis=1)


def rk4_np(S, z0, z1, field, step=STEP):
    S = S.astype(np.float64).copy()
    z = float(z0)
    if z1 == z0:
        return S
    h = step if z1 > z0 else -step
    while (z1 - z) * np.sign(h) > abs(h):
        k1 = deriv_np(S, z, field); k2 = deriv_np(S + 0.5 * h * k1, z + 0.5 * h, field)
        k3 = deriv_np(S + 0.5 * h * k2, z + 0.5 * h, field); k4 = deriv_np(S + h * k3, z + h, field)
        S = S + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4); z += h
    r = z1 - z
    if abs(r) > 1e-12:
        k1 = deriv_np(S, z, field); k2 = deriv_np(S + 0.5 * r * k1, z + 0.5 * r, field)
        k3 = deriv_np(S + 0.5 * r * k2, z + 0.5 * r, field); k4 = deriv_np(S + r * k3, z + r, field)
        S = S + (r / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return S


class ConstFieldNP:
    def __init__(self, Bx, By, Bz):
        self.B = (float(Bx), float(By), float(Bz))

    def __call__(self, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        return (np.full_like(x, self.B[0]), np.full_like(x, self.B[1]),
                np.full_like(x, self.B[2]))


# ----------------------------------------------------------------------------
# torch engine — same interpolation formula, same RK4, differentiable
# ----------------------------------------------------------------------------
class TorchFieldV8R1:
    """fp64 torch mirror of FieldV8R1.__call__ (same trunc/clip semantics)."""

    def __init__(self, f: FieldV8R1):
        self.Bx = torch.from_numpy(f.Bx); self.By = torch.from_numpy(f.By)
        self.Bz = torch.from_numpy(f.Bz)
        self.invD = torch.from_numpy(f.invD); self.min = torch.from_numpy(f.min)
        self.N = torch.from_numpy(f.N)
        self.scale = float(f.scale)

    def __call__(self, x, y, z):
        fx = (x - self.min[0]) * self.invD[0]
        fy = (y - self.min[1]) * self.invD[1]
        fz = (z - self.min[2]) * self.invD[2]
        # .long() truncates toward zero — matches numpy .astype(int64)
        ix = torch.clamp(fx.long(), 0, int(self.N[0]) - 2)
        iy = torch.clamp(fy.long(), 0, int(self.N[1]) - 2)
        iz = torch.clamp(fz.long(), 0, int(self.N[2]) - 2)
        tx = torch.clamp(fx - ix, 0.0, 1.0); ty = torch.clamp(fy - iy, 0.0, 1.0)
        tz = torch.clamp(fz - iz, 0.0, 1.0)

        def tri(G):
            c000 = G[iz, iy, ix];         c100 = G[iz, iy, ix + 1]
            c010 = G[iz, iy + 1, ix];     c110 = G[iz, iy + 1, ix + 1]
            c001 = G[iz + 1, iy, ix];     c101 = G[iz + 1, iy, ix + 1]
            c011 = G[iz + 1, iy + 1, ix]; c111 = G[iz + 1, iy + 1, ix + 1]
            c00 = c000 * (1 - tx) + c100 * tx
            c10 = c010 * (1 - tx) + c110 * tx
            c01 = c001 * (1 - tx) + c101 * tx
            c11 = c011 * (1 - tx) + c111 * tx
            c0 = c00 * (1 - ty) + c10 * ty
            c1 = c01 * (1 - ty) + c11 * ty
            return (c0 * (1 - tz) + c1 * tz) * self.scale

        return tri(self.Bx), tri(self.By), tri(self.Bz)


class ConstFieldT:
    def __init__(self, Bx, By, Bz):
        self.B = (Bx, By, Bz)

    def __call__(self, x, y, z):
        o = torch.ones_like(x)
        return (o * self.B[0], o * self.B[1], o * self.B[2])


def deriv_t(S, z, field):
    x, y, tx, ty, qop = S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4]
    Bx, By, Bz = field(x, y, torch.full_like(x, z))
    k = KAPPA * qop
    N = torch.sqrt(1 + tx * tx + ty * ty)
    return torch.stack([tx, ty,
                        k * N * (tx * ty * Bx - (1 + tx * tx) * By + ty * Bz),
                        k * N * ((1 + ty * ty) * Bx - tx * ty * By - tx * Bz),
                        torch.zeros_like(x)], dim=1)


def rk4_t(S, z0, z1, field, step=STEP):
    z = float(z0)
    if z1 == z0:
        return S
    h = step if z1 > z0 else -step
    while (z1 - z) * np.sign(h) > abs(h):
        k1 = deriv_t(S, z, field); k2 = deriv_t(S + 0.5 * h * k1, z + 0.5 * h, field)
        k3 = deriv_t(S + 0.5 * h * k2, z + 0.5 * h, field); k4 = deriv_t(S + h * k3, z + h, field)
        S = S + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4); z += h
    r = z1 - z
    if abs(r) > 1e-12:
        k1 = deriv_t(S, z, field); k2 = deriv_t(S + 0.5 * r * k1, z + 0.5 * r, field)
        k3 = deriv_t(S + 0.5 * r * k2, z + 0.5 * r, field); k4 = deriv_t(S + r * k3, z + r, field)
        S = S + (r / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return S


def jacobian_t(S_np, z0, z1, field_t):
    """J[b,i,j] = d out_i / d in_j via 5 forward-mode JVP sweeps (fp64).

    Forward mode carries dual numbers through the step loop instead of
    retaining a reverse graph over O(1000) RK steps — O(10x) lighter in time
    and memory for the long legs. Batch rows are independent, so a per-row
    tangent e_j yields column j for every row in one sweep.
    """
    S = torch.from_numpy(np.ascontiguousarray(S_np))
    B = S.shape[0]
    # one sweep for all 5 columns: stack the tangent directions along batch
    S_rep = S.repeat(5, 1)
    E = torch.zeros(5 * B, 5)
    for j in range(5):
        E[j * B:(j + 1) * B, j] = 1.0
    Y_rep, Jflat = torch.func.jvp(lambda s: rk4_t(s, z0, z1, field_t), (S_rep,), (E,))
    # Jflat[j*B + b, :] = dY[b]/d in_j  ->  J[b, out, in]
    J = Jflat.reshape(5, B, 5).permute(1, 2, 0)
    return J.numpy(), Y_rep[:B].numpy()


# ----------------------------------------------------------------------------
# C++ engine (vf_engine.cpp) — the production path: RK4 + discrete-tangent
# Jacobian, OpenMP over rows, per-row (z0, z1). ~1000x the torch JVP path.
# numpy/torch engines above are kept as REFERENCE implementations; the C++
# engine must pass every gate they pass (G0c..G4c, G5 in --selftest).
# ----------------------------------------------------------------------------
class CppEngine:
    def __init__(self):
        src, lib = HERE / "vf_engine.cpp", HERE / "vf_engine.so"
        if (not lib.exists()) or lib.stat().st_mtime < src.stat().st_mtime:
            subprocess.run(["g++", "-O3", "-march=native", "-fopenmp", "-shared",
                            "-fPIC", str(src), "-o", str(lib)], check=True)
        L = ctypes.CDLL(str(lib))
        L.vf_field_load.restype = ctypes.c_void_p
        L.vf_field_load.argtypes = [ctypes.c_char_p]
        L.vf_field_const.restype = ctypes.c_void_p
        L.vf_field_const.argtypes = [ctypes.c_double] * 3
        L.vf_propagate.restype = ctypes.c_int
        PD = ctypes.POINTER(ctypes.c_double)
        L.vf_propagate.argtypes = [ctypes.c_void_p, PD, PD, PD, ctypes.c_int,
                                   ctypes.c_double, PD, PD]
        self.L = L
        from field_v8r1 import V8R1_DOWN
        self.fld = L.vf_field_load(V8R1_DOWN.encode())
        if not self.fld:
            raise RuntimeError("vf_engine: failed to load v8r1 field")

    def const_field(self, bx, by, bz):
        return self.L.vf_field_const(bx, by, bz)

    def propagate(self, S, z0, z1, want_J=True, fld=None):
        """S[n,5], z0[n], z1[n] -> (Y[n,5], J[n,5,5]|None)."""
        S = np.ascontiguousarray(S, dtype=np.float64)
        n = S.shape[0]
        z0 = np.ascontiguousarray(np.broadcast_to(np.asarray(z0, np.float64), (n,)))
        z1 = np.ascontiguousarray(np.broadcast_to(np.asarray(z1, np.float64), (n,)))
        Y = np.empty((n, 5))
        J = np.empty((n, 25)) if want_J else None
        PD = ctypes.POINTER(ctypes.c_double)
        rc = self.L.vf_propagate(
            fld if fld is not None else self.fld,
            S.ctypes.data_as(PD), z0.ctypes.data_as(PD), z1.ctypes.data_as(PD),
            n, STEP, Y.ctypes.data_as(PD),
            J.ctypes.data_as(PD) if want_J else None)
        if rc != 0:
            raise RuntimeError("vf_propagate failed")
        return Y, (J.reshape(n, 5, 5) if want_J else None)


# ----------------------------------------------------------------------------
# exact closed forms in uniform fields (torch, differentiable) — see docstring
# ----------------------------------------------------------------------------
def closed_form_By(S, dz, By):
    x0, y0, tx0, ty0, qop = S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4]
    lam = ty0 / torch.sqrt(1 + tx0 * tx0)
    kt = KAPPA * qop * By * torch.sqrt(1 + lam * lam)
    s0 = tx0 / torch.sqrt(1 + tx0 * tx0)
    s1 = s0 - kt * dz
    th0 = torch.asin(s0); th1 = torch.asin(s1)
    x1 = x0 + (torch.cos(th1) - torch.cos(th0)) / kt
    y1 = y0 - lam * (th1 - th0) / kt
    tx1 = s1 / torch.sqrt(1 - s1 * s1)
    ty1 = lam * torch.sqrt(1 + tx1 * tx1)
    return torch.stack([x1, y1, tx1, ty1, qop], dim=1)


def closed_form_Bx(S, dz, Bx):
    x0, y0, tx0, ty0, qop = S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4]
    mu = tx0 / torch.sqrt(1 + ty0 * ty0)
    kh = KAPPA * qop * Bx * torch.sqrt(1 + mu * mu)
    s0 = ty0 / torch.sqrt(1 + ty0 * ty0)
    s1 = s0 + kh * dz
    ph0 = torch.asin(s0); ph1 = torch.asin(s1)
    y1 = y0 - (torch.cos(ph1) - torch.cos(ph0)) / kh
    x1 = x0 + mu * (ph1 - ph0) / kh
    ty1 = s1 / torch.sqrt(1 - s1 * s1)
    tx1 = mu * torch.sqrt(1 + ty1 * ty1)
    return torch.stack([x1, y1, tx1, ty1, qop], dim=1)


def jacobian_closed_form(S_np, dz, fn, Bcomp):
    S = torch.from_numpy(np.ascontiguousarray(S_np)).requires_grad_(True)
    Y = fn(S, dz, Bcomp)
    rows = []
    for k in range(5):
        g, = torch.autograd.grad(Y[:, k].sum(), S, retain_graph=(k < 4))
        rows.append(g)
    return torch.stack(rows, dim=1).detach().numpy(), Y.detach().numpy()


# ----------------------------------------------------------------------------
# samplers (spec section 4)
# ----------------------------------------------------------------------------
def _sample_kinematics(rng, n):
    """Soft momentum spectrum + acceptance angles at the vertex."""
    soft = rng.random(n) < 0.60
    p = np.where(soft,
                 np.exp(rng.uniform(np.log(3.0), np.log(15.0), n)),
                 np.exp(rng.uniform(np.log(2.0), np.log(100.0), n)))
    qop = rng.choice([-1.0, 1.0], n) * C_QP / p
    # acceptance: resample slopes until 0.015 < |t| < 0.39 (eta ~ 2..4.9)
    tx = np.empty(n); ty = np.empty(n)
    todo = np.ones(n, dtype=bool)
    while todo.any():
        m = int(todo.sum())
        tx[todo] = rng.uniform(-0.30, 0.30, m)
        ty[todo] = rng.uniform(-0.25, 0.25, m)
        r = np.hypot(tx, ty)
        todo = (r < 0.015) | (r > 0.39)
    return p, qop, tx, ty


def _vertex_origin(rng, leg):
    """Block-shared vertex z; per-track transverse spread grows with z."""
    if leg in ("A", "B"):        # displaced decays, weighted toward late z
        z_v = rng.uniform(0.0, 2300.0) if rng.random() < 0.5 else rng.uniform(1150.0, 2300.0)
    elif leg == "C":             # PV-dominated
        z_v = float(np.clip(rng.normal(0.0, 50.0), -200.0, 800.0)) if rng.random() < 0.7 \
            else rng.uniform(-200.0, 800.0)
    else:                        # D: anywhere in the vertex band
        z_v = rng.uniform(0.0, 2300.0)
    return z_v


def _z0_plane(rng, leg, z_v):
    if leg == "A":
        return rng.uniform(2300.0, 2700.0)
    if leg == "B":
        return rng.uniform(7500.0, 9410.0)
    if leg == "C":
        return rng.uniform(0.0, 770.0)
    # D: short hop away from the vertex, either direction
    return float(np.clip(z_v + rng.choice([-1.0, 1.0]) * rng.uniform(10.0, 500.0), 0.0, 2700.0))


def make_block(rng, leg, eng: CppEngine, n=BLOCK):
    """One block: physical states at a shared z0 plane, PER-TRACK z1 targets.

    v1.1 (C++ engine): targets are per-track (seed vertex + 2 Laplace(30 mm)
    perturbations each, clipped to +-200 mm and the leg band) — the per-row
    (z0, z1) API removes the v1.0 block-shared-target compromise.
    """
    z_v = _vertex_origin(rng, leg)
    z0 = _z0_plane(rng, leg, z_v)
    p, qop, tx, ty = _sample_kinematics(rng, n)
    sig = 0.5 + 2.0 * max(z_v, 0.0) / 2300.0
    Sv = np.stack([rng.normal(0.0, sig, n), rng.normal(0.0, sig, n), tx, ty, qop], axis=1)
    S0, _ = eng.propagate(Sv, z_v, z0, want_J=False)
    ok = (np.all(np.isfinite(S0), axis=1) & (np.abs(S0[:, 0]) < 3900)
          & (np.abs(S0[:, 1]) < 3900) & (np.abs(S0[:, 2]) < 0.6) & (np.abs(S0[:, 3]) < 0.6))
    S0, p = S0[ok], p[ok]
    m = S0.shape[0]
    if m == 0:
        return None
    lo, hi = (-200.0, 800.0) if leg == "C" else (0.0, 2300.0)
    Z1 = np.empty((m, N_TARGETS))
    Z1[:, 0] = np.clip(z_v, lo, hi)
    Z1[:, 1:] = np.clip(z_v + np.clip(rng.laplace(0.0, 30.0, (m, N_TARGETS - 1)),
                                      -200.0, 200.0), lo, hi)
    Srep = np.repeat(S0, N_TARGETS, axis=0)
    prep = np.repeat(p, N_TARGETS)
    z1r = Z1.reshape(-1)
    keep = np.abs(z1r - z0) >= MIN_LEG_MM
    if not keep.any():
        return None
    Srep, prep, z1r = Srep[keep], prep[keep], z1r[keep]
    Y, J = eng.propagate(Srep, z0, z1r, want_J=True)
    ok2 = (np.all(np.isfinite(Y), axis=1) & np.all(np.isfinite(J.reshape(-1, 25)), axis=1)
           & (np.abs(Y[:, 0]) < 3900) & (np.abs(Y[:, 1]) < 3900))
    if not ok2.any():
        return None
    Srep, prep, z1r, Y, J = Srep[ok2], prep[ok2], z1r[ok2], Y[ok2], J[ok2]
    mm = Srep.shape[0]
    X = np.concatenate([Srep[:, :5], np.full((mm, 1), z0), (z1r - z0)[:, None]], axis=1)
    Y5 = np.concatenate([Y[:, :4], Srep[:, 4:5]], axis=1)
    return X, Y5, J, prep


# ----------------------------------------------------------------------------
# FD cross-check (spec section 5: central differences on the numpy engine)
# ----------------------------------------------------------------------------
FD_EPS = np.array([1e-3, 1e-3, 1e-6, 1e-6, 0.0])  # qop eps is relative, below


def jacobian_fd(S, z0, z1, field):
    n = S.shape[0]
    J = np.zeros((n, 5, 5))
    for j in range(5):
        eps = np.full(n, FD_EPS[j]) if j < 4 else np.abs(S[:, 4]) * 1e-4 + 1e-9
        Sp = S.copy(); Sp[:, j] += eps
        Sm = S.copy(); Sm[:, j] -= eps
        Yp = rk4_np(Sp, z0, z1, field); Ym = rk4_np(Sm, z0, z1, field)
        J[:, :, j] = (Yp - Ym) / (2 * eps[:, None])
    return J


def rel_frob(A, B):
    num = np.linalg.norm((A - B).reshape(A.shape[0], -1), axis=1)
    den = np.linalg.norm(B.reshape(B.shape[0], -1), axis=1)
    return num / den


# ----------------------------------------------------------------------------
# gates
# ----------------------------------------------------------------------------
def selftest():
    rng = np.random.default_rng(20260702)
    field_np = FieldV8R1()
    field_t = TorchFieldV8R1(field_np)
    print("field:", field_np.info())
    results = {}

    def report(name, value, tol, desc):
        ok = value < tol
        results[name] = (value, tol, ok)
        print(f"  {name}: {desc}: {value:.3e}  (tol {tol:.0e})  {'PASS' if ok else 'FAIL'}")

    # shared random states (vertex-band kinematics, moderate slopes)
    def rand_states(n):
        p, qop, tx, ty = _sample_kinematics(rng, n)
        return np.stack([rng.uniform(-50, 50, n), rng.uniform(-50, 50, n), tx, ty, qop], axis=1)

    # --- G0: numpy engine == generate_data_v2.rk4 (module-global v8r1 field)
    import generate_data_v2 as g4
    S = rand_states(256)
    legs = [(2500.0, 400.0), (8000.0, 1200.0), (500.0, -100.0)]
    worst = 0.0
    for z0, z1 in legs:
        a = rk4_np(S, z0, z1, field_np)
        b = g4.rk4(S, z0, z1)
        worst = max(worst, float(np.abs(a - b).max()))
    report("G0", worst, 1e-9, "max |numpy engine - gen4 engine| [mm]")

    # --- G0b: torch fp64 engine == numpy engine (real field)
    worst = 0.0
    for z0, z1 in legs:
        a = rk4_np(S, z0, z1, field_np)
        b = rk4_t(torch.from_numpy(S), z0, z1, field_t).numpy()
        worst = max(worst, float(np.abs(a - b).max()))
    report("G0b", worst, 1e-9, "max |torch engine - numpy engine| [mm]")

    # --- G1: uniform By — RK4 vs exact closed form (endpoint)
    By = -1.0
    fnp = ConstFieldNP(0.0, By, 0.0)
    S1 = rand_states(4096)
    Y_rk = rk4_np(S1, 0.0, 1000.0, fnp)
    Y_cf = closed_form_By(torch.from_numpy(S1), 1000.0, By).numpy()
    report("G1", float(np.abs(Y_rk[:, :4] - Y_cf[:, :4]).max()), 1e-6,
           "uniform-By arc: max |RK4 - exact| [mm]")
    # known-number spot check: q=+1, p=10 GeV, on-axis, dz=1000 mm, By=-1 T.
    # Exact arc: x_f = (1 - sqrt(1 - (k dz)^2)) / k with k = KAPPA*qop = 2.998e-5/mm
    # => 14.993 mm (small-angle 0.3*B*L^2/(2p) = 14.99 mm).  NOTE: the gen-3
    # GENERATION_SPEC.md section 6.1 quotes "14.990 um" for this observable —
    # that value is only obtained under the kappa=1e-6 bug convention its
    # section 1.1 prescribed; the historical closed-form gate therefore
    # validated the kappa x1000 bug rather than catching it (flagged in
    # Notion, 2026-07-02).  The physical answer is mm-scale.
    spot = np.array([[0.0, 0.0, 0.0, 0.0, C_QP / 10.0]])
    k_spot = KAPPA * (C_QP / 10.0)
    xf_exact = (1.0 - np.sqrt(1.0 - (k_spot * 1000.0) ** 2)) / k_spot
    xf = rk4_np(spot, 0.0, 1000.0, fnp)[0, 0]
    print(f"      spot: x_f = {xf:.6f} mm (exact arc {xf_exact:.6f} mm)")
    if abs(xf - xf_exact) > 1e-6:
        results["G1spot"] = (abs(xf - xf_exact), 1e-6, False)
        print("      G1spot FAIL")

    # --- G2: uniform Bx — RK4 vs exact closed form (endpoint)
    Bx = 1.0
    fnp = ConstFieldNP(Bx, 0.0, 0.0)
    Y_rk = rk4_np(S1, 0.0, 1000.0, fnp)
    Y_cf = closed_form_Bx(torch.from_numpy(S1), 1000.0, Bx).numpy()
    report("G2", float(np.abs(Y_rk[:, :4] - Y_cf[:, :4]).max()), 1e-6,
           "uniform-Bx helix: max |RK4 - exact| [mm]")

    # --- G3: Jacobian gate — autodiff(RK4) vs autodiff(exact), uniform By
    ft = ConstFieldT(0.0, -1.0, 0.0)
    S3 = rand_states(512)
    J_rk, _ = jacobian_t(S3, 0.0, 1000.0, ft)
    J_cf, _ = jacobian_closed_form(S3, 1000.0, closed_form_By, -1.0)
    report("G3", float(np.max(rel_frob(J_rk, J_cf))), 1e-6,
           "uniform-By: max relF |J_rk4 - J_exact|")

    # --- G4: FD cross-check on real-field legs (A-like and B-like)
    worst_med = 0.0
    for z0, z1 in [(2500.0, 400.0), (8000.0, 1200.0)]:
        S4 = rand_states(64)
        S4 = rk4_np(S4, z1, z0, field_np)          # physical states at the z0 plane
        J_ad, _ = jacobian_t(S4, z0, z1, field_t)
        J_fd = jacobian_fd(S4, z0, z1, field_np)
        worst_med = max(worst_med, float(np.median(rel_frob(J_ad, J_fd))))
    report("G4", worst_med, 1e-4, "real field: median relF |J_autodiff - J_fd|")

    # --- C++ engine gates: the production path must match everything above
    eng = CppEngine()
    worst = 0.0
    for z0, z1 in legs:
        a = rk4_np(S, z0, z1, field_np)
        b, _ = eng.propagate(S, z0, z1, want_J=False)
        worst = max(worst, float(np.abs(a - b).max()))
    report("G0c", worst, 1e-9, "max |cpp engine - numpy engine| [mm]")

    fBy = eng.const_field(0.0, -1.0, 0.0)
    fBx = eng.const_field(1.0, 0.0, 0.0)
    Yb, _ = eng.propagate(S1, 0.0, 1000.0, want_J=False, fld=fBy)
    Yc = closed_form_By(torch.from_numpy(S1), 1000.0, -1.0).numpy()
    report("G1c", float(np.abs(Yb[:, :4] - Yc[:, :4]).max()), 1e-6,
           "cpp uniform-By arc vs exact [mm]")
    Yb, _ = eng.propagate(S1, 0.0, 1000.0, want_J=False, fld=fBx)
    Yc = closed_form_Bx(torch.from_numpy(S1), 1000.0, 1.0).numpy()
    report("G2c", float(np.abs(Yb[:, :4] - Yc[:, :4]).max()), 1e-6,
           "cpp uniform-Bx helix vs exact [mm]")

    _, Jc = eng.propagate(S3, 0.0, 1000.0, want_J=True, fld=fBy)
    report("G3c", float(np.max(rel_frob(Jc, J_cf))), 1e-6,
           "cpp J vs exact closed-form J (uniform By)")

    worst_med = 0.0
    for z0, z1 in [(2500.0, 400.0), (8000.0, 1200.0)]:
        S4 = rand_states(64)
        S4, _ = eng.propagate(S4, z1, z0, want_J=False)
        _, Jx = eng.propagate(S4, z0, z1, want_J=True)
        J_fd = jacobian_fd(S4, z0, z1, field_np)
        worst_med = max(worst_med, float(np.median(rel_frob(Jx, J_fd))))
    report("G4c", worst_med, 1e-4, "cpp J vs central FD (real field)")

    S5 = rand_states(32)
    S5, _ = eng.propagate(S5, 400.0, 2500.0, want_J=False)
    _, Jx = eng.propagate(S5, 2500.0, 400.0, want_J=True)
    Jt, _ = jacobian_t(S5, 2500.0, 400.0, field_t)
    report("G5", float(np.max(rel_frob(Jx, Jt))), 1e-8,
           "max relF |cpp J - torch autodiff J| (real field)")

    ok = all(v[2] for v in results.values())
    print("SELFTEST:", "ALL GATES PASS" if ok else "GATE FAILURE")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# shard generation
# ----------------------------------------------------------------------------
def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    shard, n_tracks, out_dir = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    rng = np.random.default_rng(41000 + shard * 7919)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    field_np = FieldV8R1()
    eng = CppEngine()

    legs = list(LEG_FRAC)
    probs = np.array([LEG_FRAC[c] for c in legs])
    Xs, Ys, Js, Ls, Ps = [], [], [], [], []
    n_blocks = max(1, n_tracks // BLOCK)
    for ib in range(n_blocks):
        leg = rng.choice(legs, p=probs)
        tb = time.time()
        blk = make_block(rng, leg, eng)
        if blk is None:
            continue
        X, Y, J, P = blk
        Xs.append(X); Ys.append(Y); Js.append(J); Ps.append(P)
        Ls.append(np.full(X.shape[0], LEG_CODE[leg], dtype=np.uint8))
        print(f"block {ib+1}/{n_blocks} leg={leg} rows={X.shape[0]} "
              f"wall={time.time()-tb:.1f}s", flush=True)

    X = np.concatenate(Xs).astype(np.float32)
    Y = np.concatenate(Ys).astype(np.float32)
    J = np.concatenate(Js).astype(np.float32)
    L = np.concatenate(Ls)
    P = np.concatenate(Ps).astype(np.float32)

    # post-hoc gates on the shard: re-propagation + FD-vs-autodiff on a subsample
    idx = rng.choice(X.shape[0], size=min(200, X.shape[0]), replace=False)
    reprop_worst, fd_meds = 0.0, []
    for i in idx[:50]:
        S = X[i, :5].astype(np.float64)[None, :]
        z0, dz = float(X[i, 5]), float(X[i, 6])
        Yr = rk4_np(S, z0, z0 + dz, field_np)
        reprop_worst = max(reprop_worst, float(np.abs(Yr[0, :4] - Y[i, :4].astype(np.float64)).max()))
    for i in idx[:20]:
        # stored (fp32) cpp J vs fp64 central FD on the independent numpy
        # engine — fp32 storage caps this at ~1e-7; fp64-level equivalence of
        # the J path is proven by selftest gates G3c/G4c/G5.
        S = X[i, :5].astype(np.float64)[None, :]
        z0, dz = float(X[i, 5]), float(X[i, 6])
        J_fd = jacobian_fd(S, z0, z0 + dz, field_np)
        fd_meds.append(float(rel_frob(J[i].astype(np.float64)[None], J_fd)[0]))

    leg_counts = {c: int((L == LEG_CODE[c]).sum()) for c in legs}
    meta = dict(shard=shard, n=int(X.shape[0]), field="v8r1.down", kappa=KAPPA,
                seed=41000 + shard * 7919, spec="PLAN 3915d544-b9d9-81b7 v1.1",
                engine="cpp (vf_engine.cpp: RK4 + discrete-tangent J, OpenMP)",
                contract="track-state->vertex-z + Jacobian",
                leg_counts=leg_counts, block=BLOCK, n_targets=N_TARGETS,
                fwd_frac=float((X[:, 6] > 0).mean()),
                reprop_worst_mm=reprop_worst,
                jac_fd_relF_median=float(np.median(fd_meds)),
                wall_s=round(time.time() - t0, 1))
    np.savez_compressed(out_dir / f"vf_shard_{shard:04d}.npz", X=X, Y=Y, J=J, LEG=L, P=P)
    (out_dir / f"vf_shard_{shard:04d}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
