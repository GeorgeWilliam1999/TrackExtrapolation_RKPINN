#!/usr/bin/env python3
"""Generate figures from ParKalman notebook for the B-field characterisation LaTeX doc."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Callable
import os

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── State & field (from notebook) ────────────────────────────────────────────
C_LIGHT = 299.792458

@dataclass
class State:
    x: float = 0.0; y: float = 0.0; z: float = 0.0
    tx: float = 0.0; ty: float = 0.0; qop: float = 0.0
    def copy(self): return State(self.x, self.y, self.z, self.tx, self.ty, self.qop)

def derivative(state, B):
    tx, ty, qop = state.tx, state.ty, state.qop
    norm = np.sqrt(1.0 + tx*tx + ty*ty)
    Bx, By, Bz = B
    ax = norm * (ty * (tx * Bx + Bz) - (1.0 + tx*tx) * By)
    ay = norm * (-tx * (ty * By + Bz) + (1.0 + ty*ty) * Bx)
    return np.array([tx, ty, 1.0, qop * ax, qop * ay])

def magnetic_field(pos):
    x, y, z = pos
    By = -1.0 * np.exp(-0.5 * ((z - 5200.0) / 1800.0) ** 2)
    Bx = 0.02 * By * (y / 1000.0)
    Bz = 0.01 * By * (x / 1000.0)
    return np.array([Bx, By, Bz])

# ── Butcher tableaux ─────────────────────────────────────────────────────────
class RK4:
    N_stages = 4
    a = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1.0,0]])
    b = np.array([1/6, 1/3, 1/3, 1/6]); b_star = None

class CashKarp:
    N_stages = 6
    a = np.array([
        [0,0,0,0,0,0],[1/5,0,0,0,0,0],[3/40,9/40,0,0,0,0],
        [3/10,-9/10,6/5,0,0,0],[-11/54,5/2,-70/27,35/27,0,0],
        [1631/55296,175/512,575/13824,44275/110592,253/4096,0]])
    b = np.array([37/378,0,250/621,125/594,0,512/1771])
    b_star = np.array([2825/27648,0,18575/48384,13525/55296,277/14336,1/4])

# ── Extrapolators ────────────────────────────────────────────────────────────
def parabolic_propagate(state, dz, field_fn):
    x_mid = state.x + state.tx * dz * 0.5
    y_mid = state.y + state.ty * dz * 0.5
    z_mid = state.z + dz * 0.5
    B = field_fn(np.array([x_mid, y_mid, z_mid]))
    d = derivative(state, B)
    state.x += dz * (d[0] + 0.5 * dz * d[3])
    state.y += dz * (d[1] + 0.5 * dz * d[4])
    state.z += dz; state.tx += dz * d[3]; state.ty += dz * d[4]

def rk_propagate(state, dz, field_fn, Table=CashKarp):
    N = Table.N_stages; k = np.zeros((N, 5))
    for stage in range(N):
        s = state.copy()
        for i in range(stage):
            deriv = k[i] * Table.a[stage, i]
            s.x += deriv[0]; s.y += deriv[1]; s.z += deriv[2]
            s.tx += deriv[3]; s.ty += deriv[4]
        B = field_fn(np.array([s.x, s.y, s.z])); k[stage] = derivative(s, B) * dz
    for i in range(N):
        state.x += k[i,0]*Table.b[i]; state.y += k[i,1]*Table.b[i]
        state.z += k[i,2]*Table.b[i]; state.tx += k[i,3]*Table.b[i]; state.ty += k[i,4]*Table.b[i]

def gamma_nystrom(t, B):
    norm = np.sqrt(1.0 + t[0]**2 + t[1]**2)
    dtx = norm * (t[0]*t[1]*B[0] - (1 + t[0]**2)*B[1] + t[1]*B[2])
    dty = norm * ((1 + t[1]**2)*B[0] - t[0]*t[1]*B[1] - t[0]*B[2])
    return np.array([dtx, dty])

def rkn_fast_step(state, dz, field_fn):
    B = field_fn(np.array([state.x+0.5*state.tx*dz, state.y+0.5*state.ty*dz, state.z+0.5*dz]))
    tn = np.array([state.tx, state.ty]); c = [0.0, 0.5, 0.5, 1.0]
    k = np.zeros((4, 2)); k[0] = gamma_nystrom(tn, B) * state.qop
    for s in range(1, 4): k[s] = gamma_nystrom(tn + k[s-1]*(dz*c[s]), B) * state.qop
    dRn = tn*dz + (k[0]+k[1]+k[2])*(dz**2/6.0)
    dTn = (k[0]+2*k[1]+2*k[2]+k[3])*(dz/6.0)
    state.x += dRn[0]; state.y += dRn[1]; state.z += dz
    state.tx += dTn[0]; state.ty += dTn[1]

def propagate_track(state0, z_targets, method="ck", step_size=10.0, table=CashKarp):
    states = []; s = state0.copy()
    for zt in z_targets:
        while abs(s.z - zt) > 0.1:
            dz_remain = zt - s.z
            dz = np.sign(dz_remain) * min(step_size, abs(dz_remain))
            if method == "parabolic": parabolic_propagate(s, dz, magnetic_field)
            elif method == "rkn": rkn_fast_step(s, dz, magnetic_field)
            else: rk_propagate(s, dz, magnetic_field, table)
        states.append(s.copy())
    return states

def extract(states, attr): return np.array([getattr(s, attr) for s in states])

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1: B-field profile
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: B-field profile...")
z_range = np.linspace(-500, 12000, 500)
By_vals = [magnetic_field(np.array([0, 0, z]))[1] for z in z_range]

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(z_range, By_vals, "b-", lw=2)
ax.set_xlabel("z [mm]"); ax.set_ylabel("$B_y$ [T]")
ax.set_title("Simplified LHCb Dipole Field Model")
ax.axvspan(-200, 770, alpha=0.15, color="green", label="VELO")
ax.axvspan(2200, 2700, alpha=0.15, color="orange", label="UT")
ax.axvspan(7500, 9500, alpha=0.15, color="red", label="SciFi")
ax.legend(loc="lower right"); plt.tight_layout()
fig.savefig(os.path.join(OUT, "bfield_profile.pdf"), bbox_inches="tight")
plt.close(fig)
print("  -> bfield_profile.pdf")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Track trajectories — extrapolator comparison
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Track trajectories...")
p_GeV = 10.0; qop0 = C_LIGHT / (p_GeV * 1e3)
init_state = State(x=1.0, y=0.5, z=0.0, tx=0.01, ty=0.005, qop=qop0)
z_grid = np.linspace(0, 9500, 1000)

truth = propagate_track(init_state, z_grid, method="ck", step_size=2.0)
para_states = propagate_track(init_state, z_grid, method="parabolic", step_size=50.0)
rk4_states = propagate_track(init_state, z_grid, method="ck", step_size=100.0, table=RK4)
rkn_states = propagate_track(init_state, z_grid, method="rkn", step_size=100.0)
z_t = extract(truth, "z")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for ax, coord in zip(axes, ["x", "y"]):
    ax.plot(z_t, extract(truth, coord), "k-", lw=2, label="Truth (CK, 2mm)", alpha=0.7)
    ax.plot(z_t, extract(para_states, coord), "--", label="Parabolic (50mm)", alpha=0.8)
    ax.plot(z_t, extract(rk4_states, coord), "-.", label="RK4 (100mm)", alpha=0.8)
    ax.plot(z_t, extract(rkn_states, coord), ":", lw=2, label="RKN (100mm)", alpha=0.8)
    ax.set_ylabel(f"{coord} [mm]"); ax.legend(fontsize=9)
    ax.axvspan(-200, 770, alpha=0.08, color="green")
    ax.axvspan(2200, 2700, alpha=0.08, color="orange")
    ax.axvspan(7500, 9500, alpha=0.08, color="red")
axes[1].set_xlabel("z [mm]")
axes[0].set_title("Track Trajectory: Extrapolator Comparison")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "track_trajectories.pdf"), bbox_inches="tight")
plt.close(fig)
print("  -> track_trajectories.pdf")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Extrapolator errors vs truth
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Extrapolator errors...")
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for col, coord in enumerate(["x", "y"]):
    truth_vals = extract(truth, coord)
    for label, states, color in [("Parabolic (50mm)", para_states, "C0"),
                                  ("RK4 (100mm)", rk4_states, "C1"),
                                  ("RKN (100mm)", rkn_states, "C2")]:
        axes[0, col].plot(z_t, extract(states, coord) - truth_vals, color=color, label=label, alpha=0.8)
    axes[0, col].set_ylabel(f"Δ{coord} [mm]"); axes[0, col].set_title(f"{coord} position error")
    axes[0, col].legend(fontsize=8)
for col, coord in enumerate(["tx", "ty"]):
    truth_vals = extract(truth, coord)
    for label, states, color in [("Parabolic (50mm)", para_states, "C0"),
                                  ("RK4 (100mm)", rk4_states, "C1"),
                                  ("RKN (100mm)", rkn_states, "C2")]:
        axes[1, col].plot(z_t, extract(states, coord) - truth_vals, color=color, label=label, alpha=0.8)
    axes[1, col].set_ylabel(f"Δ{coord}"); axes[1, col].set_xlabel("z [mm]")
    axes[1, col].set_title(f"{coord} slope error"); axes[1, col].legend(fontsize=8)
plt.suptitle("Extrapolator Errors Relative to Fine Cash-Karp Truth", fontsize=13)
plt.tight_layout(); fig.savefig(os.path.join(OUT, "extrapolator_errors.pdf"), bbox_inches="tight")
plt.close(fig)
print("  -> extrapolator_errors.pdf")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4: State element sensitivity to B-field
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: State element sensitivity...")
momenta = [3, 5, 10, 20, 50]
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

for p_GeV in momenta:
    qop = C_LIGHT / (p_GeV * 1e3)
    s0 = State(x=0, y=0, z=0, tx=0.01, ty=0.005, qop=qop)
    states = propagate_track(s0, z_grid, method="ck", step_size=5.0)
    z_v = extract(states, "z")
    label = f"p={p_GeV} GeV/c"
    axes[0,0].plot(z_v, extract(states, "x"), label=label)
    axes[0,1].plot(z_v, extract(states, "y"), label=label)
    axes[0,2].plot(z_v, extract(states, "tx"), label=label)
    axes[1,0].plot(z_v, extract(states, "ty"), label=label)
    # Compute dtx/dz, dty/dz numerically
    tx_arr = extract(states, "tx"); ty_arr = extract(states, "ty")
    dtx_dz = np.gradient(tx_arr, z_v)
    dty_dz = np.gradient(ty_arr, z_v)
    axes[1,1].plot(z_v, dtx_dz, label=label)
    axes[1,2].plot(z_v, dty_dz, label=label)

titles = ["x(z)", "y(z)", "$t_x(z)$", "$t_y(z)$", "$dt_x/dz$", "$dt_y/dz$"]
ylabels = ["x [mm]", "y [mm]", "$t_x$", "$t_y$", "$dt_x/dz$ [1/mm]", "$dt_y/dz$ [1/mm]"]
for i, ax in enumerate(axes.flat):
    ax.set_title(titles[i]); ax.set_ylabel(ylabels[i]); ax.set_xlabel("z [mm]")
    ax.legend(fontsize=7)
    ax.axvspan(-200, 770, alpha=0.06, color="green")
    ax.axvspan(2200, 2700, alpha=0.06, color="orange")
    ax.axvspan(7500, 9500, alpha=0.06, color="red")

plt.suptitle("State Element Evolution vs Momentum — Sensitivity to $B$-field", fontsize=14)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "state_sensitivity.pdf"), bbox_inches="tight")
plt.close(fig)
print("  -> state_sensitivity.pdf")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 5: By(z) overlaid with dtx/dz to show direct coupling
# ═════════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: Field-slope coupling...")
fig, ax1 = plt.subplots(figsize=(11, 4))
ax1.fill_between(z_range, 0, By_vals, alpha=0.2, color="blue", label="$B_y(z)$")
ax1.set_ylabel("$B_y$ [T]", color="blue"); ax1.set_xlabel("z [mm]")
ax1.tick_params(axis="y", labelcolor="blue")

ax2 = ax1.twinx()
for p_GeV, ls in [(5, "-"), (10, "--"), (50, ":")]:
    qop = C_LIGHT / (p_GeV * 1e3)
    s0 = State(x=0, y=0, z=0, tx=0.01, ty=0.005, qop=qop)
    st = propagate_track(s0, z_grid, method="ck", step_size=5.0)
    tx_arr = extract(st, "tx"); z_v = extract(st, "z")
    dtx = np.gradient(tx_arr, z_v)
    ax2.plot(z_v, dtx, ls, color="red", alpha=0.7, label=f"$dt_x/dz$, p={p_GeV}")
ax2.set_ylabel("$dt_x/dz$ [1/mm]", color="red")
ax2.tick_params(axis="y", labelcolor="red")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)
ax1.set_title("Direct Coupling: $B_y(z)$ Drives $dt_x/dz \\propto (q/p) \\cdot B_y$")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "field_slope_coupling.pdf"), bbox_inches="tight")
plt.close(fig)
print("  -> field_slope_coupling.pdf")

print("\nAll figures generated successfully.")
