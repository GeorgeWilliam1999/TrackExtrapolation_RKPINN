#!/usr/bin/env python3
"""F2 eval — chart + residual-MLP on the UT->T pool.

final(X) = chart_predict(X) + MLP_residual(X).
Compares chart-alone, chart+NN, vs RK truth and the locked NN, on UT->T.
"""
from __future__ import annotations
import sys, json
from datetime import datetime
from pathlib import Path
import os

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
FLAT = HERE.parent
REPO = FLAT.parent
# Big data / checkpoints live in the lab, not in this repo.
LAB = Path(os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))
LAB_FLAT = LAB.parent / "flattening"
sys.path.insert(0, str(FLAT / "charts"))
sys.path.insert(0, str(REPO / "models"))
from chart import chart_predict, load_chart  # noqa: E402
from architectures import create_model  # noqa: E402

UT_Z, T_Z = (2300.0, 3000.0), (7600.0, 9500.0)


def load_mlp(exp_dir):
    ck = torch.load(exp_dir / "best_model.pt", weights_only=False, map_location="cpu")
    c = ck["config"]
    m = create_model("mlp", hidden_dims=c["hidden_dims"], activation=c["activation"],
                     dropout=c.get("dropout", 0.0),
                     engineered_features=c.get("engineered_features", False))
    if (exp_dir / "normalization.json").exists():
        m.load_normalization(str(exp_dir / "normalization.json"))
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


def rep(dx_mm, dtx, qop, label, store):
    a = np.abs(dx_mm) * 1e3
    q = np.abs(qop); e = np.quantile(q, [.25, .5, .75]); b = np.digitize(q, e)
    byq = [round(float(np.median(a[b == i])), 1) for i in range(4)]
    store[label] = {"median_um": float(np.median(a)), "p95_um": float(np.quantile(a, .95)),
                    "p99_um": float(np.quantile(a, .99)), "byq": byq}
    print(f"{label:<32} med={np.median(a):7.1f} p95={np.quantile(a,.95):8.0f} "
          f"p99={np.quantile(a,.99):8.0f} byQ={byq}")


def main():
    d = np.load(LAB / "data" / "train_10M_gen3.npz"); X, Y = d["X"], d["Y"]
    z0, dz = X[:, 5], X[:, 6]; zf = z0 + dz
    m = (z0 >= UT_Z[0]) & (z0 <= UT_Z[1]) & (zf >= T_Z[0]) & (zf <= T_Z[1]) & (dz > 0)
    Xs, Ys = X[m].astype(np.float64), Y[m].astype(np.float64)
    print(f"UT->T pool {len(Xs):,}\n")

    chart = load_chart()
    base = chart_predict(Xs, chart)            # [N,4]
    store = {}
    rep(base[:, 0] - Ys[:, 0], base[:, 2] - Ys[:, 2], Xs[:, 4], "chart alone (0 params)", store)

    Xt = torch.from_numpy(Xs.astype(np.float32))
    for exp_name, label in (("residual_mlp_2M", "chart + full-pool MLP"),
                            ("residual_mlp_fwd", "chart + focused MLP")):
        exp = LAB_FLAT / "trained_models" / exp_name
        if not (exp / "best_model.pt").exists():
            print(f"  ({exp_name} not trained yet)")
            continue
        mlp = load_mlp(exp)
        with torch.no_grad():
            res = mlp(Xt).numpy()              # [N,5] residual
        final = base + res[:, :4]
        rep(final[:, 0] - Ys[:, 0], final[:, 2] - Ys[:, 2], Xs[:, 4], label, store)
        store[label + " :params"] = int(sum(p.numel() for p in mlp.parameters()))

    # references
    print("\nreferences:")
    print("  REF rung1.5 true-field   med=5.7  p95=159  p99=480")
    print("  NN pinn_v2_small_v1      med=293  p95=1894")
    print("  NN pinn_v2_kick_10M      med=153  p95=2827")

    out = FLAT / "results" / f"F2_chartnn_{datetime.now().strftime('%Y-%m-%d')}.json"
    json.dump(store, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
