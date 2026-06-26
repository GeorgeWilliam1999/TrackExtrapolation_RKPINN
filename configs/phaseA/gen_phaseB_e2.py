#!/usr/bin/env python3
"""Phase B / E2 — higher-order kick basis (2026-06-26).

Adds a 2nd-order (kappa*dz)^2 term to the deployable kick head (architectures.py
`kick_order`, field-free, Allen-faithful) and stacks it on the multi-step unroll.
Matched ablations against the E1 runs at the SAME epoch budget:

  e2_order2_unroll16  vs  e1_unroll16      (does order-2 help at the unroll plateau?)
  e2_order2_unroll8   vs  e1_unroll8_long  (does order-2 help at moderate steps?)

Base = e1_unroll16.yaml; only kick_order / n_unroll / experiment_name change.
"""
import copy, yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = yaml.safe_load((HERE / "e1_unroll16.yaml").read_text())

RUNS = [
    dict(experiment_name="e2_order2_unroll16", kick_order=2, n_unroll=16, epochs=250),
    dict(experiment_name="e2_order2_unroll8",  kick_order=2, n_unroll=8,  epochs=250),
]

for r in RUNS:
    cfg = copy.deepcopy(BASE)
    cfg.update(r)
    cfg["patience"] = 60
    out = HERE / f"{r['experiment_name']}.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {out.name}  (kick_order={cfg['kick_order']} n_unroll={cfg['n_unroll']} epochs={cfg['epochs']})")
