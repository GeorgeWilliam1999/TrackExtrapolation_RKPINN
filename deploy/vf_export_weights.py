#!/usr/bin/env python3
"""Export a vertex-fit PINN_v2 checkpoint to the flat fp32 blob vf_kernel.cpp
loads.

Layout (all little-endian fp32 after the 8-byte header):
  magic   "VFK1"            (4 bytes)
  H       int32             hidden width (both layers)
  mean[7] std[7]            input normalisation
  W1[H*8] b1[H]             encoder layer 1 (row-major, input order:
                            x,y,tx,ty,qop normalised, z_frac, z0n, dzn)
  W2[H*H] b2[H]             encoder layer 2
  W3[4*H] b3[4]             correction head
  g[4]                      exp(kick_loggain)  (per-channel kick gains)

Only the deployed configuration is supported: pinn_v2, two hidden layers of
equal width, tanh, kick_scaled_head, kick_order 1, n_unroll 1, z_features.

Usage: vf_export_weights.py <experiment_dir> <out.blob>
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

exp = Path(sys.argv[1])
out = Path(sys.argv[2])

ckpt = torch.load(exp / "best_model.pt", weights_only=False, map_location="cpu")
cfg = ckpt["config"]
assert cfg["model_type"] == "pinn_v2"
assert cfg.get("kick_scaled_head", False), "kernel implements the kick head only"
assert cfg.get("kick_order", 1) == 1 and cfg.get("n_unroll", 1) == 1
assert cfg.get("z_features", False), "kernel expects the 8-feature encoder"
h = cfg["hidden_dims"]
assert len(h) == 2 and h[0] == h[1], f"need two equal hidden layers, got {h}"
H = h[0]
assert cfg["activation"] == "tanh"

sd = ckpt["model_state_dict"]
norm = json.load(open(exp / "normalization.json"))

W1 = sd["encoder.0.weight"].numpy().astype(np.float32)     # [H, 8]
b1 = sd["encoder.0.bias"].numpy().astype(np.float32)
W2 = sd["encoder.2.weight"].numpy().astype(np.float32)     # [H, H]
b2 = sd["encoder.2.bias"].numpy().astype(np.float32)
W3 = sd["correction_head.weight"].numpy().astype(np.float32)  # [4, H]
b3 = sd["correction_head.bias"].numpy().astype(np.float32)
g = np.exp(sd["kick_loggain"].numpy()).astype(np.float32)
assert W1.shape == (H, 8) and W2.shape == (H, H) and W3.shape == (4, H)

with open(out, "wb") as f:
    f.write(b"VFK1")
    f.write(struct.pack("<i", H))
    for a in [np.asarray(norm["input_mean"], np.float32),
              np.asarray(norm["input_std"], np.float32),
              W1.ravel(), b1, W2.ravel(), b2, W3.ravel(), b3, g]:
        f.write(a.tobytes())
print(f"wrote {out} (H={H}, {out.stat().st_size} bytes) from {exp.name}")
