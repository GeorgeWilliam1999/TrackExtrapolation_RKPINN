#!/usr/bin/env python3
"""
Gen-3 training script.

Faithful port of the gen-2 ``v2_fixes`` trainer (which produced 0.125 mm for
``nrk4_small_1step``) with the minimum changes required by gen-3 data:

  * X layout is 7-dim (Fix H, includes ``z_start``) and Y is 5-dim (A5,
    includes ``qop_f`` as a pass-through).
  * Loss is applied only to the four learned outputs (positions + slopes);
    ``qop_f`` is an identity pass-through and contributes no gradient.
  * Stratified split on ``sign(dz)`` (50/50 fwd/bwd).
  * PINN_v2 is supported via ``model_type='pinn_v2'`` with a Fix-F physics
    warmup schedule.

The ``detector_sigma`` / ``tolerance_scaled`` losses were trialled for
milestone M1 and found to make training intractable on the gen-3 signed-dz
dataset (see ``docs/reports/gen3_protocol.tex`` §M1-Postmortem).  Gen-3
restores the gen-2 loss and defers the detector-sigma weighting to M2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# repo code (this file lives in <repo>/models; shared physics utils in <repo>/core)
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "models"))
sys.path.insert(0, str(_REPO_ROOT / "core"))

# Big data / checkpoints / mlruns live in the lab, not in this repo.
_LAB = Path(os.environ.get("TE_LAB", "/data/bfys/gscriven/TrackExtrapolation/experiments/gen_3"))

from architectures import create_model  # noqa: E402

try:
    import mlflow  # noqa: F401
    _MLFLOW_OK = True
except Exception:
    _MLFLOW_OK = False


# =============================================================================
# Config
# =============================================================================

DEFAULTS: dict = {
    "seed": 42,
    "data_path": str(_LAB / "data" / "train_10M_gen3.npz"),
    "max_samples": 1_400_000,          # 1.12M train / 140k val / 140k test
    "train_fraction": 0.8,
    "val_fraction": 0.1,

    "model_type": "neural_rk4",
    "hidden_dims": [64, 64],
    "activation": "tanh",
    "dropout": 0.0,
    "engineered_features": False,     # MLP-v2 only: append log10|dz| + sign(dz)

    # NeuralRK4-specific
    "n_rk_steps": 1,
    "correction_scale_init": 1e-3,

    # PINN_v2-specific
    "lambda_pde": 1.0,
    "lambda_ic": 0.1,
    "n_collocation": 2,
    "physics_warmup_epochs": 5,

    "batch_size": 2048,
    "epochs": 200,
    "learning_rate": 5e-4,       # gen-2 v2_fixes used 5e-4 for all PINN/RK4 variants.
    "weight_decay": 1e-4,
    "warmup_epochs": 5,
    "grad_clip": 1.0,
    "patience": 30,
    "min_delta": 1e-7,

    "checkpoint_dir": str(_LAB / "trained_models"),
    "experiment_name": None,

    "use_mlflow": True,
    "mlflow_tracking_uri": None,            # None -> local file store
    "mlflow_experiment_name": "gen_3_track_extrapolation",

    "eval_jacobian": False,

    "num_workers": 0,
    "pin_memory": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "loss": "log_cosh",   # R1: log-cosh is the default; use "mse" for legacy compat
    # --- Wave-2 (2026-06-14) range-aware residual loss + restratified-corpus knobs ---
    # loss: "residual_rel" -> log-cosh on the (pred-truth) error normalised by a
    #   per-track, per-component RESIDUAL scale  sqrt(floor^2 + (alpha*bend)^2).
    #   Fixes the gen-4 mis-scaling (the legacy loss divides by the *endpoint* std
    #   ~1.2 m so a 100 um error registers as ~1e-7 and only metre-scale tail tracks
    #   produce gradient). floors set the "don't care below" absolute accuracy;
    #   alpha sets how relative the hard tail is. Components are equally weighted in
    #   their own relative units -> x,y(mm) balanced against tx,ty(rad).
    "resid_scale_pos": 0.05,     # mm  (x,y absolute Huber scale; quad core -> um accuracy)
    "resid_scale_slope": 2.0e-5, # rad (tx,ty absolute Huber scale)
    "resid_alpha": 0.0,          # optional relative blend sqrt(scale^2+(alpha*bend)^2); 0 = pure absolute
    "resid_huber_delta": 8.0,    # Huber transition (in z units) -- linear tail gives the metre-scale bend a capped, sustained gradient
    # split: balance forward/backward 50/50 (legacy) vs random (preserve the
    #   deployment-weighted corpus mix incl. the >=10% UT->T fraction).
    "balance_sign": True,
    # checkpoint selection metric: "median_dx_mm" (legacy, bulk-dominated),
    #   "utt_median_dx_um" (the UT->T regime we are judged on), or "val_loss_sel".
    "select_metric": "median_dx_mm",
}


# =============================================================================
# Helpers
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_path = Path(config["data_path"])
    print(f"Loading {data_path} ...")
    with np.load(data_path, allow_pickle=False) as d:
        X = d["X"].astype(np.float32)
        Y = d["Y"].astype(np.float32)
        P = d["P"].astype(np.float32)
    assert X.shape[1] == 7, f"Expected 7 inputs, got {X.shape[1]}"
    assert Y.shape[1] == 5, f"Expected 5 outputs, got {Y.shape[1]}"
    print(f"  Loaded {X.shape[0]:,} tracks.  X={X.shape}  Y={Y.shape}")
    return X, Y, P


def stratified_split(
    X: np.ndarray,
    Y: np.ndarray,
    P: np.ndarray,
    config: dict,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Deterministic stratified split on ``sign(dz)``.

    Subsampling (when ``max_samples`` < N) is also stratified so the final
    split has ~50/50 forward/backward tracks.
    """
    N = X.shape[0]
    rng = np.random.default_rng(config["seed"])

    max_n = config.get("max_samples") or N
    max_n = min(int(max_n), N)

    if config.get("balance_sign", True):
        sign_dz = np.sign(X[:, 6]).astype(np.int8)
        idx_pos = np.flatnonzero(sign_dz > 0)
        idx_neg = np.flatnonzero(sign_dz < 0)
        rng.shuffle(idx_pos)
        rng.shuffle(idx_neg)
        n_each = max_n // 2
        keep = np.concatenate([idx_pos[:n_each], idx_neg[:n_each]])
        rng.shuffle(keep)
        print(f"  subsampled {len(keep):,}/{N:,}  (stratified 50/50 on sign(dz))")
    else:
        # Wave-2: random subsample preserves the deployment-weighted corpus mix
        # (incl. the >=10% UT->T fraction), which a 50/50 sign balance would dilute.
        keep = rng.permutation(N)[:max_n]
        print(f"  subsampled {len(keep):,}/{N:,}  (random; preserves corpus mix)")

    n_train = int(len(keep) * config["train_fraction"])
    n_val = int(len(keep) * config["val_fraction"])
    n_test = len(keep) - n_train - n_val

    sl = {
        "train": keep[:n_train],
        "val":   keep[n_train : n_train + n_val],
        "test":  keep[n_train + n_val : n_train + n_val + n_test],
    }
    out = {}
    for name, ix in sl.items():
        frac_fwd = float((X[ix, 6] > 0).mean())
        z0 = X[ix, 5]; zf = z0 + X[ix, 6]
        frac_utt = float(((z0 >= 2300) & (z0 <= 3000) & (zf >= 7600)
                          & (zf <= 9500) & (X[ix, 6] > 0)).mean())
        print(f"  {name}: {len(ix):,} tracks  (forward={frac_fwd:.3f}  UT->T={frac_utt*100:.1f}%)")
        out[name] = (X[ix], Y[ix], P[ix], ix)
    return out


# =============================================================================
# Training
# =============================================================================

def _make_loader(
    X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool, config: dict
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        drop_last=shuffle,
    )


def cosine_with_warmup(optimizer, steps_total: int, steps_warmup: int):
    def lr_lambda(step: int) -> float:
        if steps_warmup > 0 and step < steps_warmup:
            return max(0.01, step / steps_warmup)
        rem = steps_total - steps_warmup
        if rem <= 0:
            return 1.0
        prog = (step - steps_warmup) / rem
        return max(0.01, 0.5 * (1 + float(np.cos(np.pi * min(prog, 1.0)))))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EarlyStop:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.count = 0

    def step(self, val: float) -> bool:
        if val < self.best - self.min_delta:
            self.best = val
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


def _logcosh(z: torch.Tensor) -> torch.Tensor:
    # log(cosh(x)) = |x| + softplus(-2|x|) - log(2); numerically stable for all |x|.
    return z.abs() + torch.nn.functional.softplus(-2.0 * z.abs()) - 0.6931471805599453


def log_cosh_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Log-cosh loss, scale-normalised over the four learned outputs.

    Quadratic near zero (same gradient signal as MSE for typical residuals),
    linear in the tails (caps the gradient contribution of outlier tracks).
    ``scale`` should be ``model.output_std[:4]``.

    This is the R1 replacement for the MSE data loss.  MSE is still available
    via ``--loss=mse`` for backwards compatibility.
    """
    return torch.mean(_logcosh((y_pred[:, :4] - y_true[:, :4]) / scale))


def _straight_line(x: torch.Tensor) -> torch.Tensor:
    """Field-free straight-line prediction of (x,y,tx,ty) at z0+dz from the input.

    The kick/residual head predicts the *bend* relative to this line; the loss
    uses it only to size the per-track residual scale (the straight line cancels
    in the numerator: y_pred - y_true = bend_pred - bend_true)."""
    dz = x[:, 6]
    return torch.stack([x[:, 0] + x[:, 2] * dz,   # x  = x0 + tx0*dz
                        x[:, 1] + x[:, 3] * dz,   # y  = y0 + ty0*dz
                        x[:, 2],                  # tx = tx0  (slope unchanged)
                        x[:, 3]], dim=1)          # ty = ty0


def _huber(z: torch.Tensor, delta: float) -> torch.Tensor:
    az = z.abs()
    return torch.where(az < delta, 0.5 * z * z, delta * (az - 0.5 * delta))


def residual_rel_loss(
    x: torch.Tensor, y_pred: torch.Tensor, y_true: torch.Tensor,
    scales: torch.Tensor, alpha: float, delta: float,
) -> torch.Tensor:
    """Range-aware HUBER on the residual (Wave-2 default).

    Normalises the (pred - truth) error by a per-component **absolute** scale
    ``sqrt(scale_c^2 + (alpha*bend)^2)`` (alpha=0 -> pure absolute) and applies
    Huber with a large transition ``delta`` placed well beyond the bulk |z|.

    Why absolute, not relative: a relative scale (alpha*bend) divides the huge
    UT->T error by the huge bend, shrinking its gradient ~1000x below the bulk's
    -> the optimiser ignores the hard regime (observed: 2% of the bend recovered).
    A small absolute scale + Huber's LINEAR TAIL instead gives the metre-scale
    bend a *sustained, capped* gradient (learnability, no outlier explosion), while
    the quadratic core drives um-level accuracy where the residual is small.
    Per-component scales (mm for x,y; rad for tx,ty) balance positions vs slopes.
    """
    err = y_pred[:, :4] - y_true[:, :4]
    if alpha > 0.0:
        bend = y_true[:, :4] - _straight_line(x)
        scale = torch.sqrt(scales.unsqueeze(0) ** 2 + (alpha * bend) ** 2)
    else:
        scale = scales.unsqueeze(0)
    return torch.mean(_huber(err / scale, delta))


def _resid_scales(model, config: dict) -> torch.Tensor:
    sp = float(config.get("resid_scale_pos", 0.05))     # mm
    ss = float(config.get("resid_scale_slope", 2.0e-5)) # rad
    return torch.tensor([sp, sp, ss, ss], device=model.output_std.device, dtype=model.output_std.dtype)


def _data_loss(
    model,
    x: torch.Tensor,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    loss_fn: str = "log_cosh",
    config: dict | None = None,
) -> torch.Tensor:
    """Dispatch to the range-aware residual loss (Wave-2), log-cosh (R1) or MSE.

    ``y_pred[:, 4]`` and ``y_true[:, 4]`` are both ``qop`` (pass-through);
    including them would either dilute the gradient by 20% or, worse, if
    the normalisation is stale, introduce spurious noise.
    """
    if loss_fn == "residual_rel":
        cfg = config or {}
        return residual_rel_loss(x, y_pred, y_true, _resid_scales(model, cfg),
                                 float(cfg.get("resid_alpha", 0.0)),
                                 float(cfg.get("resid_huber_delta", 8.0)))
    scale = model.output_std[:4]
    if loss_fn == "mse":
        inv = 1.0 / scale
        return ((y_pred[:, :4] - y_true[:, :4]) * inv).pow(2).mean()
    return log_cosh_loss(y_pred, y_true, scale)


def _build_model(config: dict):
    mt = config["model_type"]
    if mt == "mlp":
        return create_model(
            "mlp",
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            dropout=config["dropout"],
            engineered_features=config.get("engineered_features", False),
        )
    if mt == "pinn_v2":
        return create_model(
            "pinn_v2",
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            dropout=config["dropout"],
            lambda_pde=config["lambda_pde"],
            lambda_ic=config["lambda_ic"],
            n_collocation=config["n_collocation"],
            kick_scaled_head=config.get("kick_scaled_head", False),
            pde_scale_mode=config.get("pde_scale_mode", "legacy"),
            pde_ref_length=config.get("pde_ref_length", 5213.0),
        )
    if mt == "neural_rk4":
        return create_model(
            "neural_rk4",
            hidden_dims=config["hidden_dims"],
            activation=config["activation"],
            n_rk_steps=config["n_rk_steps"],
            correction_scale_init=config["correction_scale_init"],
        )
    raise ValueError(f"Unknown model_type {mt!r}")


def train_epoch(model, loader, optim_, scheduler, device, grad_clip, phys_ramp, loss_fn="log_cosh", config=None):
    model.train()
    tot = 0.0; tot_data = 0.0; tot_pde = 0.0; tot_ic = 0.0
    n = 0
    skipped = 0
    # lambda_pde = lambda_ic = 0 is a pure-data ablation: skip the (expensive)
    # JVP physics evaluation entirely, it would contribute zero gradient.
    use_physics = hasattr(model, "compute_physics_loss") and (
        getattr(model, "lambda_pde", 0.0) > 0.0 or getattr(model, "lambda_ic", 0.0) > 0.0
    )
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        optim_.zero_grad(set_to_none=True)
        y_pred = model(x)
        data_loss = _data_loss(model, x, y_pred, y, loss_fn, config)
        if use_physics:
            phys = model.compute_physics_loss(x, y_pred)
            pde_loss = phys.get("pde", torch.zeros((), device=device))
            ic_loss = phys.get("ic", torch.zeros((), device=device))
            loss = data_loss + phys_ramp * (pde_loss + ic_loss)
            tot_pde += float(pde_loss.item())
            tot_ic += float(ic_loss.item())
        else:
            loss = data_loss
        if not torch.isfinite(loss):
            skipped += 1
            continue
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim_.step()
        if scheduler is not None:
            scheduler.step()
        tot += float(loss.item())
        tot_data += float(data_loss.item())
        n += 1
    return {
        "loss": tot / max(1, n),
        "data_loss": tot_data / max(1, n),
        "pde_loss": tot_pde / max(1, n),
        "ic_loss": tot_ic / max(1, n),
        "skipped": skipped,
        "lr": optim_.param_groups[0]["lr"],
    }


@torch.no_grad()
def validate(model, loader, device, loss_fn="log_cosh", config=None) -> Dict[str, float]:
    """Evaluate on *loader* and return a full suite of robust metrics (R1).

    Adds (Wave-2) ``val_loss_sel`` = the *configured* training loss on val and
    ``utt_median_dx_um`` = median |dx| restricted to the UT->T regime, so the
    checkpoint can be selected on the regime it is actually judged on rather
    than the bulk-dominated whole-val median.
    """
    model.eval()
    # Accumulate residuals in-memory to compute percentiles.
    all_dx: list[torch.Tensor] = []
    all_dy: list[torch.Tensor] = []
    all_dtx: list[torch.Tensor] = []
    all_dty: list[torch.Tensor] = []
    all_z0: list[torch.Tensor] = []
    all_dz: list[torch.Tensor] = []
    tot_logcosh = 0.0; tot_mse = 0.0; tot_sel = 0.0; n_batches = 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        y_pred = model(x)
        tot_logcosh += float(_data_loss(model, x, y_pred, y, "log_cosh").item())
        tot_mse     += float(_data_loss(model, x, y_pred, y, "mse").item())
        tot_sel     += float(_data_loss(model, x, y_pred, y, loss_fn, config).item())
        n_batches += 1
        r = (y_pred - y).cpu()
        all_dx.append(r[:, 0].abs())
        all_dy.append(r[:, 1].abs())
        all_dtx.append(r[:, 2].abs())
        all_dty.append(r[:, 3].abs())
        all_z0.append(x[:, 5].cpu())
        all_dz.append(x[:, 6].cpu())

    dx  = torch.cat(all_dx)
    dy  = torch.cat(all_dy)
    dtx = torch.cat(all_dtx)
    dty = torch.cat(all_dty)
    z0  = torch.cat(all_z0)
    dz  = torch.cat(all_dz)
    zf  = z0 + dz
    pos = torch.sqrt(dx**2 + dy**2)  # scalar position error per track

    def _q(t: torch.Tensor, q: float) -> float:
        return float(torch.quantile(t, q).item())

    # UT->T regime mask (matches gates/run_r7_utt_eval.py + the frozen pool)
    utt = (z0 >= 2300.0) & (z0 <= 3000.0) & (zf >= 7600.0) & (zf <= 9500.0) & (dz > 0)
    n_utt = int(utt.sum().item())
    if n_utt >= 50:
        utt_med_um = float(torch.quantile(dx[utt], 0.50).item()) * 1e3
        utt_p95_um = float(torch.quantile(dx[utt], 0.95).item()) * 1e3
    else:
        utt_med_um = float("nan")
        utt_p95_um = float("nan")

    return {
        # --- selection metrics ---
        "val_median_dx_mm":   _q(dx,  0.50),
        "val_loss_sel":       tot_sel / max(1, n_batches),
        "utt_median_dx_um":   utt_med_um,
        "utt_p95_dx_um":      utt_p95_um,
        "n_utt":              n_utt,
        # --- full quantile suite ---
        "median_dx_mm":       _q(dx,  0.50),
        "p68_dx_mm":          _q(dx,  0.68),
        "p95_dx_mm":          _q(dx,  0.95),
        "p99_dx_mm":          _q(dx,  0.99),
        "median_dy_mm":       _q(dy,  0.50),
        "p95_dy_mm":          _q(dy,  0.95),
        "median_pos_mm":      _q(pos, 0.50),
        "p95_pos_mm":         _q(pos, 0.95),
        "median_dtx_mrad":    _q(dtx, 0.50) * 1e3,
        "median_dty_mrad":    _q(dty, 0.50) * 1e3,
        "p95_dtx_mrad":       _q(dtx, 0.95) * 1e3,
        "p95_dty_mrad":       _q(dty, 0.95) * 1e3,
        # --- legacy / debug (not used for selection) ---
        "loss":      tot_logcosh / max(1, n_batches),
        "mse_loss":  tot_mse     / max(1, n_batches),
        "pos_rms_mm": float(torch.sqrt((dx**2 + dy**2).mean()).item()),
    }


def train(config: dict):
    set_seed(config["seed"])
    device = torch.device(config["device"])
    data_path = Path(config["data_path"])

    X, Y, P = load_data(config)
    splits = stratified_split(X, Y, P, config)

    # ------------------------------------------------------------------
    # Experiment dir + MLflow
    # ------------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = config["experiment_name"] or f"{config['model_type']}_{ts}"
    exp_dir = Path(config["checkpoint_dir"]) / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Experiment dir: {exp_dir}")

    # Compute data hash once (~1s for 568 MB) to log provenance.
    data_sha = _sha256_file(data_path)
    print(f"data_sha256 = {data_sha[:16]}...")

    # Model
    model = _build_model(config)
    X_train_t = torch.from_numpy(splits["train"][0])
    Y_train_t = torch.from_numpy(splits["train"][1])
    model.set_normalization(X_train_t, Y_train_t)
    model = model.to(device)
    n_params = model.count_parameters()
    print(f"Model: {config['model_type']}  params={n_params:,}")

    # Loaders
    train_loader = _make_loader(
        splits["train"][0], splits["train"][1], config["batch_size"], True, config
    )
    val_loader = _make_loader(
        splits["val"][0], splits["val"][1], config["batch_size"], False, config
    )
    test_loader = _make_loader(
        splits["test"][0], splits["test"][1], config["batch_size"], False, config
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = cosine_with_warmup(
        optimizer,
        steps_total=config["epochs"] * len(train_loader),
        steps_warmup=config["warmup_epochs"] * len(train_loader),
    )
    early = EarlyStop(config["patience"], config["min_delta"])

    # MLflow
    mlflow_run = None
    if config["use_mlflow"] and _MLFLOW_OK:
        import mlflow
        uri = config["mlflow_tracking_uri"] or f"file://{(_LAB / 'mlruns').resolve()}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config["mlflow_experiment_name"])
        mlflow_run = mlflow.start_run(run_name=name)
        flat = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool)) or v is None}
        flat["hidden_dims"] = str(config["hidden_dims"])
        mlflow.log_params(flat)
        mlflow.set_tags({
            "generation": "gen_3",
            "model_type": config["model_type"],
            "applied_fixes": "C1,C2,H,I,A5",
            "milestone": "M1",
            "data_sha256": data_sha,
        })
        mlflow.log_param("n_parameters", n_params)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------
    history = {"train": [], "val": [], "best_epoch": 0, "best_val_median_dx_mm": float("inf")}
    best = float("inf")  # tracks val_median_dx_mm
    t0 = time.time()
    loss_fn = config.get("loss", "log_cosh")

    print("\n" + "=" * 72)
    print(f"Training {config['model_type']}  ({config['epochs']} epochs max)  loss={loss_fn}")
    print("=" * 72)
    use_physics = hasattr(model, "compute_physics_loss") and (
        getattr(model, "lambda_pde", 0.0) > 0.0 or getattr(model, "lambda_ic", 0.0) > 0.0
    )
    phys_warm = int(config.get("physics_warmup_epochs", 0) or 0) if use_physics else 0
    for epoch in range(config["epochs"]):
        phys_ramp = (
            float(min(1.0, (epoch + 1) / phys_warm)) if phys_warm > 0 else 1.0
        )
        t_m = train_epoch(model, train_loader, optimizer, scheduler, device, config["grad_clip"], phys_ramp, loss_fn, config)
        v_m = validate(model, val_loader, device, loss_fn, config)
        history["train"].append(t_m)
        history["val"].append(v_m)
        elapsed = time.time() - t0
        extra = (
            f" [pde={t_m['pde_loss']:.3e} ic={t_m['ic_loss']:.3e} ramp={phys_ramp:.2f}]"
            if use_physics else ""
        )
        utt_str = (f"  utt_med={v_m['utt_median_dx_um']:.0f}µm(n={v_m['n_utt']})"
                   if v_m.get("n_utt", 0) >= 50 else "")
        print(
            f"[{epoch+1:3d}/{config['epochs']}]  "
            f"tr={t_m['loss']:.4f}  "
            f"median_dx={v_m['median_dx_mm']*1e3:.1f}µm  "
            f"p95_dx={v_m['p95_dx_mm']*1e3:.1f}µm  "
            f"pos_rms={v_m['pos_rms_mm']*1e3:.1f}µm{utt_str}  "
            f"lr={t_m['lr']:.2e}  wall={elapsed/60:.1f}m{extra}"
        )
        if mlflow_run is not None:
            import mlflow
            mlflow.log_metrics(
                {
                    "train_loss":          t_m["loss"],
                    "val_loss":            v_m["loss"],
                    "val_mse_loss":        v_m["mse_loss"],
                    "val_median_dx_mm":    v_m["median_dx_mm"],
                    "val_p68_dx_mm":       v_m["p68_dx_mm"],
                    "val_p95_dx_mm":       v_m["p95_dx_mm"],
                    "val_p99_dx_mm":       v_m["p99_dx_mm"],
                    "val_median_dy_mm":    v_m["median_dy_mm"],
                    "val_median_pos_mm":   v_m["median_pos_mm"],
                    "val_p95_pos_mm":      v_m["p95_pos_mm"],
                    "val_median_dtx_mrad": v_m["median_dtx_mrad"],
                    "val_median_dty_mrad": v_m["median_dty_mrad"],
                    "val_pos_rms_mm":      v_m["pos_rms_mm"],
                    "lr":                  t_m["lr"],
                },
                step=epoch,
            )

        # Checkpoint selection metric (Wave-2: select on the UT->T regime when the
        # corpus carries enough of it; else fall back to the bulk median).
        select_metric = config.get("select_metric", "median_dx_mm")
        if select_metric == "utt_median_dx_um" and v_m.get("n_utt", 0) >= 50:
            sel = v_m["utt_median_dx_um"]
        elif select_metric == "val_loss_sel":
            sel = v_m["val_loss_sel"]
        else:
            sel = v_m["val_median_dx_mm"]
        if sel < best:
            best = sel
            history["best_epoch"] = epoch + 1
            history["best_val_median_dx_mm"] = best
            history["best_select_metric"] = select_metric
            history["best_val_full"] = v_m
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "val_median_dx_mm": sel,
                    "val_loss": v_m["loss"],
                    "config": config,
                },
                exp_dir / "best_model.pt",
            )
            model.save_normalization(str(exp_dir / "normalization.json"))
        if early.step(sel):
            print(f"  early stopping at epoch {epoch+1}")
            break

    history["training_time_s"] = time.time() - t0

    # ------------------------------------------------------------------
    # Final test
    # ------------------------------------------------------------------
    ckpt = torch.load(exp_dir / "best_model.pt", weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    t_final = validate(model, test_loader, device, loss_fn, config)
    history["test_final"] = t_final
    print("\nTest set (R1 metrics):")
    for k, v in t_final.items():
        print(f"  {k:28s} = {v:.6e}")

    with open(exp_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    # Also dump test split indices so Stage-1 eval can reuse them.
    np.save(exp_dir / "test_indices.npy", splits["test"][3])

    if mlflow_run is not None:
        import mlflow
        mlflow.log_metrics(
            {
                "test_loss":            t_final["loss"],
                "test_mse_loss":        t_final["mse_loss"],
                "test_median_dx_mm":    t_final["median_dx_mm"],
                "test_p68_dx_mm":       t_final["p68_dx_mm"],
                "test_p95_dx_mm":       t_final["p95_dx_mm"],
                "test_p99_dx_mm":       t_final["p99_dx_mm"],
                "test_median_pos_mm":   t_final["median_pos_mm"],
                "test_p95_pos_mm":      t_final["p95_pos_mm"],
                "test_median_dtx_mrad": t_final["median_dtx_mrad"],
                "test_median_dty_mrad": t_final["median_dty_mrad"],
                "test_pos_rms_mm":      t_final["pos_rms_mm"],
                "best_val_median_dx_mm": history["best_val_median_dx_mm"],
                "best_epoch":            history["best_epoch"],
                "training_time_min":     history["training_time_s"] / 60.0,
            }
        )
        for art in ("config.json", "history.json", "best_model.pt", "normalization.json", "test_indices.npy"):
            if (exp_dir / art).exists():
                mlflow.log_artifact(str(exp_dir / art))
        mlflow.end_run()

    print(f"\nExperiment saved: {exp_dir}")
    return model, history, exp_dir


# =============================================================================
# CLI
# =============================================================================

def _load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        if config_path.suffix in (".yaml", ".yml"):
            import yaml  # lazy import — only needed for yaml configs
            override = yaml.safe_load(f)
        else:
            override = json.load(f)
    cfg = DEFAULTS.copy()
    cfg.update(override)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=False)
    p.add_argument("--max-samples", type=int, default=None, help="override max_samples")
    p.add_argument("--epochs", type=int, default=None, help="override epochs")
    p.add_argument("--experiment-name", type=str, default=None)
    p.add_argument("--loss", type=str, default=None, choices=["log_cosh", "mse"],
                   help="training loss (default: log_cosh as per R1)")
    args = p.parse_args()

    cfg = DEFAULTS.copy()
    if args.config:
        cfg.update(_load_config(Path(args.config)))
    if args.max_samples is not None:
        cfg["max_samples"] = args.max_samples
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.experiment_name is not None:
        cfg["experiment_name"] = args.experiment_name
    if args.loss is not None:
        cfg["loss"] = args.loss

    train(cfg)


if __name__ == "__main__":
    main()
