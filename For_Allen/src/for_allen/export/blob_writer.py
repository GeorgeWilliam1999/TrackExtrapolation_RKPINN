"""V3 NN-extrapolator blob writer and loader.

The single source of truth for the byte layout is
``For_Allen/pins/loader_v3_spec.md``.  Both functions in this module assert
every field against that spec; **do not** change the layout here without
bumping the spec version and rewriting both the spec and this file.

Public API
----------
- :func:`write_v3_blob`   : serialise a trained ``PINN_v2`` instance to bytes.
- :func:`read_v3_blob`    : parse a blob back into a Python dict (no torch
  dependency on the read path beyond what's already imported).
- :func:`load_v3_blob_into_model` : convenience — restore a fresh
  ``PINN_v2`` from a blob and verify the round-trip is bit-identical.

The CUDA-side parser (R6) reads the same bytes via a ``constexpr`` shape
table; the bit-bound parity gate (spec §4) ties the two paths together.
"""

from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn

# -----------------------------------------------------------------------------
# Spec constants — must match loader_v3_spec.md verbatim.
# -----------------------------------------------------------------------------

MAGIC: bytes = b"NRKv3\0\0\0"
VERSION: int = 3

ARCH_PINN_V2_DIPOLE: int = 1
ACTIVATION_TANH: int = 1

# Locked architecture for v3.  Any deviation -> bump the spec.
EXPECTED_INPUT_DIM: int = 7
EXPECTED_OUTPUT_DIM: int = 5
EXPECTED_ENCODER_IN_DIM: int = 6  # 5 normalised state + 1 z_frac
EXPECTED_N_NORM: int = 5

# Header is sections A + B = 16 + 32 = 48 bytes, then 40-byte normalisation
# block, then the layer table.  Weight payload starts after the table; the
# spec requires the start-of-payload offset to land on a 16-byte boundary.
SECTION_AB_BYTES: int = 48
SECTION_C_BYTES: int = 40
LAYER_TABLE_ENTRY_BYTES: int = 8
PAYLOAD_ALIGNMENT: int = 16


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _activation_id_for(name: str) -> int:
    name = name.lower()
    if name == "tanh":
        return ACTIVATION_TANH
    raise ValueError(
        f"v3 blob spec only supports activation 'tanh' (got {name!r}). "
        "If you really want another activation, bump the spec to v4."
    )


def _activation_name_for(act_id: int) -> str:
    if act_id == ACTIVATION_TANH:
        return "tanh"
    raise ValueError(f"unknown activation_id {act_id} (v3 only allows 1=TANH)")


def _extract_pinn_v2_layers(
    model: nn.Module,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Walk a ``PINN_v2`` and return its three Linear layers in inference order.

    Returns a list of ``(W, b)`` numpy arrays with ``W.shape == (out, in)``.
    """
    # PINN_v2 lays out:
    #   self.encoder = Sequential(Linear, act, [Dropout], Linear, act, [Dropout], ...)
    #   self.correction_head = Linear(prev, 4)
    linears: List[nn.Linear] = [m for m in model.encoder if isinstance(m, nn.Linear)]
    linears.append(model.correction_head)
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for lin in linears:
        W = lin.weight.detach().cpu().float().numpy()
        b = lin.bias.detach().cpu().float().numpy()
        assert W.ndim == 2 and b.ndim == 1, "non-standard Linear shape"
        assert W.shape[0] == b.shape[0], "weight/bias out_dim mismatch"
        out.append((W, b))
    return out


def _extract_input_norm(model: nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """Pull the 5-channel normalisation that the network actually consumes."""
    mean_full = model.input_mean.detach().cpu().float().numpy()
    std_full = model.input_std.detach().cpu().float().numpy()
    if mean_full.shape[0] < EXPECTED_N_NORM or std_full.shape[0] < EXPECTED_N_NORM:
        raise ValueError(
            f"model.input_mean / input_std must have at least {EXPECTED_N_NORM} "
            f"entries (got {mean_full.shape[0]} / {std_full.shape[0]})"
        )
    mean5 = mean_full[:EXPECTED_N_NORM].astype(np.float32, copy=True)
    std5 = std_full[:EXPECTED_N_NORM].astype(np.float32, copy=True)
    if not np.all(std5 > 0):
        raise ValueError(f"input_std must be strictly positive, got {std5}")
    return mean5, std5


# -----------------------------------------------------------------------------
# Writer
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class V3BlobSummary:
    """Diagnostic summary returned by :func:`write_v3_blob`."""

    n_layers: int
    total_params: int
    layer_shapes: List[Tuple[int, int]]
    payload_bytes: int
    file_bytes: int
    crc32: int
    fits_64kib: bool


def write_v3_blob(
    model: nn.Module,
    out_path: str | Path,
    *,
    expect_n_params: int | None = None,
    max_bytes: int = 64 * 1024,
) -> V3BlobSummary:
    """Serialise a ``PINN_v2`` instance to a v3 blob on disk.

    Parameters
    ----------
    model
        A ``PINN_v2`` (or duck-compatible) module with ``encoder``,
        ``correction_head``, ``input_mean``, ``input_std`` attributes.
    out_path
        Destination path. Parents must already exist.
    expect_n_params
        If given, the writer asserts the model parameter count matches.
        Used to catch silent architecture drift; pass ``10372`` for
        ``pinn_v2_ALLEN_v1``.
    max_bytes
        Hard cap on the total blob size. Defaults to 64 kiB (Allen budget).
    """
    out_path = Path(out_path)
    activation = getattr(model, "activation_name", "tanh")
    activation_id = _activation_id_for(activation)

    layers = _extract_pinn_v2_layers(model)
    layer_shapes = [(int(W.shape[0]), int(W.shape[1])) for W, _ in layers]

    # Architecture invariants required by the v3 spec.
    assert layer_shapes[0][1] == EXPECTED_ENCODER_IN_DIM, (
        f"layer 0 input must be {EXPECTED_ENCODER_IN_DIM} dims (got {layer_shapes[0][1]})"
    )
    assert layer_shapes[-1][0] == 4, (
        f"final layer output must be 4 (correction head), got {layer_shapes[-1][0]}"
    )

    total_params = sum(W.size + b.size for W, b in layers)
    if expect_n_params is not None and total_params != expect_n_params:
        raise AssertionError(
            f"param count mismatch: model has {total_params}, expected "
            f"{expect_n_params} -- architecture drifted?"
        )

    mean5, std5 = _extract_input_norm(model)

    # ---- Compute layer-table padding so the payload is 16-B aligned. ----
    table_bytes = LAYER_TABLE_ENTRY_BYTES * len(layers)
    table_offset = SECTION_AB_BYTES + SECTION_C_BYTES  # = 88
    raw_payload_offset = table_offset + table_bytes
    pad_bytes = (-raw_payload_offset) % PAYLOAD_ALIGNMENT
    payload_offset = raw_payload_offset + pad_bytes
    assert payload_offset % PAYLOAD_ALIGNMENT == 0

    # ---- Pack sections A, B, C ----
    buf = bytearray()
    buf += MAGIC  # bytes 0..8
    buf += struct.pack("<II", VERSION, 0)  # version + _pad0
    buf += struct.pack(
        "<IIIIIIII",
        ARCH_PINN_V2_DIPOLE,
        activation_id,
        EXPECTED_INPUT_DIM,
        EXPECTED_OUTPUT_DIM,
        EXPECTED_ENCODER_IN_DIM,
        EXPECTED_N_NORM,
        len(layers),
        total_params,
    )
    assert len(buf) == SECTION_AB_BYTES, (len(buf), SECTION_AB_BYTES)

    buf += mean5.astype("<f4").tobytes()
    buf += std5.astype("<f4").tobytes()
    assert len(buf) == SECTION_AB_BYTES + SECTION_C_BYTES

    # ---- Section D: layer table (+ pad to 16-B boundary) ----
    for out_dim, in_dim in layer_shapes:
        buf += struct.pack("<II", out_dim, in_dim)
    buf += b"\0" * pad_bytes
    assert len(buf) == payload_offset

    # ---- Section E: weights, row-major (matches torch nn.Linear storage). ----
    for W, b in layers:
        W32 = np.ascontiguousarray(W, dtype="<f4")
        b32 = np.ascontiguousarray(b, dtype="<f4")
        buf += W32.tobytes()
        buf += b32.tobytes()

    # ---- Section F: CRC32 trailer ----
    crc = zlib.crc32(bytes(buf)) & 0xFFFFFFFF
    buf += struct.pack("<I", crc)

    if len(buf) > max_bytes:
        raise AssertionError(
            f"blob is {len(buf)} bytes, exceeds budget {max_bytes}"
        )

    out_path.write_bytes(bytes(buf))

    return V3BlobSummary(
        n_layers=len(layers),
        total_params=total_params,
        layer_shapes=layer_shapes,
        payload_bytes=len(buf) - 4 - payload_offset,
        file_bytes=len(buf),
        crc32=crc,
        fits_64kib=len(buf) <= 64 * 1024,
    )


# -----------------------------------------------------------------------------
# Reader
# -----------------------------------------------------------------------------


def read_v3_blob(path: str | Path) -> Dict[str, object]:
    """Parse a v3 blob into a plain dict.  CRC32 is verified."""
    path = Path(path)
    raw = path.read_bytes()

    if len(raw) < SECTION_AB_BYTES + SECTION_C_BYTES + LAYER_TABLE_ENTRY_BYTES + 4:
        raise ValueError(f"blob {path} too small ({len(raw)} bytes)")

    payload_no_crc = raw[:-4]
    crc_stored = struct.unpack("<I", raw[-4:])[0]
    crc_computed = zlib.crc32(payload_no_crc) & 0xFFFFFFFF
    if crc_stored != crc_computed:
        raise ValueError(
            f"CRC32 mismatch in {path}: stored=0x{crc_stored:08x} "
            f"computed=0x{crc_computed:08x}"
        )

    magic = raw[:8]
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r} (expected {MAGIC!r})")
    version, _pad0 = struct.unpack_from("<II", raw, 8)
    if version != VERSION:
        raise ValueError(f"unsupported version {version} (this loader is v{VERSION})")
    if _pad0 != 0:
        raise ValueError(f"_pad0 must be zero, got {_pad0}")

    (
        arch_id,
        activation_id,
        input_dim,
        output_dim,
        encoder_in_dim,
        n_norm,
        n_layers,
        total_params,
    ) = struct.unpack_from("<IIIIIIII", raw, 16)

    if arch_id != ARCH_PINN_V2_DIPOLE:
        raise ValueError(f"unsupported arch_id {arch_id}")
    if input_dim != EXPECTED_INPUT_DIM or output_dim != EXPECTED_OUTPUT_DIM:
        raise ValueError(f"unexpected input/output dims: {input_dim}/{output_dim}")
    if encoder_in_dim != EXPECTED_ENCODER_IN_DIM or n_norm != EXPECTED_N_NORM:
        raise ValueError(f"unexpected encoder_in/n_norm: {encoder_in_dim}/{n_norm}")

    mean5 = np.frombuffer(raw, dtype="<f4", count=5, offset=48).copy()
    std5 = np.frombuffer(raw, dtype="<f4", count=5, offset=68).copy()
    if not np.all(std5 > 0):
        raise ValueError(f"input_std must be strictly positive, got {std5}")

    table_offset = SECTION_AB_BYTES + SECTION_C_BYTES  # 88
    table_bytes = LAYER_TABLE_ENTRY_BYTES * n_layers
    raw_payload_offset = table_offset + table_bytes
    pad_bytes = (-raw_payload_offset) % PAYLOAD_ALIGNMENT
    payload_offset = raw_payload_offset + pad_bytes

    layer_shapes: List[Tuple[int, int]] = []
    for i in range(n_layers):
        out_dim, in_dim = struct.unpack_from("<II", raw, table_offset + i * 8)
        layer_shapes.append((out_dim, in_dim))

    # Pad bytes must be zero.
    if raw[raw_payload_offset:payload_offset] != b"\0" * pad_bytes:
        raise ValueError("non-zero bytes in layer-table padding")

    cursor = payload_offset
    layers: List[Tuple[np.ndarray, np.ndarray]] = []
    counted_params = 0
    for out_dim, in_dim in layer_shapes:
        W = np.frombuffer(raw, dtype="<f4", count=out_dim * in_dim, offset=cursor)
        W = W.reshape(out_dim, in_dim).copy()
        cursor += out_dim * in_dim * 4
        b = np.frombuffer(raw, dtype="<f4", count=out_dim, offset=cursor).copy()
        cursor += out_dim * 4
        layers.append((W, b))
        counted_params += W.size + b.size

    if counted_params != total_params:
        raise ValueError(
            f"weight payload param count {counted_params} != header.total_params {total_params}"
        )
    if cursor + 4 != len(raw):
        raise ValueError(
            f"trailing bytes after weight payload: cursor={cursor}, file_size={len(raw)}"
        )

    return {
        "magic": magic,
        "version": version,
        "arch_id": arch_id,
        "activation_id": activation_id,
        "activation": _activation_name_for(activation_id),
        "input_dim": input_dim,
        "output_dim": output_dim,
        "encoder_in_dim": encoder_in_dim,
        "n_norm": n_norm,
        "n_layers": n_layers,
        "total_params": total_params,
        "input_mean5": mean5,
        "input_std5": std5,
        "layer_shapes": layer_shapes,
        "layers": layers,
        "crc32": crc_stored,
        "file_bytes": len(raw),
    }


# -----------------------------------------------------------------------------
# Convenience: load blob back into a fresh PINN_v2 and check equivalence.
# -----------------------------------------------------------------------------


def load_v3_blob_into_model(blob_path: str | Path, model_factory) -> nn.Module:
    """Build a ``PINN_v2`` from ``model_factory()``, fill it from the blob, and
    return it in eval mode.

    ``model_factory`` is a zero-arg callable that returns a freshly initialised
    PINN_v2 with the right hidden_dims/activation. The caller is responsible
    for matching the architecture to the blob; this function asserts the match.
    """
    parsed = read_v3_blob(blob_path)
    model = model_factory()

    file_layers = parsed["layers"]
    model_linears: List[nn.Linear] = [
        m for m in model.encoder if isinstance(m, nn.Linear)
    ]
    model_linears.append(model.correction_head)

    if len(model_linears) != len(file_layers):
        raise ValueError(
            f"blob has {len(file_layers)} layers, model has {len(model_linears)}"
        )
    for i, (lin, (W, b)) in enumerate(zip(model_linears, file_layers)):
        if tuple(lin.weight.shape) != W.shape:
            raise ValueError(
                f"layer {i} shape mismatch: model={tuple(lin.weight.shape)} blob={W.shape}"
            )
        with torch.no_grad():
            lin.weight.copy_(torch.from_numpy(W))
            lin.bias.copy_(torch.from_numpy(b))

    with torch.no_grad():
        full_mean = model.input_mean.clone()
        full_std = model.input_std.clone()
        full_mean[:5] = torch.from_numpy(parsed["input_mean5"])
        full_std[:5] = torch.from_numpy(parsed["input_std5"])
        model.input_mean.copy_(full_mean)
        model.input_std.copy_(full_std)

    model.eval()
    return model


# -----------------------------------------------------------------------------
# Pure-Python reference forward pass (no torch).
# -----------------------------------------------------------------------------


def reference_forward_from_blob(
    blob_path: str | Path, x: np.ndarray
) -> np.ndarray:
    """fp32 numpy forward pass directly from a v3 blob.

    Used by the bit-bound parity test as the "canonical" Python reference:
    the CUDA loader must match this byte-for-byte (to within 1 ULP, see
    loader_v3_spec.md §4).

    ``x`` must be ``(N, 7)`` float32.  Output is ``(N, 5)`` float32.
    """
    parsed = read_v3_blob(blob_path)
    x = np.ascontiguousarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != EXPECTED_INPUT_DIM:
        raise ValueError(f"x must be (N, {EXPECTED_INPUT_DIM}); got shape {x.shape}")

    mean5 = parsed["input_mean5"].astype(np.float32)
    std5 = parsed["input_std5"].astype(np.float32)
    layers = parsed["layers"]

    x_norm = (x[:, :5] - mean5) / std5
    z_frac = np.ones((x.shape[0], 1), dtype=np.float32)
    h = np.concatenate([x_norm, z_frac], axis=1).astype(np.float32)

    n_layers = len(layers)
    for idx, (W, b) in enumerate(layers):
        W = W.astype(np.float32, copy=False)
        b = b.astype(np.float32, copy=False)
        h = h @ W.T + b
        if idx < n_layers - 1:
            h = np.tanh(h).astype(np.float32)

    corr = h  # (N, 4)
    dz = x[:, 6]
    y = np.empty((x.shape[0], 5), dtype=np.float32)
    y[:, 0] = x[:, 0] + x[:, 2] * dz + corr[:, 2] * dz
    y[:, 1] = x[:, 1] + x[:, 3] * dz + corr[:, 3] * dz
    y[:, 2] = x[:, 2] + corr[:, 0]
    y[:, 3] = x[:, 3] + corr[:, 1]
    y[:, 4] = x[:, 4]
    return y
