"""Round-trip + spec-conformance tests for the v3 blob format.

These tests are the Python half of the R5 acceptance gate.  The other half
is the CUDA-vs-Python bit-bound parity test (loader_v3_spec.md §4), which
lives in the Allen MR `!2497` (R6).

Tested invariants
-----------------
1. The locked candidate ``pinn_v2_ALLEN_v1`` produces a blob whose header
   matches the spec exactly.
2. The blob round-trips byte-for-byte: writing then reading reproduces the
   model's Linear weights, biases, and first-5 normalisation entries
   without any drift.
3. ``reference_forward_from_blob`` (numpy fp32) matches the PyTorch fp32
   forward pass to within the tolerance allowed by FP reassociation.
4. CRC32 is enforced: a single flipped byte must be rejected by the loader.
5. The 64 kiB Allen budget holds.
"""

from __future__ import annotations

import shutil
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Reach into the gen_3 model package for the PINN_v2 architecture.
HERE = Path(__file__).resolve()
GEN3_ROOT = HERE.parents[2]  # .../experiments/gen_3/
sys.path.insert(0, str(GEN3_ROOT / "models"))
sys.path.insert(0, str(GEN3_ROOT / "core"))

from architectures import create_model  # noqa: E402

from for_allen.export import (  # noqa: E402
    MAGIC,
    VERSION,
    load_v3_blob_into_model,
    read_v3_blob,
    reference_forward_from_blob,
    write_v3_blob,
)


CANDIDATE_DIR = GEN3_ROOT / "trained_models" / "_for_allen" / "pinn_v2_ALLEN_v1"
EXPECTED_PARAMS = 10372
EXPECTED_HIDDEN = [96, 96]
EXPECTED_ACTIVATION = "tanh"


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _load_candidate(device: str = "cpu"):
    ckpt_path = CANDIDATE_DIR / "best_model.pt"
    if not ckpt_path.exists():
        pytest.skip(f"candidate checkpoint not found at {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    cfg = ckpt["config"]
    assert cfg["model_type"] == "pinn_v2"
    assert cfg["hidden_dims"] == EXPECTED_HIDDEN
    assert cfg["activation"] == EXPECTED_ACTIVATION
    model = create_model(
        "pinn_v2",
        hidden_dims=cfg["hidden_dims"],
        activation=cfg["activation"],
        dropout=cfg.get("dropout", 0.0),
        lambda_pde=cfg.get("lambda_pde", 0.1),
        lambda_ic=cfg.get("lambda_ic", 0.1),
        n_collocation=cfg.get("n_collocation", 2),
    )
    norm = CANDIDATE_DIR / "normalization.json"
    if norm.exists():
        model.load_normalization(str(norm))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model, cfg


@pytest.fixture(scope="module")
def candidate_model():
    model, _ = _load_candidate()
    return model


@pytest.fixture(scope="module")
def written_blob(candidate_model, tmp_path_factory):
    out = tmp_path_factory.mktemp("blob") / "pinn_v2_ALLEN_v1.bin"
    summary = write_v3_blob(
        candidate_model, out, expect_n_params=EXPECTED_PARAMS
    )
    return out, summary


# -----------------------------------------------------------------------------
# 1. Spec conformance
# -----------------------------------------------------------------------------


def test_blob_header_matches_spec(written_blob):
    path, summary = written_blob
    parsed = read_v3_blob(path)
    assert parsed["magic"] == MAGIC
    assert parsed["version"] == VERSION
    assert parsed["arch_id"] == 1
    assert parsed["activation"] == "tanh"
    assert parsed["input_dim"] == 7
    assert parsed["output_dim"] == 5
    assert parsed["encoder_in_dim"] == 6
    assert parsed["n_norm"] == 5
    assert parsed["n_layers"] == 3
    assert parsed["total_params"] == EXPECTED_PARAMS

    assert parsed["layer_shapes"] == [(96, 6), (96, 96), (4, 96)]
    assert summary.fits_64kib
    assert summary.file_bytes == 41604, (
        f"file_bytes drifted from spec value: got {summary.file_bytes}"
    )


# -----------------------------------------------------------------------------
# 2. Round-trip
# -----------------------------------------------------------------------------


def test_roundtrip_weights_bitexact(candidate_model, written_blob):
    path, _ = written_blob
    factory = lambda: create_model(  # noqa: E731
        "pinn_v2",
        hidden_dims=EXPECTED_HIDDEN,
        activation=EXPECTED_ACTIVATION,
        dropout=0.0,
    )
    restored = load_v3_blob_into_model(path, factory)

    # Compare each Linear pair bit-exactly.
    src_linears = [m for m in candidate_model.encoder if isinstance(m, torch.nn.Linear)]
    src_linears.append(candidate_model.correction_head)
    dst_linears = [m for m in restored.encoder if isinstance(m, torch.nn.Linear)]
    dst_linears.append(restored.correction_head)

    assert len(src_linears) == len(dst_linears) == 3
    for i, (s, d) in enumerate(zip(src_linears, dst_linears)):
        # fp32 storage with no intermediate ops -> byte-for-byte equality.
        assert torch.equal(s.weight.float(), d.weight.float()), f"weight {i} drift"
        assert torch.equal(s.bias.float(), d.bias.float()), f"bias {i} drift"

    # Normalisation: blob only carries first 5 entries.
    assert torch.equal(
        candidate_model.input_mean[:5].float(), restored.input_mean[:5].float()
    )
    assert torch.equal(
        candidate_model.input_std[:5].float(), restored.input_std[:5].float()
    )


# -----------------------------------------------------------------------------
# 3. Numpy reference forward matches torch forward
# -----------------------------------------------------------------------------


def test_numpy_reference_matches_torch(candidate_model, written_blob):
    path, _ = written_blob
    rng = np.random.default_rng(20260520)
    # Realistic input ranges from the dataset (see notebook §3).
    N = 256
    x = np.empty((N, 7), dtype=np.float32)
    x[:, 0] = rng.uniform(-3500.0, 3500.0, N)
    x[:, 1] = rng.uniform(-2500.0, 2500.0, N)
    x[:, 2] = rng.uniform(-0.4, 0.4, N)
    x[:, 3] = rng.uniform(-0.35, 0.35, N)
    x[:, 4] = rng.uniform(-0.3, 0.3, N)
    x[:, 5] = rng.uniform(0.0, 14_000.0, N)
    x[:, 6] = rng.uniform(-10_000.0, 10_000.0, N)

    y_numpy = reference_forward_from_blob(path, x)

    with torch.no_grad():
        y_torch = candidate_model(torch.from_numpy(x)).cpu().numpy()

    # Reduction order differs between numpy.matmul and torch.matmul on CPU,
    # so we tolerate ~few ULP per output channel.  Median should be tiny.
    diff = np.abs(y_torch - y_numpy)
    rel = diff / (np.abs(y_torch) + 1e-6)
    assert diff.max() < 5e-2, (
        f"max abs diff {diff.max():.3e} too large; "
        f"numpy<->torch forward divergence"
    )
    assert np.median(rel) < 1e-5, f"median rel diff {np.median(rel):.3e} too large"


# -----------------------------------------------------------------------------
# 4. CRC32 guard
# -----------------------------------------------------------------------------


def test_crc_rejects_single_bit_flip(written_blob, tmp_path):
    src, _ = written_blob
    tampered = tmp_path / "tampered.bin"
    shutil.copyfile(src, tampered)

    raw = bytearray(tampered.read_bytes())
    # Flip a single bit in the weight payload (anywhere past the layer table).
    target = 200
    raw[target] ^= 0x01
    tampered.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="CRC32"):
        read_v3_blob(tampered)


def test_bad_magic_rejected(written_blob, tmp_path):
    src, _ = written_blob
    tampered = tmp_path / "badmagic.bin"
    raw = bytearray(src.read_bytes())
    raw[0] = ord("X")
    # Re-CRC so we don't trip the CRC check first.
    import zlib

    new_crc = zlib.crc32(bytes(raw[:-4])) & 0xFFFFFFFF
    raw[-4:] = struct.pack("<I", new_crc)
    tampered.write_bytes(bytes(raw))

    with pytest.raises(ValueError, match="bad magic"):
        read_v3_blob(tampered)


# -----------------------------------------------------------------------------
# 5. Size budget
# -----------------------------------------------------------------------------


def test_allen_size_budget(written_blob):
    _, summary = written_blob
    assert summary.file_bytes <= 64 * 1024, (
        f"blob {summary.file_bytes} B exceeds Allen 64 kiB budget"
    )
