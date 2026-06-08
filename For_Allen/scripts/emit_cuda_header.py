"""Emit the ``PINN_V2_UTT.cuh`` constexpr-weights header from a v3 blob.

Usage
-----

    python -m for_allen.scripts.emit_cuda_header \
        --blob For_Allen/artifacts/blobs/v3/pinn_v2_ALLEN_v1.bin \
        --out  Allen/device/kalman/ParKalman/include/PINN_V2_UTT.cuh

The emitted header is hermetic: no I/O at runtime, all weights live in
``constexpr float`` arrays.  The header carries the source blob's CRC32
so a future loader-side ``static_assert`` (or Catch2 test) can pin the
weights to the locked artefact and refuse to compile against a drifted
blob.

This script is the single seam between the Python-side spec (the v3
blob) and the CUDA-side build artefact.  Both sides remain pinned to
``loader_v3_spec.md``: this script reads the blob via
``for_allen.export.read_v3_blob`` (which itself asserts every spec
invariant), so any divergence between spec and blob is caught here,
before any C++ is generated.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import textwrap
from pathlib import Path
from typing import List, Sequence

import numpy as np

# Make the in-tree ``for_allen`` package importable when running this
# script directly from a checkout (mirrors the convention in
# tests/test_blob_roundtrip.py).
_HERE = Path(__file__).resolve()
_FOR_ALLEN_SRC = _HERE.parents[1] / "src"
if str(_FOR_ALLEN_SRC) not in sys.path:
    sys.path.insert(0, str(_FOR_ALLEN_SRC))

from for_allen.export import read_v3_blob  # noqa: E402


# ---------------------------------------------------------------------------
# Float emission
# ---------------------------------------------------------------------------

def _fmt_f32(x: float) -> str:
    """Round-trip-safe float32 literal.

    ``repr`` on a Python float (which is fp64) is *not* sufficient — once
    cast to fp32 it can collide with a neighbouring fp32 value.  We use
    the bit-pattern via numpy to be certain the literal we emit parses
    back to the *exact* fp32 value already stored in the blob.  ``%.9g``
    is the minimum precision that guarantees fp32 round-trip; we use
    ``%.9e`` (always normalised) for legibility.
    """
    f = float(np.float32(x))
    if np.isnan(f):
        return "NAN_F"
    if np.isinf(f):
        return "INFINITY" if f > 0 else "-INFINITY"
    s = f"{f:.9e}f"
    return s


def _emit_array(name: str, values: Sequence[float], per_line: int = 6) -> str:
    """Emit a ``constexpr float NAME[N] = { ... };`` block."""
    lines: List[str] = []
    lines.append(f"constexpr float {name}[{len(values)}] = {{")
    for i in range(0, len(values), per_line):
        chunk = values[i : i + per_line]
        lines.append("    " + ", ".join(_fmt_f32(v) for v in chunk) + ",")
    # Strip the trailing comma on the final element line to keep -Wpedantic happy.
    lines[-1] = lines[-1].rstrip(",")
    lines.append("};")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main emitter
# ---------------------------------------------------------------------------

_HEADER_BANNER = """\
/***************************************************************************** \\
* (c) Copyright 2026 CERN for the benefit of the LHCb Collaboration            *
*                                                                              *
* This software is distributed under the terms of the Apache License           *
* version 2, copied verbatim in the file "LICENSE".                            *
*                                                                              *
* In applying this licence, CERN does not waive the privileges and immunities  *
* granted to it by virtue of its Statute or as an Institution or individual    *
* with the agreement of CERN.                                                  *
\\*****************************************************************************/

// =============================================================================
//  PINN_V2 UT->T extrapolator weights (read-only, generated artefact)
//
//  GENERATED FILE -- do not edit by hand.
//
//  Source blob       : {blob_path}
//  Blob SHA256       : {blob_sha}
//  Blob CRC32        : 0x{crc32:08x}
//  Spec              : For_Allen/pins/loader_v3_spec.md (v3, magic "NRKv3")
//  Emitter           : For_Allen/scripts/emit_cuda_header.py
//  Generated         : {generated_at}
//
//  Architecture (locked, arch_id=1 = PINN_V2_DIPOLE):
//    encoder : Linear(6, 96) -> tanh -> Linear(96, 96) -> tanh
//    head    : Linear(96, 4)              (no activation)
//    inputs  : (x, y, tx, ty, qop) z-score normalised; z_frac = 1.0f
//              constant appended internally; (z_start, dz) consumed
//              outside the network.
//    outputs : (c0=delta_tx, c1=delta_ty, c2=x_corr_per_dz, c3=y_corr_per_dz)
//              wrapped by the envelope in pinn_v2_utt_state() below.
//
//  All MACs use fmaf.  The reduction order matches the canonical Python
//  reference in For_Allen/src/for_allen/export/blob_writer.py once that
//  reference is converted to scalar fmaf order (R6 parity gate).
// =============================================================================
"""


def emit_header(blob_path: Path, out_path: Path) -> dict:
    parsed = read_v3_blob(blob_path)

    # --- sanity-check the parsed blob against the locked invariants ----
    assert parsed["arch_id"] == 1, "v3 spec requires arch_id == 1 (PINN_V2_DIPOLE)"
    assert parsed["activation_id"] == 1, "v3 spec requires activation_id == 1 (TANH)"
    assert parsed["input_dim"] == 7
    assert parsed["output_dim"] == 5
    assert parsed["encoder_in_dim"] == 6
    assert parsed["n_norm"] == 5
    assert parsed["n_layers"] == 3
    layers = parsed["layers"]
    expected_shapes = [(96, 6), (96, 96), (4, 96)]
    for (W, _b), (e_out, e_in) in zip(layers, expected_shapes):
        assert W.shape == (e_out, e_in), f"unexpected layer shape {W.shape} (expected {(e_out, e_in)})"

    mean5 = np.asarray(parsed["input_mean5"], dtype=np.float32)
    std5 = np.asarray(parsed["input_std5"], dtype=np.float32)

    # --- compute provenance ---
    import hashlib

    raw = Path(blob_path).read_bytes()
    blob_sha = hashlib.sha256(raw).hexdigest()
    blob_crc = int(parsed.get("crc32", 0))
    if blob_crc == 0:
        # Fall back to recomputing from the trailer.
        import struct
        blob_crc = struct.unpack("<I", raw[-4:])[0]
    generated_at = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    # --- emit ---
    banner = _HEADER_BANNER.format(
        blob_path=blob_path.as_posix(),
        blob_sha=blob_sha,
        crc32=blob_crc,
        generated_at=generated_at,
    )

    parts: List[str] = [banner, ""]
    parts.append("#pragma once")
    parts.append("")
    parts.append('#include "ParKalmanDefinitions.cuh"  // KalmanFloat')
    parts.append("")
    parts.append("namespace ParKalmanFilter {")
    parts.append("namespace PINN_V2_UTT_Weights {")
    parts.append("")
    parts.append("// CRC32 of the source v3 blob; pin this in a Catch2 test.")
    parts.append(f"constexpr unsigned int kSourceBlobCRC32 = 0x{blob_crc:08x}u;")
    parts.append("")
    parts.append("// --- input normalisation (5 channels: x, y, tx, ty, qop) ---")
    parts.append(_emit_array("kInputMean", mean5.tolist(), per_line=5))
    parts.append("")
    parts.append(_emit_array("kInputStd", std5.tolist(), per_line=5))
    parts.append("")

    W0, b0 = layers[0]
    W1, b1 = layers[1]
    W2, b2 = layers[2]
    # Row-major flatten matches torch nn.Linear storage and the v3 spec.
    parts.append("// --- encoder layer 0: Linear(6 -> 96) ---")
    parts.append(_emit_array("kW0", W0.ravel(order="C").tolist(), per_line=6))
    parts.append("")
    parts.append(_emit_array("kB0", b0.ravel().tolist(), per_line=6))
    parts.append("")
    parts.append("// --- encoder layer 1: Linear(96 -> 96) ---")
    parts.append(_emit_array("kW1", W1.ravel(order="C").tolist(), per_line=6))
    parts.append("")
    parts.append(_emit_array("kB1", b1.ravel().tolist(), per_line=6))
    parts.append("")
    parts.append("// --- correction head: Linear(96 -> 4) ---")
    parts.append(_emit_array("kW2", W2.ravel(order="C").tolist(), per_line=6))
    parts.append("")
    parts.append(_emit_array("kB2", b2.ravel().tolist(), per_line=6))
    parts.append("")
    parts.append("}  // namespace PINN_V2_UTT_Weights")
    parts.append("")

    # ------------------------------------------------------------------
    # Inline device function: pinn_v2_utt_state.
    #
    # This is the canonical CUDA implementation of the spec §3 envelope.
    # The Python reference in blob_writer.py::reference_forward_from_blob
    # must mirror this MAC order (left-to-right scalar fma) for the
    # 1-ULP parity gate; see EXECUTION_PLAN.md §6 for the gate
    # definition.
    # ------------------------------------------------------------------
    parts.append(textwrap.dedent("""\
        // Forward pass of the locked PINN_v2 UT->T extrapolator.
        //
        // Inputs:  state at z = 2642.5 mm (end of UT layers)
        // Outputs: state at z = 7855   mm (begin of T stations)
        //
        // dz = z_end - z_start. The spec §3 envelope wraps a straight-line
        // drift plus a per-dz correction; we therefore divide-by-dz is *not*
        // needed -- the network's c2, c3 are already scaled by dz internally.
        //
        // qop is unchanged (spec envelope rule).
        __device__ __host__ inline void pinn_v2_utt_state(
          KalmanFloat x_in,
          KalmanFloat y_in,
          KalmanFloat tx_in,
          KalmanFloat ty_in,
          KalmanFloat qop_in,
          KalmanFloat dz,
          KalmanFloat& x_out,
          KalmanFloat& y_out,
          KalmanFloat& tx_out,
          KalmanFloat& ty_out)
        {
          using namespace PINN_V2_UTT_Weights;

          // --- normalise the 5 physical inputs ----------------------------
          const float n0 = (float(x_in)   - kInputMean[0]) / kInputStd[0];
          const float n1 = (float(y_in)   - kInputMean[1]) / kInputStd[1];
          const float n2 = (float(tx_in)  - kInputMean[2]) / kInputStd[2];
          const float n3 = (float(ty_in)  - kInputMean[3]) / kInputStd[3];
          const float n4 = (float(qop_in) - kInputMean[4]) / kInputStd[4];

          // --- 6-dim encoder input: (norm[0..4], z_frac = 1.0f) ------------
          float in6[6] = { n0, n1, n2, n3, n4, 1.0f };

          // --- layer 0: Linear(6 -> 96) ----------------------------------
          float h0[96];
          for (int o = 0; o < 96; ++o) {
            float acc = kB0[o];
            #pragma unroll
            for (int i = 0; i < 6; ++i) {
              acc = fmaf(kW0[o * 6 + i], in6[i], acc);
            }
            h0[o] = tanhf(acc);
          }

          // --- layer 1: Linear(96 -> 96) ---------------------------------
          float h1[96];
          for (int o = 0; o < 96; ++o) {
            float acc = kB1[o];
            for (int i = 0; i < 96; ++i) {
              acc = fmaf(kW1[o * 96 + i], h0[i], acc);
            }
            h1[o] = tanhf(acc);
          }

          // --- correction head: Linear(96 -> 4), no activation ------------
          float c[4];
          for (int o = 0; o < 4; ++o) {
            float acc = kB2[o];
            for (int i = 0; i < 96; ++i) {
              acc = fmaf(kW2[o * 96 + i], h1[i], acc);
            }
            c[o] = acc;
          }

          // --- envelope (spec §3) ----------------------------------------
          //   x'  = x  + tx * dz + c[2] * dz
          //   y'  = y  + ty * dz + c[3] * dz
          //   tx' = tx + c[0]
          //   ty' = ty + c[1]
          //   qop' = qop  (handled by caller)
          const float dzf = float(dz);
          float xo = float(x_in);
          xo = fmaf(float(tx_in), dzf, xo);
          xo = fmaf(c[2], dzf, xo);

          float yo = float(y_in);
          yo = fmaf(float(ty_in), dzf, yo);
          yo = fmaf(c[3], dzf, yo);

          x_out  = KalmanFloat(xo);
          y_out  = KalmanFloat(yo);
          tx_out = KalmanFloat(float(tx_in) + c[0]);
          ty_out = KalmanFloat(float(ty_in) + c[1]);
        }
    """))

    parts.append("}  // namespace ParKalmanFilter")
    parts.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))

    return {
        "blob_sha256": blob_sha,
        "blob_crc32": blob_crc,
        "out_path": str(out_path),
        "out_bytes": out_path.stat().st_size,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blob", type=Path, required=True, help="Path to v3 blob (.bin).")
    parser.add_argument("--out", type=Path, required=True, help="Destination .cuh path.")
    args = parser.parse_args(argv)

    info = emit_header(args.blob, args.out)
    print(
        "emitted {out_path} ({out_bytes} bytes); source CRC32=0x{blob_crc32:08x}, SHA256={blob_sha256}".format(**info)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
