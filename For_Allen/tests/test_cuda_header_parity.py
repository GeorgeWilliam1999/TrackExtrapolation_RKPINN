"""Bit-exact parity test: emitted PINN_V2_UTT.cuh <-> v3 blob.

Re-parses the constexpr-weights C++ header back into numpy arrays and
asserts byte-for-byte equality with the locked v3 blob.  This catches
any drift between ``emit_cuda_header.py`` (the generator) and
``read_v3_blob`` / the v3 spec (the source of truth) before the C++
ever sees the header.

The C++ side of the parity gate (a Catch2 test asserting that
``PINN_V2_UTT_Weights::kSourceBlobCRC32`` matches the blob's CRC32,
and that a host call to ``pinn_v2_utt_state`` reproduces a numpy
reference within tolerance) belongs in the Allen unit-test layer and
is tracked separately in EXECUTION_PLAN.md §6.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_FOR_ALLEN_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_FOR_ALLEN_ROOT / "src"))

from for_allen.export import read_v3_blob  # noqa: E402


_ARRAY_RE = re.compile(
    r"constexpr\s+float\s+(\w+)\s*\[\s*(\d+)\s*\]\s*=\s*\{([^}]*)\}\s*;",
    re.DOTALL,
)


def parse_header_arrays(header_path: Path) -> dict[str, np.ndarray]:
    text = header_path.read_text()
    out: dict[str, np.ndarray] = {}
    for match in _ARRAY_RE.finditer(text):
        name = match.group(1)
        n_expected = int(match.group(2))
        body = match.group(3)
        # Pull every float literal (handles 1.234e-5f, -inf, etc.).
        token_re = re.compile(r"-?\d+\.\d+e[+-]?\d+f|-?\d+\.\d+f|-?\d+f|NAN_F|INFINITY|-INFINITY")
        toks = token_re.findall(body)
        if len(toks) != n_expected:
            # Fall back to comma-split parse if the regex missed a literal.
            toks = [t.strip().rstrip("f") for t in body.split(",") if t.strip()]
        vals: list[float] = []
        for t in toks:
            t_clean = t.rstrip("f")
            if t_clean == "NAN_F":
                vals.append(float("nan"))
            elif t_clean == "INFINITY":
                vals.append(float("inf"))
            elif t_clean == "-INFINITY":
                vals.append(-float("inf"))
            else:
                vals.append(float(t_clean))
        if len(vals) != n_expected:
            raise AssertionError(
                f"{name}: parsed {len(vals)} floats, header declared {n_expected}"
            )
        out[name] = np.asarray(vals, dtype=np.float32)
    return out


def main() -> int:
    blob_path = _FOR_ALLEN_ROOT / "artifacts" / "blobs" / "v3" / "pinn_v2_ALLEN_v1.bin"
    header_path = (
        Path("/data/bfys/gscriven/Allen/device/kalman/ParKalman/include/PINN_V2_UTT.cuh")
    )
    assert blob_path.exists(), f"blob not found: {blob_path}"
    assert header_path.exists(), f"header not found: {header_path}"

    parsed_blob = read_v3_blob(blob_path)
    parsed_header = parse_header_arrays(header_path)

    expected = {
        "kInputMean": np.asarray(parsed_blob["input_mean5"], dtype=np.float32),
        "kInputStd": np.asarray(parsed_blob["input_std5"], dtype=np.float32),
    }
    layers = parsed_blob["layers"]
    for i, (W, b) in enumerate(layers):
        expected[f"kW{i}"] = W.astype(np.float32).ravel(order="C")
        expected[f"kB{i}"] = b.astype(np.float32).ravel()

    failures: list[str] = []
    for name, ref in expected.items():
        if name not in parsed_header:
            failures.append(f"  missing array in header: {name}")
            continue
        got = parsed_header[name]
        if got.shape != ref.shape:
            failures.append(
                f"  {name}: shape {got.shape} != reference shape {ref.shape}"
            )
            continue
        if not np.array_equal(got.view(np.uint32), ref.view(np.uint32)):
            n_diff = int(np.sum(got.view(np.uint32) != ref.view(np.uint32)))
            max_abs = float(np.max(np.abs(got - ref)))
            failures.append(
                f"  {name}: {n_diff}/{ref.size} fp32 bit-patterns differ (max |delta| = {max_abs:.3e})"
            )

    crc_re = re.search(r"kSourceBlobCRC32\s*=\s*0x([0-9a-fA-F]+)u", header_path.read_text())
    if crc_re is None:
        failures.append("  header is missing kSourceBlobCRC32 constant")
    else:
        crc_header = int(crc_re.group(1), 16)
        crc_blob = int(parsed_blob["crc32"])
        if crc_header != crc_blob:
            failures.append(
                f"  CRC32 mismatch: header=0x{crc_header:08x} blob=0x{crc_blob:08x}"
            )

    if failures:
        print("PARITY FAILED:")
        for f in failures:
            print(f)
        return 1

    n_floats = sum(v.size for v in expected.values())
    print(
        f"PARITY PASS: {len(expected)} arrays ({n_floats} fp32 values) bit-exact;"
        f" CRC32 0x{int(parsed_blob['crc32']):08x} matched."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
