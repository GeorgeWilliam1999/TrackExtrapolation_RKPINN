"""
sanity_check.py — runs the 6-test per-checkpoint smoke battery from
PLAN.md §"Per-checkpoint smoke battery".

This is the Phase-0 STUB: contains the test contracts and a `--toy`
mode that runs against a 1 k-param dummy model so the test scaffold
itself is green before any of Phase 1+ exists.

Real implementations land progressively:
  - Phase 0:  determinism (1), zero-dz identity (2), manifest round-trip (6, against the V3 stub from Phase 1b)
  - Phase 1a: A4-lite (3) and forward/backward closure (4) wired against the cached RK45 reference
  - Phase 2a: 100-track stage-1 subset (5) wired against test_v1_frozen.npy

Every implemented test must:
  - run in < 5 s on the toy model,
  - return a (passed: bool, value: float, gate: float) tuple,
  - log the tuple to MLflow under metric `sanity.<test_name>`.
"""
from __future__ import annotations

import argparse
import sys
from typing import Callable, Tuple


SanityFn = Callable[[object], Tuple[bool, float, float]]


def t1_determinism(model) -> Tuple[bool, float, float]:
    raise NotImplementedError("Phase 0 task: same 8 inputs forwarded twice -> bit-identical fp32.")


def t2_zero_dz_identity(model) -> Tuple[bool, float, float]:
    raise NotImplementedError("Phase 0 task: dz=0 inputs -> outputs match inputs to <1e-5 fp32.")


def t3_a4_lite(model) -> Tuple[bool, float, float]:
    raise NotImplementedError("Phase 1a task: 16 random states, autograd Jacobian vs 5-point FD, Frobenius rel-err < 0.05.")


def t4_fwd_bwd_closure(model) -> Tuple[bool, float, float]:
    raise NotImplementedError("Phase 1a task: 64 tracks, fwd then bwd, ||round-trip - x_0|| < 50 um avg.")


def t5_stage1_100tracks(model) -> Tuple[bool, float, float]:
    raise NotImplementedError("Phase 2a task: 100-track subset of test_v1_frozen.npy, soft warn outside gate, hard fail at >3x.")


def t6_manifest_roundtrip(model) -> Tuple[bool, float, float]:
    raise NotImplementedError("Phase 1b task: write .bin, read back via Python V3 mirror, assert metadata + outputs.")


SUITE: dict[str, SanityFn] = {
    "determinism": t1_determinism,
    "zero_dz_identity": t2_zero_dz_identity,
    "a4_lite": t3_a4_lite,
    "fwd_bwd_closure": t4_fwd_bwd_closure,
    "stage1_100tracks": t5_stage1_100tracks,
    "manifest_roundtrip": t6_manifest_roundtrip,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", help="Path to a .pt checkpoint.")
    ap.add_argument("--toy", action="store_true", help="Run against a 1k-param toy model (Phase 0).")
    args = ap.parse_args()

    if not args.toy and not args.checkpoint:
        ap.error("provide --checkpoint or --toy")

    print("[sanity_check] Phase-0 stub. Test contracts defined; implementations land per PLAN.md.")
    print("[sanity_check] Available tests:")
    for name in SUITE:
        print(f"  - {name}")
    print("[sanity_check] Toy-mode pass: scaffold OK." if args.toy else "[sanity_check] Real-mode not yet implemented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
