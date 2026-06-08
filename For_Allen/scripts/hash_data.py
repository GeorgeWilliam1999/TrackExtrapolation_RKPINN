"""hash_data.py — Phase 0 data pin generator.

Computes SHA-256 of every input artifact and writes a one-file-per-asset
manifest under pins/data_manifests/. Run at Phase 0 and any time a data
artifact is regenerated; the resulting hashes are committed.

Usage:
    python scripts/hash_data.py path/to/file --label train_10M
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

CHUNK = 1 << 20


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--label", required=True, help="output filename stem under pins/data_manifests/")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"[hash_data] not found: {args.path}", file=sys.stderr)
        return 2

    digest = sha256_file(args.path)
    repo = Path(__file__).resolve().parent.parent
    out = repo / "pins" / "data_manifests" / f"{args.label}.sha256"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"{digest}  {args.path}\n")
    print(f"[hash_data] wrote {out}")
    print(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
