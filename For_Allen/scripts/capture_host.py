"""capture_host.py — Phase 0 host pin generator.

Writes a pins/host.txt with hostname, kernel, GPU info, CUDA & cuDNN
versions, conda info. Run once at Phase 0; re-run any time the host
changes. Output is committed.
"""
from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str:
    if shutil.which(cmd[0]) is None:
        return f"(not available: {cmd[0]})"
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=30).strip()
    except subprocess.CalledProcessError as e:
        return f"(non-zero exit: {e.returncode})\n{e.output}"
    except Exception as e:  # noqa: BLE001
        return f"(error: {e})"


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    out = repo / "pins" / "host.txt"
    lines: list[str] = []
    lines.append(f"# captured: {_dt.datetime.utcnow().isoformat(timespec='seconds')}Z")
    lines.append("")
    lines.append("## hostname")
    lines.append(_run(["hostname"]))
    lines.append("")
    lines.append("## uname -a")
    lines.append(_run(["uname", "-a"]))
    lines.append("")
    lines.append("## nvidia-smi")
    lines.append(_run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv"]))
    lines.append("")
    lines.append("## nvcc --version")
    lines.append(_run(["nvcc", "--version"]))
    lines.append("")
    lines.append("## torch versions")
    try:
        import torch  # type: ignore[import-not-found]

        cudnn = torch.backends.cudnn.version()
        lines.append(f"torch={torch.__version__} cuda={torch.version.cuda} cudnn={cudnn}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"(torch import failed: {e})")
    lines.append("")
    lines.append("## conda")
    lines.append(_run(["conda", "info"]))

    out.write_text("\n".join(lines) + "\n")
    print(f"[capture_host] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
