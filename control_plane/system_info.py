from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_total_mb: int
    memory_used_mb: int


def _run_cmd(cmd: list[str], timeout_s: float = 1.5) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=timeout_s)


def get_gpu_info() -> list[GPUInfo]:
    """
    Best-effort NVIDIA GPU query via nvidia-smi.
    Returns [] if nvidia-smi is unavailable.
    """
    if shutil.which("nvidia-smi") is None:
        return []

    try:
        out = _run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
    except Exception:
        return []

    gpus: list[GPUInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_total_mb=int(parts[2]),
                    memory_used_mb=int(parts[3]),
                )
            )
        except ValueError:
            continue
    return gpus


def get_system_info() -> dict[str, Any]:
    return {
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
        },
        "os": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "gpus": [gpu.__dict__ for gpu in get_gpu_info()],
    }

