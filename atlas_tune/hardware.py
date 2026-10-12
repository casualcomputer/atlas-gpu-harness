"""Detect the GPU, memory and Docker setup that every decision depends on."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], timeout: float = 10) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True).stdout


def gpus() -> list[dict]:
    try:
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                    "--format=csv,noheader,nounits"])
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.strip().splitlines():
        name, total, used, driver = [part.strip() for part in line.split(",")]
        rows.append({"name": name, "vram_gib": round(float(total) / 1024, 2),
                     "used_gib": round(float(used) / 1024, 2), "driver": driver})
    return rows


def ram_gib() -> tuple[float, float]:
    total = available = 0.0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key == "MemTotal":
                total = int(value.split()[0]) / 2**20
            elif key == "MemAvailable":
                available = int(value.split()[0]) / 2**20
    except (OSError, ValueError):
        pass
    return round(total, 1), round(available, 1)


def docker() -> dict:
    if not shutil.which("docker"):
        return {"ok": False, "detail": "docker not installed"}
    try:
        info = json.loads(_run(["docker", "info", "--format", "{{json .}}"], timeout=15))
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"ok": False, "detail": "docker daemon not reachable"}
    runtimes = set((info.get("Runtimes") or {}).keys())
    return {"ok": True, "version": info.get("ServerVersion"), "nvidia_runtime": "nvidia" in runtimes,
            "detail": f"docker {info.get('ServerVersion')}"}


def detect() -> dict:
    gpu_list = gpus()
    total, available = ram_gib()
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    cache.mkdir(parents=True, exist_ok=True)
    return {"gpus": gpu_list, "gpu": gpu_list[0] if gpu_list else None,
            "ram_total_gib": total, "ram_available_gib": available,
            "disk_free_gib": round(shutil.disk_usage(cache).free / 2**30, 1),
            "cpu_count": os.cpu_count()}


def fingerprint(hw: dict) -> str:
    """Identity of the machine for memory: same GPU model, VRAM and driver → comparable trials."""
    gpu = hw.get("gpu") or {}
    key = {"gpu": gpu.get("name"), "vram": gpu.get("vram_gib"), "driver": gpu.get("driver"),
           "gpus": len(hw.get("gpus") or [])}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
