"""Detect the GPU, memory and Docker setup that every decision depends on."""

from __future__ import annotations

import hashlib
import json
import os
import platform
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
    """(total, available) GiB on Linux, WSL, Windows and macOS."""
    import psutil

    memory = psutil.virtual_memory()
    return round(memory.total / 2**30, 1), round(memory.available / 2**30, 1)


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


def cpu_model() -> str:
    info = Path("/proc/cpuinfo")
    if info.is_file():
        for line in info.read_text().splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown")


def detect() -> dict:
    gpu_list = gpus()
    total, available = ram_gib()
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    cache.mkdir(parents=True, exist_ok=True)
    return {"gpus": gpu_list, "gpu": gpu_list[0] if gpu_list else None,
            "ram_total_gib": total, "ram_available_gib": available,
            "disk_free_gib": round(shutil.disk_usage(cache).free / 2**30, 1),
            "cpu_count": os.cpu_count(), "cpu_model": cpu_model()}


def fingerprint(hw: dict) -> str:
    """Identity of the machine for memory: include CPU and RAM because offloaded inference depends on both."""
    gpu = hw.get("gpu") or {}
    key = {"gpu": gpu.get("name"), "vram": gpu.get("vram_gib"), "driver": gpu.get("driver"),
           "gpus": len(hw.get("gpus") or []), "cpu_model": hw.get("cpu_model"),
           "cpu_count": hw.get("cpu_count"), "ram_total_gib": hw.get("ram_total_gib")}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]
