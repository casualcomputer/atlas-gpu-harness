"""Inspect local GGUF files and propose a bounded, explicitly estimated MoE trial."""

import csv
import hashlib
import math
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


SUPPORTED_ARCHITECTURES = ("qwen35moe", "qwen3moe", "kimi-linear")
MIB = 1024 ** 2


def inspect_gguf(path):
    """Read GGUF metadata through a read-only memory map, without reading weights."""
    from gguf import GGUFReader, LlamaFileType

    model_path = Path(path).expanduser().absolute()
    if not model_path.is_file():
        raise ValueError(f"Model is not an existing file: {model_path}")
    if model_path.suffix.lower() != ".gguf":
        raise ValueError("Expected a local .gguf model file")
    try:
        reader = GGUFReader(model_path, mode="r")
    except (ValueError, IndexError, OverflowError) as exc:
        raise ValueError(f"Cannot read GGUF metadata: {exc}") from exc

    def field(name, default=None):
        item = reader.fields.get(name)
        return item.contents() if item is not None else default

    if field("split.count", 1) > 1:
        raise ValueError("This prototype requires a single-file GGUF; split models are unsupported")
    architecture = field("general.architecture", "unknown")
    file_type = field("general.file_type")
    try:
        quantization_name = LlamaFileType(file_type).name
    except (ValueError, TypeError):
        quantization_name = "unknown"
    # This deliberately excludes tensor data: identity is not a weight checksum.
    header_bytes = int(reader.data_offset)
    size = model_path.stat().st_size
    digest = hashlib.sha256()
    with model_path.open("rb") as source:
        remaining = header_bytes
        while remaining:
            chunk = source.read(min(remaining, MIB))
            if not chunk:
                raise ValueError("GGUF header is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    digest.update(str(size).encode("ascii"))
    revision = re.search(r"/snapshots/([0-9a-f]{40})(?:/|$)", model_path.as_posix())
    return {
        "path": str(model_path),
        "name": field("general.name", model_path.stem),
        "architecture": architecture,
        "block_count": field(f"{architecture}.block_count", 0),
        "expert_count": field(f"{architecture}.expert_count", 0),
        "expert_used_count": field(f"{architecture}.expert_used_count", 0),
        "context_length": field(f"{architecture}.context_length"),
        "file_bytes": size,
        "quantization": {"file_type": file_type, "name": quantization_name},
        "identity": {
            "kind": "sha256_gguf_header_and_file_size_not_weight_checksum",
            "digest": digest.hexdigest(), "header_bytes": header_bytes,
            "file_bytes": size, "hf_revision": revision.group(1) if revision else None,
        },
    }


def probe_hardware():
    """Inspect this process's GPU, RAM and workspace filesystem, without model loads."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    rows = list(csv.reader(result.stdout.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1:
        raise ValueError("This prototype requires exactly one visible NVIDIA GPU")
    name, total, used, free, driver = rows[0]
    memory = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemTotal", "MemAvailable"):
            memory[key] = int(value.split()[0]) / 1024
    workspace = Path.cwd()
    disk = shutil.disk_usage(workspace)
    return {
        "gpu": {"name": name, "total_mib": float(total), "used_mib": float(used),
                "free_mib": float(free), "driver": driver},
        "ram": {"total_mib": memory["MemTotal"], "available_mib": memory["MemAvailable"],
                "scope": "this Linux process; WSL limits apply"},
        "cpu_count": os.cpu_count(), "platform": platform.platform(),
        "disk": {"path": str(workspace), "free_bytes": disk.free, "total_bytes": disk.total,
                 "scope": "workspace filesystem; virtual disk capacity may exceed physical backing"},
    }


def plan_model(model, hardware, gpu_budget_mib):
    """Estimate a starting expert-offload count; execution must establish actual fit."""
    if model.get("expert_count", 0) <= 0:
        raise ValueError("A mixture-of-experts model with expert_count > 0 is required")
    if model.get("architecture") not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture; supported: {', '.join(SUPPORTED_ARCHITECTURES)}")
    if (isinstance(gpu_budget_mib, bool) or not isinstance(gpu_budget_mib, (int, float))
            or not math.isfinite(gpu_budget_mib) or gpu_budget_mib <= 2048):
        raise ValueError("GPU budget must be finite and greater than the 2048 MiB reserve")
    if gpu_budget_mib > hardware["gpu"]["total_mib"]:
        raise ValueError("GPU budget exceeds physical GPU memory")
    blocks = model.get("block_count", 0)
    file_mib = model.get("file_bytes", 0) / MIB
    if not isinstance(blocks, int) or blocks <= 0 or file_mib <= 0:
        raise ValueError("Positive block count and model file size are required")
    usable = gpu_budget_mib - 2048
    initial = min(blocks, math.ceil(blocks * max(0, file_mib - usable) / file_mib))
    # Even a predicted fit needs a measured baseline; a second trial explores headroom.
    alternate = initial if initial else max(1, math.ceil(blocks / 4))
    return {
        "supported": True, "engine_architecture_allowlist": list(SUPPORTED_ARCHITECTURES),
        "gpu_budget_mib": gpu_budget_mib, "reserve_mib": 2048,
        "initial_n_cpu_moe": initial,
        "candidates": [{"name": "engine_auto_baseline", "n_cpu_moe": None},
                       {"name": "expert_offload", "n_cpu_moe": alternate}],
        "estimate_only": True,
        "reason": ("Estimate assumes file bytes are spread evenly across blocks. Expert bytes, "
                   "KV cache, context length and engine allocations vary; measured inference "
                   "must establish memory fit, speed and answer quality. No fit is guaranteed."),
        "available_gpu_below_budget": hardware["gpu"]["free_mib"] < gpu_budget_mib,
        "model": model, "hardware": hardware,
    }
