"""Fixed exact top-k evaluator. PyTorch operators are upstream, not our kernels."""

from __future__ import annotations

import hashlib
import json
import platform
import re
import statistics
import subprocess
from typing import Any

CONTRACT = {
    "version": "topk-exact-v1",
    "operator": "1d top-k indices, score descending, lower index first on ties",
    "shape": [16384],
    "dtype": "float32",
    "layout": "contiguous",
    "k": 64,
    "seed": 20261011,
    "warmup": 10,
    "samples": 30,
    "unit": "microseconds",
    "timing": "CUDA events, one operator call per sample, end event synchronized",
}
EVALUATOR_VERSION = "3"
CANDIDATES = ("sort_stable", "argsort_stable", "packed_topk", "topk_sorted", "sort_unstable")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def preflight() -> dict:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot use CUDA on this machine")
    # Prove actual execution, rather than trusting the device name alone.
    x = torch.tensor([2.0, 3.0], device="cuda")
    if (x * x).sum().item() != 13.0:
        raise RuntimeError("CUDA smoke computation failed")
    device = torch.cuda.get_device_properties(0)
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        smi = "unavailable"
    try:
        smi_banner = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                    check=True, timeout=10).stdout
        match = re.search(r"CUDA Version:\s*([0-9.]+)", smi_banner)
        driver_cuda = match.group(1) if match else "unavailable"
    except (OSError, subprocess.SubprocessError):
        driver_cuda = "unavailable"
    try:
        nvcc = subprocess.run(["nvcc", "--version"], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        nvcc = "not installed"
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "gpu": device.name,
        "vram_bytes": device.total_memory,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "driver_and_vram_query": smi,
        "driver_supported_cuda": driver_cuda,
        "nvcc": nvcc,
        "torch": torch.__version__,
        "torch_cuda_toolkit": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repository_commit": commit,
        "evaluator_version": EVALUATOR_VERSION,
        "contract": CONTRACT,
    }


def _operator(scores, k: int, candidate: str):
    import torch

    if candidate == "sort_stable":
        return torch.sort(scores, descending=True, stable=True).indices[:k]
    if candidate == "argsort_stable":
        return torch.argsort(scores, descending=True, stable=True)[:k]
    if candidate == "packed_topk":
        # Monotonic float32 bit rank in the high word; lower original index wins ties.
        # Normalize signed zero because Python's oracle treats +0 and -0 equally.
        bits = torch.where(scores == 0, torch.zeros_like(scores), scores).view(torch.int32)
        unsigned = bits.to(torch.int64) & 0xFFFFFFFF
        ordered = torch.where(bits < 0, (~unsigned) & 0xFFFFFFFF,
                              unsigned ^ 0x80000000)
        positions = torch.arange(scores.numel(), device=scores.device, dtype=torch.int64)
        keys = ((ordered - 0x80000000) << 32) | (0xFFFFFFFF - positions)
        return torch.topk(keys, k, sorted=True).indices
    if candidate == "topk_sorted":
        return torch.topk(scores, k, sorted=True).indices
    if candidate == "sort_unstable":
        return torch.sort(scores, descending=True, stable=False).indices[:k]
    raise ValueError(f"unknown candidate: {candidate}")


def _oracle(values: list[float], k: int) -> list[int]:
    return sorted(range(len(values)), key=lambda i: (-values[i], i))[:k]


def _samples(scores, k: int, candidate: str, count: int) -> list[float]:
    import torch

    results = []
    for _ in range(count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _operator(scores, k, candidate)
        end.record()
        end.synchronize()
        results.append(start.elapsed_time(end) * 1000.0)
    return results


def _summary(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "samples_us": samples,
        "count": len(samples),
        "median_us": statistics.median(samples),
        "min_us": ordered[0],
        "max_us": ordered[-1],
        "p05_us": ordered[max(0, int(len(ordered) * 0.05) - 1)],
        "p95_us": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
    }


def evaluate(candidate: str) -> dict:
    import torch

    if candidate not in CANDIDATES:
        raise ValueError(candidate)
    preflight()
    fixtures = [
        ([3.0, 3.0, 2.0, 1.0], 2, "ties"),
        ([1.0, 4.0, 2.0, 3.0], 1, "k_one"),
        ([4.0, 1.0, 3.0, 2.0], 4, "k_all"),
        ([0.0, -1.0, 0.0, -2.0], 2, "negative_and_ties"),
    ]
    checks = []
    for values, k, name in fixtures:
        actual = _operator(torch.tensor(values, device="cuda"), k, candidate).cpu().tolist()
        expected = _oracle(values, k)
        checks.append({"case": name, "expected": expected, "actual": actual, "passed": actual == expected})
    generator = torch.Generator(device="cpu").manual_seed(CONTRACT["seed"])
    # Random input is fixed once per evaluation and copied to the GPU before timing.
    cpu = torch.rand(CONTRACT["shape"], generator=generator, dtype=torch.float32)
    expected = _oracle(cpu.tolist(), CONTRACT["k"])
    scores = cpu.cuda()
    actual = _operator(scores, CONTRACT["k"], candidate).cpu().tolist()
    checks.append({"case": "fixed_workload", "expected_sha256": fingerprint(expected),
                   "actual_sha256": fingerprint(actual), "passed": actual == expected})
    correct = all(item["passed"] for item in checks)
    evidence = {"candidate": candidate, "correct": correct, "checks": checks,
                "input_sha256": fingerprint(cpu.tolist()), "contract_sha256": fingerprint(CONTRACT)}
    if not correct:
        return evidence
    for _ in range(CONTRACT["warmup"]):
        _operator(scores, CONTRACT["k"], candidate)
    torch.cuda.synchronize()
    samples = _samples(scores, CONTRACT["k"], candidate, CONTRACT["samples"])
    evidence["timing"] = _summary(samples)
    if candidate != "sort_stable":
        # Both sides need warmup: the baseline may not have run in this worker.
        for _ in range(CONTRACT["warmup"]):
            _operator(scores, CONTRACT["k"], "sort_stable")
            _operator(scores, CONTRACT["k"], candidate)
        torch.cuda.synchronize()
        # Alternating order guards against a single hot/cold run being called a win.
        base, contender = [], []
        for i in range(12):
            if i % 2:
                contender += _samples(scores, CONTRACT["k"], candidate, 1)
                base += _samples(scores, CONTRACT["k"], "sort_stable", 1)
            else:
                base += _samples(scores, CONTRACT["k"], "sort_stable", 1)
                contender += _samples(scores, CONTRACT["k"], candidate, 1)
        evidence["paired_recheck"] = {
            "baseline": _summary(base), "candidate": _summary(contender),
            "speedup_supported": max(contender) < min(base),
        }
    return evidence
