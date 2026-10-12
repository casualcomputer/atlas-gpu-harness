"""Small, evidence-based failure descriptions; these are hints, not root-cause proof."""

import re


def describe(trial: dict) -> dict | None:
    text = f"{trial.get('error') or ''}\n{trial.get('logs') or ''}"
    rules = [
        ("memory_reservation", r"Free memory on device[^\n]*less than desired GPU memory utilization",
         "The engine requested more GPU memory than is currently free; lower its reservation or free VRAM.", False),
        ("oom", r"out of memory|OutOfMemory|\bOOM\b|failed to allocate|cudaMalloc|No available memory for the cache|Not enough memory|maximum number of tokens that can be stored",
         "GPU allocation or KV capacity failed; reduce concurrency, quantize KV, or offload weights.", False),
        ("disk_full", r"no space left on device|disk quota exceeded",
         "Storage is full; free space in the relevant model cache or Docker storage before retrying.", True),
        ("port_conflict", r"address already in use|port is already allocated",
         "The serving port is occupied; free it or choose another port before retrying.", True),
        ("authentication", r"\b401\b|\b403\b|gated repo|unauthorized|authentication|access denied",
         "Model or service access was denied; check credentials and model access, then retry.", True),
        ("unsupported", r"unrecognized arguments|unknown argument|not supported|unsupported|no kernel image|invalid choice",
         "The image rejected an option or lacks model/GPU support; inspect its help and compatibility.", False),
        ("timeout", r"Timeout|timed out|not ready after",
         "Startup or a request exceeded its deadline; inspect download progress, disk pressure and engine logs.", True),
        ("network", r"ConnectionError|ConnectError|connection refused|connection reset|name resolution|SSL|certificate|\b502\b|\b503\b",
         "A network or TLS operation failed; check connectivity and retry after correcting it.", True),
    ]
    if trial.get("status") == "failed":
        for category, pattern, explanation, retryable in rules:
            if re.search(pattern, text, re.IGNORECASE):
                return {"category": category, "explanation": explanation, "retryable": retryable}
        return {"category": "unknown", "explanation": "Failure is not classified; inspect the error and engine logs.",
                "retryable": False}
    if trial.get("status") == "quality_failed":
        return {"category": "quality", "explanation": "Answers failed the quality gate; check model, template and precision.",
                "retryable": False}
    levels = (trial.get("metrics") or {}).get("levels") or []
    if any(lv.get("ok") != lv.get("requests") for lv in levels):
        return {"category": "load_failure", "explanation": "Some load-test requests failed; inspect concurrency-level errors.",
                "retryable": False}
    limit = (trial.get("metrics") or {}).get("max_ttft_ms")
    if limit is not None and any(lv.get("ttft_p99_ms") is None or lv["ttft_p99_ms"] > limit for lv in levels):
        return {"category": "latency_limit", "explanation": "TTFT p99 exceeded the requested limit or was not measured.",
                "retryable": False}
    return None
