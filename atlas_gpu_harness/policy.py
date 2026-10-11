"""Deterministic memory policy; no LLM is claimed or required."""

from __future__ import annotations

from .evaluator import CANDIDATES


def propose(run_id: str, retrieved: list[dict]) -> tuple[str, str, str] | None:
    own_done = {e["candidate"] for e in retrieved
                if e.get("run_id") == run_id and e.get("status") in ("completed", "failed")}
    prior_failures = {e["candidate"] for e in retrieved
                      if e.get("status") == "failed" and
                      any(not c.get("passed") for c in e.get("result", {}).get("checks", []))}
    tie_failures = [e for e in retrieved if e.get("status") == "failed" and
                    any(not c.get("passed") and "ties" in c.get("case", "")
                        for c in e.get("result", {}).get("checks", []))]
    for candidate in CANDIDATES:
        if candidate in own_done:
            continue
        if candidate != "sort_stable" and candidate in prior_failures:
            continue
        if candidate == "sort_unstable" and tie_failures:
            continue
        if candidate == "sort_stable":
            return candidate, "Establish a stable exact top-k baseline", "No baseline committed for this run"
        if candidate == "argsort_stable":
            return candidate, "Stable argsort may execute faster with the same tie contract", \
                "The stored baseline provides a correctness and latency reference"
        if candidate == "packed_topk":
            return candidate, "Encode score and tie index into a sortable 64-bit key, then partial top-k", \
                "Stable alternatives are measured; this tests a bounded partial selection with exact IDs"
        if candidate == "topk_sorted":
            return candidate, "Partial top-k may be faster, but tie behavior needs checking", \
                "Try a distinct PyTorch operator and reject it if exact IDs differ"
        return candidate, "Unstable sort may be faster", "No stored tie failure rules it out yet"
    return None
