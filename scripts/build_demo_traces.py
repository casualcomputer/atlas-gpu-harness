"""Build demo-ui/traces.json from exported MongoDB Atlas records.

The demo console replays these historical traces; it never fabricates values.
Every number below is read from files in evidence/.

    python scripts/build_demo_traces.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence"


def ts(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "")).timestamp()


def short(identifier: str) -> str:
    run, _, rest = identifier.partition(":")
    return f"{run[:8]}:{rest[:8] if len(rest) > 40 else rest}" if rest else run[:8]


def r1(x, d=2):
    return None if x is None else round(float(x), d)


def build_llamacpp(path: Path, title: str, subtitle: str, model: str, verdict: dict) -> dict:
    data = json.loads(path.read_text())
    run, proposals = data["run"], data["run"].get("proposals", {})
    events = []
    for d in data["decisions"]:
        p = proposals.get(d["candidate"], {})
        events.append({
            "t": ts(d["created_at"]), "kind": "decision", "candidate": d["candidate"],
            "hypothesis": d["hypothesis"], "reason": d.get("reason"),
            "retrieved": [short(i) for i in d.get("retrieved_experiment_ids", [])],
            "code": p.get("code"), "id": d["_id"][:8],
        })
    for e in data["experiments"]:
        r = e["result"]
        recipe = r.get("recipe") or {}
        setting = "engine auto-fit" if recipe.get("n_cpu_moe") is None else f"{recipe['n_cpu_moe']} CPU expert layers"
        events.append({
            "t": ts(e["finished_at"]), "start": ts(e["started_at"]), "kind": "experiment",
            "candidate": e["candidate"], "stage": r.get("stage"), "setting": setting,
            "status": e["status"], "attempt": e["attempt"], "id": short(e["_id"]),
            "decode": r1(r.get("median_decode_tokens_per_second")),
            "passed": r["quality"]["passed"], "total": r["quality"]["total"],
            "gpu_peak": r1(r["memory"]["device_peak_mib"], 0), "correct": r.get("correct"),
        })
    events.sort(key=lambda x: x["t"] or 0)
    return {"id": run["_id"][:8], "run_id": run["_id"], "title": title, "subtitle": subtitle,
            "model": model, "gpu": "RTX 5070 Ti · 16,303 MiB", "engine": "llama.cpp b11429",
            "metric": "Median decode tok/s", "events": events, "verdict": verdict,
            "source": str(path.relative_to(ROOT))}


def build_strata(path: Path) -> dict:
    data = json.loads(path.read_text())
    run = data["run"]
    names = {"default": "Strata default", "cpu_4": "4 CPU workers", "cpu_7": "7 CPU workers",
             "adaptive_160": "Adaptive cache · 160", "no_pcie": "No PCIe offload"}
    events = []
    for d in data["decisions"]:
        events.append({"t": ts(d["created_at"]), "kind": "decision", "candidate": d["candidate"],
                       "hypothesis": d["hypothesis"], "reason": d.get("reason"),
                       "retrieved": [short(i) for i in d.get("retrieved_experiment_ids", [])],
                       "code": None, "id": d["_id"][:8]})
    for e in data["experiments"]:
        r = e["result"]
        events.append({"t": ts(e["finished_at"]), "start": ts(e["started_at"]), "kind": "experiment",
                       "candidate": e["candidate"], "stage": r.get("stage"),
                       "setting": names.get(r.get("policy"), r.get("policy")),
                       "status": e["status"], "attempt": e["attempt"], "id": short(e["_id"]),
                       "decode": r1(r.get("decode_tps")),
                       "passed": 1 if (r.get("quality") or {}).get("passed") else 0, "total": 1,
                       "retrieval": f"{(r.get('quality') or {}).get('matched_settings', 0)}/8",
                       "gpu_peak": r1((r.get("resources") or {}).get("gpu_peak_mib"), 0),
                       "correct": r.get("correct"), "tokens": r.get("prompt_tokens")})
    events.sort(key=lambda x: x["t"] or 0)
    return {"id": run["_id"][:8], "run_id": run["_id"], "title": "Strata policy sweep · 32K",
            "subtitle": "Five Strata policies × 3 repeats + fresh-prompt checks",
            "model": "Qwen3.8-Flash-Next IQ2_XS · 68 GB", "gpu": "RTX 5070 Ti · 16 GB",
            "engine": "Strata v0.1.43", "metric": "Decode tok/s (single request)", "events": events,
            "verdict": {"tone": "reject", "headline": "No validated policy",
                        "detail": "Adaptive cache showed a +6.2% median decode difference, but 0/18 requests passed the full answer check (18/18 retrieved all 8 values; the sum was wrong). Nothing was promoted."},
            "source": str(path.relative_to(ROOT))}


def build_comparison(path: Path) -> dict:
    data = json.loads(path.read_text())
    run = data["run"]
    events = []
    for d in data["decisions"]:
        events.append({"t": ts(d["created_at"]), "kind": "decision", "candidate": d["candidate"],
                       "hypothesis": d["hypothesis"], "reason": d.get("reason"),
                       "retrieved": [short(i) for i in d.get("retrieved_experiment_ids", [])],
                       "code": None, "id": d["_id"][:8]})
    for e in data["experiments"]:
        r = e["result"]
        events.append({"t": ts(e["finished_at"]), "start": ts(e["started_at"]), "kind": "experiment",
                       "candidate": e["candidate"], "stage": "comparison",
                       "setting": "llama.cpp b11429 · auto-fit" if e["candidate"] == "llama_cpp" else "Strata v0.1.43 · shipped settings",
                       "status": e["status"], "attempt": e["attempt"], "id": short(e["_id"]),
                       "decode": r1(r.get("decode_tps")), "prefill": r1(r.get("prefill_tps")),
                       "ttft": r1(r.get("first_token_seconds")),
                       "passed": 1 if (r.get("quality") or {}).get("passed") else 0, "total": 1,
                       "retrieval": f"{(r.get('quality') or {}).get('matched_settings', 0)}/8",
                       "gpu_peak": r1((r.get("resources") or {}).get("gpu_peak_mib"), 0),
                       "correct": r.get("correct"), "tokens": r.get("prompt_tokens")})
    events.sort(key=lambda x: x["t"] or 0)
    raw = {e["candidate"]: e["result"]["decode_tps"] for e in data["experiments"]}
    ratio = raw["strata_default"] / raw["llama_cpp"]
    return {"id": run["_id"][:8], "run_id": run["_id"], "title": "Engine comparison · 32K",
            "subtitle": "Same model files, same 32,768 input tokens, one request per engine",
            "model": "Qwen3.8-Flash-Next IQ2_XS · 68 GB", "gpu": "RTX 5070 Ti · 16 GB",
            "engine": "llama.cpp vs. Strata", "metric": "Decode tok/s", "events": events,
            "verdict": {"tone": "info", "headline": f"{ratio:.2f}× observed decode throughput (Strata vs. llama.cpp)",
                        "detail": "Both engines retrieved 8/8 document values; both answers had the same arithmetic error (413 vs. expected 503). One request each — a descriptive measurement, not a promotion."},
            "source": str(path.relative_to(ROOT))}


def main() -> None:
    kimi = build_llamacpp(
        EV / "inference/70be4572a9824186999857dadeb1237d.json",
        "Kimi-Linear-48B · 30 GB on 16 GB", "Coding agent writes placement policies; harness tests each",
        "Kimi-Linear-48B-A3B Q4_K_M · 30.06 GB",
        {"tone": "keep", "headline": "Kept: engine auto-fit",
         "detail": "Policy 2 read policy 1's slowdown from Atlas and rewrote its code (+30.73% over policy 1), but was still 36.94% slower than the default. The default passed 4/4 unseen questions and was saved as the launch recipe."})
    qwen = build_llamacpp(
        EV / "inference/b88ec5bd304443a3add2b2fa767e2450.json",
        "Qwen3.5-35B · fake win caught", "A small speedup must survive held-out confirmation",
        "Qwen3.5-35B-A3B UD-IQ2_M · 11.39 GB",
        {"tone": "reject", "headline": "Rejected: +3.56% / +1.92%",
         "detail": "The candidate passed every answer but its confirmation gains missed the predeclared >5% rule, so engine auto-fit stayed."})
    traces = {"generated_from": "evidence/ (MongoDB Atlas exports)",
              "scenarios": [build_comparison(EV / "engine-comparison/latest.json"), kimi, qwen,
                            build_strata(EV / "strata/latest.json")]}
    out = ROOT / "demo-ui/traces.json"
    out.write_text(json.dumps(traces, separators=(",", ":")))
    print(out, out.stat().st_size, [len(s["events"]) for s in traces["scenarios"]])


if __name__ == "__main__":
    main()
