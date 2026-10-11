"""Reproducible CLI for the hackathon demonstration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .evaluator import CONTRACT, evaluate, fingerprint, preflight
from .policy import propose

ARTIFACTS = Path("artifacts")


def _save(name: str, data: dict) -> Path:
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / name
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def _store():
    from .store import AtlasStore

    try:
        store = AtlasStore()
        store.init()
    except Exception:
        raise RuntimeError("Atlas connection or write/read check failed. Check the Sandbox URI, "
                           "database user, and public IP allowlist.") from None
    return store


def _context() -> tuple[dict, str]:
    info = preflight()
    relevant = {key: info[key] for key in ("gpu", "compute_capability", "torch",
                                            "torch_cuda_toolkit", "evaluator_version", "contract")}
    return info, fingerprint(relevant)


def _summary(store, run_id: str) -> dict:
    run = store.run(run_id)
    experiments = store.experiments_for(run_id)
    completed = [e for e in experiments if e["status"] == "completed" and
                 e.get("result", {}).get("correct") and e.get("result", {}).get("timing")]
    best = min(completed, key=lambda e: e["result"]["timing"]["median_us"], default=None)
    terminal = sum(e["status"] in ("completed", "failed") for e in experiments)
    return {"run_id": run_id, "status": run["status"], "budget_evals": run["budget_evals"],
            "terminal_evaluations": terminal, "best_experiment": best["_id"] if best else None,
            "experiments": [{"id": e["_id"], "candidate": e["candidate"],
                             "status": e["status"], "attempt": e.get("attempt", 0),
                             "correct": e.get("result", {}).get("correct"),
                             "median_us": e.get("result", {}).get("timing", {}).get("median_us")}
                            for e in experiments]}


def _worker(candidate: str) -> dict:
    from filelock import FileLock

    ARTIFACTS.mkdir(exist_ok=True)
    # The previous orphan worker must release this lock before a retry reaches CUDA.
    with FileLock(str(ARTIFACTS / "gpu.lock"), timeout=120):
        return evaluate(candidate)


def _run_worker(candidate: str) -> dict:
    try:
        child = subprocess.run([sys.executable, "-m", "atlas_gpu_harness", "worker", candidate],
                               capture_output=True, text=True, timeout=180, check=False)
        if child.returncode != 0:
            return {"candidate": candidate, "correct": False, "error":
                    f"worker exit {child.returncode}: {child.stderr[-1200:]}"}
        return json.loads(child.stdout)
    except subprocess.TimeoutExpired:
        return {"candidate": candidate, "correct": False, "error": "worker timeout after 180 seconds"}
    except (OSError, json.JSONDecodeError) as exc:
        return {"candidate": candidate, "correct": False, "error": str(exc)}


def _drive(store, run_id: str, fault_after_commit: bool = False) -> dict:
    from filelock import FileLock, Timeout

    ARTIFACTS.mkdir(exist_ok=True)
    try:
        lock = FileLock(str(ARTIFACTS / f"run-{run_id}.lock"), timeout=0)
        with lock:
            run = store.run(run_id)
            _, current_hash = _context()
            if current_hash != run["context_hash"]:
                raise RuntimeError("GPU, PyTorch, or evaluation contract changed; resume refused")
            while True:
                current = store.experiments_for(run_id)
                terminal = sum(e["status"] in ("completed", "failed") for e in current)
                retrieved = store.retrieve(run["context_hash"])
                # The history query is bounded; always include this run's own records.
                known = {e["_id"] for e in retrieved}
                retrieved.extend(e for e in current if e["_id"] not in known)
                next_step = propose(run_id, retrieved)
                if terminal >= run["budget_evals"] or next_step is None:
                    summary = _summary(store, run_id)
                    store.update_progress(run_id, "completed", summary["best_experiment"])
                    return _summary(store, run_id)
                candidate, hypothesis, reason = next_step
                identity = fingerprint({"context": run["context_hash"], "candidate": candidate})
                store.decide(run_id, [e["_id"] for e in retrieved], candidate, hypothesis, reason)
                token = store.start(run_id, identity, run["context_hash"], candidate)
                if token is None:
                    continue  # A committed identity won the race; reload Atlas state.
                print(f"evaluating {candidate}; retrieved {len(retrieved)} records", flush=True)
                result = _run_worker(candidate)
                store.finish(run_id, identity, token, result)
                print(f"committed {candidate}: correct={result.get('correct')} ", flush=True)
                if fault_after_commit:
                    print("RECOVERY TEST: exiting after Atlas commit, before progress update", flush=True)
                    os._exit(86)
                summary = _summary(store, run_id)
                store.update_progress(run_id, "running", summary["best_experiment"])
    except Timeout:
        raise RuntimeError("another controller is already driving this run") from None


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m atlas_gpu_harness")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("baseline")
    sub.add_parser("atlas-check")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--budget-evals", type=int, default=4)
    run_parser.add_argument("--fault-after-commit", action="store_true",
                            help="recovery test: exit 86 after one Atlas result commit")
    for name in ("resume", "status", "export"):
        p = sub.add_parser(name)
        p.add_argument("--run-id", required=True)
    worker = sub.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("candidate")
    args = parser.parse_args()

    if args.command == "worker":
        print(json.dumps(_worker(args.candidate)))
        return
    if args.command == "preflight":
        info = preflight()
        print(json.dumps({"manifest": str(_save("preflight.json", info)), "gpu": info["gpu"],
                          "compute_capability": info["compute_capability"]}, indent=2))
        return
    if args.command == "baseline":
        result = _worker("sort_stable")
        print(json.dumps({"artifact": str(_save("local-baseline.json", result)),
                          "correct": result["correct"],
                          "median_us": result.get("timing", {}).get("median_us")}, indent=2))
        return
    store = _store()
    if args.command == "atlas-check":
        print(json.dumps({"ping": "ok", "write_read": "ok", "database": store.db.name}))
        return
    if args.command == "run":
        if not 1 <= args.budget_evals <= 5:
            parser.error("--budget-evals must be from 1 to 5")
        info, context_hash = _context()
        run_id = store.create_run(info, context_hash, args.budget_evals)
        print(f"RUN_ID={run_id}", flush=True)
        print(json.dumps(_drive(store, run_id, args.fault_after_commit), indent=2))
        return
    if args.command == "resume":
        print(json.dumps(_drive(store, args.run_id), indent=2))
        return
    if args.command == "status":
        print(json.dumps(_summary(store, args.run_id), indent=2))
        return
    if args.command == "export":
        report = {"run": store.run(args.run_id),
                  "experiments": store.experiments_for(args.run_id),
                  "decisions": store.decisions_for(args.run_id),
                  "note": "Only measured CUDA results are evidence; no LLM or custom kernel is claimed."}
        path = _save(f"report-{args.run_id}.json", report)
        print(path)


if __name__ == "__main__":
    main()
