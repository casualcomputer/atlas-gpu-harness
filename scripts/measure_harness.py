"""Controlled live Atlas comparison: hide or expose one run's historical evidence.

Run from the repository root. This creates four new Atlas runs; it never changes
the seed run. GPU evaluation and the experiment policy are used without changes.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atlas_gpu_harness.__main__ import _context, _drive
from atlas_gpu_harness.evaluator import CANDIDATES, CONTRACT, EVALUATOR_VERSION
from atlas_gpu_harness.store import AtlasStore, now


class HistoryArm:
    """Override only cross-run retrieval; _drive still includes its own records."""

    def __init__(self, store, history):
        self.store = store
        self.history = history

    def __getattr__(self, name):
        return getattr(self.store, name)

    def retrieve(self, context_hash):
        return copy.deepcopy(self.history)


def compare_blocks(arms):
    comparisons = []
    for block in (1, 2):
        group = {arm["condition"]: arm for arm in arms if arm["block"] == block}
        if len(group) != 2:
            continue
        off, on = group["history_off"], group["history_on"]
        off_count, on_count = off["evaluation_count"], on["evaluation_count"]
        comparisons.append({
            "block": block,
            "off_run_id": off["run"]["_id"], "on_run_id": on["run"]["_id"],
            "off_evaluations": off_count, "on_evaluations": on_count,
            "evaluation_reduction_fraction": (off_count - on_count) / off_count if off_count else None,
            "off_incorrect": off["incorrect_count"], "on_incorrect": on["incorrect_count"],
            "equal_correct_candidate_sets": off["correct_candidates"] == on["correct_candidates"],
            "correct_candidates": {"off": off["correct_candidates"], "on": on["correct_candidates"]},
            "off_elapsed_seconds": off["elapsed_seconds"], "on_elapsed_seconds": on["elapsed_seconds"],
        })
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-run-id", required=True)
    parser.add_argument("--output", type=Path, default=Path("evidence/harness-ablation.json"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path to preserve prior evidence")

    context, context_hash = _context()
    store = AtlasStore()
    try:
        store.init()
        seed_run = store.run(args.seed_run_id)
        seed = store.experiments_for(args.seed_run_id)
        if seed_run["status"] != "completed" or seed_run["context_hash"] != context_hash:
            raise ValueError("seed must be completed and match the current GPU/software/evaluator context")
        if not seed or any(e["status"] not in ("completed", "failed") or
                           e["context_hash"] != context_hash for e in seed):
            raise ValueError("every seed experiment must be terminal and match the current context")
        if not any(e["candidate"] == "topk_sorted" and e["status"] == "failed" and
                   any(not c["passed"] and "ties" in c["case"]
                       for c in e.get("result", {}).get("checks", [])) for e in seed):
            raise ValueError("seed must contain the observed direct top-k tie-contract failure")

        ablation_id = uuid.uuid4().hex
        expected_correct = sorted(e["candidate"] for e in seed if e.get("result", {}).get("correct"))
        report = {
            "ablation_id": ablation_id, "started_at": now(),
            "design": {
                "intervention": "Expose only seed-run history versus no cross-run history",
                "within_run_evidence": "Enabled in both arms by the unchanged _drive loop",
                "order": [["history_off", "history_on"], ["history_on", "history_off"]],
                "budget_evals": 5, "candidates": list(CANDIDATES),
                "hypothesis": "History reduces repeated invalid evaluations without losing correct candidates",
                "primary_metrics": ["evaluation_count", "incorrect_count", "equal_correct_candidate_sets"],
                "reference_correct_candidates": expected_correct,
                "secondary_metric": "Elapsed _drive wall time, including worker and Atlas overhead",
                "limitations": "Two counterbalanced blocks on one fixed workload; no GPU-hour or kernel-speedup claim",
            },
            "context": context, "context_hash": context_hash,
            "contract": CONTRACT, "evaluator_version": EVALUATOR_VERSION,
            "seed_run": seed_run, "seed_experiments": seed,
            "arms": [], "comparisons": [], "protocol_pass": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        for block, order in enumerate(report["design"]["order"], 1):
            for position, condition in enumerate(order, 1):
                history = seed if condition == "history_on" else []
                run_id = store.create_run(context, context_hash, budget=5)
                metadata = {"ablation_id": ablation_id, "condition": condition,
                            "seed_run_id": args.seed_run_id, "block": block, "order": position,
                            "history_experiment_ids": [e["_id"] for e in history]}
                store.runs.update_one({"_id": run_id}, {"$set": {"experiment_design": metadata}})
                print(f"BLOCK={block} ORDER={position} CONDITION={condition} RUN_ID={run_id}", flush=True)
                started = time.perf_counter()
                _drive(HistoryArm(store, history), run_id)
                elapsed = time.perf_counter() - started
                experiments = store.experiments_for(run_id)
                decisions = store.decisions_for(run_id)
                own_ids = {e["_id"] for e in experiments}
                history_ids = {e["_id"] for e in history}
                arm = {
                    **metadata, "elapsed_seconds": elapsed, "run": store.run(run_id),
                    "experiments": experiments, "decisions": decisions,
                    "evaluation_count": len(experiments),
                    "candidate_counts": {c: sum(e["candidate"] == c for e in experiments) for c in CANDIDATES},
                    "incorrect_count": sum(any(not c["passed"] for c in e.get("result", {}).get("checks", []))
                                           for e in experiments),
                    "worker_error_count": sum(bool(e.get("result", {}).get("error")) for e in experiments),
                    "incorrect_accepted_count": sum(e["status"] == "completed" and not
                                                    e.get("result", {}).get("correct") for e in experiments),
                    "correct_candidates": sorted(e["candidate"] for e in experiments if e.get("result", {}).get("correct")),
                    "history_scope_pass": bool(decisions) and all(
                        set(d["retrieved_experiment_ids"]) - own_ids == history_ids for d in decisions),
                    "single_attempt_pass": all(e.get("attempt") == 1 for e in experiments),
                    "terminal_pass": all(e["status"] in ("completed", "failed") for e in experiments),
                }
                report["arms"].append(arm)
                report["comparisons"] = compare_blocks(report["arms"])
                args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

        report["protocol_pass"] = len(report["arms"]) == 4 and all(
            arm["history_scope_pass"] and arm["single_attempt_pass"] and arm["terminal_pass"] and
            arm["worker_error_count"] == 0 and arm["incorrect_accepted_count"] == 0 and
            arm["correct_candidates"] == expected_correct and arm["run"]["status"] == "completed"
            for arm in report["arms"]) and all(c["equal_correct_candidate_sets"] for c in report["comparisons"])
        report["hypothesis_supported"] = report["protocol_pass"] and all(
            c["off_evaluations"] > c["on_evaluations"] and c["off_incorrect"] > c["on_incorrect"]
            for c in report["comparisons"])
        report["finished_at"] = now()
        args.output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "protocol_pass": report["protocol_pass"],
                          "hypothesis_supported": report["hypothesis_supported"],
                          "comparisons": report["comparisons"]}, indent=2))
        if not report["protocol_pass"]:
            raise SystemExit(1)
    finally:
        store.client.close()


if __name__ == "__main__":
    main()
