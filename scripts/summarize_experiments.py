"""Recompute presentation metrics from the tracked raw Atlas exports."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def summarize():
    ablation_path = ROOT / "evidence/harness-ablation.json"
    data = json.loads(ablation_path.read_text())
    seed_path = ROOT / f"evidence/report-{data['seed_run']['_id']}.json"
    seed = json.loads(seed_path.read_text())
    blocks = []
    for block in (1, 2):
        arms = {a["condition"]: a for a in data["arms"] if a["block"] == block}
        off, on = arms["history_off"], arms["history_on"]
        counts = [sum(e["attempt"] for e in a["experiments"]) for a in (off, on)]
        correct = [{e["candidate"] for e in a["experiments"] if e["result"]["correct"]}
                   for a in (off, on)]
        blocks.append({
            "block": block, "off_attempts": counts[0], "on_attempts": counts[1],
            "fewer_attempts_percent": 100 * (counts[0] - counts[1]) / counts[0],
            "same_correct_alternatives": correct[0] == correct[1],
            "correct_alternatives_count": len(correct[0]),
            "off_elapsed_seconds": off["elapsed_seconds"],
            "on_elapsed_seconds": on["elapsed_seconds"],
            "elapsed_reduction_percent": 100 * (1 - on["elapsed_seconds"] / off["elapsed_seconds"]),
        })
    timing = []
    for run in [seed, *data["arms"]]:
        baseline = next(e["result"]["timing"]["median_us"] for e in run["experiments"]
                        if e["candidate"] == "sort_stable")
        for e in run["experiments"]:
            paired = e["result"].get("paired_recheck")
            if paired is None:
                continue
            base = paired["baseline"]["samples_us"]
            contender = paired["candidate"]["samples_us"]
            assert len(base) == len(contender) == 12
            reductions = [100 * (b - c) / b for b, c in zip(base, contender)]
            timing.append({
                "run_id": run["run"]["_id"], "candidate": e["candidate"],
                "initial_unpaired_reduction_percent": 100 * (1 - e["result"]["timing"]["median_us"] / baseline),
                "median_paired_reduction_percent": statistics.median(reductions),
                "pair_reductions_percent": reductions,
                "pairs_faster": sum(c < b for b, c in zip(base, contender)),
                "pairs": len(base),
                "speedup_supported": max(contender) < min(base),
            })
    return {
        "source_files": [str(p.relative_to(ROOT)) for p in (ablation_path, seed_path)],
        "definitions": {
            "fewer_attempts_percent": "100 * (without-history attempts - with-history attempts) / without-history attempts",
            "paired_reduction_percent": "100 * (reference latency - alternative latency) / reference latency within each alternating pair; positive means faster",
        },
        "limitations": [
            "One fixed workload and GPU; two counterbalanced blocks, not a population estimate.",
            "Prior failure history is supplied; its original acquisition cost is excluded.",
            "Wall time includes startup and Atlas operations, not just GPU execution.",
            "Latency pairs within a worker are not independent workload replications.",
        ],
        "blocks": blocks, "timing": timing,
    }


if __name__ == "__main__":
    output = ROOT / "evidence/experiment-summary.json"
    result = summarize()
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("SAVED REAL RUNS: SAME TASK, ONLY ACCESS TO PREVIOUS NOTES CHANGED")
    print("Block | Without notes | With notes | Same correct alternatives | Fewer attempts")
    for block in result["blocks"]:
        coverage = block["correct_alternatives_count"] if block["same_correct_alternatives"] else "MISMATCH"
        print(f"{block['block']:5} | {block['off_attempts']:13} | {block['on_attempts']:10} | "
              f"{str(coverage):25} | {block['fewer_attempts_percent']:.1f}%")
    supported = sum(row["speedup_supported"] for row in result["timing"])
    print(f"GPU speed comparisons passing the current gate: {supported}/{len(result['timing'])}")
    print("Scope: one workload, two comparison blocks, prior evidence already available.")
    print(f"Raw-derived summary: {output}")
