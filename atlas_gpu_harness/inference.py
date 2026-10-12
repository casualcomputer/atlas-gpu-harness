"""Measured, resumable code-generation loop for one supported local MoE GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
from pathlib import Path

from filelock import FileLock

from .inference_backend import CONTRACT_VERSION, HOLDOUT_TASKS, TUNING_TASKS, evaluate_recipe
from .model_probe import inspect_gguf, plan_model, probe_hardware
from .policy_codegen import PolicyError, propose_policy
from .store import AtlasStore

DEFAULT_ENGINE = Path("artifacts/engines/llama-b11429/launch-llama.sh")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def context_for(model_path, engine_path, budget):
    model, hardware = inspect_gguf(model_path), probe_hardware()
    if budget is None:
        budget = int(hardware["gpu"]["total_mib"])
    plan = plan_model(model, hardware, budget)
    engine = Path(engine_path).resolve()
    version = subprocess.run([str(engine), "--version"], text=True, capture_output=True,
                             timeout=20, check=True)
    source = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ("inference.py", "inference_backend.py", "model_probe.py", "policy_codegen.py")}
    fixed = {"model_identity": model["identity"], "model_mtime_ns": Path(model_path).stat().st_mtime_ns,
             "engine_version": (version.stdout + version.stderr).strip(),
             "engine_launcher_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
             "gpu": {k: hardware["gpu"][k] for k in ("name", "total_mib", "driver")},
             "ram_total_mib": hardware["ram"]["total_mib"], "cpu_count": hardware["cpu_count"],
             "contract_version": CONTRACT_VERSION, "context_size": 4096,
             "gpu_budget_mib": budget, "tuning": TUNING_TASKS, "holdout": HOLDOUT_TASKS,
             "source_sha256": source}
    return {"model": model, "hardware": hardware, "plan": plan, "fixed": fixed}, digest(fixed)


def trial_summary(experiment):
    result = experiment["result"]
    return {"experiment_id": experiment["_id"], "candidate": experiment["candidate"],
            "n_cpu_moe": result.get("recipe", {}).get("n_cpu_moe"),
            "engine_success": result.get("engine_success"), "correct": result.get("correct"),
            "quality": result.get("quality"), "quality_pass": result.get("quality_pass"),
            "resource_pass": result.get("resource_pass"),
            "device_peak_mib": result.get("memory", {}).get("device_peak_mib"),
            "median_decode_tokens_per_second": result.get("median_decode_tokens_per_second"),
            "median_request_latency_seconds": result.get("median_request_latency_seconds"),
            "error": result.get("error"),
            "engine_error_tail": (result.get("server_log_tail") or "")[-1500:]}


def _evaluate(store, run, name, recipe, stage, directory, evidence_ids=(), proposal=None):
    identity = digest({"context": run["context_hash"], "name": name, "recipe": recipe, "stage": stage})
    document_id = f"{run['_id']}:{identity}"
    existing = store.experiments.find_one({"_id": document_id})
    if existing and existing["status"] in ("completed", "failed"):
        print(f"reuse committed {name}", flush=True)
        return existing
    store.decide(run["_id"], list(evidence_ids), name,
                 proposal["hypothesis"] if proposal else "Measure the fixed engine reference or confirmation",
                 "Measured code-generated placement" if proposal else "Fixed evaluation protocol",
                 policy_version="local-model-codegen-v1")
    token = store.start(run["_id"], identity, run["context_hash"], name)
    if token is None:
        return store.experiments.find_one({"_id": document_id})
    print(f"evaluate {name}: {recipe}; stage={stage}", flush=True)
    try:
        result = evaluate_recipe(run["engine_path"], run["model_path"], recipe, directory,
                                 port=run["port"], gpu_budget_mib=run["context"]["fixed"]["gpu_budget_mib"],
                                 stage=stage)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        # Commit preflight failures too. An occupied port is never cleared by killing an unknown process.
        result = {"recipe": recipe, "stage": stage, "correct": False, "engine_success": False,
                  "quality_pass": False, "resource_pass": False, "tasks": [],
                  "error": {"type": type(error).__name__, "message": str(error)[:1000],
                            "phase": "startup_or_evaluator"}}
    if proposal:
        result["proposal"] = proposal
    store.finish(run["_id"], identity, token, result)
    experiment = store.experiments.find_one({"_id": document_id})
    save(directory / f"{name}.json", experiment)
    print(json.dumps(trial_summary(experiment)), flush=True)
    return experiment


def _rate(experiment):
    return experiment["result"].get("median_decode_tokens_per_second") or 0


def _confirmation(comparisons):
    effects = []
    for baseline, candidate in comparisons:
        by_task = {x["task_id"]: x for x in baseline["result"]["tasks"]}
        paired = []
        for task in candidate["result"]["tasks"]:
            reference = by_task.get(task["task_id"], {}).get("timings", {}).get("predicted_per_second")
            rate = task.get("timings", {}).get("predicted_per_second")
            if reference and rate:
                paired.append(100 * (rate / reference - 1))
        effects.append({"baseline_id": baseline["_id"], "candidate_id": candidate["_id"],
                        "median_paired_decode_change_pct": statistics.median(paired) if paired else None,
                        "pairs": len(paired), "both_pass": baseline["result"]["correct"] and candidate["result"]["correct"]})
    supported = len(effects) == 2 and all(e["both_pass"] and e["pairs"] == len(HOLDOUT_TASKS)
        and e["median_paired_decode_change_pct"] > 5 for e in effects)
    return {"blocks": effects, "directionally_supported_speedup": supported,
            "rule": "Both reversed-order holdout blocks pass quality/resource checks and median paired decode gain >5%; descriptive, not a confidence interval"}


def evolve(store, run, directory):
    latest = store.run(run["_id"])
    if latest.get("status") == "completed" and "recommendation" in latest:
        return latest["recommendation"]
    baseline_recipe = {"name": "engine_autofit", "n_cpu_moe": None}
    baseline = _evaluate(store, run, "baseline", baseline_recipe, "tuning", directory)
    for index in range(run["budget_evals"] - 1):
        name = f"policy_{index + 1}"
        latest = store.run(run["_id"])
        proposal = latest.get("proposals", {}).get(name)
        attempt = latest.get("proposal_attempts", {}).get(name)
        if proposal is None and (attempt or name in latest.get("proposal_errors", {})):
            if attempt and attempt["status"] == "running":
                failure = {"message": "Previous process ended before the proposal was committed; this budget slot is consumed"}
                store.runs.update_one({"_id": run["_id"]}, {"$set": {
                    f"proposal_attempts.{name}": {"status": "interrupted"},
                    f"proposal_errors.{name}": failure}})
            print(f"reuse terminal proposal attempt {name}; no extra generation budget", flush=True)
            continue
        current = [e for e in store.experiments_for(run["_id"])
                   if e["status"] in ("completed", "failed") and e["result"].get("stage") == "tuning"]
        # Only compatible tuning evidence is exposed to the proposer. Holdout stays hidden.
        previous = [e for e in store.retrieve(run["context_hash"])
                    if e["run_id"] != run["_id"] and e.get("result", {}).get("stage") == "tuning"][:8]
        known = previous + current
        if proposal is None:
            feedback = {"objective": "Maximize decode speed while every task passes and sampled total-device memory stays within budget",
                        "gpu_budget_mib": run["context"]["fixed"]["gpu_budget_mib"],
                        "initial_estimate_n_cpu_moe": run["context"]["plan"]["initial_n_cpu_moe"],
                        "avoid_duplicate_placements": [e["result"]["recipe"]["n_cpu_moe"] for e in known],
                        "previous_errors": {key: ({field: value[field] for field in ("message", "execution_error") if field in value}
                                                   if isinstance(value, dict) else str(value)[:1000])
                                            for key, value in latest.get("proposal_errors", {}).items()}}
            context = {"model": {k: run["context"]["model"][k] for k in
                                  ("name", "architecture", "block_count", "expert_count", "expert_used_count", "file_bytes")},
                       "hardware": probe_hardware(), "previous_trials": [trial_summary(e) for e in known],
                       "feedback": feedback}
            for previous_index in range(index - 1, -1, -1):
                previous_name = f"policy_{previous_index + 1}"
                last = latest.get("proposals", {}).get(previous_name)
                rejected = latest.get("proposal_errors", {}).get(previous_name)
                if last:
                    context["previous_policy"] = {"code": last["code"], "hypothesis": last["hypothesis"]}
                    break
                if isinstance(rejected, dict) and rejected.get("code"):
                    context["previous_policy"] = {"code": rejected["code"], "hypothesis": rejected.get("hypothesis", ""),
                                                  "validation_status": "failed"}
                    break
            print(f"coding agent proposing {name}", flush=True)
            # Reserve before calling the model: a crash cannot silently create a fresh budget on resume.
            store.runs.update_one({"_id": run["_id"]}, {"$set": {f"proposal_attempts.{name}": {"status": "running"}}})
            try:
                proposal = propose_policy(context, directory, model=run["proposer_model"])
            except PolicyError as error:
                message = str(error)
                failure = {"message": message[:1000], **{key: error.evidence[key] for key in
                           ("code", "hypothesis", "source_hash", "path", "receipt_path", "execution_error", "worker_returncode")
                           if key in error.evidence}}
                store.runs.update_one({"_id": run["_id"]}, {"$set": {
                    f"proposal_errors.{name}": failure, f"proposal_attempts.{name}": {"status": "failed"}}})
                print(message, flush=True)
                continue
            store.runs.update_one({"_id": run["_id"]}, {"$set": {
                f"proposals.{name}": proposal, f"proposal_attempts.{name}": {"status": "completed"}}})
        recipe = {"name": name, "n_cpu_moe": proposal["n_cpu_moe"]}
        _evaluate(store, run, name, recipe, "tuning", directory, [e["_id"] for e in known], proposal)
    trials = [e for e in store.experiments_for(run["_id"])
              if e["status"] in ("completed", "failed") and e["result"].get("stage") == "tuning"]
    passing = [e for e in trials if e["result"].get("correct")]
    selected = max(passing, key=_rate, default=None)
    confirmation = None
    recommended = None
    validated_experiment_id = None
    selection_reason = "No tuning recipe passed quality and resource checks."
    if selected is not None:
        recipe = selected["result"]["recipe"]
        first_baseline = _evaluate(store, run, "holdout_baseline_1", baseline_recipe, "holdout", directory)
        if selected["_id"] == baseline["_id"]:
            selection_reason = "Selected tuning baseline failed held-out validation."
            if first_baseline["result"]["correct"]:
                recommended = baseline_recipe
                validated_experiment_id = first_baseline["_id"]
                selection_reason = "Engine baseline passed held-out validation."
        else:
            first_candidate = _evaluate(store, run, "holdout_candidate_1", recipe, "holdout", directory)
            second_candidate = _evaluate(store, run, "holdout_candidate_2", recipe, "holdout", directory)
            second_baseline = _evaluate(store, run, "holdout_baseline_2", baseline_recipe, "holdout", directory)
            confirmation = _confirmation([(first_baseline, first_candidate), (second_baseline, second_candidate)])
            candidate_pass = first_candidate["result"]["correct"] and second_candidate["result"]["correct"]
            selection_reason = "Neither recipe passed the required held-out confirmation."
            if confirmation["directionally_supported_speedup"]:
                recommended = recipe
                validated_experiment_id = first_candidate["_id"]
                selection_reason = "Candidate passed both held-out comparisons and exceeded the fixed speedup rule."
            elif first_baseline["result"]["correct"] and second_baseline["result"]["correct"]:
                recommended = baseline_recipe
                validated_experiment_id = second_baseline["_id"]
                selection_reason = "Both held-out baselines passed; candidate did not establish a speedup."
            elif candidate_pass:
                recommended = recipe
                validated_experiment_id = first_candidate["_id"]
                selection_reason = ("Candidate passed both held-out runs; baseline did not pass both. "
                                    "Feasibility only; speedup unproven.")
    decision = {"run_id": run["_id"], "recipe": recommended, "confirmation": confirmation,
                "validated_experiment_id": validated_experiment_id, "selection_reason": selection_reason,
                "model_path": run["model_path"], "engine_path": run["engine_path"],
                "gpu_budget_mib": run["context"]["fixed"]["gpu_budget_mib"], "context_size": 4096,
                "context_hash": run["context_hash"],
                "scope": "One quantized local MoE, short exact-answer tasks, sampled memory budget on this physical GPU; no general model quality or arbitrary model support claim"}
    save(directory / "recommendation.json", decision)
    store.runs.update_one({"_id": run["_id"]}, {"$set": {"recommendation": decision}})
    store.update_progress(run["_id"], "completed", validated_experiment_id)
    report = {"run": store.run(run["_id"]), "experiments": store.experiments_for(run["_id"]),
              "decisions": store.decisions_for(run["_id"])}
    save(directory / "report.json", report)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "evolve"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine", default=str(DEFAULT_ENGINE) if DEFAULT_ENGINE.is_file() else None,
                        help=f"llama-server launcher (default: {DEFAULT_ENGINE} if installed)")
    parser.add_argument("--gpu-budget-mib", type=int, default=None,
                        help="optional lower experiment limit; default uses the physical card capacity")
    parser.add_argument("--budget-evals", type=int, default=3, choices=range(2, 6))
    parser.add_argument("--proposer-model", default="gemma3:12b")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--resume-run")
    args = parser.parse_args()
    if args.command == "inspect":
        hardware = probe_hardware()
        budget = args.gpu_budget_mib if args.gpu_budget_mib is not None else int(hardware["gpu"]["total_mib"])
        print(json.dumps(plan_model(inspect_gguf(args.model), hardware, budget), indent=2))
        return
    if not args.engine:
        parser.error(f"no engine: run `atlas-harness setup-engine` to install {DEFAULT_ENGINE}, "
                     "or pass --engine with a working llama-server launcher")
    Path("artifacts").mkdir(exist_ok=True)
    with FileLock("artifacts/gpu.lock", timeout=0):
        context, context_hash = context_for(args.model, args.engine, args.gpu_budget_mib)
        try:
            store = AtlasStore()
            store.init()
        except Exception:
            raise RuntimeError("Atlas connection failed; check the configured local .env and network access") from None
        if args.resume_run:
            run = store.run(args.resume_run)
            if run["context_hash"] != context_hash:
                raise RuntimeError("Model, engine, machine, source or evaluation contract changed; resume refused")
        else:
            run_id = store.create_run(
                context, context_hash, args.budget_evals,
                objective="Generate and test MoE placement code under a GPU memory budget",
                policy_version="local-model-codegen-v1", workflow=CONTRACT_VERSION,
                model_path=str(Path(args.model).absolute()), engine_path=str(Path(args.engine).absolute()),
                proposer_model=args.proposer_model, port=args.port,
                budget_scope="baseline plus bounded code proposals; held-out confirmation adds up to four engine launches")
            run = store.run(run_id)
        directory = Path("artifacts/inference") / run["_id"]
        print(f"INFERENCE_RUN_ID={run['_id']}", flush=True)
        print(json.dumps(evolve(store, run, directory), indent=2))


if __name__ == "__main__":
    main()
