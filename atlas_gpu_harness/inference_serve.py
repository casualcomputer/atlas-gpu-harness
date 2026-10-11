"""Launch a measured recipe without rerunning the search."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from filelock import FileLock

from .inference_backend import CONTRACT_VERSION, _assert_port_free, _child_environment, _command
from .model_probe import inspect_gguf, probe_hardware


def _validated_command(path, port):
    """Refuse changed inputs and reconstruct the command that passed holdout.

    The recorded model identity covers its header, size and mtime, not a full
    weight checksum. Engine identity covers the launcher hash and version, not
    every shared library. These are reproducibility guards, not signed records.
    """
    path = Path(path).resolve()
    recommendation = json.loads(path.read_text())
    if recommendation.get("recipe") is None:
        raise ValueError("this run did not produce a validated recommendation")
    report = json.loads(path.with_name("report.json").read_text())
    run = report["run"]
    fixed = run["context"]["fixed"]
    if recommendation != run.get("recommendation") or recommendation["run_id"] != run["_id"]:
        raise ValueError("recipe does not match the recorded run recommendation")
    context_hash = hashlib.sha256(json.dumps(fixed, sort_keys=True, default=str).encode()).hexdigest()
    if context_hash != run["context_hash"] or context_hash != recommendation["context_hash"]:
        raise ValueError("saved evaluation context does not match its recorded identity")
    if fixed["contract_version"] != CONTRACT_VERSION:
        raise ValueError("evaluation contract changed; rerun the experiment")
    backend_hash = hashlib.sha256(Path(__file__).with_name("inference_backend.py").read_bytes()).hexdigest()
    if backend_hash != fixed["source_sha256"]["inference_backend.py"]:
        raise ValueError("inference backend changed since validation; rerun the experiment")
    model = inspect_gguf(recommendation["model_path"])
    if (model["identity"] != fixed["model_identity"]
            or model["identity"] != run["context"]["model"]["identity"]
            or Path(recommendation["model_path"]).stat().st_mtime_ns != fixed["model_mtime_ns"]):
        raise ValueError("model metadata or modification time changed since validation; rerun the experiment")
    engine = Path(recommendation["engine_path"]).resolve()
    if hashlib.sha256(engine.read_bytes()).hexdigest() != fixed["engine_launcher_sha256"]:
        raise ValueError("engine launcher changed since validation; rerun the experiment")
    version = subprocess.run([str(engine), "--version"], text=True, capture_output=True,
                             timeout=20, check=True, env=_child_environment(engine))
    if (version.stdout + version.stderr).strip() != fixed["engine_version"]:
        raise ValueError("engine version changed since validation; rerun the experiment")
    hardware = probe_hardware()
    gpu = {key: hardware["gpu"][key] for key in ("name", "total_mib", "driver")}
    if gpu != fixed["gpu"]:
        raise ValueError("GPU identity, capacity or driver changed since validation; rerun the experiment")
    if hardware["cpu_count"] != fixed["cpu_count"] or hardware["ram"]["total_mib"] != fixed["ram_total_mib"]:
        raise ValueError("CPU count or host memory capacity changed since validation; rerun the experiment")
    matches = [item for item in report["experiments"]
               if item["_id"] == recommendation.get("validated_experiment_id")]
    if len(matches) != 1:
        raise ValueError("the successful holdout experiment is missing or duplicated")
    experiment = matches[0]
    result = experiment["result"]
    if (experiment["run_id"] != run["_id"] or experiment["context_hash"] != context_hash
            or experiment["status"] != "completed" or result.get("stage") != "holdout"
            or not all(result.get(key) is True for key in ("correct", "engine_success", "quality_pass", "resource_pass"))
            or result.get("recipe") != recommendation["recipe"]):
        raise ValueError("recommendation is not backed by a successful matching holdout experiment")
    for key in ("context_size", "gpu_budget_mib"):
        if recommendation[key] != fixed[key] or result.get(key) != fixed[key]:
            raise ValueError(f"saved {key} differs between recommendation, context and validation")
    if result.get("gpu_total_mib") != fixed["gpu"]["total_mib"]:
        raise ValueError("validated GPU capacity does not match the saved context")
    recipe = recommendation["recipe"]
    expected_target = max(1024, int(fixed["gpu"]["total_mib"] - fixed["gpu_budget_mib"]))
    target = result.get("baseline_fit_target_mib") if recipe["n_cpu_moe"] is None else expected_target
    if target != expected_target:
        raise ValueError("recorded baseline fit target does not match the validated memory budget")
    command = _command(engine, Path(model["path"]).resolve(), recipe, port, fixed["context_size"], target)
    recorded = list(result["command"])
    try:
        recorded[recorded.index("--port") + 1] = str(port)
    except (ValueError, IndexError):
        raise ValueError("validated launch command has no valid port") from None
    if command != recorded:
        raise ValueError("launch command differs from the successful holdout command; rerun the experiment")
    return command, _child_environment(engine)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="recommendation.json produced by the inference loop")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be in 1024..65535")
    Path("artifacts").mkdir(exist_ok=True)
    with FileLock("artifacts/gpu.lock", timeout=0):
        try:
            command, environment = _validated_command(args.recipe, args.port)
            _assert_port_free(args.port)
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            parser.error(str(error))
        print(f"Launching saved recipe. API will be http://127.0.0.1:{args.port}/v1 after loading.", flush=True)
        process = subprocess.Popen(command, env=environment)
        try:
            raise SystemExit(process.wait())
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
