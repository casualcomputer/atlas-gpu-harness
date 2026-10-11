"""Bounded llama.cpp inference experiments; no model output is executed.

Task banks and scoring are fixed before search. This measures a small exact-answer
suite, not general intelligence. Device memory is sampled, includes other GPU
processes, and is not an allocator-enforced memory limit.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import threading
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener
import uuid


TUNING_TASKS = (
    ("add", "What is 37 + 58? Return the integer answer.", 95),
    ("multiply", "What is 14 * 6? Return the integer answer.", 84),
    ("extract_city", 'Record: name=Mira; city=Oslo; role=engineer. Return the city.', "Oslo"),
    ("extract_id", 'Record: {"id":"AX-204","status":"ready"}. Return the id.', "AX-204"),
    ("code_sum", 'Python code: values = [2, 4, 6]; result = sum(values). What is result?', 12),
    ("code_filter", 'Python code: result = len([x for x in [1, 2, 3, 4] if x % 2 == 0]). What is result?', 2),
)
HOLDOUT_TASKS = (
    ("subtract", "What is 91 - 46? Return the integer answer.", 45),
    ("divide", "What is 144 divided by 12? Return the integer answer.", 12),
    ("extract_color", 'Record: item=bag; color=teal; count=3. Return the color.', "teal"),
    ("code_index", 'Python code: letters = ["a", "b", "c"]; result = letters[1]. What is result?', "b"),
)
SYSTEM_PROMPT = 'Return only one JSON object with exactly one key, "answer". No explanation or Markdown.'
CONTRACT_VERSION = "consumer-inference-v1"


def _request_json(port: int, route: str, payload: dict | None = None, timeout: float = 120) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(f"http://127.0.0.1:{port}{route}", data=data,
                      headers={"Content-Type": "application/json"})
    # A local experiment must not travel through a user's HTTP proxy.
    with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def _score(content: str, expected) -> bool:
    try:
        value = json.loads(content)
    except (ValueError, TypeError):
        return False
    return (isinstance(value, dict) and set(value) == {"answer"}
            and type(value["answer"]) is type(expected) and value["answer"] == expected)


def _child_environment(engine: Path) -> dict:
    env = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "TMPDIR", "LANG", "LC_ALL")
           if key in os.environ}
    library_dirs = [str(engine.parent)]
    library_dirs += [str(p) for p in engine.parent.parent.glob("cudart-*") if p.is_dir()]
    if env.get("LD_LIBRARY_PATH"):
        library_dirs.append(env["LD_LIBRARY_PATH"])
    env.update(LD_LIBRARY_PATH=":".join(library_dirs), OMP_NUM_THREADS="8", CUDA_VISIBLE_DEVICES="0")
    return env


def _gpu_memory() -> tuple[float, float]:
    result = subprocess.run(
        ["nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=3,
    )
    used, total = result.stdout.strip().splitlines()[0].split(",")
    return float(used), float(total)


class _MemorySampler:
    def __init__(self):
        self.pid = None
        self.samples = []
        self.errors = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.start_time = time.monotonic()

    def sample(self):
        item = {"elapsed_seconds": time.monotonic() - self.start_time}
        try:
            used, total = _gpu_memory()
            item.update(device_used_mib=used, device_total_mib=total)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            self.errors.append(type(error).__name__ + ": " + str(error))
        if self.pid is not None:
            try:
                status = Path(f"/proc/{self.pid}/status").read_text()
                match = re.search(r"^VmRSS:\s+(\d+)\s+kB", status, re.MULTILINE)
                if match:
                    item["process_rss_mib"] = int(match.group(1)) / 1024
            except (FileNotFoundError, ProcessLookupError):
                pass
        self.samples.append(item)

    def _run(self):
        while not self.stop_event.is_set():
            tick = time.monotonic()
            self.sample()
            self.stop_event.wait(max(0, 0.25 - (time.monotonic() - tick)))

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=4)

    def summary(self) -> dict:
        gpu = [s["device_used_mib"] for s in self.samples if "device_used_mib" in s]
        rss = [s["process_rss_mib"] for s in self.samples if "process_rss_mib" in s]
        return {"sampling_interval_seconds": 0.25, "device_peak_mib": max(gpu) if gpu else None,
                "process_rss_peak_mib": max(rss) if rss else None, "samples": self.samples,
                "sampling_errors": self.errors,
                "scope": "sampled total GPU-0 memory including other processes and model startup; peaks between samples can be missed"}


def _assert_port_free(port: int):
    with socket.socket() as probe:
        # Completed requests can leave TIME_WAIT after our previous server exits.
        # Reuse those sockets, while still refusing a live listener.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"Refusing to use occupied/unavailable port {port}") from error


def _command(engine: Path, model: Path, recipe: dict, port: int, context_size: int,
             fit_target_mib: int) -> list[str]:
    command = [str(engine), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
               "--ctx-size", str(context_size), "--parallel", "1", "--threads", "8",
               "--threads-batch", "8", "--load-mode", "mmap", "--no-cache-prompt", "--jinja",
               "--chat-template-kwargs", '{"enable_thinking":false}', "--metrics"]
    if recipe["n_cpu_moe"] is None:
        command += ["--gpu-layers", "auto", "--fit", "on", "--fit-target", str(fit_target_mib)]
    else:
        command += ["--gpu-layers", "all", "--n-cpu-moe", str(recipe["n_cpu_moe"]), "--fit", "off"]
    return command


def _wait_healthy(process, port: int, timeout: float = 600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited before readiness with code {process.returncode}")
        try:
            if _request_json(port, "/health", timeout=2).get("status") == "ok":
                return
        except (OSError, ValueError, URLError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"llama-server was not healthy within {timeout} seconds")


def _complete_task(port: int, task: tuple, round_index: int) -> dict:
    name, prompt, expected = task
    payload = {"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt + "\n/no_think"}],
               "temperature": 0, "seed": 20261011, "max_tokens": 96, "stream": False,
               "cache_prompt": False}
    start = time.monotonic()
    response = _request_json(port, "/v1/chat/completions", payload)
    latency = time.monotonic() - start
    choice = response["choices"][0]
    content = choice["message"].get("content") or ""
    return {"task_id": name, "round": round_index, "prompt": prompt, "expected": expected,
            "raw_output": content, "reasoning_content": choice["message"].get("reasoning_content"),
            "passed": _score(content, expected), "request_latency_seconds": latency,
            "finish_reason": choice.get("finish_reason"), "usage": response.get("usage", {}),
            "timings": response.get("timings", {}),
            "latency_scope": "nonstream request completion; not time to first token"}


def evaluate_recipe(engine_path, model_path, recipe, output_dir, port=8081,
                    context_size=4096, gpu_budget_mib=None, stage="tuning") -> dict:
    """Run one fixed recipe in an owned process, then stop it before returning.

    Only n_cpu_moe is tunable. A None value selects llama.cpp's automatic-fit
    baseline with the same total-device memory budget as every candidate. The
    default is physical device capacity; a lower explicit budget is optional.
    """
    if (not isinstance(recipe, dict) or set(recipe) != {"name", "n_cpu_moe"}
            or not isinstance(recipe["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", recipe["name"])
            or (recipe["n_cpu_moe"] is not None
                and (type(recipe["n_cpu_moe"]) is not int or not 0 <= recipe["n_cpu_moe"] <= 64))):
        raise ValueError("recipe must contain only a safe name and n_cpu_moe=None or integer 0..64")
    if stage not in ("tuning", "holdout"):
        raise ValueError("stage must be tuning or holdout")
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("port must be an integer in 1024..65535")
    if context_size != 4096 or (gpu_budget_mib is not None
            and (type(gpu_budget_mib) not in (int, float) or not math.isfinite(gpu_budget_mib)
                 or gpu_budget_mib <= 0)):
        raise ValueError("the fixed contract requires context_size=4096 and a positive finite memory budget or None")
    engine, model = Path(engine_path).resolve(), Path(model_path).resolve()
    if not engine.is_file() or not model.is_file():
        raise FileNotFoundError("engine_path and model_path must be existing local files")
    _assert_port_free(port)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"{recipe['name']}-{stage}-{uuid.uuid4().hex[:8]}.server.log"
    started = time.monotonic()
    result = {"contract_version": CONTRACT_VERSION, "recipe": dict(recipe), "stage": stage,
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "model_path": str(model),
              "model_size_bytes": model.stat().st_size, "engine_path": str(engine),
              "context_size": context_size, "gpu_budget_mib": gpu_budget_mib,
              "budget_scope": "sampled total device memory; an explicit limit below physical capacity is an optional simulated constraint",
              "server_log_path": str(log_path.resolve()), "engine_success": False,
              "quality_pass": False, "resource_pass": False, "correct": False, "tasks": []}
    process = None
    sampler = _MemorySampler()
    try:
        used, total = _gpu_memory()
        if gpu_budget_mib is None:
            gpu_budget_mib = int(total)
        result["gpu_budget_mib"] = gpu_budget_mib
        fit_target = max(1024, int(total - gpu_budget_mib))
        command = _command(engine, model, recipe, port, context_size, fit_target)
        result.update(command=command, gpu_total_mib=total, device_memory_before_mib=used,
                      baseline_fit_target_mib=fit_target if recipe["n_cpu_moe"] is None else None)
        sampler.start()
        with log_path.open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                       env=_child_environment(engine), stdin=subprocess.DEVNULL)
            sampler.pid = process.pid
            _wait_healthy(process, port)
            result["load_seconds"] = time.monotonic() - started
            # Warmup is a separate fixed task; neither timed nor scored for comparison.
            _complete_task(port, ("warmup", "What is 1 + 1? Return the integer answer.", 2), -1)
            tasks = TUNING_TASKS if stage == "tuning" else HOLDOUT_TASKS
            rounds = (tasks, tuple(reversed(tasks))) if stage == "tuning" else (tasks,)
            for round_index, ordered in enumerate(rounds):
                for task in ordered:
                    result["tasks"].append(_complete_task(port, task, round_index))
            result["server_healthy_after"] = (process.poll() is None
                                               and _request_json(port, "/health", timeout=2).get("status") == "ok")
            result["engine_success"] = result["server_healthy_after"]
    except (OSError, ValueError, KeyError, IndexError, RuntimeError, subprocess.SubprocessError) as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        try:
            if process is not None and process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except ProcessLookupError:
                    pass  # It exited between poll() and terminate().
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        finally:
            if sampler.thread.ident is not None:
                sampler.stop()
        result["wallclock_seconds"] = time.monotonic() - started
        result["memory"] = sampler.summary()
        if result.get("error") and log_path.exists():
            with log_path.open("rb") as log:
                log.seek(max(0, log_path.stat().st_size - 5000))
                result["server_log_tail"] = log.read().decode(errors="replace")
    expected_count = 12 if stage == "tuning" else 4
    passed = sum(task["passed"] for task in result["tasks"])
    result["quality"] = {"passed": passed, "total": expected_count,
                         "completed": len(result["tasks"]), "unique_tasks": 6 if stage == "tuning" else 4}
    result["quality_pass"] = len(result["tasks"]) == expected_count and passed == expected_count
    peak = result["memory"]["device_peak_mib"]
    result["resource_pass"] = gpu_budget_mib is not None and peak is not None and peak <= gpu_budget_mib
    result["correct"] = result["engine_success"] and result["quality_pass"] and result["resource_pass"]
    if result["tasks"]:
        result["median_request_latency_seconds"] = statistics.median(t["request_latency_seconds"] for t in result["tasks"])
        rates = [t["timings"]["predicted_per_second"] for t in result["tasks"]
                 if isinstance(t["timings"].get("predicted_per_second"), (int, float))]
        result["median_decode_tokens_per_second"] = statistics.median(rates) if rates else None
        result["output_tokens"] = sum(t["usage"].get("completion_tokens", 0) for t in result["tasks"])
    return result
