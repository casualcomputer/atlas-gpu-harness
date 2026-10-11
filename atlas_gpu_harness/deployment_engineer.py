"""Plan source-backed serving experiments and measure compatible local APIs.

This workflow does not install engines or assume that an HF checkpoint is
compatible with every server. Run one configuration at a time, then benchmark
the same model and task contract before comparing results.
"""

from __future__ import annotations

import argparse
import datetime as dt
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time
from urllib.parse import urljoin, urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .inference_backend import SYSTEM_PROMPT, TUNING_TASKS, _score


SOURCES = {
    "vllm": "https://docs.vllm.ai/en/latest/configuration/engine_args/",
    "sglang": "https://docs.sglang.io/docs/advanced_features/server_arguments",
    "ollama": "https://docs.ollama.com/faq",
    "unsloth": "https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide.md",
}
SERVING_RUNTIMES = ("vllm", "sglang", "ollama")
EP_SOURCES = {
    "vllm": "https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/",
    "sglang": "https://docs.sglang.io/docs/advanced_features/expert_parallelism",
}
DOC_INDEXES = {
    "vllm": "https://docs.vllm.ai/en/latest/",
    "sglang": "https://docs.sglang.io/llms.txt",
    "ollama": "https://docs.ollama.com/llms.txt",
    "unsloth": "https://unsloth.ai/docs/llms.txt",
}
WORKLOAD = "deployment-short-json-v1"


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.href = None
        self.label = []
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.href = dict(attrs).get("href")
            self.label = []

    def handle_data(self, data):
        if self.href is not None:
            self.label.append(data)

    def handle_endtag(self, tag):
        if tag == "a" and self.href is not None:
            self.links.append((" ".join(" ".join(self.label).split()), self.href))
            self.href = None
            self.label = []


def _index_links(index_url: str, body: str) -> list[dict]:
    if index_url.endswith(".txt"):
        raw = re.findall(r"^\s*-\s*\[([^]]+)\]\((https?://[^)]+)\)([^\n]*)",
                         body, re.MULTILINE)
        links = [(title + " " + description.strip(), url) for title, url, description in raw]
    else:
        parser = _Links()
        parser.feed(body)
        links = parser.links
    host = urlsplit(index_url).hostname
    return [{"title": title, "url": urljoin(index_url, href)} for title, href in links
            if title and urlsplit(urljoin(index_url, href)).hostname == host]


def discover_docs(query: str, fetch=None) -> dict:
    """Search first-party documentation indexes without fetching untrusted links."""
    terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 1]
    if not terms:
        raise ValueError("query needs at least one word of two or more characters")
    if fetch is None:
        opener = build_opener(ProxyHandler({}))

        def fetch(url):
            request = Request(url, headers={"User-Agent": "AtlasGPUHarness/1.0 documentation research",
                                            "Accept": "text/plain,text/html"})
            with opener.open(request, timeout=15) as response:
                return response.read(3_000_000).decode("utf-8", errors="replace")

    matches, errors = [], {}
    for runtime, index in DOC_INDEXES.items():
        try:
            for link in _index_links(index, fetch(index)):
                score = sum(term in (link["title"] + " " + link["url"]).lower() for term in terms)
                if score:
                    matches.append({"runtime": runtime, **link, "matched_terms": score})
        except (OSError, UnicodeError, ValueError) as error:
            errors[runtime] = f"{type(error).__name__}: {error}"
    matches.sort(key=lambda row: (-row["matched_terms"], row["runtime"], row["title"]))
    return {"query": query, "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "indexes": DOC_INDEXES, "matches": matches[:40], "total_matches": len(matches),
            "errors": errors, "curated_sources": SOURCES,
            "scope": "Searches official documentation indexes by title and description; read matched pages before applying a flag."}


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _experiment(runtime, model, name, argv, env, hypothesis, source, prerequisite=None,
                reference=None):
    return {"id": f"{runtime}:{name}", "runtime": runtime, "model": model,
            "argv": argv, "env": env, "hypothesis": hypothesis,
            "compare_to": reference if reference is not None else
                          (f"{runtime}:baseline" if name != "baseline" else None),
            "source": source, "prerequisite": prerequisite,
            "measure": ["quality", "request latency", "completion throughput", "GPU peak",
                        "RAM available minimum"],
            "measure_externally": ["startup/load time", "SSD read bytes"]}


def _launch(runtime, model, context, extra=(), env=None):
    if runtime == "vllm":
        return ["vllm", "serve", model, "--max-model-len", str(context), *extra], env or {}
    if runtime == "sglang":
        return ["python", "-m", "sglang.launch_server", "--model-path", model,
                "--context-length", str(context), *extra], env or {}
    if runtime == "ollama":
        settings = {"OLLAMA_CONTEXT_LENGTH": str(context), **(env or {})}
        return ["ollama", "serve"], settings
    raise ValueError(f"unsupported runtime: {runtime}")


def plan(spec: dict, observations: list[dict] | None = None) -> dict:
    """Create one-variable experiments from model formats and observed pressure.

    Inputs are user supplied inventory and measurements, not detected support.
    A baseline for each supplied runtime is always measured before a variant.
    """
    if not isinstance(spec, dict):
        raise ValueError("spec must be an object")
    hardware = spec.get("hardware")
    targets = spec.get("targets")
    if not isinstance(hardware, dict) or not isinstance(targets, dict) or not targets:
        raise ValueError("spec needs hardware and nonempty targets objects")
    gpu = _positive(hardware.get("gpu_total_gib"), "gpu_total_gib")
    ram = _positive(hardware.get("ram_available_gib"), "ram_available_gib")
    disk = _positive(hardware.get("ssd_free_gib"), "ssd_free_gib")
    context = spec.get("context_tokens", 4096)
    if type(context) is not int or not 256 <= context <= 131072:
        raise ValueError("context_tokens must be an integer in 256..131072")
    model = spec.get("model", {})
    if not isinstance(model, dict) or model.get("architecture") not in ("moe", "dense"):
        raise ValueError("model.architecture must be moe or dense")
    weight_gib = _positive(model.get("weights_gib"), "model.weights_gib")
    unknown = set(targets) - set(SERVING_RUNTIMES)
    if unknown:
        raise ValueError(f"unsupported targets: {sorted(unknown)}")
    resolved_targets = {}
    for runtime, target in targets.items():
        if isinstance(target, str):
            model_id, target_weight = target, weight_gib
        elif isinstance(target, dict):
            model_id = target.get("model")
            target_weight = _positive(target.get("weights_gib"), f"targets.{runtime}.weights_gib")
        else:
            raise ValueError("each target must be a model ID or an object with model and weights_gib")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"targets.{runtime}.model must be a nonempty ID or path")
        resolved_targets[runtime] = {"model": model_id, "weights_gib": target_weight}
    gpu_count = hardware.get("gpu_count", 1)
    if type(gpu_count) is not int or gpu_count < 1:
        raise ValueError("gpu_count must be a positive integer")
    observations = observations or []
    if not isinstance(observations, list) or not all(isinstance(row, dict) for row in observations):
        raise ValueError("observations must be a list of objects")
    experiments = []
    diagnostics = []
    pressure = {}
    for runtime, target in resolved_targets.items():
        model_id, target_weight = target["model"], target["weights_gib"]
        seen = [row for row in observations if row.get("runtime") == runtime
                and row.get("model") == model_id and row.get("workload") == WORKLOAD]
        failures = [str(message) for row in seen for message in
                    ([row["error"]] if row.get("error") else []) +
                    [task["error"] for task in row.get("tasks", []) if task.get("error")]]
        failed = bool(failures)
        memory_failure = any(re.search(r"out of memory|\boom\b|insufficient memory|memory allocation",
                                       message, re.IGNORECASE) for message in failures)
        quality_failed = any(row.get("quality_pass") is False and
                             not any(task.get("error") for task in row.get("tasks", [])) for row in seen)
        gpu_tight = any(row.get("gpu_peak_mib") is not None and
                        row["gpu_peak_mib"] >= gpu * 1024 * 0.9 for row in seen)
        ram_tight = any(row.get("ram_available_min_mib") is not None and
                        row["ram_available_min_mib"] < 2048 for row in seen)
        pressure[runtime] = {"gpu_tight": gpu_tight, "ram_tight": ram_tight,
                             "observed_failure": failed, "quality_failed": quality_failed,
                             "observations": len(seen)}
        if quality_failed:
            diagnostics.append({"runtime": runtime, "model": model_id,
                "finding": "The server answered but failed fixed JSON checks. Review outputs, chat template, special tokens, and model compatibility before tuning speed.",
                "source": "https://docs.unsloth.ai/basics/saving-and-using-models/troubleshooting"})
        if failed and not memory_failure:
            diagnostics.append({"runtime": runtime, "model": model_id,
                "finding": "A request or server failed without a documented memory error. Inspect the saved error and server log before changing memory settings."})
        base_extra = (["--tensor-parallel-size", str(gpu_count)] if runtime == "vllm" else
                      ["--tp", str(gpu_count)] if runtime == "sglang" else []) if gpu_count > 1 else []
        argv, env = _launch(runtime, model_id, context, base_extra)
        experiments.append(_experiment(runtime, model_id, "baseline", argv, env,
            "Measure the unmodified runtime at the fixed context and task contract.", SOURCES[runtime],
            "Confirm this checkpoint, tokenizer, and chat template are supported by this runtime."))
        if runtime in ("vllm", "sglang") and target_weight > gpu * 0.8 and ram > 8 and not ram_tight:
            offload = min(math.ceil(target_weight - gpu * 0.8), math.floor(ram - 4))
            if offload > 0:
                argv, env = _launch(runtime, model_id, context,
                    [*base_extra, "--cpu-offload-gb", str(offload)])
                experiments.append(_experiment(runtime, model_id, "cpu_offload", argv, env,
                    "Test whether RAM can hold weight spill while preserving answer quality and useful speed.",
                    SOURCES[runtime], "Check CPU-GPU interconnect, host RAM, model support, and cold load time."))
                segment = model.get("expert_parameter_segment")
                if runtime == "vllm" and model["architecture"] == "moe" and segment:
                    argv, env = _launch(runtime, model_id, context,
                        [*base_extra, "--cpu-offload-gb", str(offload), "--cpu-offload-params", segment])
                    experiments.append(_experiment(runtime, model_id, "expert_only_offload", argv, env,
                        "Compare selective expert offload with general weight offload.", SOURCES[runtime],
                        "Verify the segment exactly matches checkpoint parameter names before launch.",
                        reference=f"{runtime}:cpu_offload"))
        if (gpu_tight or memory_failure) and runtime in ("vllm", "sglang"):
            kv_type = "fp8" if runtime == "vllm" else "fp8_e4m3"
            argv, env = _launch(runtime, model_id, context, [*base_extra, "--kv-cache-dtype", kv_type])
            experiments.append(_experiment(runtime, model_id, "fp8_kv", argv, env,
                "Test whether a smaller KV cache relieves GPU pressure at fixed context.",
                SOURCES[runtime], "Confirm model/device support; recheck all answers after changing KV precision."))
        if (gpu_tight or memory_failure) and runtime == "ollama":
            argv, env = _launch(runtime, model_id, context,
                env={"OLLAMA_FLASH_ATTENTION": "1"})
            experiments.append(_experiment(runtime, model_id, "flash_attention", argv, env,
                "Establish the Flash Attention control before changing KV precision.", SOURCES[runtime],
                "Confirm backend support and compare quality with the baseline."))
            argv, env = _launch(runtime, model_id, context,
                env={"OLLAMA_FLASH_ATTENTION": "1", "OLLAMA_KV_CACHE_TYPE": "q8_0"})
            experiments.append(_experiment(runtime, model_id, "q8_kv", argv, env,
                "Test lower KV memory use while checking output quality.", SOURCES[runtime],
                "Confirm Flash Attention and KV quantization support on the selected backend.",
                reference=f"{runtime}:flash_attention"))
        if runtime == "sglang" and (gpu_tight or memory_failure):
            argv, env = _launch(runtime, model_id, context, [*base_extra, "--mem-fraction-static", "0.7"])
            experiments.append(_experiment(runtime, model_id, "smaller_static_pool", argv, env,
                "Test whether reducing static allocation prevents startup or prefill OOM.", SOURCES[runtime]))
        if model["architecture"] == "moe" and gpu_count > 1 and runtime in EP_SOURCES:
            ep = ["--enable-expert-parallel"] if runtime == "vllm" else ["--ep", str(gpu_count)]
            argv, env = _launch(runtime, model_id, context, [*base_extra, *ep])
            experiments.append(_experiment(runtime, model_id, "expert_parallel", argv, env,
                "Compare expert placement across the same GPUs with the tensor-parallel baseline.",
                EP_SOURCES[runtime], "Confirm model support and inter-GPU communication; benchmark at production concurrency."))
            if runtime == "vllm" and model.get("per_expert_checkpoint") is True:
                argv, env = _launch(runtime, model_id, context,
                                    [*base_extra, *ep, "--enable-ep-weight-filter"])
                experiments.append(_experiment(runtime, model_id, "expert_weight_filter", argv, env,
                    "Compare startup disk reads and load time against expert parallel without weight filtering.",
                    SOURCES[runtime], "Only use with verified per-expert checkpoint tensors; measure cold load and SSD reads externally.",
                    reference=f"{runtime}:expert_parallel"))
    storage = []
    if disk >= 2 * max(target["weights_gib"] for target in resolved_targets.values()):
        storage.append({"id": "ssd:cold_vs_warm_load", "hypothesis":
            "Compare cold and warm load times with the same verified model bytes on local SSD.",
            "control": "Keep runtime, model hash, quantization, context, and RAM cache state recorded.",
            "measure": ["startup/load time", "SSD read bytes", "RAM available minimum"],
            "note": "This is a measurement protocol, not evidence that SSD changes token decode speed."})
    return {"schema": "deployment-plan-v1", "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": model, "hardware": hardware, "targets": resolved_targets,
            "context_tokens": context,
            "source_catalog": SOURCES, "source_scope": "official documentation; check installed runtime --help/version before launch",
            "pressure": pressure, "diagnostics": diagnostics,
            "experiments": experiments, "storage_experiments": storage,
            "artifact_preparation": {"unsloth": "Export a compatible merged checkpoint for vLLM/SGLang or GGUF for Ollama; preserve the training chat template and verify model bytes before comparing runtimes.",
                                     "source": SOURCES["unsloth"]},
            "rule": "Run each runtime baseline first, then change one setting per referenced control. Keep artifact bytes, tasks, and hardware fixed within a runtime; record format and quantization differences across runtimes."}


def _host_snapshot():
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                memory["ram_available_mib"] = int(line.split()[1]) / 1024
                break
    except (OSError, ValueError, IndexError):
        pass
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=3, check=True).stdout
        memory["gpu_used_mib_by_device"] = [float(line) for line in output.strip().splitlines()]
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return memory


def benchmark(endpoint: str, runtime: str, model: str, config: dict | None = None) -> dict:
    """Measure a running local OpenAI-compatible chat endpoint, without owning it."""
    parsed = urlsplit(endpoint)
    if (parsed.scheme != "http" or parsed.hostname not in ("localhost", "127.0.0.1")
            or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("endpoint must be a local http://localhost or http://127.0.0.1 origin")
    if runtime not in SERVING_RUNTIMES or not model:
        raise ValueError("runtime and model must identify a catalog runtime and nonempty model")
    route = endpoint.rstrip("/") + "/v1/chat/completions"
    opener = build_opener(ProxyHandler({}))
    rows, samples = [], []
    for round_index, tasks in enumerate((TUNING_TASKS, tuple(reversed(TUNING_TASKS)))):
        for name, prompt, expected in tasks:
            payload = {"model": model, "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                                     {"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": 96, "stream": False}
            request = Request(route, data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            samples.append(_host_snapshot())
            start = time.monotonic()
            try:
                with opener.open(request, timeout=120) as response:
                    answer = json.load(response)
                latency = time.monotonic() - start
                content = answer["choices"][0]["message"].get("content") or ""
                usage = answer.get("usage") or {}
                rows.append({"task_id": name, "round": round_index, "passed": _score(content, expected),
                             "raw_output": content, "latency_seconds": latency,
                             "completion_tokens": usage.get("completion_tokens")})
            except (OSError, ValueError, KeyError, IndexError) as error:
                rows.append({"task_id": name, "round": round_index, "passed": False,
                             "error": f"{type(error).__name__}: {error}"})
            samples.append(_host_snapshot())
    latency = [row["latency_seconds"] for row in rows if "latency_seconds" in row]
    rates = [row["completion_tokens"] / row["latency_seconds"] for row in rows
             if isinstance(row.get("completion_tokens"), int) and row.get("latency_seconds", 0) > 0]
    gpu = [sample["gpu_used_mib_by_device"] for sample in samples if "gpu_used_mib_by_device" in sample]
    ram = [sample["ram_available_mib"] for sample in samples if "ram_available_mib" in sample]
    gpu_devices = [max(values[index] for values in gpu if len(values) > index)
                   for index in range(max(map(len, gpu), default=0))]
    return {"schema": "deployment-benchmark-v1", "workload": WORKLOAD, "runtime": runtime,
            "model": model, "config": config or {}, "endpoint": endpoint,
            "measured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "quality_pass": len(rows) == 12 and all(row["passed"] for row in rows), "tasks": rows,
            "median_request_latency_seconds": statistics.median(latency) if latency else None,
            "median_completion_tokens_per_response_second": statistics.median(rates) if rates else None,
            "gpu_peak_mib": max(map(sum, gpu)) if gpu else None,
            "gpu_per_device_peak_mib": gpu_devices,
            "ram_available_min_mib": min(ram) if ram else None,
            "resource_scope": "total host observations around requests; peaks between samples can be missed; external server startup, per-process use, and SSD IO are not sampled"}


def compare(results: list[dict]) -> dict:
    if not results:
        raise ValueError("results must contain at least one benchmark")
    if any(row.get("workload") != WORKLOAD for row in results):
        raise ValueError("results must share the same workload contract")
    ranked = sorted(results, key=lambda row: (
        not row.get("quality_pass", False),
        -(row.get("median_completion_tokens_per_response_second") or 0)))
    return {"models": sorted({row["model"] for row in results}), "workload": WORKLOAD,
            "ranking": [{"runtime": row["runtime"], "model": row["model"],
                         "config": row.get("config", {}),
                         "outcome": "worked" if row.get("quality_pass") else "did_not_work",
                         "quality_pass": row.get("quality_pass", False),
                         "response_tokens_per_second": row.get("median_completion_tokens_per_response_second"),
                         "gpu_peak_mib": row.get("gpu_peak_mib"),
                         "ram_available_min_mib": row.get("ram_available_min_mib")}
                        for row in ranked],
            "caution": "Throughput is completion tokens divided by full response time, not decode-only speed. Different model IDs or quantizations can be compared as deployments, but cannot isolate a runtime effect. Check hardware, model bytes, tokenizer, chat template, and load conditions before a causal claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    docs = sub.add_parser("docs", help="search official runtime documentation indexes")
    docs.add_argument("--query", required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--spec", type=Path, required=True)
    planning.add_argument("--results", type=Path, nargs="*", default=[])
    measuring = sub.add_parser("benchmark")
    measuring.add_argument("--endpoint", required=True)
    measuring.add_argument("--runtime", choices=SERVING_RUNTIMES, required=True)
    measuring.add_argument("--model", required=True)
    measuring.add_argument("--config-id", default="baseline")
    measuring.add_argument("--config-json", type=Path,
                           help="plan experiment JSON to retain exact launch argv, env and source in evidence")
    measuring.add_argument("--plan", type=Path,
                           help="plan JSON containing the experiment to retain in evidence")
    measuring.add_argument("--experiment-id", help="experiment ID within --plan")
    comparing = sub.add_parser("compare")
    comparing.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    if args.command == "docs":
        result = discover_docs(args.query)
    elif args.command == "plan":
        result = plan(json.loads(args.spec.read_text()), [json.loads(p.read_text()) for p in args.results])
    elif args.command == "benchmark":
        if bool(args.plan) != bool(args.experiment_id) or (args.plan and args.config_json):
            parser.error("use --plan with --experiment-id, or --config-json, or --config-id")
        if args.plan:
            saved_plan = json.loads(args.plan.read_text())
            matches = [row for row in saved_plan["experiments"] if row["id"] == args.experiment_id]
            if len(matches) != 1:
                parser.error("experiment ID not found exactly once in the plan")
            config = matches[0]
        else:
            config = json.loads(args.config_json.read_text()) if args.config_json else {"id": args.config_id}
        if config.get("runtime", args.runtime) != args.runtime or config.get("model", args.model) != args.model:
            parser.error("config runtime and model must match the benchmark target")
        result = benchmark(args.endpoint, args.runtime, args.model, config)
    else:
        result = compare([json.loads(p.read_text()) for p in args.results])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
