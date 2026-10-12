"""Engines as pinned Docker images plus a short allowlist of the flags worth tuning.

Configs are plain dicts of knob → value. Anything outside the allowlist is rejected, so
the agent (or a web page it read) can never inject arbitrary container arguments.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from .models import Artifact


@dataclass(frozen=True)
class Knob:
    flag: str
    kind: type            # int, float, str or bool (bool = presence flag)
    low: float | None = None
    high: float | None = None
    choices: tuple = ()
    help: str = ""

    def validate(self, name: str, value):
        if self.kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true/false")
            return value
        if self.choices:
            if value not in self.choices:
                raise ValueError(f"{name} must be one of {list(self.choices)}")
            return value
        try:
            number = self.kind(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be {self.kind.__name__}") from None
        if isinstance(value, bool) or (isinstance(number, float) and not math.isfinite(number)):
            raise ValueError(f"{name} must be a finite number")
        if (self.low is not None and number < self.low) or (self.high is not None and number > self.high):
            raise ValueError(f"{name} must be in [{self.low}, {self.high}]")
        return number

    def argv(self, value) -> list[str]:
        if self.kind is bool:
            return [self.flag] if value else []
        return [self.flag, str(value)]


@dataclass(frozen=True)
class Engine:
    name: str
    image: str
    port: int
    format: str                        # artifact format it loads: "hf" or "gguf"
    knobs: dict[str, Knob] = field(default_factory=dict)
    docs: str = ""

    def validate(self, config: dict) -> dict:
        unknown = set(config) - set(self.knobs)
        if unknown:
            raise ValueError(f"{self.name}: unknown settings {sorted(unknown)}")
        return {k: self.knobs[k].validate(k, v) for k, v in sorted(config.items())}

    def args(self, artifact: Artifact, config: dict, context: int) -> list[str]:
        """Container command (everything after the image name)."""
        config = self.validate(config)
        tuned = [arg for key, value in config.items() for arg in self.knobs[key].argv(value)]
        if self.name == "vllm":
            return ["--model", artifact.repo, "--host", "0.0.0.0", "--port", str(self.port),
                    "--max-model-len", str(context), *tuned]
        if self.name == "sglang":
            return ["python3", "-m", "sglang.launch_server", "--model-path", artifact.repo,
                    "--host", "0.0.0.0", "--port", str(self.port), "--context-length", str(context), *tuned]
        if self.name == "llamacpp":
            # -c is shared by all parallel slots, so give each slot the requested context.
            slots = int(config.get("parallel", 1))
            return ["-hf", f"{artifact.repo}:{artifact.quant}", "--host", "0.0.0.0", "--port", str(self.port),
                    "-c", str(context * slots), "--jinja", *tuned]
        raise ValueError(f"unknown engine {self.name}")


def _image(name: str, default: str) -> str:
    return os.environ.get(f"ATLAS_{name.upper()}_IMAGE", default)


ENGINES: dict[str, Engine] = {
    "vllm": Engine("vllm", _image("vllm", "vllm/vllm-openai:latest"), 8000, "hf", {
        "gpu_memory_utilization": Knob("--gpu-memory-utilization", float, 0.5, 0.97, help="fraction of VRAM vLLM may claim"),
        "max_num_seqs": Knob("--max-num-seqs", int, 1, 1024, help="max concurrent sequences"),
        "max_num_batched_tokens": Knob("--max-num-batched-tokens", int, 256, 131072, help="tokens per scheduler step (chunked prefill)"),
        "kv_cache_dtype": Knob("--kv-cache-dtype", str, choices=("auto", "fp8"), help="fp8 halves KV memory"),
        "cpu_offload_gb": Knob("--cpu-offload-gb", float, 0, 512, help="weights kept in CPU RAM (slow, for models that don't fit)"),
        "enforce_eager": Knob("--enforce-eager", bool, help="disable CUDA graphs (less memory, slower)"),
    }, "https://docs.vllm.ai/en/latest/configuration/engine_args/"),
    "sglang": Engine("sglang", _image("sglang", "lmsysorg/sglang:latest"), 30000, "hf", {
        "mem_fraction_static": Knob("--mem-fraction-static", float, 0.5, 0.95, help="fraction of VRAM for weights + KV pool"),
        "max_running_requests": Knob("--max-running-requests", int, 1, 1024),
        "chunked_prefill_size": Knob("--chunked-prefill-size", int, 256, 65536),
        "kv_cache_dtype": Knob("--kv-cache-dtype", str, choices=("auto", "fp8_e4m3")),
        "cpu_offload_gb": Knob("--cpu-offload-gb", int, 0, 512),
        "disable_cuda_graph": Knob("--disable-cuda-graph", bool),
    }, "https://docs.sglang.io/docs/advanced_features/server_arguments"),
    "llamacpp": Engine("llamacpp", _image("llamacpp", "ghcr.io/ggml-org/llama.cpp:server-cuda"), 8080, "gguf", {
        "n_gpu_layers": Knob("-ngl", int, 0, 999, help="layers on GPU (999 = all)"),
        "n_cpu_moe": Knob("--n-cpu-moe", int, 0, 512, help="keep MoE experts of the first N layers in CPU RAM"),
        "parallel": Knob("-np", int, 1, 64, help="parallel request slots"),
        "flash_attn": Knob("-fa", str, choices=("on", "off", "auto")),
        "cache_type_k": Knob("-ctk", str, choices=("f16", "q8_0", "q4_0")),
        "cache_type_v": Knob("-ctv", str, choices=("f16", "q8_0", "q4_0")),
        "batch_size": Knob("-b", int, 32, 8192),
        "ubatch_size": Knob("-ub", int, 32, 8192),
    }, "https://github.com/ggml-org/llama.cpp/tree/master/tools/server"),
}


def docker_run(engine: Engine, artifact: Artifact, config: dict, context: int, *, host_port: int,
               name: str, production: bool = False, labels: dict | None = None) -> list[str]:
    """Full `docker run` argv. Caches are mounted so weights download once.

    Trials bind to 127.0.0.1 only; the production recipe publishes the port and restarts.
    """
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    llama_cache = Path(os.environ.get("LLAMA_CACHE", Path.home() / ".cache/llama.cpp"))
    publish = f"{host_port}:{engine.port}" if production else f"127.0.0.1:{host_port}:{engine.port}"
    argv = ["docker", "run", "-d", *(["--restart", "unless-stopped"] if production else []),
            "--gpus", "all", "--ipc=host", "--name", name, "-p", publish,
            "-v", f"{hf_cache}:/root/.cache/huggingface", "-v", f"{llama_cache}:/root/.cache/llama.cpp",
            "-e", "LLAMA_CACHE=/root/.cache/llama.cpp", "-e", "HF_TOKEN"]
    for key, value in (labels or {}).items():
        argv += ["--label", f"{key}={value}"]
    return [*argv, engine.image, *engine.args(artifact, config, context)]
