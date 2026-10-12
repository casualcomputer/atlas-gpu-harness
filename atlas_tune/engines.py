"""Engines as Docker images plus an allowlist of the flags worth tuning.

Configs are plain dicts of knob → value. Anything outside the allowlist is rejected, so
the agent (or a web page it read) can never inject arbitrary container arguments.

The allowlist is the union of tuning flags across recent engine versions, taken from each
engine's official docs. Before a run, atlas reads the pulled image's own `--help` and keeps
only the knobs that image actually has (`Engine.supported`), so flags that were renamed or
removed in that version are never sent, and the agent sees that version's help text.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .models import Artifact


@dataclass(frozen=True)
class Knob:
    flag: str
    kind: type            # int, float, str or bool
    low: float | None = None
    high: float | None = None
    choices: tuple = ()
    help: str = ""
    off_flag: str = ""    # bool knobs with an explicit "off" form, e.g. --no-enable-prefix-caching

    def validate(self, name: str, value):
        if self.kind is bool:
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be true/false")
            return value
        if self.choices:
            if self.kind in (int, float) and not isinstance(value, bool):
                try:
                    value = self.kind(value)
                except (TypeError, ValueError):
                    pass
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
            return [self.flag] if value else ([self.off_flag] if self.off_flag else [])
        return [self.flag, str(value)]

    def in_help(self, help_text: str) -> bool:
        flags = [self.flag] + ([self.off_flag] if self.off_flag else [])
        return all(re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", help_text) for f in flags)


@dataclass(frozen=True)
class Engine:
    name: str
    image: str
    port: int
    format: str                        # artifact format it loads: "hf" or "gguf"
    knobs: dict[str, Knob] = field(default_factory=dict)
    docs: str = ""
    # How to read this image's own documentation: (entrypoint or "", args) for --help and version.
    help_cmd: tuple = ()
    version_cmd: tuple = ()

    def supported(self, help_text: str | None) -> dict[str, Knob]:
        """Knobs present in this image's --help (all of them if the help is unknown)."""
        if not help_text:
            return dict(self.knobs)
        return {name: knob for name, knob in self.knobs.items() if knob.in_help(help_text)}

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


VLLM_DOCS = "https://docs.vllm.ai/en/latest/configuration/engine_args/"
SGLANG_DOCS = "https://docs.sglang.io/docs/advanced_features/server_arguments"
LLAMA_DOCS = "https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md"

ENGINES: dict[str, Engine] = {
    "vllm": Engine("vllm", _image("vllm", "vllm/vllm-openai:latest"), 8000, "hf", {
        "gpu_memory_utilization": Knob("--gpu-memory-utilization", float, 0.5, 0.98, help="share of VRAM vLLM may claim"),
        "max_num_seqs": Knob("--max-num-seqs", int, 1, 2048, help="max concurrent sequences per step"),
        "max_num_batched_tokens": Knob("--max-num-batched-tokens", int, 256, 262144, help="tokens per scheduler step; trades prefill speed vs decode latency"),
        "kv_cache_dtype": Knob("--kv-cache-dtype", str, choices=("auto", "fp8", "fp8_e4m3", "fp8_e5m2"), help="fp8 roughly halves KV memory"),
        "enable_prefix_caching": Knob("--enable-prefix-caching", bool, off_flag="--no-enable-prefix-caching", help="reuse KV for shared prompt prefixes"),
        "block_size": Knob("--block-size", int, choices=(8, 16, 32, 64, 128), help="KV cache block size in tokens"),
        "dtype": Knob("--dtype", str, choices=("auto", "bfloat16", "float16"), help="weight/activation dtype; float16 recommended for AWQ"),
        "enforce_eager": Knob("--enforce-eager", bool, help="disable CUDA graphs and torch.compile: less memory, slower decode"),
        "max_cudagraph_capture_size": Knob("--max-cudagraph-capture-size", int, 1, 1024, help="largest batch captured in a CUDA graph"),
        "async_scheduling": Knob("--async-scheduling", bool, help="overlap scheduling with execution"),
        "cpu_offload_gb": Knob("--cpu-offload-gb", float, 0, 512, help="weights kept in CPU RAM; needs fast CPU-GPU link"),
        "kv_offloading_size": Knob("--kv-offloading-size", float, 0, 512, help="GiB of KV cache offloaded to CPU RAM"),
    }, VLLM_DOCS, help_cmd=("vllm", ("serve", "--help=all")), version_cmd=("vllm", ("--version",))),
    "sglang": Engine("sglang", _image("sglang", "lmsysorg/sglang:latest"), 30000, "hf", {
        "mem_fraction_static": Knob("--mem-fraction-static", float, 0.4, 0.95, help="share of VRAM for weights + KV pool; lower on OOM"),
        "max_running_requests": Knob("--max-running-requests", int, 1, 4096),
        "chunked_prefill_size": Knob("--chunked-prefill-size", int, -1, 65536, help="max tokens per prefill chunk; -1 disables chunking"),
        "max_prefill_tokens": Knob("--max-prefill-tokens", int, 512, 262144, help="max tokens in one prefill batch"),
        "kv_cache_dtype": Knob("--kv-cache-dtype", str, choices=("auto", "fp8_e4m3", "fp8_e5m2", "bf16"), help="fp8 shrinks KV cache"),
        "schedule_policy": Knob("--schedule-policy", str, choices=("fcfs", "lpm", "random", "dfs-weight", "lof"), help="lpm favours prefix-cache hits"),
        "schedule_conservativeness": Knob("--schedule-conservativeness", float, 0.1, 10, help="raise if requests get retracted often"),
        "disable_radix_cache": Knob("--disable-radix-cache", bool, help="turn off prefix (radix) caching"),
        "page_size": Knob("--page-size", int, choices=(1, 16, 32, 64), help="tokens per KV page"),
        "cuda_graph_max_bs": Knob("--cuda-graph-max-bs", int, 1, 1024, help="largest batch captured in a CUDA graph"),
        "disable_cuda_graph": Knob("--disable-cuda-graph", bool, help="less memory, slower decode"),
        "dtype": Knob("--dtype", str, choices=("auto", "bfloat16", "float16")),
        "cpu_offload_gb": Knob("--cpu-offload-gb", int, 0, 512),
    }, SGLANG_DOCS, help_cmd=("", ("python3", "-m", "sglang.launch_server", "--help")),
       version_cmd=("", ("python3", "-c", "import sglang; print(sglang.__version__)"))),
    "llamacpp": Engine("llamacpp", _image("llamacpp", "ghcr.io/ggml-org/llama.cpp:server-cuda"), 8080, "gguf", {
        "n_gpu_layers": Knob("-ngl", int, 0, 999, help="layers on GPU (999 = all)"),
        "n_cpu_moe": Knob("--n-cpu-moe", int, 0, 512, help="keep MoE experts of the first N layers in CPU RAM"),
        "cpu_moe": Knob("--cpu-moe", bool, help="keep all MoE experts in CPU RAM"),
        "parallel": Knob("-np", int, 1, 64, help="parallel request slots; context is split across them"),
        "flash_attn": Knob("-fa", str, choices=("on", "off", "auto")),
        "cache_type_k": Knob("-ctk", str, choices=("f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl"), help="quantized K cache needs flash attention"),
        "cache_type_v": Knob("-ctv", str, choices=("f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "iq4_nl")),
        "batch_size": Knob("-b", int, 32, 16384, help="logical batch (prompt processing)"),
        "ubatch_size": Knob("-ub", int, 32, 8192, help="physical batch; larger speeds prefill, costs VRAM"),
        "threads": Knob("-t", int, 1, 256, help="CPU threads for generation (matters with MoE offload)"),
        "threads_batch": Knob("-tb", int, 1, 256, help="CPU threads for prompt processing"),
        "kv_unified": Knob("--kv-unified", bool, off_flag="--no-kv-unified", help="one KV buffer shared by all slots"),
        "kv_offload": Knob("--kv-offload", bool, off_flag="--no-kv-offload", help="keep KV cache on GPU (off moves it to RAM)"),
        "cache_reuse": Knob("--cache-reuse", int, 0, 4096, help="min chunk size to reuse cached prompt via KV shifting"),
    }, LLAMA_DOCS, help_cmd=("", ("--help",)), version_cmd=("", ("--version",))),
}


def docker_run(engine: Engine, artifact: Artifact, config: dict, context: int, *, host_port: int,
               name: str, production: bool = False, labels: dict | None = None) -> list[str]:
    """Full `docker run` argv.

    Only the two model caches are mounted, never the working directory or home folder.
    Trials bind to 127.0.0.1 only; the production recipe publishes the port and restarts.
    """
    hf_cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    llama_cache = Path(os.environ.get("LLAMA_CACHE", Path.home() / ".cache/llama.cpp"))
    publish = f"{host_port}:{engine.port}" if production else f"127.0.0.1:{host_port}:{engine.port}"
    argv = ["docker", "run", "-d", *(["--restart", "unless-stopped"] if production else []),
            "--gpus", "all", "--ipc=host", "--security-opt", "no-new-privileges", "--name", name, "-p", publish,
            "-v", f"{hf_cache}:/root/.cache/huggingface", "-v", f"{llama_cache}:/root/.cache/llama.cpp",
            "-e", "LLAMA_CACHE=/root/.cache/llama.cpp", "-e", "HF_TOKEN"]
    for key, value in (labels or {}).items():
        argv += ["--label", f"{key}={value}"]
    return [*argv, engine.image, *engine.args(artifact, config, context)]
