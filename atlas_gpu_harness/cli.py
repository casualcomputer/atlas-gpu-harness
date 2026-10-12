"""Hackathon-era experiments: `atlas-harness <command>`.

Each command delegates to an existing module CLI, so `python -m
atlas_gpu_harness.<module>` keeps working. Run from the repository root:
artifacts, the engine runtime and `.env` are read relative to it.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

DEFAULT_ENGINE = Path("artifacts/engines/llama-b11429/launch-llama.sh")
OLLAMA = "http://127.0.0.1:11434"
DEFAULT_PROPOSER = "gemma3:12b"

# name -> (module, fixed leading args, help)
DELEGATES = {
    "inspect": ("atlas_gpu_harness.inference", ["inspect"],
                "Read GGUF metadata and the machine; print an initial placement estimate"),
    "tune": ("atlas_gpu_harness.inference", ["evolve"],
             "Baseline, generate and test placement policies, confirm on held-out tasks (Atlas)"),
    "serve": ("atlas_gpu_harness.inference_serve", [],
              "Relaunch a confirmed recommendation.json with llama.cpp"),
    "deploy": ("atlas_gpu_harness.deployment_engineer", [],
               "vLLM/SGLang: docs search, experiment plan, endpoint benchmark, compare"),
    "strata": ("atlas_gpu_harness.strata_experiment", [],
               "Native Strata policy sweep at 32K context (Atlas)"),
    "setup-engine": ("atlas_gpu_harness.setup_engine", [],
                     "Install the pinned llama.cpp b11429 CUDA runtime under artifacts/engines"),
    "topk": ("atlas_gpu_harness.__main__", [],
             "Original exact top-k operator demo (needs the torch extra)"),
}


def _delegate(name: str, argv: list[str]) -> None:
    module_name, fixed, _ = DELEGATES[name]
    module = importlib.import_module(module_name)
    sys.argv = [f"atlas-harness {name}", *fixed, *argv]
    module.main()


# ---------------------------------------------------------------- doctor ---

def _row(status: str, name: str, detail: str, hint: str = "") -> dict:
    return {"status": status, "check": name, "detail": detail, "hint": hint}


def _check_atlas() -> dict:
    from dotenv import load_dotenv

    load_dotenv(override=False)
    if not os.environ.get("MONGODB_URI"):
        return _row("fail", "Atlas", "MONGODB_URI is not set",
                    "cp .env.example .env and paste your Atlas connection string")
    try:
        from .store import AtlasStore

        store = AtlasStore()
        store.init()
        return _row("ok", "Atlas", f"ping and write/read ok (database {store.db.name})")
    except Exception as error:  # noqa: BLE001 - report any connection failure
        return _row("fail", "Atlas", f"{type(error).__name__}: {str(error)[:160]}",
                    "check the URI, database user and the IP access list in Atlas")


def _check_gpu() -> dict:
    if not shutil.which("nvidia-smi"):
        return _row("fail", "NVIDIA GPU", "nvidia-smi not found", "install the NVIDIA driver (WSL: Windows driver)")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10, check=True).stdout.strip()
        return _row("ok", "NVIDIA GPU", out.replace("\n", "; "))
    except (OSError, subprocess.SubprocessError) as error:
        return _row("fail", "NVIDIA GPU", f"nvidia-smi failed: {error}")


def _check_engine(path: Path) -> dict:
    if not path.is_file():
        return _row("fail", "llama.cpp", f"{path} missing", "run: uv run atlas-harness setup-engine")
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                             timeout=30, check=True)
        version = (out.stdout + out.stderr).strip().splitlines()
        return _row("ok", "llama.cpp", version[0] if version else str(path))
    except (OSError, subprocess.SubprocessError) as error:
        return _row("fail", "llama.cpp", f"{path} did not run: {error}",
                    "run: uv run atlas-harness setup-engine --check")


def _check_ollama(model: str) -> dict:
    try:
        with build_opener(ProxyHandler({})).open(f"{OLLAMA}/api/tags", timeout=3) as response:
            names = [m.get("name") for m in json.load(response).get("models", [])]
    except (OSError, ValueError):
        return _row("fail", "Ollama", f"not reachable at {OLLAMA}",
                    f"install Ollama, then: ollama serve & ollama pull {model}")
    if model not in names and f"{model}:latest" not in names:
        return _row("warn", "Ollama", f"running; {model} not pulled", f"ollama pull {model}")
    return _row("ok", "Ollama", f"running with {model}")


def _check_torch() -> dict:
    try:
        import torch
    except ImportError:
        return _row("skip", "PyTorch", "not installed (only needed for `atlas-harness topk`)",
                    "uv sync --extra torch")
    cuda = torch.cuda.is_available()
    return _row("ok" if cuda else "warn", "PyTorch",
                f"{torch.__version__}, CUDA {'available' if cuda else 'NOT available'}")


def doctor(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="atlas-harness doctor",
                                     description="Check everything `atlas-harness tune` needs, with fixes.")
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--proposer-model", default=DEFAULT_PROPOSER)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)
    py = sys.version_info
    rows = [
        _row("ok" if py >= (3, 11) else "fail", "Python",
             f"{platform.python_version()} ({sys.executable})"),
        _check_atlas(), _check_gpu(), _check_engine(args.engine),
        _check_ollama(args.proposer_model), _check_torch(),
    ]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        marks = {"ok": "✓", "fail": "✗", "warn": "!", "skip": "–"}
        width = max(len(r["check"]) for r in rows)
        for r in rows:
            print(f" {marks[r['status']]} {r['check']:<{width}}  {r['detail']}")
            if r["hint"] and r["status"] != "ok":
                print(f"   {'':<{width}}  → {r['hint']}")
    return 1 if any(r["status"] == "fail" and r["check"] != "PyTorch" for r in rows) else 0


# -------------------------------------------------------------------- ui ---

def ui(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="atlas-harness ui", description="Serve the static demo site.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not Path("demo-ui/index.html").is_file():
        parser.error("run from the repository root (demo-ui/ not found)")
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(SimpleHTTPRequestHandler, directory="."))
    print(f"Demo site: http://127.0.0.1:{args.port}/demo-ui/   (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# ------------------------------------------------------------------ main ---

def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"doctor": "Check Python, Atlas, GPU, llama.cpp and Ollama, with fixes",
                **{name: spec[2] for name, spec in DELEGATES.items()},
                "ui": "Serve the static demo site on 127.0.0.1"}
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in commands:
        width = max(map(len, commands))
        print("usage: atlas-harness <command> [options]   (atlas-harness <command> -h for details)\n")
        for name, text in commands.items():
            print(f"  {name:<{width}}  {text}")
        if argv and argv[0] not in ("-h", "--help"):
            print(f"\natlas-harness: unknown command {argv[0]!r}", file=sys.stderr)
            raise SystemExit(2)
        return
    name, rest = argv[0], argv[1:]
    if name == "doctor":
        raise SystemExit(doctor(rest))
    if name == "ui":
        ui(rest)
        return
    _delegate(name, rest)


if __name__ == "__main__":
    main()
