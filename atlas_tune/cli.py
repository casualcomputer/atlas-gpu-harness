"""atlas: find the fastest verified way to serve a model on your GPU.

  atlas doctor                          check Docker, GPU, keys, local DB and Atlas
  atlas run Qwen/Qwen3-8B               try every engine, keep the best verified config
  atlas run MODEL --dry-run --gpu-gib 16   show the trials and docker commands only
  atlas history [MODEL]                 past runs and trials from local memory
  atlas recipe MODEL                    best launch command found on this machine
  atlas sync                            copy local memory to MongoDB Atlas
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys

from dotenv import load_dotenv

from . import bench, hardware
from .engines import ENGINES


def _doctor(args) -> int:
    from .store import AtlasMirror, Store

    rows = []
    d = hardware.docker()
    rows.append(("Docker", d["ok"], d["detail"], "install Docker Engine (WSL: Docker Desktop with WSL integration)"))
    if d["ok"]:
        # Docker Desktop (Windows/WSL2) passes GPUs through without listing an "nvidia" runtime.
        rows.append(("NVIDIA runtime", True if d.get("nvidia_runtime") else None,
                     "nvidia" if d.get("nvidia_runtime") else "not listed (normal on Docker Desktop)",
                     "Linux: install the NVIDIA Container Toolkit; Docker Desktop: enable the WSL2 backend"))
    gpus = hardware.gpus()
    rows.append(("GPU", bool(gpus), "; ".join(f"{g['name']} {g['vram_gib']} GiB ({g['used_gib']} used)" for g in gpus)
                 or "nvidia-smi found no GPU", "install the NVIDIA driver"))
    busy = sum(g["used_gib"] for g in gpus)
    if busy > 1:
        rows.append(("GPU idle", None, f"{busy:.1f} GiB already in use", "stop other GPU apps (e.g. Ollama) for clean results"))
    rows.append(("HF token", None if not os.environ.get("HF_TOKEN") else True,
                 "set" if os.environ.get("HF_TOKEN") else "not set (only gated models need it)", "export HF_TOKEN=…"))
    from .research import native_web_search

    llm = os.environ.get("ATLAS_LLM")
    search = "with built-in web search" if native_web_search(llm) else "no web search (model card + GitHub only)"
    rows.append(("Agent LLM", True if llm else None, f"{llm}, {search}" if llm else "not set (stock configs only)",
                 "optional: set ATLAS_LLM, e.g. anthropic/claude-sonnet-5-5, gemini/gemini-2.5-flash, ollama_chat/qwen3:8b"))
    try:
        store = Store(mirror=False)
        rows.append(("Local memory", True, f"{store.path} ({len(store.trials())} trials)", ""))
    except Exception as error:  # noqa: BLE001
        rows.append(("Local memory", False, str(error), "check ATLAS_HOME is writable"))
    if os.environ.get("MONGODB_URI"):
        try:
            AtlasMirror(os.environ["MONGODB_URI"]).db.client.admin.command("ping")
        except ImportError:
            rows.append(("Atlas mirror", False, "MONGODB_URI set but pymongo not installed", "uv sync --extra atlas"))
        except Exception as error:  # noqa: BLE001
            rows.append(("Atlas mirror", False, f"{type(error).__name__}", "check MONGODB_URI and the Atlas IP access list"))
        else:
            rows.append(("Atlas mirror", True, "connected", ""))
    else:
        rows.append(("Atlas mirror", None, "off (local only)", "set MONGODB_URI in .env to share results"))
    width = max(len(r[0]) for r in rows)
    for name, ok, detail, fix in rows:
        mark = "✓" if ok else ("✗" if ok is False else "–")
        print(f" {mark} {name:<{width}}  {detail}")
        if not ok and fix:
            print(f"   {'':<{width}}  → {fix}")
    return 1 if any(ok is False for name, ok, *_ in rows if name in ("Docker", "GPU", "Local memory")) else 0


def _run(args) -> int:
    from .tuner import Options, Tuner

    opts = Options(model=args.model, engines=args.engines, goal=args.goal, workload=args.workload,
                   context=args.context, budget_minutes=args.budget_minutes, max_trials=args.max_trials,
                   llm=args.llm or os.environ.get("ATLAS_LLM"), research=not args.no_research,
                   gguf_repo=args.gguf_repo, extra_hf=tuple(args.hf_repo), dry_run=args.dry_run,
                   vram_gib=args.gpu_gib, rounds=max(1, args.rounds), share_logs=not args.no_share_logs)
    if not args.dry_run and not hardware.docker()["ok"]:
        print("Docker is not available; run `atlas doctor`.", file=sys.stderr)
        return 1
    summary = Tuner(opts).run()
    if args.dry_run:
        return 0
    print()
    winner = summary.get("winner")
    if winner:
        print(f"Best: {winner['engine']} · score {winner['score']} ({args.goal})")
        print(f"Report: {summary['report']}\nLaunch:\n  {winner['launch']}")
    else:
        print(f"No configuration passed every check. Report: {summary['report']}")
    return 0 if winner else 2


def _history(args) -> int:
    from .store import Store

    store = Store(mirror=False)
    for run in store.runs(args.model):
        winner = (run.get("summary") or {}).get("winner") or {}
        print(f"{run['created']}  {run['id']}  {run['model']}  {run['goal']}/{run['workload']}  "
              f"{run['status']}  best={winner.get('engine', '–')} {winner.get('score', '')}")
    if args.model:
        print()
        for t in store.trials(args.model):
            s = bench.score(t.get("metrics"), "throughput")
            print(f"  {t['created']}  {t['engine']:<8} {t['status']:<14} {s or '':>8}  {json.dumps(t['config'])}"
                  + (f"  ! {t['error'][:80]}" if t.get("error") else ""))
    return 0


def _recipe(args) -> int:
    from .store import Store

    store = Store(mirror=False)
    machine = hardware.fingerprint(hardware.detect())
    recipes = [r for r in store.recipes(args.model, machine) if r["goal"] == args.goal]
    if not recipes:
        print(f"No verified recipe for {args.model} ({args.goal}) on this machine. Run: atlas run {args.model}")
        return 1
    best = recipes[0]
    print(f"# {best['engine']} · {best['workload']} · score {best['score']} · verified {best['updated']}")
    print(shlex.join(best["launch"]))
    return 0


def _sync(args) -> int:
    from .store import Store

    print(json.dumps(Store().sync()))
    return 0


def main(argv: list[str] | None = None) -> None:
    load_dotenv(override=False)
    parser = argparse.ArgumentParser(prog="atlas", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check Docker, GPU, keys, local DB and Atlas")
    run = sub.add_parser("run", help="benchmark every engine for a model and keep the best")
    run.add_argument("model", help="Hugging Face model ID, e.g. Qwen/Qwen3-8B")
    run.add_argument("--engines", type=lambda s: s.split(","), default=list(ENGINES),
                     help=f"comma-separated subset of {','.join(ENGINES)}")
    run.add_argument("--goal", choices=("throughput", "latency"), default="throughput")
    run.add_argument("--workload", choices=list(bench.WORKLOADS), default="chat")
    run.add_argument("--context", type=int, help="max context per request (default: workload size)")
    run.add_argument("--budget-minutes", type=float, default=90)
    run.add_argument("--max-trials", type=int, default=5, help="trials per engine")
    run.add_argument("--llm", help="LiteLLM model for the agent (default: $ATLAS_LLM; none = defaults only)")
    run.add_argument("--rounds", type=int, default=3,
                     help="1 = planned trials only; each extra round fixes failures and refines (default 3)")
    run.add_argument("--no-research", action="store_true", help="agent proposes without web/GitHub search")
    run.add_argument("--no-share-logs", action="store_true",
                     help="don't send redacted engine log tails of failed trials to the agent LLM")
    run.add_argument("--gguf-repo", help="GGUF repo for llama.cpp (default: found automatically)")
    run.add_argument("--hf-repo", action="append", default=[], help="extra HF repo, e.g. an AWQ/FP8 variant")
    run.add_argument("--dry-run", action="store_true", help="plan only: print trials and docker commands")
    run.add_argument("--gpu-gib", type=float, help="plan for this VRAM size instead of the detected GPU")
    history = sub.add_parser("history", help="past runs and trials")
    history.add_argument("model", nargs="?")
    recipe = sub.add_parser("recipe", help="best verified launch command on this machine")
    recipe.add_argument("model")
    recipe.add_argument("--goal", choices=("throughput", "latency"), default="throughput")
    sub.add_parser("sync", help="copy local memory to MongoDB Atlas (needs MONGODB_URI)")
    args = parser.parse_args(argv)
    if args.command == "run":
        unknown = set(args.engines) - set(ENGINES)
        if unknown:
            parser.error(f"unknown engines {sorted(unknown)}; choose from {list(ENGINES)}")
    handler = {"doctor": _doctor, "run": _run, "history": _history, "recipe": _recipe, "sync": _sync}[args.command]
    raise SystemExit(handler(args))


if __name__ == "__main__":
    main()
