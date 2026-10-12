"""Real Strata policy trials, long tokenized inputs, and a resumable Atlas ledger.

Run with Strata's Python environment, on the OS that owns the Strata install.
The workload is synthetic; inference, engine counters, and timings are real.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

from filelock import FileLock

from .store import AtlasStore, now

VERSION = "strata-32k-policy-v1"
POLICIES = {
    "default": {},
    "cpu_4": {"--pool-workers": "4"},
    "cpu_7": {"--pool-workers": "7"},
    "adaptive_160": {"--adapt-every": "1", "--adapt-swaps": "160", "--adapt-decay": "0.97"},
    "no_pcie": {"--pcie-frac": "0"},
}
HYPOTHESES = {
    "default": "Measure Strata's shipped hardware policy with prompt reuse disabled.",
    "cpu_4": "Four expert workers may avoid contention and slow efficiency cores on this hybrid CPU.",
    "cpu_7": "Seven expert workers target the performance-core count minus the engine's host thread.",
    "adaptive_160": "More frequent expert promotion may increase VRAM hits enough to repay swap overhead.",
    "no_pcie": "Ablate PCIe compute of missed experts to measure the contribution of CPU/GPU work sharing.",
}
SOURCES = [
    "https://github.com/Niko1221/Strata/blob/v0.1.43/tools/calibrate.py",
    "https://github.com/Niko1221/Strata/blob/v0.1.43/docs/DETAILS.md",
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def with_args(args, settings):
    out = list(args)
    for flag, value in settings.items():
        while flag in out:
            i = out.index(flag)
            del out[i:i + 2]
        if value is not None:
            out.extend([flag, str(value)])
    return out


def workload(tok, target, seed):
    """Exactly target input tokens; eight settings plus two later amendments.

    Each policy sees identical token IDs. Expected answers are generated before
    inference, and held-out seeds/lengths never enter policy selection.
    """
    if target < 32768:
        raise ValueError("This suite requires at least 32768 actual input tokens")
    rng = random.Random(seed)
    keys = [f"service_{rng.randrange(0x100000, 0xffffff):06x}" for _ in range(8)]
    values = [rng.randrange(10, 90) for _ in keys]
    final = dict(zip(keys, values))
    prefix = tok.encode(
        "<|im_start|>user\nReview this generated deployment repository and its change log. "
        "DEPLOY_SETTING entries are authoritative; a later entry for the same service supersedes an earlier one. "
        "Other code is background material. Read the entire document.\nDOCUMENT START\n",
        parse_special=True)
    suffix = tok.encode(
        "\nDOCUMENT END\nReturn one JSON object, no markdown, with keys settings, total, review. "
        "settings must map each of these services to its LAST DEPLOY_SETTING integer: " + ", ".join(keys) +
        ". total must be the sum of those eight final integers. review must be a 200-word explanation "
        "of how you resolved the amendments and how an operator should validate this deployment. "
        "Use integers, not strings, for the settings and total.\n<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
    entries = [f"\nDEPLOY_SETTING {key} = {value}\n" for key, value in zip(keys, values)]
    for i in (1, 5):
        final[keys[i]] = values[i] + 7
        entries.append(f"\nAMENDMENT: DEPLOY_SETTING {keys[i]} = {final[keys[i]]}\n")
    encoded = [tok.encode(s) for s in entries]
    filler_count = target - len(prefix) - len(suffix) - sum(map(len, encoded))
    if filler_count <= 0:
        raise ValueError("Prompt overhead exceeds requested length")
    # Varied code avoids a document made from a single repeated sentence.
    lines = [f"# module {i}: deployment telemetry\ndef probe_{i}(samples):\n"
             f"    return sum(samples[-{rng.randrange(8, 96)}:]) / max(1, len(samples))\n"
             for i in range(target // 16)]
    filler = tok.encode("\n".join(lines))
    if len(filler) < filler_count:
        raise ValueError("Insufficient filler tokens")
    ids, positions, cursor = list(prefix), [], 0
    for i, entry in enumerate(encoded):
        end = round(filler_count * (i + 1) / (len(encoded) + 1))
        ids.extend(filler[cursor:end])
        positions.append(len(ids))
        ids.extend(entry)
        cursor = end
    ids.extend(filler[cursor:filler_count])
    ids.extend(suffix)
    assert len(ids) == target
    return {"input_ids": ids, "prompt_tokens": target, "seed": seed,
            "expected": {"settings": final, "total": sum(final.values())},
            "entry_token_positions": positions, "generator": VERSION,
            "description": "Synthetic repository review: 8 dispersed settings, 2 later overrides, aggregation"}


def score(text, expected):
    try:
        value = json.loads(text.strip())
    except (ValueError, TypeError):
        return {"passed": False, "reason": "Output is not a complete JSON object", "matched_settings": 0}
    if not isinstance(value, dict):
        return {"passed": False, "reason": "Output is not an object", "matched_settings": 0}
    settings = value.get("settings")
    matched = sum(type(settings.get(k)) is int and settings[k] == v for k, v in expected["settings"].items()) \
        if isinstance(settings, dict) else 0
    passed = (isinstance(settings, dict) and set(settings) == set(expected["settings"]) and matched == 8
              and type(value.get("total")) is int and value["total"] == expected["total"]
              and isinstance(value.get("review"), str) and bool(value["review"].strip()))
    return {"passed": passed, "matched_settings": matched, "required_settings": 8,
            "total_correct": type(value.get("total")) is int and value["total"] == expected["total"],
            "scope": "Exact lookup, override and sum checks; prose quality is not graded"}


def answer_text(tok, output):
    # The native GEN protocol emits EOS before DONE; the HTTP layer normally
    # removes it. Strip only terminal control tokens and preserve the raw stream.
    eos = {tok.encode(s, parse_special=True)[0] for s in ("<|im_end|>", "<|endoftext|>")}
    content = list(output)
    while content and content[-1] in eos:
        content.pop()
    return tok.decode(content)


def select_policy(experiments, repeats):
    """Only complete, passing tuning arms are eligible. Never consult holdout."""
    eligible = {}
    for policy in POLICIES:
        rows = [e for e in experiments if e["result"].get("stage") == "tuning"
                and e["result"].get("policy") == policy]
        if len(rows) == repeats and all(e["result"].get("correct") for e in rows):
            eligible[policy] = statistics.median(e["result"]["decode_tps"] for e in rows)
    if "default" not in eligible:
        return "default", {"eligible_medians": eligible, "reason": "Reference failed a quality or execution gate"}
    best = max(eligible, key=eligible.get)
    winner = best if eligible[best] > eligible["default"] * 1.03 else "default"
    return winner, {"eligible_medians": eligible, "reason": "Require every repeat to pass and >3% median decode gain"}


def paired_summary(rows, candidate, length, repeats):
    paired = []
    for block in range(repeats):
        arms = {r["policy"]: r for r in rows if r.get("stage") == "holdout"
                and r.get("prompt_tokens") == length and r.get("block") == block}
        if "default" not in arms or candidate not in arms or candidate == "default":
            continue
        b, c = arms["default"], arms[candidate]
        if b.get("correct") and c.get("correct") and b.get("decode_tps", 0) > 0:
            paired.append(100 * (c["decode_tps"] / b["decode_tps"] - 1))
    return {"prompt_tokens": length, "candidate": candidate, "paired_decode_gain_pct": paired,
            "median_gain_pct": statistics.median(paired) if paired else None,
            "min_gain_pct": min(paired) if paired else None, "max_gain_pct": max(paired) if paired else None,
            "supported": len(paired) == repeats and min(paired) > 3,
            "rule": "All matched holdout pairs must pass and each decode gain must exceed 3%; descriptive, not statistical significance"}


class Resources:
    """Native process RSS/I/O and total GPU usage, sampled once per second."""
    def __init__(self, pid):
        self.pid, self.samples, self.errors = pid, [], []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        import psutil
        process = psutil.Process(self.pid)
        start = time.monotonic()
        while not self.stop_event.is_set():
            try:
                io = process.io_counters()
                mem = psutil.virtual_memory()
                sample = {"seconds": time.monotonic() - start, "rss_mib": process.memory_info().rss / 2**20,
                          "host_available_mib": mem.available / 2**20,
                          "process_read_bytes": io.read_bytes, "process_write_bytes": io.write_bytes}
                p = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                if p.returncode == 0:
                    gpu, util = p.stdout.strip().splitlines()[0].split(",")
                    sample.update(gpu_used_mib=float(gpu), gpu_util_pct=float(util))
                self.samples.append(sample)
            except Exception as exc:
                self.errors.append(type(exc).__name__)
            self.stop_event.wait(1)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
        return {"samples": self.samples, "errors": self.errors,
                "gpu_peak_mib": max((s.get("gpu_used_mib", 0) for s in self.samples), default=None),
                "rss_peak_mib": max((s["rss_mib"] for s in self.samples), default=None),
                "scope": "1-second samples during requests; GPU includes other processes; process I/O is not physical SSD traffic"}


def evaluate(cfg, engine_class, engine_args, child_env, tok, prompt, policy, log_path):
    args = with_args(engine_args(cfg), {**POLICIES[policy], "--max-context": prompt["prompt_tokens"] + 1024,
                                      "--prompt-cache": 0, "--conversation-cache-mib": 0})
    result = {"policy": policy, "settings": POLICIES[policy], "command": [cfg["exe"], "--serve", *args],
              "prompt_tokens": prompt["prompt_tokens"], "seed": prompt["seed"], "correct": False,
              "engine_success": False, "log_path": str(log_path)}
    engine, sampler = None, None
    try:
        start = time.monotonic()
        engine = engine_class(cfg["exe"], args, cwd=cfg.get("cwd"), log=str(log_path), env=child_env(cfg))
        result.update(load_seconds=time.monotonic() - start, engine_info=dict(engine.info))
        # Same short warmup for every fresh engine; no measured prompt has been seen.
        warm = tok.encode("<|im_start|>user\nSay ready.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", parse_special=True)
        list(engine.generate(warm, 16, {"temperature": 0}, threading.Event()))
        sampler = Resources(engine.proc.pid)
        sampler.start()
        output, first = [], None
        start = time.monotonic()
        for token in engine.generate(prompt["input_ids"], 768, {"temperature": 0, "seed": prompt["seed"]}, threading.Event()):
            if token is not None:
                if first is None:
                    first = time.monotonic() - start
                output.append(token)
        result.update(request_seconds=time.monotonic() - start, first_token_seconds=first,
                      raw_output=tok.decode(output), answer_text=answer_text(tok, output),
                      output_ids=output, telemetry=dict(engine.last),
                      engine_success=True)
        t = result["telemetry"]
        result["quality"] = score(result["answer_text"], prompt["expected"])
        result["decode_tps"] = t["generated"] / (t["decode_ms"] / 1000) if t.get("decode_ms", 0) > 0 else None
        result["prefill_tps"] = t["prompt_read"] / (t["prompt_ms"] / 1000) if t.get("prompt_ms", 0) > 0 and "prompt_read" in t else None
        result["expert_hit_rate"] = t["hits"] / t["lookups"] if t.get("lookups") else None
        result["input_verified"] = t.get("prompt_tokens") == len(prompt["input_ids"]) and t.get("reused", 0) == 0
        result["correct"] = bool(result["quality"]["passed"] and result["input_verified"]
                                 and result["decode_tps"] and math.isfinite(result["decode_tps"]))
    except Exception as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)[:1500]}
    finally:
        if sampler:
            result["resources"] = sampler.stop()
        if engine:
            try:
                engine.close()
            except Exception as exc:
                result.update(correct=False, cleanup_error={"type": type(exc).__name__, "message": str(exc)[:1000]})
    return result


def export(store, run_id, path):
    run = store.run(run_id)
    experiments = store.experiments_for(run_id)
    decisions = store.decisions_for(run_id)
    ids = {e["_id"] for e in experiments}
    audit = {"experiments": len(experiments), "decisions": len(decisions),
             "dangling_references": sum(ref not in ids for d in decisions for ref in d["retrieved_experiment_ids"]),
             "attempts": [e["attempt"] for e in experiments], "read_back_at": str(now())}
    save(path, {"run": run, "experiments": experiments, "decisions": decisions, "atlas_audit": audit})


class CheckpointStop(Exception):
    """An intentional stop after a durable commit, for recovery demonstrations."""


def execute(store, run, cfg, tok, api, directory, export_path, stop_after=None):
    run_id, repeats = run["_id"], run["repeats"]
    tuning_ids = []
    committed = 0

    def trial(policy, length, block, stage):
        nonlocal committed
        identity = f"{stage}-{length}-{block}-{policy}"
        old = store.experiments.find_one({"_id": f"{run_id}:{identity}"})
        if old and old["status"] in ("completed", "failed"):
            store.runs.update_one({"_id": run_id}, {"$addToSet": {"reused_experiment_ids": old["_id"]}})
            print("reuse committed", identity, flush=True)
            return old
        seed = (20261011 if stage == "tuning" else 20262011) + block + length
        prompt = workload(tok, length, seed)
        artifact = directory / f"prompt-{stage}-{length}-{block}.json.gz"
        if not artifact.exists():
            artifact.write_bytes(gzip.compress(json.dumps(prompt).encode(), mtime=0))
        reference = {"artifact": artifact.as_posix(), "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                     "expected": prompt["expected"], "entry_token_positions": prompt["entry_token_positions"]}
        evidence = [e["_id"] for e in store.experiments_for(run_id) if e["status"] in ("completed", "failed")]
        store.decide(run_id, evidence, identity, HYPOTHESES[policy],
                     "Fixed tuning grid" if stage == "tuning" else "Frozen tuning choice on unseen seed at the same 32K input length",
                     policy_version=VERSION)
        owner = store.start(run_id, identity, run["context_hash"], policy)
        if owner is None:
            return store.experiments.find_one({"_id": f"{run_id}:{identity}"})
        print("evaluate", identity, flush=True)
        result = evaluate(cfg, *api, tok, prompt, policy, directory / f"{identity}.log")
        result.update(stage=stage, block=block, prompt=reference)
        store.finish(run_id, identity, owner, result)
        export(store, run_id, export_path)
        print(json.dumps({k: result.get(k) for k in ("policy", "prompt_tokens", "decode_tps", "prefill_tps", "correct", "quality", "error")}), flush=True)
        committed += 1
        if stop_after and committed >= stop_after:
            raise CheckpointStop("Stopped after committed trial; resume with the printed run ID")
        return store.experiments.find_one({"_id": f"{run_id}:{identity}"})

    for block in range(repeats):
        arms = list(POLICIES)
        arms = arms[block % len(arms):] + arms[:block % len(arms)]
        for policy in arms:
            tuning_ids.append(trial(policy, 32768, block, "tuning")["_id"])
    # Persist the selection before any holdout. A resume cannot retune on holdout.
    latest = store.run(run_id)
    if "selection" not in latest:
        winner, why = select_policy(store.experiments_for(run_id), repeats)
        selection = {"policy": winner, "evidence_ids": tuning_ids, **why}
        store.runs.update_one({"_id": run_id}, {"$set": {"selection": selection}})
    else:
        selection = latest["selection"]
        winner = selection["policy"]
    for length in (32768,):
        for block in range(repeats):
            arms = ["default"] if winner == "default" else ["default", winner]
            for policy in (arms if block % 2 == 0 else list(reversed(arms))):
                trial(policy, length, block, "holdout")
    rows = [e["result"] for e in store.experiments_for(run_id)]
    confirmation = [paired_summary(rows, winner, length, repeats) for length in (32768,)]
    accepted = winner != "default" and all(c["supported"] for c in confirmation)
    reference = [r for r in rows if r.get("stage") == "holdout" and r.get("policy") == "default"]
    baseline_valid = len(reference) == repeats and all(r.get("correct") for r in reference)
    recommendation = winner if accepted else "default" if baseline_valid else None
    store.runs.update_one({"_id": run_id}, {"$set": {
        "status": "completed", "finished_at": now(), "confirmation": confirmation,
        "recommendation": recommendation, "improvement_supported": accepted,
        "conclusion": ("Matched 32K confirmation passed" if accepted else "No confirmed improvement over Strata default"
                       if baseline_valid else "No policy passed fresh-prompt confirmation")}})
    export(store, run_id, export_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strata-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int, help="Exit after this many new committed trials; resume reuses them")
    parser.add_argument("--output", type=Path, default=Path("evidence/strata"))
    args = parser.parse_args()
    if args.repeats < 3 or (args.stop_after is not None and args.stop_after < 1):
        parser.error("At least three repeats are required")
    root = args.strata_root.resolve()
    sys.path[:0] = [str(root), str(root / "tools")]
    from serve.server import StrataEngine, engine_args, child_env
    from strata_tokenizer import Tokenizer
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig"))
    # Preserve setup's recommended hardware defaults (including hybrid-CPU
    # worker limits). Each candidate changes only its declared controls.
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for text, i in vocab.items():
        tokens[i] = text
    tok = Tokenizer(tokens, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                    json.loads((tpath / "token_type.json").read_text(encoding="utf-8")))
    fixed = {"contract": VERSION, "engine_sha256": hashlib.sha256(Path(cfg["exe"]).read_bytes()).hexdigest(),
             "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "base_args": engine_args(cfg), "policies": POLICIES, "repeats": args.repeats,
             "prompt_tokens": [32768], "max_new": 768, "sources": SOURCES}
    fixed["strata_git_head"] = (root / ".git/HEAD").read_text().strip()
    fixed["engine_build"] = json.loads((Path(cfg["exe"]).parent / "BUILD.json").read_text(encoding="utf-8"))
    fixed["model_files"] = [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                            for flag in ("--native", "--ple-gguf") if flag in cfg["args"]
                            for p in [Path(cfg["args"][cfg["args"].index(flag) + 1])]]
    store = AtlasStore()
    store.init()
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock("artifacts/gpu.lock", timeout=0):
        if args.resume:
            run = store.run(args.resume)
            if run["context_hash"] != digest(fixed):
                raise ValueError("Resume rejected: engine, config, source or protocol changed")
        else:
            run_id = store.create_run(
                fixed, digest(fixed), args.repeats * (len(POLICIES) + 2),
                objective="Strata hardware policy calibration on long-context repository review",
                policy_version=VERSION, repeats=args.repeats,
                evidence_kind="real_native_Strata_inference_on_synthetic_workloads")
            run = store.run(run_id)
        print("ATLAS_RUN_ID=" + run["_id"], flush=True)
        directory = args.output / run["_id"]
        directory.mkdir(parents=True, exist_ok=True)
        export(store, run["_id"], args.output / "latest.json")
        if run["status"] != "completed":
            try:
                execute(store, run, cfg, tok, (StrataEngine, engine_args, child_env), directory,
                        args.output / "latest.json", args.stop_after)
            except CheckpointStop as exc:
                print(str(exc), flush=True)
                return
        print("Complete:", run["_id"], flush=True)


if __name__ == "__main__":
    main()
