"""The loop: detect → resolve → propose → run each trial → judge → remember → ship the best."""

from __future__ import annotations

import json
import shlex
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import agent, bench, hardware, models, research
from .engines import ENGINES, docker_run
from .runner import DockerRunner, EngineFailed, GpuSampler
from .store import Store, trial_key


@dataclass
class Options:
    model: str
    engines: list[str] = field(default_factory=lambda: ["vllm", "sglang", "llamacpp"])
    goal: str = "throughput"
    workload: str = "chat"
    context: int | None = None
    budget_minutes: float = 90
    max_trials: int = 5            # per engine
    llm: str | None = None
    research: bool = True
    gguf_repo: str | None = None
    extra_hf: tuple[str, ...] = ()
    dry_run: bool = False
    vram_gib: float | None = None  # plan for a different GPU size (with --dry-run)
    rounds: int = 3                # 1 = planned trials only; more = fix failures and refine
    share_logs: bool = True        # send redacted log tails of failed trials to the agent LLM
    out_dir: Path = Path("atlas-out")


def _context(opts: Options) -> int:
    w = bench.WORKLOADS[opts.workload]
    return opts.context or (w.input_tokens + w.output_tokens + 512)


def _summary_row(trial: dict, with_logs: bool = False) -> dict:
    """What the agent sees about a trial. Failed trials of this run carry a redacted log tail."""
    m = trial.get("metrics") or {}
    q = trial.get("quality") or {}
    row = {"engine": trial["engine"], "artifact": (trial.get("artifact") or {}).get("id"),
           "config": trial.get("config"), "status": trial["status"], "hypothesis": trial.get("hypothesis"),
           "quality": f"{q.get('passed')}/{q.get('total')}",
           "wrong_answers": [t.get("output") or t.get("error") for t in q.get("tasks", []) if not t.get("passed")][:3],
           "levels": [{k: lv.get(k) for k in ("concurrency", "ok", "requests", "output_tok_s", "ttft_p50_ms",
                                              "per_user_tok_s", "errors")} for lv in m.get("levels", [])],
           "gpu_peak_gib": trial.get("gpu_peak_gib"), "error": (trial.get("error") or "")[:300],
           "related_issues": trial.get("related") or []}
    if with_logs and trial["status"] != "ok":
        row["log_tail"] = research.redact(trial.get("logs"))
    return row


class Tuner:
    def __init__(self, opts: Options, store: Store | None = None, runner=None, log=print):
        self.opts, self.log = opts, log
        self.store = store or Store()
        self.runner = runner or DockerRunner(log=log)
        self.run_id = uuid.uuid4().hex[:12]
        self.deadline = time.monotonic() + opts.budget_minutes * 60
        self.info: dict[str, dict] = {}       # engine → {"version", "help"} of the pulled image
        self.allowed: dict[str, dict] = {}    # engine → knobs this image supports
        self.tried: set = set()

    # ---------------------------------------------------------------- engines
    def inspect_engines(self, engines: list[str]) -> None:
        """Read each image's own --help and version (cached by digest) and keep its knobs."""
        for name in engines:
            engine = ENGINES[name]
            if self.opts.dry_run and not self.runner.has_image(engine):
                self.allowed[name] = dict(engine.knobs)
                self.log(f"  {name}: {engine.image} not pulled yet; flags are checked against its --help at run time")
                continue
            image = self.runner.image_id(engine)
            info = self.store.engine_info(image)
            if not info:
                self.log(f"  reading {name} --help from {engine.image}")
                described = self.runner.describe(engine)
                self.store.put_engine_info(image, name, described["version"], described["help"])
                info = {"version": described["version"], "help": described["help"]}
            self.info[name] = info
            self.allowed[name] = engine.supported(info.get("help"))
            dropped = sorted(set(engine.knobs) - set(self.allowed[name]))
            self.log(f"  {name} {info.get('version')}: {len(self.allowed[name])} tunable flags"
                     + (f" (not in this version: {', '.join(dropped)})" if dropped else ""))

    def _fit(self, c: agent.Candidate) -> agent.Candidate | None:
        """Drop settings this image doesn't have; skip anything already tried."""
        allowed = self.allowed.get(c.engine, ENGINES[c.engine].knobs)
        c.config = {k: v for k, v in c.config.items() if k in allowed}
        key = (c.engine, c.artifact_id, json.dumps(c.config, sort_keys=True))
        if key in self.tried:
            return None
        self.tried.add(key)
        return c

    def _cap(self, candidates: list[agent.Candidate]) -> list[agent.Candidate]:
        order = {"fix": 0, "baseline": 0, "llm": 1, "default": 2}
        candidates.sort(key=lambda c: order.get(c.origin, 3))
        kept, per_engine = [], {}
        for c in candidates:
            c = self._fit(c)
            if c is None:
                continue
            per_engine[c.engine] = per_engine.get(c.engine, 0) + 1
            if per_engine[c.engine] <= self.opts.max_trials:
                kept.append(c)
            else:
                self.tried.discard((c.engine, c.artifact_id, json.dumps(c.config, sort_keys=True)))
        return kept

    def _agent(self, hw, artifacts, engines, previous, round_no) -> list[agent.Candidate]:
        o = self.opts
        if not (o.llm and engines):
            return []
        self.log(f"Agent ({o.llm}) round {round_no}: researching…")
        try:
            return agent.propose(o.llm, model=o.model, engines=engines, artifacts=artifacts, hw=hw, goal=o.goal,
                                 workload=bench.WORKLOADS[o.workload].__dict__, previous=previous,
                                 max_new=o.max_trials * len(engines), use_research=o.research,
                                 allowed=self.allowed, info=self.info, round_no=round_no, log=self.log)
        except Exception as error:  # noqa: BLE001 - a broken LLM must not stop the benchmark
            self.log(f"  agent unavailable ({type(error).__name__}: {str(error)[:160]})")
            return []

    # ------------------------------------------------------------------ steps
    def plan(self) -> tuple[dict, list[models.Artifact], list[str], list[agent.Candidate]]:
        o = self.opts
        hw = hardware.detect()
        if o.vram_gib:
            hw["gpu"] = {**(hw.get("gpu") or {"name": "planned GPU"}), "vram_gib": o.vram_gib}
        gpu = hw.get("gpu") or {}
        self.log(f"GPU: {gpu.get('name', 'none detected')} {gpu.get('vram_gib', '?')} GiB · "
                 f"RAM {hw['ram_available_gib']}/{hw['ram_total_gib']} GiB free · disk {hw['disk_free_gib']} GiB free")
        all_artifacts = models.resolve(o.model, o.gguf_repo, o.extra_hf)
        artifacts = models.shortlist(all_artifacts, gpu.get("vram_gib") or 0, hw["ram_available_gib"])
        for a in artifacts:
            self.log(f"  artifact {a.id}  {a.size_gib} GiB")
        engines = [e for e in o.engines if any(a.engine_format == ENGINES[e].format for a in artifacts)]
        for skipped in sorted(set(o.engines) - set(engines)):
            self.log(f"  skip {skipped}: no {ENGINES[skipped].format} artifact fits this machine "
                     f"(pass --hf-repo with an AWQ/FP8 variant, or a smaller model)")
        self.inspect_engines(engines)
        workload = bench.WORKLOADS[o.workload]
        candidates = agent.defaults(engines, artifacts, hw, max(workload.concurrency), o.goal)
        previous = [_summary_row(t) for t in self.store.trials(o.model, hardware.fingerprint(hw))]
        candidates += self._agent(hw, artifacts, engines, previous, 1)
        return hw, artifacts, engines, self._cap(candidates)

    def next_round(self, hw, artifacts, engines, trials, round_no) -> list[agent.Candidate]:
        """Fix what failed (rules, then the agent with logs and issues) and push on what worked."""
        by_id = {a.id: a for a in artifacts}
        fixes = [f for t in trials if not t.get("reused")
                 for f in agent.rule_fixes(t, by_id.get((t.get("artifact") or {}).get("id")))]
        current = [_summary_row(t, with_logs=self.opts.share_logs) for t in trials]
        memory = [_summary_row(t) for t in self.store.trials(self.opts.model, hardware.fingerprint(hw))
                  if t.get("run_id") != self.run_id]
        return self._cap(fixes + self._agent(hw, artifacts, engines, memory[-20:] + current, round_no))

    def trial(self, hw: dict, artifacts: list[models.Artifact], c: agent.Candidate) -> dict:
        o, engine = self.opts, ENGINES[c.engine]
        art = next(a for a in artifacts if a.id == c.artifact_id)
        context, machine = _context(o), hardware.fingerprint(hw)
        image = self.runner.image_id(engine)
        key = trial_key(machine=machine, engine=c.engine, image=image, artifact=art.id, config=c.config,
                        context=context, workload=f"{o.workload}-{bench.WORKLOAD_VERSION}")
        if (seen := self.store.trial(key)) and seen["status"] in ("ok", "quality_failed", "failed"):
            self.log(f"  memory: already measured on this machine → reuse ({seen['status']})")
            return {**seen, "reused": True}
        record = {"key": key, "run_id": self.run_id, "model": o.model, "machine": machine, "engine": c.engine,
                  "image": image, "artifact": art.to_dict(), "config": c.config, "context": context,
                  "workload": o.workload, "hypothesis": f"[{c.origin}] {c.hypothesis}"[:800], "sources": c.sources,
                  "launch": docker_run(engine, art, c.config, context, host_port=8000,
                                       name=f"{c.engine}-serve", production=True)}
        server = None
        try:
            with GpuSampler() as gpu:
                server = self.runner.start(engine, art, c.config, context)
                self.runner.wait_ready(server)
                q = bench.quality(server.base_url, server.model)
                self.log(f"  quality {q['passed']}/{q['total']}")
                metrics = bench.load(server.base_url, server.model, bench.WORKLOADS[o.workload]) if q["pass"] else None
            record.update(status="ok" if q["pass"] else "quality_failed", quality=q, metrics=metrics,
                          gpu_peak_gib=gpu.peak_gib)
        except EngineFailed as error:
            record.update(status="failed", error=str(error), logs=error.logs)
        except Exception as error:  # noqa: BLE001 - one bad trial must not stop the run
            record.update(status="failed", error=f"{type(error).__name__}: {error}"[:1000],
                          logs=self.runner.logs(server) if server else None)
        finally:
            if server:
                self.runner.stop(server)
        if record["status"] == "failed":
            record["related"] = research.related_issues(c.engine, record.get("logs"), record.get("error", ""))
            for issue in record["related"]:
                self.log(f"  related issue: {issue['title'][:90]}  {issue['url']}")
        self.store.put_trial(record)
        return record

    # ------------------------------------------------------------------- run
    def run(self) -> dict:
        o = self.opts
        hw, artifacts, engines, candidates = self.plan()
        if o.dry_run:
            for c in candidates:
                art = next(a for a in artifacts if a.id == c.artifact_id)
                argv = docker_run(ENGINES[c.engine], art, c.config, _context(o), host_port=8000,
                                  name=f"{c.engine}-trial")
                self.log(f"\n# {c.engine} · {c.hypothesis}\n{shlex.join(argv)}")
            return {"run_id": None, "dry_run": True, "candidates": [c.to_dict() for c in candidates]}
        self.store.start_run(self.run_id, o.model, o.goal, o.workload, hw)
        removed = self.runner.cleanup()
        if removed:
            self.log(f"  removed {removed} leftover atlas-tune container(s)")
        trials: list[dict] = []
        for round_no in range(1, o.rounds + 1):
            if not candidates:
                break
            self.log(f"\nRound {round_no}: {len(candidates)} trial(s), "
                     f"{max(0, (self.deadline - time.monotonic()) / 60):.0f} min left")
            for c in candidates:
                if time.monotonic() > self.deadline:
                    break
                self.log(f"\n[{c.engine}:{c.origin}] {c.artifact_id} {json.dumps(c.config)}  — {c.hypothesis[:200]}")
                t = self.trial(hw, artifacts, c)
                s = bench.score(t.get("metrics"), o.goal)
                self.log(f"  → {t['status']}" + (f", score {s}" if s is not None else "")
                         + (f": {t['error'][:160]}" if t.get("error") else ""))
                trials.append(t)
            if time.monotonic() > self.deadline:
                self.log("Time budget reached; stopping with what was measured.")
                break
            if round_no < o.rounds:
                candidates = self.next_round(hw, artifacts, engines, trials, round_no + 1)
        return self.finish(hw, trials)

    def finish(self, hw: dict, trials: list[dict]) -> dict:
        o = self.opts
        best: dict[str, dict] = {}
        for t in trials:
            s = bench.score(t.get("metrics"), o.goal)
            if t["status"] == "ok" and s is not None and s > best.get(t["engine"], {}).get("score", -1):
                best[t["engine"]] = {**t, "score": s}
        for engine, t in best.items():
            self.store.put_recipe({"model": o.model, "machine": t["machine"], "engine": engine, "goal": o.goal,
                                   "workload": o.workload, "trial_key": t["key"], "score": t["score"],
                                   "launch": t["launch"]})
        winner = max(best.values(), key=lambda t: t["score"], default=None)
        summary = {"run_id": self.run_id, "model": o.model, "goal": o.goal, "workload": o.workload,
                   "trials": len(trials), "best": {e: {"score": t["score"], "config": t["config"],
                                                       "artifact": t["artifact"]["id"]} for e, t in best.items()},
                   "winner": winner and {"engine": winner["engine"], "score": winner["score"],
                                         "launch": shlex.join(winner["launch"])}}
        self.store.finish_run(self.run_id, "completed", summary)
        out = o.out_dir / self.run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(report(o, hw, trials, winner), encoding="utf-8")
        if winner:
            (out / "serve.sh").write_text("#!/usr/bin/env bash\n# " + f"{o.model} on {winner['engine']}, "
                                          f"verified by atlas-tune run {self.run_id}\n" + shlex.join(winner["launch"]) + "\n")
            (out / "serve.sh").chmod(0o755)
        summary["report"] = str(out / "report.md")
        return summary


def report(o: Options, hw: dict, trials: list[dict], winner: dict | None) -> str:
    unit = "per-user tok/s at lowest concurrency" if o.goal == "latency" else "total output tok/s"
    gpu = hw.get("gpu") or {}
    lines = [f"# {o.model}", "", f"GPU {gpu.get('name')} ({gpu.get('vram_gib')} GiB) · goal **{o.goal}** "
             f"({unit}) · workload **{o.workload}**", "",
             "| engine | artifact | config | status | quality | score | TTFT p50 | GPU peak | why tried |",
             "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |"]
    for t in sorted(trials, key=lambda t: -(bench.score(t.get("metrics"), o.goal) or -1)):
        levels = (t.get("metrics") or {}).get("levels") or [{}]
        q = t.get("quality") or {}
        lines.append(f"| {t['engine']} | {(t.get('artifact') or {}).get('id')} | `{json.dumps(t['config'])}` | "
                     f"{t['status']}{' (memory)' if t.get('reused') else ''} | {q.get('passed', '–')}/{q.get('total', '–')} | "
                     f"{bench.score(t.get('metrics'), o.goal) or '–'} | {levels[0].get('ttft_p50_ms', '–')} ms | "
                     f"{t.get('gpu_peak_gib') or '–'} GiB | {(t.get('hypothesis') or '').replace('|', '/')[:160]} |")
    failures = [t for t in trials if t.get("error")]
    if failures:
        lines += ["", "## Failures", ""]
        for t in failures:
            lines.append(f"- **{t['engine']}** `{json.dumps(t['config'])}`: {t['error'][:300]}")
            lines += [f"  - related: [{i['title'][:100]}]({i['url']}) ({i['state']})" for i in t.get("related") or []]
    lines += ["", "## Launch the winner", ""]
    lines += (["```bash", shlex.join(winner["launch"]), "```"] if winner else ["No configuration passed every check."])
    sources = sorted({s for t in trials for s in t.get("sources") or []})
    if sources:
        lines += ["", "## Sources the agent used", ""] + [f"- {s}" for s in sources]
    lines += ["", "Scores are from one machine and a synthetic workload; GPU peak is total device memory sampled "
              "every 0.5 s and includes other processes. Quality is an 8-question exact-answer smoke test."]
    return "\n".join(lines) + "\n"
