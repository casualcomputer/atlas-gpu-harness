"""End-to-end atlas-tune loop against a fake OpenAI-compatible engine (no Docker, GPU or network)."""

import json
import re
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from atlas_tune import agent, bench, engines, models, tuner
from atlas_tune.runner import EngineFailed, Server
from atlas_tune.store import Store

ANSWERS = {prompt: expected for _, prompt, expected in bench.QUALITY_TASKS}


def fake_server(delay: float, wrong: bool = False):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._json({"data": [{"id": "fake-model"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            prompt = body["messages"][-1]["content"]
            if not body.get("stream"):
                answer = "nope" if wrong else ANSWERS.get(prompt, 0)
                return self._json({"choices": [{"message": {"content": json.dumps({"answer": answer})}}]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            n = min(body["max_tokens"], 20)
            for i in range(n):
                time.sleep(delay)
                chunk = {"choices": [{"delta": {"content": f"t{i} "}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': n}})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


FAKE_VLLM_HELP = """usage: vllm serve [model_tag] [options]
  --gpu-memory-utilization GPU_MEMORY_UTILIZATION
                        The fraction of GPU memory to be used for the model executor.
  --max-num-seqs MAX_NUM_SEQS
                        Maximum number of sequences per iteration.
  --kv-cache-dtype {auto,fp8,fp8_e4m3,fp8_e5m2}
                        Data type for kv cache storage.
  --enforce-eager       Always use eager-mode PyTorch.
"""


class FakeRunner:
    """Speed depends on config so the tuner has something to find; one config 'OOMs'.

    Its vLLM image is an older build without --max-num-batched-tokens, --async-scheduling, etc.
    """

    def __init__(self):
        self.started, self.servers, self.described = [], {}, 0

    def image_id(self, engine, pull=True):
        return f"{engine.image}@sha256:test"

    def has_image(self, engine):
        return True

    def docker_free_gib(self):
        return 500.0

    def describe(self, engine):
        self.described += 1
        return {"version": "0.0.test", "help": FAKE_VLLM_HELP}

    def cleanup(self):
        return 0

    def start(self, engine, artifact, config, context):
        self.started.append((engine.name, dict(config)))
        if config.get("max_num_seqs") == 999:
            raise EngineFailed("container exited during startup",
                               logs="loading\nHF_TOKEN=hf_secretsecretsecret\ntorch.OutOfMemoryError: CUDA out of memory.")
        delay = 0.001 if config.get("kv_cache_dtype") == "fp8" else 0.004
        httpd = fake_server(delay, wrong=config.get("enforce_eager", False))
        server = Server(name=f"s{len(self.started)}", base_url=f"http://127.0.0.1:{httpd.server_port}")
        self.servers[server.name] = httpd
        return server

    def wait_ready(self, server):
        server.model = "fake-model"
        return server

    def logs(self, server):
        return ""

    def stop(self, server):
        httpd = self.servers.pop(server.name)
        httpd.shutdown()
        httpd.server_close()


HW = {"gpu": {"name": "Test GPU", "vram_gib": 16.0, "used_gib": 0.0, "driver": "1"}, "gpus": [{}],
      "ram_total_gib": 64, "ram_available_gib": 60, "disk_free_gib": 500, "cpu_count": 8}
ARTIFACT = models.Artifact(id="hf:org/m", engine_format="hf", repo="org/m", size_gib=8)
FAST = bench.Workload("chat", 32, 16, (1, 2))


class TunerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(Path(self.tmp.name) / "atlas.db", mirror=False)
        patches = [mock.patch("atlas_tune.hardware.detect", return_value=dict(HW)),
                   mock.patch("atlas_tune.hardware.gpus", return_value=[]),
                   mock.patch("atlas_tune.models.resolve", return_value=[ARTIFACT]),
                   mock.patch.dict(bench.WORKLOADS, {"chat": FAST})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _run(self, runner, **kw):
        opts = tuner.Options(model="org/m", engines=["vllm"], out_dir=Path(self.tmp.name) / "out", **kw)
        return tuner.Tuner(opts, store=self.store, runner=runner, log=lambda *_: None).run()

    def test_full_loop_picks_fastest_verified_config_and_writes_recipe(self):
        summary = self._run(FakeRunner())
        self.assertEqual(summary["winner"]["engine"], "vllm")
        self.assertEqual(summary["best"]["vllm"]["config"]["kv_cache_dtype"], "fp8")
        self.assertIn("--kv-cache-dtype fp8", summary["winner"]["launch"])
        self.assertIn("--restart unless-stopped", summary["winner"]["launch"])
        recipe = self.store.recipes("org/m")[0]
        self.assertEqual(recipe["engine"], "vllm")
        report = Path(summary["report"]).read_text()
        self.assertIn("Deploy the winner", report)
        self.assertIn("(runs)", report)
        out = Path(summary["report"]).parent
        serve = (out / "serve.sh").read_text()
        self.assertIn('--api-key "$API_KEY"', serve)
        self.assertIn("@sha256:test", serve)                     # pinned to the measured image digest
        import yaml
        compose = yaml.safe_load((out / "compose.yaml").read_text())
        self.assertIn("--kv-cache-dtype", compose["services"]["llm"]["command"])
        docs = list(yaml.safe_load_all((out / "k8s.yaml").read_text()))
        container = docs[0]["spec"]["template"]["spec"]["containers"][0]
        self.assertEqual(container["args"][-2:], ["--api-key", "$(API_KEY)"])
        self.assertEqual(container["startupProbe"]["httpGet"]["path"], "/health")
        self.assertEqual(docs[1]["kind"], "Service")

    def test_second_run_reuses_memory_instead_of_relaunching(self):
        self._run(FakeRunner(), rounds=1, confirm=0)
        runner = FakeRunner()
        summary = self._run(runner, rounds=1, confirm=0)
        self.assertEqual(runner.started, [])
        self.assertEqual(summary["winner"]["engine"], "vllm")

    def test_failures_and_wrong_answers_are_recorded_never_selected(self):
        bad = [agent.Candidate("vllm", ARTIFACT.id, {"max_num_seqs": 999}, "oom", origin="llm"),
               agent.Candidate("vllm", ARTIFACT.id, {"enforce_eager": True, "kv_cache_dtype": "fp8"}, "wrong", origin="llm")]
        issue = {"title": "[Bug]: OOM at startup", "url": "https://github.com/vllm-project/vllm/issues/1", "state": "open"}
        with mock.patch("atlas_tune.agent.defaults", return_value=bad), \
                mock.patch("atlas_tune.research.search_github_issues", return_value=[issue]) as search:
            summary = self._run(FakeRunner(), rounds=1)
        self.assertIn("OutOfMemoryError", search.call_args.args[1])
        statuses = {t["status"] for t in self.store.trials("org/m")}
        self.assertEqual(statuses, {"failed", "quality_failed"})
        self.assertIsNone(summary["winner"])
        failed = next(t for t in self.store.trials("org/m") if t["status"] == "failed")
        self.assertIn("exited during startup", failed["error"])
        self.assertIn("OutOfMemoryError", failed["logs"])
        self.assertEqual(failed["related"], [issue])
        self.assertIn("issues/1", Path(summary["report"]).read_text())

    def test_image_help_decides_which_flags_are_used(self):
        runner = FakeRunner()
        self._run(runner, rounds=1)
        used = {k for _, cfg in runner.started for k in cfg}
        self.assertNotIn("max_num_batched_tokens", used)   # not in this image's --help
        self.assertNotIn("async_scheduling", used)
        self.assertIn("kv_cache_dtype", used)
        self.assertEqual(runner.described, 1)
        self._run(FakeRunner(), rounds=1)                  # second run: help comes from the cache
        self.assertEqual(self.store.engine_info("vllm/vllm-openai:latest@sha256:test")["version"], "0.0.test")

    def test_oom_is_fixed_automatically_in_the_next_round(self):
        start = [agent.Candidate("vllm", ARTIFACT.id, {"max_num_seqs": 999}, "too many sequences", origin="baseline")]
        with mock.patch("atlas_tune.agent.defaults", return_value=start), \
                mock.patch("atlas_tune.research.search_github_issues", return_value=[]):
            summary = self._run(FakeRunner(), rounds=2)
        fixed = [t for t in self.store.trials("org/m") if t["hypothesis"].startswith("[fix]")]
        self.assertTrue(fixed)
        self.assertEqual(summary["winner"]["engine"], "vllm")
        self.assertLess(summary["best"]["vllm"]["config"]["max_num_seqs"], 999)

    def test_agent_round_two_sees_redacted_logs_of_failures(self):
        start = [agent.Candidate("vllm", ARTIFACT.id, {"max_num_seqs": 999}, "too many sequences", origin="baseline")]
        seen = []

        def fake_propose(llm, **kw):
            seen.append(kw)
            return []

        with mock.patch("atlas_tune.agent.defaults", return_value=start), \
                mock.patch("atlas_tune.agent.propose", side_effect=fake_propose), \
                mock.patch("atlas_tune.research.search_github_issues", return_value=[]):
            self._run(FakeRunner(), rounds=2, llm="openai/test")
        round2 = next(kw for kw in seen if kw["round_no"] == 2)
        failed = next(r for r in round2["previous"] if r["status"] == "failed")
        self.assertIn("OutOfMemoryError", failed["log_tail"])
        self.assertNotIn("hf_secret", failed["log_tail"])
        self.assertIn("--max-num-seqs", round2["info"]["vllm"]["help"])
        self.assertNotIn("max_num_batched_tokens", round2["allowed"]["vllm"])

    def test_dry_run_starts_nothing(self):
        runner = FakeRunner()
        summary = self._run(runner, dry_run=True)
        self.assertTrue(summary["dry_run"])
        self.assertEqual(runner.started, [])
        self.assertEqual(self.store.runs(), [])


class EngineTests(unittest.TestCase):
    def test_allowlist_rejects_unknown_flags_and_out_of_range_values(self):
        vllm = engines.ENGINES["vllm"]
        with self.assertRaises(ValueError):
            vllm.validate({"--entrypoint": "sh"})
        with self.assertRaises(ValueError):
            vllm.validate({"gpu_memory_utilization": 1.5})
        with self.assertRaises(ValueError):
            vllm.validate({"kv_cache_dtype": "int2"})
        self.assertEqual(vllm.validate({"max_num_seqs": "32"}), {"max_num_seqs": 32})

    def test_renamed_flag_is_mapped_to_the_images_new_name(self):
        from atlas_tune.tuner import Options, Tuner
        tuner_ = Tuner(Options(model="m"), store=mock.Mock(), runner=mock.Mock(), log=lambda *_: None)
        sglang = engines.ENGINES["sglang"]
        newer_help = "--mem-fraction-static F\n--cuda-graph-max-bs-decode N\n"
        tuner_.allowed["sglang"] = sglang.supported(newer_help)
        c = tuner_._fit(agent.Candidate("sglang", "hf:x", {"mem_fraction_static": 0.85, "cuda_graph_max_bs": 32}))
        self.assertEqual(c.config, {"mem_fraction_static": 0.85, "cuda_graph_max_bs_decode": 32})

    def test_llamacpp_context_scales_with_slots(self):
        art = models.Artifact("gguf:r:Q4_K_M", "gguf", "r", 5, quant="Q4_K_M")
        args = engines.ENGINES["llamacpp"].args(art, {"parallel": 4}, 2048)
        self.assertEqual(args[args.index("-c") + 1], "8192")
        self.assertIn("r:Q4_K_M", args)

    def test_trial_binds_localhost_production_publishes(self):
        trial = engines.docker_run(engines.ENGINES["vllm"], ARTIFACT, {}, 4096, host_port=9000, name="t")
        prod = engines.docker_run(engines.ENGINES["vllm"], ARTIFACT, {}, 4096, host_port=8000, name="p",
                                  production=True)
        self.assertIn("127.0.0.1:9000:8000", trial)
        self.assertIn("8000:8000", prod)
        self.assertIn("unless-stopped", prod)
        self.assertIn("no-new-privileges", trial)
        mounts = [trial[i + 1] for i, arg in enumerate(trial) if arg == "-v"]
        self.assertTrue(all(m.endswith(("/root/.cache/huggingface", "/root/.cache/llama.cpp")) for m in mounts))


class ShortlistTests(unittest.TestCase):
    def test_prefers_common_quants_and_adds_moe_offload(self):
        arts = [models.Artifact(f"gguf:u/m:{q}", "gguf", "u/m", s, quant=q, moe=True, layers=48)
                for q, s in (("Q8_0", 30), ("Q4_K_M", 17), ("Q4_1", 18), ("Q3_K_M", 13), ("Q2_K", 10))]
        picked = [a.quant for a in models.shortlist(arts, vram_gib=16, ram_gib=32)]
        self.assertEqual(picked, ["Q3_K_M", "Q2_K", "Q4_K_M"])


class AgentTests(unittest.TestCase):
    def _reply(self, name, args, call_id="c1"):
        call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        message = mock.Mock(tool_calls=[call])
        message.model_dump.return_value = {"role": "assistant"}
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def test_agent_rejects_flags_the_image_does_not_have(self):
        allowed = {"vllm": {"max_num_seqs": engines.ENGINES["vllm"].knobs["max_num_seqs"]}}
        good, errors = agent.validate([{"engine": "vllm", "artifact_id": ARTIFACT.id, "hypothesis": "x",
                                        "config": {"async_scheduling": True}}], ["vllm"], [ARTIFACT], allowed)
        self.assertEqual(good, [])
        self.assertIn("not supported by this vllm image", errors[0])

    def test_agent_researches_then_only_valid_configs_survive(self):
        replies = [
            self._reply("search_github_issues", {"engine": "vllm", "query": "org/m OOM"}),
            self._reply("submit_configs", {"configs": [
                {"engine": "vllm", "artifact_id": ARTIFACT.id, "config": {"max_num_seqs": 48},
                 "hypothesis": "issue says 48 fits", "sources": ["https://github.com/x/1"]},
                {"engine": "vllm", "artifact_id": ARTIFACT.id, "config": {"rm -rf": True}, "hypothesis": "bad"}]}),
        ]
        fake_litellm = mock.Mock()
        fake_litellm.completion.side_effect = replies
        search = mock.Mock(return_value=[{"title": "OOM with max-num-seqs 256", "url": "https://github.com/x/1"}])
        with mock.patch.dict("sys.modules", {"litellm": fake_litellm}), \
                mock.patch.dict("atlas_tune.research.TOOLS", {"search_github_issues": search}):
            out = agent.propose("openai/test", model="org/m", engines=["vllm"], artifacts=[ARTIFACT], hw=HW,
                                goal="throughput", workload={}, previous=[], max_new=4, log=lambda *_: None)
        search.assert_called_once_with(engine="vllm", query="org/m OOM")
        self.assertEqual([c.config for c in out], [{"max_num_seqs": 48}])
        self.assertEqual(out[0].sources, ["https://github.com/x/1"])
        self.assertEqual(out[0].origin, "llm")


class ResearchSafetyTests(unittest.TestCase):
    def test_fetch_refuses_local_private_and_metadata_addresses(self):
        from atlas_tune import research

        for url in ("http://127.0.0.1:8000/", "http://169.254.169.254/latest/meta-data", "http://10.1.2.3/",
                    "file:///etc/passwd", "http://localhost/"):
            with self.assertRaises(ValueError, msg=url):
                research.fetch_page(url)

    def test_model_card_rejects_path_tricks(self):
        from atlas_tune import research

        with self.assertRaises(ValueError):
            research.model_card("../../etc/passwd")

    def test_agent_has_no_local_file_or_shell_tools(self):
        from atlas_tune import research

        names = {s["function"]["name"] for s in research.SCHEMAS} | {agent.SUBMIT["function"]["name"]}
        names.add(agent.ENGINE_HELP["function"]["name"])
        self.assertEqual(names, {"search_github_issues", "fetch_page", "model_card", "engine_help", "submit_configs"})

    def test_engine_help_search_returns_the_flag_entry(self):
        text = agent.search_help(FAKE_VLLM_HELP, "kv-cache-dtype")
        self.assertIn("fp8_e5m2", text)
        self.assertNotIn("--enforce-eager", text)


class BenchTests(unittest.TestCase):
    def test_score_answer_judges_correctness_not_format(self):
        self.assertTrue(bench.score_answer('<think>hmm</think>```json\n{"answer": 95}\n```', 95))
        self.assertTrue(bench.score_answer('{"answer": "95"}', 95))
        self.assertTrue(bench.score_answer("Oslo", "Oslo"))          # real llama.cpp output that was wrongly failed
        self.assertTrue(bench.score_answer("AX-204.", "AX-204"))
        self.assertFalse(bench.score_answer("195", 95))               # real wrong answer from a 0.5B model
        self.assertFalse(bench.score_answer('{"answer": 94}', 84))

    def test_quality_is_judged_against_the_best_config(self):
        good = {"quality": {"pass": True, "passed": 6}}
        worse = {"quality": {"pass": True, "passed": 4}}
        self.assertTrue(bench.quality_ok(good, reference_passed=6))
        self.assertTrue(bench.quality_ok({"quality": {"pass": True, "passed": 5}}, reference_passed=6))
        self.assertFalse(bench.quality_ok(worse, reference_passed=6))

    def test_score_ignores_levels_with_failed_requests(self):
        metrics = {"levels": [{"concurrency": 1, "requests": 8, "ok": 8, "output_tok_s": 50, "per_user_tok_s": 50},
                              {"concurrency": 16, "requests": 32, "ok": 30, "output_tok_s": 900}]}
        self.assertEqual(bench.score(metrics, "throughput"), 50)
        self.assertEqual(bench.score(metrics, "latency"), 50)


if __name__ == "__main__":
    unittest.main()
