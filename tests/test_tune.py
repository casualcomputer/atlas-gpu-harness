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


class FakeRunner:
    """Speed depends on config so the tuner has something to find; one config 'OOMs'."""

    def __init__(self):
        self.started, self.servers = [], {}

    def image_id(self, engine):
        return f"{engine.image}@sha256:test"

    def cleanup(self):
        return 0

    def start(self, engine, artifact, config, context):
        self.started.append((engine.name, dict(config)))
        if config.get("max_num_seqs") == 999:
            raise EngineFailed("CUDA out of memory", logs="torch.OutOfMemoryError")
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
        self.assertIn("Launch the winner", report)
        self.assertTrue((Path(summary["report"]).parent / "serve.sh").exists())

    def test_second_run_reuses_memory_instead_of_relaunching(self):
        self._run(FakeRunner())
        runner = FakeRunner()
        summary = self._run(runner)
        self.assertEqual(runner.started, [])
        self.assertEqual(summary["winner"]["engine"], "vllm")

    def test_failures_and_wrong_answers_are_recorded_never_selected(self):
        bad = [agent.Candidate("vllm", ARTIFACT.id, {"max_num_seqs": 999}, "oom", origin="llm"),
               agent.Candidate("vllm", ARTIFACT.id, {"enforce_eager": True, "kv_cache_dtype": "fp8"}, "wrong", origin="llm")]
        issue = {"title": "[Bug]: OOM at startup", "url": "https://github.com/vllm-project/vllm/issues/1", "state": "open"}
        with mock.patch("atlas_tune.agent.defaults", return_value=bad), \
                mock.patch("atlas_tune.research.search_github_issues", return_value=[issue]) as search:
            summary = self._run(FakeRunner())
        self.assertIn("OutOfMemoryError", search.call_args.args[1])
        statuses = {t["status"] for t in self.store.trials("org/m")}
        self.assertEqual(statuses, {"failed", "quality_failed"})
        self.assertIsNone(summary["winner"])
        failed = next(t for t in self.store.trials("org/m") if t["status"] == "failed")
        self.assertIn("out of memory", failed["error"])
        self.assertIn("OutOfMemoryError", failed["logs"])
        self.assertEqual(failed["related"], [issue])
        self.assertIn("issues/1", Path(summary["report"]).read_text())

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


class BenchTests(unittest.TestCase):
    def test_score_answer_tolerates_think_blocks_and_fences(self):
        self.assertTrue(bench.score_answer('<think>hmm</think>```json\n{"answer": 95}\n```', 95))
        self.assertFalse(bench.score_answer('{"answer": "95"}', 95))

    def test_score_ignores_levels_with_failed_requests(self):
        metrics = {"levels": [{"concurrency": 1, "requests": 8, "ok": 8, "output_tok_s": 50, "per_user_tok_s": 50},
                              {"concurrency": 16, "requests": 32, "ok": 30, "output_tok_s": 900}]}
        self.assertEqual(bench.score(metrics, "throughput"), 50)
        self.assertEqual(bench.score(metrics, "latency"), 50)


if __name__ == "__main__":
    unittest.main()
