"""Inference contract and process ownership tests; no weights or GPU are loaded."""

import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import Mock, patch

from atlas_gpu_harness import inference_backend as backend


class FakeSampler:
    peak = 9000

    def __init__(self):
        self.thread = Mock(ident=None)
        self.pid = None

    def start(self):
        self.thread.ident = 1

    def stop(self):
        pass

    def summary(self):
        return {"device_peak_mib": self.peak, "process_rss_peak_mib": 12000,
                "samples": [{"device_used_mib": self.peak}], "sampling_errors": []}


class InferenceBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.engine = self.root / "llama-server"
        self.model = self.root / "model.gguf"
        self.engine.write_text("placeholder")
        self.model.write_text("placeholder")
        self.process = Mock(pid=12345, returncode=None)
        self.process.poll.return_value = None
        self.calls = []
        self.wrong_answer = False
        self.fail_request = False
        FakeSampler.peak = 9000
        self.answers = {prompt: expected for _, prompt, expected in backend.TUNING_TASKS + backend.HOLDOUT_TASKS}
        self.answers["What is 1 + 1? Return the integer answer."] = 2
        for name, replacement in (
            ("_MemorySampler", FakeSampler), ("_gpu_memory", lambda: (500, 16000)),
            ("_wait_healthy", Mock()), ("_assert_port_free", Mock()),
            ("_request_json", self.request),
        ):
            patcher = patch.object(backend, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.popen = patch.object(backend.subprocess, "Popen", return_value=self.process).start()
        self.addCleanup(patch.stopall)

    def request(self, port, route, payload=None, timeout=120):
        if route == "/health":
            return {"status": "ok"}
        self.calls.append(payload)
        if self.fail_request:
            raise TimeoutError("synthetic request timeout")
        prompt = payload["messages"][1]["content"].removesuffix("\n/no_think")
        answer = -999 if self.wrong_answer else self.answers[prompt]
        return {"choices": [{"message": {"content": json.dumps({"answer": answer})},
                             "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5}, "timings": {"predicted_per_second": 20.0}}

    def evaluate(self, stage="tuning", recipe=None):
        return backend.evaluate_recipe(self.engine, self.model,
            recipe or {"name": "baseline", "n_cpu_moe": None}, self.root / "results",
            gpu_budget_mib=10000, stage=stage)

    def test_same_budget_baseline_and_fixed_two_round_tuning(self):
        result = self.evaluate()
        self.assertTrue(result["correct"])
        self.assertEqual(result["baseline_fit_target_mib"], 6000)
        command = result["command"]
        self.assertEqual(command[command.index("--fit-target") + 1], "6000")
        self.assertEqual(command[command.index("--gpu-layers") + 1], "auto")
        ids = [row["task_id"] for row in result["tasks"]]
        expected = [task[0] for task in backend.TUNING_TASKS]
        self.assertEqual(ids, expected + expected[::-1])
        self.assertEqual(len(self.calls), 13)  # one unscored warmup plus two rounds
        self.assertEqual(result["quality"], {"passed": 12, "total": 12, "completed": 12, "unique_tasks": 6})
        self.assertFalse({t[0] for t in backend.HOLDOUT_TASKS} & set(ids))
        self.process.terminate.assert_called_once()
        self.process.wait.assert_called_once_with(timeout=10)

    def test_default_uses_physical_capacity_without_a_lower_artificial_limit(self):
        FakeSampler.peak = 15000
        result = backend.evaluate_recipe(self.engine, self.model,
            {"name": "baseline", "n_cpu_moe": None}, self.root / "results")
        self.assertTrue(result["correct"])
        self.assertEqual(result["gpu_budget_mib"], 16000)
        self.assertEqual(result["baseline_fit_target_mib"], 1024)

    def test_wrong_answers_are_not_engine_failures_or_accepted_results(self):
        self.wrong_answer = True
        result = self.evaluate()
        self.assertTrue(result["engine_success"])
        self.assertFalse(result["quality_pass"])
        self.assertFalse(result["correct"])
        self.assertEqual(result["quality"]["passed"], 0)
        self.assertIn('"answer": -999', result["tasks"][0]["raw_output"])

    def test_memory_overrun_fails_even_when_every_answer_is_correct(self):
        FakeSampler.peak = 10001
        result = self.evaluate(recipe={"name": "candidate_12", "n_cpu_moe": 12})
        self.assertTrue(result["quality_pass"])
        self.assertFalse(result["resource_pass"])
        self.assertFalse(result["correct"])
        command = result["command"]
        self.assertEqual(command[command.index("--n-cpu-moe") + 1], "12")
        self.assertNotIn("--fit-target", command)
        self.assertEqual(command[command.index("--fit") + 1], "off")

    def test_holdout_is_separate_and_uses_each_task_once(self):
        result = self.evaluate(stage="holdout")
        self.assertEqual([t["task_id"] for t in result["tasks"]], [t[0] for t in backend.HOLDOUT_TASKS])
        self.assertEqual(result["quality"]["total"], 4)
        self.assertEqual(len(self.calls), 5)

    def test_request_failure_still_stops_only_owned_process(self):
        self.fail_request = True
        result = self.evaluate()
        self.assertFalse(result["engine_success"])
        self.assertFalse(result["correct"])
        self.assertEqual(result["error"]["type"], "TimeoutError")
        self.process.terminate.assert_called_once()
        self.process.kill.assert_not_called()

    def test_unknown_recipe_flags_are_rejected_before_start(self):
        with self.assertRaises(ValueError):
            self.evaluate(recipe={"name": "candidate", "n_cpu_moe": 12, "extra_flags": "--anything"})
        self.popen.assert_not_called()

    def test_environment_does_not_forward_credentials_or_engine_overrides(self):
        with patch.dict(os.environ, {"MONGODB_URI": "secret", "HF_TOKEN": "secret", "LLAMA_ARG_CTX_SIZE": "1"}):
            env = backend._child_environment(self.engine)
        self.assertNotIn("MONGODB_URI", env)
        self.assertNotIn("HF_TOKEN", env)
        self.assertNotIn("LLAMA_ARG_CTX_SIZE", env)
        self.assertEqual(env["OMP_NUM_THREADS"], "8")
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")


class IndependentGuardsTests(unittest.TestCase):
    def test_cold_start_can_finish_after_three_minutes_but_is_bounded(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(backend.time, "monotonic", side_effect=[0, 200]), patch.object(
                backend, "_request_json", return_value={"status": "ok"}):
            backend._wait_healthy(process, 8081)
        with patch.object(backend.time, "monotonic", side_effect=[0, 601]):
            with self.assertRaisesRegex(TimeoutError, "600 seconds"):
                backend._wait_healthy(process, 8081)

    def test_json_scoring_rejects_extra_text_wrong_type_and_extra_keys(self):
        self.assertTrue(backend._score('{"answer":1}', 1))
        for content in ('{"answer":true}', '{"answer":"1"}', '{"answer":1,"other":2}',
                        '```json\n{"answer":1}\n```', 'Explanation: {"answer":1}'):
            self.assertFalse(backend._score(content, 1))

    def test_existing_listener_is_refused(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            with self.assertRaises(RuntimeError):
                backend._assert_port_free(listener.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
