"""Planning and endpoint contract tests; no model or GPU is required."""

import io
import json
import unittest
from unittest.mock import patch

from atlas_gpu_harness import deployment_engineer as engineer
from atlas_gpu_harness.inference_backend import TUNING_TASKS


class PlanningTests(unittest.TestCase):
    def spec(self):
        return {"model": {"architecture": "moe", "weights_gib": 30},
                "hardware": {"gpu_total_gib": 32, "gpu_count": 2,
                             "ram_available_gib": 64, "ssd_free_gib": 100},
                "targets": {"vllm": "org/model", "sglang": "org/model",
                            "ollama": "model:tag"}}

    def test_moe_experiments_preserve_gpu_count_and_one_setting_changes(self):
        plan = engineer.plan(self.spec())
        by_id = {row["id"]: row for row in plan["experiments"]}
        self.assertIn("vllm:expert_parallel", by_id)
        self.assertIn("sglang:expert_parallel", by_id)
        self.assertIn("vllm:cpu_offload", by_id)
        self.assertEqual(by_id["vllm:baseline"]["argv"][-2:], ["--tensor-parallel-size", "2"])
        self.assertEqual(by_id["sglang:expert_parallel"]["argv"][-4:],
                         ["--tp", "2", "--ep", "2"])
        self.assertNotIn("ollama:expert_parallel", by_id)
        self.assertEqual(plan["storage_experiments"][0]["id"], "ssd:cold_vs_warm_load")

    def test_gpu_pressure_adds_kv_experiments_after_baseline_evidence(self):
        spec = self.spec()
        prior = [{"runtime": "sglang", "model": "org/model", "workload": engineer.WORKLOAD,
                  "gpu_peak_mib": 31000, "quality_pass": True}]
        by_id = {row["id"]: row for row in engineer.plan(spec, prior)["experiments"]}
        self.assertEqual(by_id["sglang:fp8_kv"]["argv"][-1], "fp8_e4m3")
        self.assertNotIn("vllm:fp8_kv", by_id)

    def test_ssd_weight_filter_requires_verified_per_expert_layout(self):
        spec = self.spec()
        ids = {row["id"] for row in engineer.plan(spec)["experiments"]}
        self.assertNotIn("vllm:expert_weight_filter", ids)
        spec["model"]["per_expert_checkpoint"] = True
        by_id = {row["id"]: row for row in engineer.plan(spec)["experiments"]}
        self.assertEqual(by_id["vllm:expert_weight_filter"]["argv"][-2:],
                         ["--enable-expert-parallel", "--enable-ep-weight-filter"])
        self.assertEqual(by_id["vllm:expert_weight_filter"]["compare_to"],
                         "vllm:expert_parallel")

    def test_runtime_artifact_sizes_drive_separate_offload_choices(self):
        spec = self.spec()
        spec["targets"]["vllm"] = {"model": "org/model", "weights_gib": 60}
        spec["targets"]["sglang"] = {"model": "org/model", "weights_gib": 12}
        ids = {row["id"] for row in engineer.plan(spec)["experiments"]}
        self.assertIn("vllm:cpu_offload", ids)
        self.assertNotIn("sglang:cpu_offload", ids)

    def test_wrong_answers_prompt_template_diagnosis_not_memory_tuning(self):
        prior = [{"runtime": "vllm", "model": "org/model", "workload": engineer.WORKLOAD,
                  "quality_pass": False, "tasks": [{"passed": False, "raw_output": "wrong"}]}]
        plan = engineer.plan(self.spec(), prior)
        self.assertEqual(plan["diagnostics"][0]["runtime"], "vllm")
        self.assertNotIn("vllm:fp8_kv", {row["id"] for row in plan["experiments"]})

    def test_endpoint_timeout_is_not_assumed_to_be_gpu_pressure(self):
        prior = [{"runtime": "vllm", "model": "org/model", "workload": engineer.WORKLOAD,
                  "quality_pass": False, "tasks": [{"error": "TimeoutError: request timed out"}]}]
        plan = engineer.plan(self.spec(), prior)
        self.assertIn("server failed", plan["diagnostics"][0]["finding"])
        self.assertNotIn("vllm:fp8_kv", {row["id"] for row in plan["experiments"]})

    def test_ollama_kv_change_follows_flash_attention_control(self):
        prior = [{"runtime": "ollama", "model": "model:tag", "workload": engineer.WORKLOAD,
                  "gpu_peak_mib": 31000, "quality_pass": True}]
        by_id = {row["id"]: row for row in engineer.plan(self.spec(), prior)["experiments"]}
        self.assertEqual(by_id["ollama:q8_kv"]["compare_to"], "ollama:flash_attention")
        self.assertNotIn("OLLAMA_KV_CACHE_TYPE", by_id["ollama:flash_attention"]["env"])

    def test_rejects_unknown_runtime_and_invalid_inventory(self):
        spec = self.spec()
        spec["targets"]["other"] = "some/model"
        with self.assertRaisesRegex(ValueError, "unsupported targets"):
            engineer.plan(spec)
        spec["targets"].pop("other")
        spec["hardware"]["gpu_total_gib"] = 0
        with self.assertRaisesRegex(ValueError, "gpu_total_gib"):
            engineer.plan(spec)


class DocumentationTests(unittest.TestCase):
    def test_discovers_first_party_links_and_ranks_matches(self):
        pages = {
            "vllm": '<a href="/en/latest/serving/expert_parallel_deployment/">Expert parallel deployment</a>',
            "sglang": '- [Expert Parallelism](https://docs.sglang.io/docs/expert.md): MoE weights\n',
            "ollama": '- [External](https://bad.example/moe): unrelated\n',
            "unsloth": '- [Export](https://unsloth.ai/docs/export.md): MoE model export\n',
        }
        by_url = {url: pages[runtime] for runtime, url in engineer.DOC_INDEXES.items()}
        result = engineer.discover_docs("moe expert", fetch=by_url.__getitem__)
        self.assertEqual(result["total_matches"], 3)
        self.assertTrue(all(row["url"].startswith(("https://docs.vllm.ai/",
                      "https://docs.sglang.io/", "https://unsloth.ai/"))
                            for row in result["matches"]))
        self.assertEqual(result["errors"], {})


class FakeResponse:
    def __init__(self, payload):
        self.body = io.BytesIO(json.dumps(payload).encode())

    def __enter__(self):
        return self.body

    def __exit__(self, *_):
        self.body.close()


class FakeOpener:
    def __init__(self):
        self.answers = {prompt: expected for _, prompt, expected in TUNING_TASKS}

    def open(self, request, timeout):
        payload = json.loads(request.data)
        prompt = payload["messages"][1]["content"]
        return FakeResponse({"choices": [{"message": {"content": json.dumps(
            {"answer": self.answers[prompt]})}}], "usage": {"completion_tokens": 5}})


class BenchmarkTests(unittest.TestCase):
    def test_local_endpoint_scores_all_tasks_and_records_resource_scope(self):
        with patch.object(engineer, "build_opener", return_value=FakeOpener()), patch.object(
                engineer, "_host_snapshot", return_value={"gpu_used_mib_by_device": [1000, 2000],
                                                          "ram_available_mib": 32000}):
            result = engineer.benchmark("http://127.0.0.1:8000", "vllm", "org/model")
        self.assertTrue(result["quality_pass"])
        self.assertEqual(len(result["tasks"]), 12)
        self.assertEqual(result["gpu_peak_mib"], 3000)
        self.assertEqual(result["gpu_per_device_peak_mib"], [1000, 2000])
        self.assertIn("peaks between samples", result["resource_scope"])

    def test_remote_endpoint_is_rejected_before_request(self):
        with self.assertRaisesRegex(ValueError, "local"):
            engineer.benchmark("https://example.com", "vllm", "org/model")

    def test_comparison_separates_pass_and_failure_and_requires_same_workload(self):
        good = {"runtime": "vllm", "model": "org/model", "workload": engineer.WORKLOAD,
                "quality_pass": True, "median_completion_tokens_per_response_second": 10}
        bad = {**good, "runtime": "sglang", "model": "local/other-format",
               "quality_pass": False,
               "median_completion_tokens_per_response_second": 20}
        comparison = engineer.compare([bad, good])
        ranking = comparison["ranking"]
        self.assertEqual([row["outcome"] for row in ranking], ["worked", "did_not_work"])
        self.assertEqual(comparison["models"], ["local/other-format", "org/model"])
        with self.assertRaisesRegex(ValueError, "same workload"):
            engineer.compare([good, {**bad, "workload": "other"}])


if __name__ == "__main__":
    unittest.main()
