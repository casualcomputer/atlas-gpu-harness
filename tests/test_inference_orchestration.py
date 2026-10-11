"""Persistence and recommendation checks without launching an inference engine."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mongomock

from atlas_gpu_harness import inference
from atlas_gpu_harness.policy_codegen import PolicyError
from atlas_gpu_harness.store import AtlasStore


def make_store(budget=2):
    store = AtlasStore.__new__(AtlasStore)
    db = mongomock.MongoClient().db
    store.runs, store.experiments, store.decisions = db.runs, db.experiments, db.decisions
    context = {"fixed": {"gpu_budget_mib": 10000}, "plan": {"initial_n_cpu_moe": 16},
               "model": {"name": "test", "architecture": "qwen3moe", "block_count": 48,
                         "expert_count": 128, "expert_used_count": 8, "file_bytes": 18000000000}}
    run_id = store.create_run(context, "test-context", budget)
    store.runs.update_one({"_id": run_id}, {"$set": {"model_path": "test.gguf", "engine_path": "test-server",
                                                     "port": 8081, "proposer_model": "test-local"}})
    return store, store.run(run_id)


def result(recipe, stage, correct=True, rate=10):
    return {"recipe": recipe, "stage": stage, "correct": correct, "engine_success": True,
            "quality_pass": correct, "resource_pass": True,
            "median_decode_tokens_per_second": rate,
            "tasks": [{"task_id": task[0], "timings": {"predicted_per_second": rate}}
                      for task in inference.HOLDOUT_TASKS]}


class InferenceOrchestrationTests(unittest.TestCase):
    def test_failed_slot_is_terminal_across_resume(self):
        store, run = make_store()
        store.runs.update_one({"_id": run["_id"]}, {"$set": {
            "proposal_attempts.policy_1": {"status": "failed"},
            "proposal_errors.policy_1": {"message": "Rejected code"}}})
        with tempfile.TemporaryDirectory() as directory, patch.object(inference, "propose_policy") as proposer, patch.object(
                inference, "evaluate_recipe", side_effect=lambda engine, model, recipe, output, **kwargs:
                result(recipe, kwargs["stage"], False)):
            inference.evolve(store, run, Path(directory))
            inference.evolve(store, run, Path(directory))
        proposer.assert_not_called()
        self.assertEqual(store.experiments.count_documents({}), 1)

    def test_interrupted_generation_consumes_slot(self):
        store, run = make_store()
        store.runs.update_one({"_id": run["_id"]}, {"$set": {"proposal_attempts.policy_1": {"status": "running"}}})
        with tempfile.TemporaryDirectory() as directory, patch.object(inference, "propose_policy") as proposer, patch.object(
                inference, "evaluate_recipe", side_effect=lambda engine, model, recipe, output, **kwargs:
                result(recipe, kwargs["stage"], False)):
            inference.evolve(store, run, Path(directory))
        proposer.assert_not_called()
        self.assertEqual(store.run(run["_id"])["proposal_attempts"]["policy_1"]["status"], "interrupted")

    def test_rejected_source_and_exception_reach_next_proposal(self):
        store, run = make_store(3)
        code = "def choose(context):\n    return {'n_cpu_moe': None + 1}"
        failure = PolicyError("Bad null arithmetic", {"code": code, "hypothesis": "Try one more layer",
                                                       "execution_error": "TypeError: NoneType + int"})
        good = {"code": "def choose(context):\n    return {'n_cpu_moe': 16}", "hypothesis": "Repair null handling",
                "n_cpu_moe": 16}
        contexts = []

        def propose(context, output, model):
            contexts.append(context)
            if len(contexts) == 1:
                raise failure
            return good

        with tempfile.TemporaryDirectory() as directory, patch.object(inference, "probe_hardware", return_value={}), patch.object(
                inference, "propose_policy", side_effect=propose), patch.object(
                inference, "evaluate_recipe", side_effect=lambda engine, model, recipe, output, **kwargs:
                result(recipe, kwargs["stage"], False)):
            inference.evolve(store, run, Path(directory))
        self.assertEqual(contexts[1]["previous_policy"]["code"], code)
        self.assertIn("NoneType", contexts[1]["feedback"]["previous_errors"]["policy_1"]["execution_error"])
        self.assertEqual(store.run(run["_id"])["proposal_errors"]["policy_1"]["code"], code)

    def test_holdout_rejection_cannot_be_best_validated(self):
        store, run = make_store()
        proposal = {"code": "def choose(context):\n    return {'n_cpu_moe': 16}",
                    "hypothesis": "Try expert offload", "n_cpu_moe": 16}

        def evaluate(engine, model, recipe, output, **kwargs):
            candidate = recipe["n_cpu_moe"] is not None
            return result(recipe, kwargs["stage"], not (candidate and kwargs["stage"] == "holdout"),
                          20 if candidate else 10)

        with tempfile.TemporaryDirectory() as directory, patch.object(inference, "probe_hardware", return_value={}), patch.object(
                inference, "propose_policy", return_value=proposal), patch.object(inference, "evaluate_recipe", side_effect=evaluate):
            decision = inference.evolve(store, run, Path(directory))
        self.assertEqual(decision["recipe"]["name"], "engine_autofit")
        best = store.experiments.find_one({"_id": store.run(run["_id"])["best_validated_experiment"]})
        self.assertEqual(best["candidate"], "holdout_baseline_2")
        self.assertTrue(best["result"]["correct"])

    def test_recovered_holdout_baseline_wins_over_slower_candidate(self):
        store, run = make_store()
        proposal = {"code": "def choose(context):\n    return {'n_cpu_moe': 16}",
                    "hypothesis": "Try expert offload", "n_cpu_moe": 16}

        def evaluate(engine, model, recipe, output, **kwargs):
            candidate = recipe["n_cpu_moe"] is not None
            # A transient tuning baseline failure must not override fresh comparisons.
            correct = candidate or kwargs["stage"] == "holdout"
            return result(recipe, kwargs["stage"], correct, 20 if candidate else 100)

        with tempfile.TemporaryDirectory() as directory, patch.object(inference, "probe_hardware", return_value={}), patch.object(
                inference, "propose_policy", return_value=proposal), patch.object(inference, "evaluate_recipe", side_effect=evaluate):
            decision = inference.evolve(store, run, Path(directory))
        self.assertEqual(decision["recipe"]["name"], "engine_autofit")
        self.assertIn("Both held-out baselines passed", decision["selection_reason"])
        self.assertFalse(decision["confirmation"]["directionally_supported_speedup"])

    def test_twice_validated_candidate_remains_usable_when_baseline_confirmation_fails(self):
        store, run = make_store()
        proposal = {"code": "def choose(context):\n    return {'n_cpu_moe': 16}",
                    "hypothesis": "Try expert offload", "n_cpu_moe": 16}

        def evaluate(engine, model, recipe, output, **kwargs):
            candidate = recipe["n_cpu_moe"] is not None
            correct = candidate or kwargs["stage"] == "tuning"
            return result(recipe, kwargs["stage"], correct, 20 if candidate else 10)

        with tempfile.TemporaryDirectory() as directory, patch.object(inference, "probe_hardware", return_value={}), patch.object(
                inference, "propose_policy", return_value=proposal), patch.object(inference, "evaluate_recipe", side_effect=evaluate):
            decision = inference.evolve(store, run, Path(directory))
        self.assertEqual(decision["recipe"]["n_cpu_moe"], 16)
        self.assertIn("Feasibility only; speedup unproven", decision["selection_reason"])
        self.assertFalse(decision["confirmation"]["directionally_supported_speedup"])

    def test_engine_failure_feedback_keeps_a_bounded_cause_tail(self):
        summary = inference.trial_summary({"_id": "experiment", "candidate": "candidate",
            "result": {"server_log_tail": "x" * 2000 + "CUDA allocation failed: out of memory"}})
        self.assertEqual(len(summary["engine_error_tail"]), 1500)
        self.assertTrue(summary["engine_error_tail"].endswith("CUDA allocation failed: out of memory"))

    def test_legacy_decision_policy_version_remains_default(self):
        store, run = make_store()
        store.decide(run["_id"], [], "sort_stable", "fixed baseline", "legacy operator experiment")
        self.assertEqual(store.decisions_for(run["_id"])[0]["policy_version"], "deterministic-memory-v1")

    def test_startup_exception_is_committed_and_reused(self):
        store, run = make_store()
        recipe = {"name": "engine_autofit", "n_cpu_moe": None}
        with tempfile.TemporaryDirectory() as directory, patch.object(
                inference, "evaluate_recipe", side_effect=RuntimeError("Refusing occupied port")) as evaluate:
            first = inference._evaluate(store, run, "baseline", recipe, "tuning", Path(directory))
            second = inference._evaluate(store, run, "baseline", recipe, "tuning", Path(directory))
        self.assertEqual(first["status"], "failed")
        self.assertEqual(second["attempt"], 1)
        self.assertIn("occupied port", first["result"]["error"]["message"])
        self.assertEqual(store.decisions_for(run["_id"])[0]["policy_version"], "local-model-codegen-v1")
        evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
