"""Focused controller tests; MongoDB is mocked here, GPU is not exercised."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mongomock

from atlas_gpu_harness import __main__ as cli
from atlas_gpu_harness.policy import propose
from atlas_gpu_harness.store import AtlasStore


class FakeStore:
    def __init__(self, budget=2):
        self.run_doc = {"_id": "run1", "status": "running", "budget_evals": budget,
                        "context_hash": "ctx"}
        self.items = {}
        self.decisions = []

    def run(self, run_id):
        return self.run_doc

    def experiments_for(self, run_id):
        return list(self.items.values())

    def retrieve(self, context_hash):
        return [x for x in self.items.values() if x["status"] in ("completed", "failed")]

    def decide(self, run_id, evidence_ids, candidate, hypothesis, reason):
        self.decisions.append({"retrieved": evidence_ids, "candidate": candidate})

    def start(self, run_id, identity, context_hash, candidate):
        key = f"{run_id}:{identity}"
        item = self.items.setdefault(key, {"_id": key, "run_id": run_id,
                                           "candidate": candidate, "attempt": 0})
        if item.get("status") in ("completed", "failed"):
            return None
        item["status"] = "running"
        item["attempt"] += 1
        return "token"

    def finish(self, run_id, identity, token, result):
        item = self.items[f"{run_id}:{identity}"]
        item["result"] = result
        item["status"] = "completed" if result["correct"] else "failed"

    def update_progress(self, run_id, status, best_id):
        self.run_doc["status"] = status


class HarnessTests(unittest.TestCase):
    def test_duplicate_identity_cannot_commit_twice(self):
        store = AtlasStore.__new__(AtlasStore)
        store.experiments = mongomock.MongoClient().db.experiments
        token = store.start("run1", "same", "ctx", "sort_stable")
        store.finish("run1", "same", token, {"correct": True, "timing": {"median_us": 10}})
        self.assertIsNone(store.start("run1", "same", "ctx", "sort_stable"))
        self.assertEqual(store.experiments.count_documents({}), 1)

    def test_failed_fast_candidate_never_wins(self):
        store = FakeStore()
        store.items = {
            "a": {"_id": "a", "run_id": "run1", "candidate": "sort_stable",
                  "status": "completed", "result": {"correct": True, "timing": {"median_us": 50}}},
            "b": {"_id": "b", "run_id": "run1", "candidate": "topk_sorted",
                  "status": "failed", "result": {"correct": False, "timing": {"median_us": 1}}},
        }
        self.assertEqual(cli._summary(store, "run1")["best_experiment"], "a")

    def test_committed_result_survives_fault_and_is_not_rerun(self):
        store = FakeStore(budget=2)
        calls = []

        def worker(candidate):
            calls.append(candidate)
            return {"candidate": candidate, "correct": True, "timing": {"median_us": 10}}

        with tempfile.TemporaryDirectory() as temp:
            with patch.object(cli, "ARTIFACTS", Path(temp)), patch.object(cli, "_context", return_value=({}, "ctx")), \
                 patch.object(cli, "_run_worker", side_effect=worker), \
                 patch.object(cli.os, "_exit", side_effect=SystemExit(86)):
                with self.assertRaises(SystemExit):
                    cli._drive(store, "run1", fault_after_commit=True)
                self.assertEqual(calls, ["sort_stable"])
                cli._drive(store, "run1")
        self.assertEqual(calls, ["sort_stable", "argsort_stable"])
        self.assertEqual(len(store.items), 2)
        self.assertTrue(store.decisions[1]["retrieved"])

    def test_retrieved_tie_failure_blocks_risky_neighbor(self):
        evidence = [
            {"_id": "base", "run_id": "run1", "candidate": "sort_stable", "status": "completed"},
            {"_id": "stable", "run_id": "run1", "candidate": "argsort_stable", "status": "completed"},
            {"_id": "packed", "run_id": "run1", "candidate": "packed_topk", "status": "completed"},
            {"_id": "bad", "run_id": "run1", "candidate": "topk_sorted", "status": "failed",
             "result": {"correct": False, "checks": [{"case": "ties", "passed": False}]}},
        ]
        self.assertIsNone(propose("run1", evidence))


if __name__ == "__main__":
    unittest.main()
