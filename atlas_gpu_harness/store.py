"""Small Atlas repository. MongoDB is the source of truth after a restart."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone


def now() -> datetime:
    return datetime.now(timezone.utc)


class AtlasStore:
    def __init__(self):
        from dotenv import load_dotenv
        from pymongo import MongoClient

        load_dotenv(override=False)
        uri = os.environ.get("MONGODB_URI")
        if not uri:
            raise RuntimeError("MONGODB_URI is unset. Set the hackathon Atlas Sandbox URI in your shell.")
        self.client = MongoClient(uri, serverSelectionTimeoutMS=8000, w="majority",
                                  appname="AtlasGPUHarness")
        self.db = self.client["atlas_gpu_harness"]
        self.runs = self.db.runs
        self.experiments = self.db.experiments
        self.decisions = self.db.decisions

    def init(self) -> dict:
        self.client.admin.command("ping")
        self.experiments.create_index([("context_hash", 1), ("finished_at", -1)])
        self.experiments.create_index([("run_id", 1), ("status", 1)])
        self.decisions.create_index("run_id")
        probe_id = "probe:" + uuid.uuid4().hex
        self.db.health.insert_one({"_id": probe_id, "created_at": now()})
        read_ok = self.db.health.find_one({"_id": probe_id}) is not None
        self.db.health.delete_one({"_id": probe_id})
        if not read_ok:
            raise RuntimeError("Atlas write/read probe failed")
        return {"ping": "ok", "write_read": "ok", "database": self.db.name}

    def create_run(self, context: dict, context_hash: str, budget: int) -> str:
        run_id = uuid.uuid4().hex
        self.runs.insert_one({"_id": run_id, "objective": "exact top-k GPU operator search",
                              "status": "running", "budget_evals": budget,
                              "context": context, "context_hash": context_hash,
                              "policy_version": "deterministic-memory-v1", "created_at": now()})
        return run_id

    def run(self, run_id: str) -> dict:
        result = self.runs.find_one({"_id": run_id})
        if result is None:
            raise ValueError(f"unknown run: {run_id}")
        return result

    def experiments_for(self, run_id: str) -> list[dict]:
        return list(self.experiments.find({"run_id": run_id}).sort("created_at", 1))

    def retrieve(self, context_hash: str) -> list[dict]:
        return list(self.experiments.find({"context_hash": context_hash,
                                           "status": {"$in": ["completed", "failed"]}})
                    .sort("finished_at", -1).limit(50))

    def decide(self, run_id: str, evidence_ids: list[str], candidate: str,
               hypothesis: str, reason: str, policy_version: str = "deterministic-memory-v1") -> None:
        self.decisions.insert_one({"_id": uuid.uuid4().hex, "run_id": run_id,
                                   "retrieved_experiment_ids": evidence_ids,
                                   "candidate": candidate, "hypothesis": hypothesis,
                                   "reason": reason, "predicted_outcome": "unknown; evaluator decides",
                                   "policy_version": policy_version, "created_at": now()})

    def start(self, run_id: str, identity: str, context_hash: str, candidate: str) -> str | None:
        from pymongo.errors import DuplicateKeyError

        document_id = f"{run_id}:{identity}"
        try:
            self.experiments.insert_one({"_id": document_id, "run_id": run_id,
                                         "context_hash": context_hash, "candidate": candidate,
                                         "status": "queued", "attempt": 0, "created_at": now()})
        except DuplicateKeyError:
            pass
        token = uuid.uuid4().hex
        result = self.experiments.update_one(
            {"_id": document_id, "status": {"$in": ["queued", "running"]}},
            {"$set": {"status": "running", "owner_token": token, "started_at": now()},
             "$inc": {"attempt": 1}},
        )
        return token if result.modified_count else None

    def finish(self, run_id: str, identity: str, token: str, result: dict) -> None:
        document_id = f"{run_id}:{identity}"
        status = "completed" if result.get("correct") else "failed"
        update = self.experiments.update_one(
            {"_id": document_id, "status": "running", "owner_token": token},
            {"$set": {"status": status, "result": result, "finished_at": now()},
             "$unset": {"owner_token": ""}},
        )
        if update.modified_count != 1:
            raise RuntimeError("stale or duplicate worker completion rejected")

    def update_progress(self, run_id: str, status: str, best_id: str | None) -> None:
        self.runs.update_one({"_id": run_id}, {"$set": {"status": status,
                              "best_validated_experiment": best_id, "updated_at": now()}})

    def decisions_for(self, run_id: str) -> list[dict]:
        return list(self.decisions.find({"run_id": run_id}).sort("created_at", 1))
