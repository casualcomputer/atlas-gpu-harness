"""Local memory in SQLite, with MongoDB Atlas as an optional mirror.

Local is always on, so the tool works offline and remembers every trial. When
MONGODB_URI is set, runs, trials and recipes are mirrored to Atlas (best effort) so a
team, or other machines, can share verified results. `atlas sync` backfills Atlas.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, created TEXT, model TEXT, goal TEXT, workload TEXT,
  hardware TEXT, status TEXT, summary TEXT);
CREATE TABLE IF NOT EXISTS trials (
  key TEXT PRIMARY KEY, run_id TEXT, created TEXT, model TEXT, machine TEXT, engine TEXT,
  image TEXT, artifact TEXT, config TEXT, context INTEGER, workload TEXT, status TEXT,
  quality TEXT, metrics TEXT, gpu_peak_gib REAL, error TEXT, logs TEXT, hypothesis TEXT,
  sources TEXT, launch TEXT, related TEXT);
CREATE INDEX IF NOT EXISTS trials_model ON trials(model, machine);
CREATE TABLE IF NOT EXISTS recipes (
  model TEXT, machine TEXT, engine TEXT, goal TEXT, workload TEXT, trial_key TEXT,
  score REAL, launch TEXT, updated TEXT, PRIMARY KEY (model, machine, engine, goal, workload));
"""
JSON_FIELDS = {"hardware", "summary", "artifact", "config", "quality", "metrics", "sources", "launch", "related"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def trial_key(**identity) -> str:
    """Same machine, engine image digest, artifact, config, context and workload → same trial."""
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:20]


def default_path() -> Path:
    return Path(os.environ.get("ATLAS_HOME", Path.home() / ".atlas-tune")) / "atlas.db"


class Store:
    def __init__(self, path: Path | str | None = None, mirror=None):
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(trials)")}
        if "related" not in columns:  # databases created by 0.3.0
            self.db.execute("ALTER TABLE trials ADD COLUMN related TEXT")
        self.mirror = mirror if mirror is not None else AtlasMirror.from_env()

    # -- helpers
    def _write(self, table: str, row: dict, conflict: str = "REPLACE") -> None:
        encoded = {k: json.dumps(v, default=str) if k in JSON_FIELDS and v is not None else v for k, v in row.items()}
        cols = ", ".join(encoded)
        self.db.execute(f"INSERT OR {conflict} INTO {table} ({cols}) VALUES ({', '.join('?' * len(encoded))})",
                        tuple(encoded.values()))
        self.db.commit()
        if self.mirror:
            self.mirror.put(table, row)

    @staticmethod
    def _decode(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        return {k: json.loads(row[k]) if k in JSON_FIELDS and row[k] is not None else row[k] for k in row.keys()}

    def _all(self, sql: str, args=()) -> list[dict]:
        return [self._decode(r) for r in self.db.execute(sql, args).fetchall()]

    # -- runs
    def start_run(self, run_id: str, model: str, goal: str, workload: str, hardware: dict) -> None:
        self._write("runs", {"id": run_id, "created": now(), "model": model, "goal": goal,
                             "workload": workload, "hardware": hardware, "status": "running", "summary": None})

    def finish_run(self, run_id: str, status: str, summary: dict) -> None:
        run = self._decode(self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone())
        self._write("runs", {**run, "status": status, "summary": summary})

    def runs(self, model: str | None = None, limit: int = 20) -> list[dict]:
        where, args = ("WHERE model=?", (model,)) if model else ("", ())
        return self._all(f"SELECT * FROM runs {where} ORDER BY created DESC LIMIT ?", (*args, limit))

    # -- trials
    def trial(self, key: str) -> dict | None:
        return self._decode(self.db.execute("SELECT * FROM trials WHERE key=?", (key,)).fetchone())

    def put_trial(self, trial: dict) -> None:
        self._write("trials", {"created": now(), **trial})

    def trials(self, model: str | None = None, machine: str | None = None, run_id: str | None = None) -> list[dict]:
        clauses, args = [], []
        for column, value in (("model", model), ("machine", machine), ("run_id", run_id)):
            if value:
                clauses.append(f"{column}=?")
                args.append(value)
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return self._all(f"SELECT * FROM trials {where} ORDER BY created", args)

    # -- recipes
    def put_recipe(self, recipe: dict) -> None:
        self._write("recipes", {**recipe, "updated": now()})

    def recipes(self, model: str | None = None, machine: str | None = None) -> list[dict]:
        clauses = [c for c, v in (("model=?", model), ("machine=?", machine)) if v]
        args = [v for v in (model, machine) if v]
        where = "WHERE " + " AND ".join(clauses) if clauses else ""
        return self._all(f"SELECT * FROM recipes {where} ORDER BY score DESC", args)

    def sync(self) -> dict:
        if not self.mirror:
            raise RuntimeError("MONGODB_URI is not set; nothing to sync to")
        counts = {}
        for table in ("runs", "trials", "recipes"):
            rows = self._all(f"SELECT * FROM {table}")
            for row in rows:
                self.mirror.put(table, row)
            counts[table] = len(rows)
        return counts


class AtlasMirror:
    """Best-effort copy of local rows into Atlas collections tune_runs / tune_trials / tune_recipes."""

    KEYS = {"runs": ("id",), "trials": ("key",), "recipes": ("model", "machine", "engine", "goal", "workload")}

    def __init__(self, uri: str, database: str = "atlas_gpu_harness"):
        from pymongo import MongoClient

        self.db = MongoClient(uri, serverSelectionTimeoutMS=5000, appname="atlas-tune")[database]
        self.failed = False

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv

        load_dotenv(override=False)
        uri = os.environ.get("MONGODB_URI")
        return cls(uri) if uri else None

    def put(self, table: str, row: dict) -> None:
        if self.failed:
            return
        document = {k: v for k, v in row.items() if k != "logs"}
        document["_id"] = ":".join(str(row[k]) for k in self.KEYS[table])
        try:
            self.db[f"tune_{table}"].replace_one({"_id": document["_id"]}, document, upsert=True)
        except Exception as error:  # noqa: BLE001 - mirror must never break a local run
            self.failed = True
            print(f"  (Atlas mirror disabled for this run: {type(error).__name__})")
