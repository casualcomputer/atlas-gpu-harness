"""Read Atlas decisions joined to the experiment evidence each decision retrieved.

Run from the repository root. This executes the checked-in read-only aggregation;
it does not create an Atlas UI saved query, indexes, or health/probe documents.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas_gpu_harness.store import AtlasStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{32}", args.run_id):
        parser.error("run-id must be a 32-character lowercase hexadecimal run ID")
    path = ROOT / "queries/decision-evidence.json"
    pipeline = json.loads(path.read_text(encoding="utf-8"))
    allowed = {"$match", "$sort", "$lookup", "$project"}
    if (not isinstance(pipeline, list) or not pipeline
            or any(not isinstance(stage, dict) or len(stage) != 1
                   or not set(stage) <= allowed for stage in pipeline)
            or pipeline[0] != {"$match": {"run_id": "REPLACE_WITH_RUN_ID"}}):
        raise SystemExit("Query must retain the run placeholder and read-only aggregation stages")
    # Only the exact match value changes; the checked-in query remains untouched.
    pipeline[0]["$match"]["run_id"] = args.run_id
    store = None
    try:
        store = AtlasStore()
        # Do not call init(): its index and health-probe operations would write.
        decisions = list(store.decisions.aggregate(pipeline, maxTimeMS=15000))
    except Exception as error:
        # Connection errors can contain connection details; never echo credentials.
        raise SystemExit(f"Atlas evidence read failed ({type(error).__name__}); check local configuration and access") from None
    finally:
        if store is not None:
            store.client.close()
    result = {
        "run_id": args.run_id,
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "database": "atlas_gpu_harness", "collection": "decisions",
        "query_file": "queries/decision-evidence.json",
        "read_only": True, "atlas_ui_saved_query_created": False,
        "decision_count": len(decisions), "decisions": decisions,
        "scope": "Live aggregation of stored decision-to-evidence links; links show evidence retrieved, not proof that a proposal improved",
    }
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
