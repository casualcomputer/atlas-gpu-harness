"""Saved recipe guards, with no inference engine or model loaded."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from atlas_gpu_harness import inference_serve as serve
from atlas_gpu_harness.inference_backend import CONTRACT_VERSION, _command


class SavedRecipeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.engine = self.root / "launch-engine"
        self.engine.write_text("trusted launcher placeholder")
        self.model = self.root / "model.gguf"
        self.model.write_text("mocked metadata and weights")
        self.identity = {"digest": "fixed-model-metadata"}
        self.hardware = {"gpu": {"name": "test GPU", "total_mib": 16303.0, "driver": "test-driver"},
                         "ram": {"total_mib": 48000}, "cpu_count": 20}
        backend = Path(serve.__file__).with_name("inference_backend.py")
        self.fixed = {"contract_version": CONTRACT_VERSION, "model_identity": self.identity,
                      "model_mtime_ns": self.model.stat().st_mtime_ns,
                      "engine_launcher_sha256": hashlib.sha256(self.engine.read_bytes()).hexdigest(),
                      "engine_version": "pinned build", "gpu": dict(self.hardware["gpu"]),
                      "cpu_count": 20, "ram_total_mib": 48000,
                      "context_size": 4096, "gpu_budget_mib": 10000,
                      "source_sha256": {"inference_backend.py": hashlib.sha256(backend.read_bytes()).hexdigest()}}
        context_hash = hashlib.sha256(json.dumps(self.fixed, sort_keys=True).encode()).hexdigest()
        self.recipe = {"name": "engine_autofit", "n_cpu_moe": None}
        self.recommendation = {"run_id": "run", "recipe": self.recipe,
                               "model_path": str(self.model), "engine_path": str(self.engine),
                               "context_hash": context_hash, "context_size": 4096,
                               "gpu_budget_mib": 10000, "validated_experiment_id": "run:validated"}
        self.result = {"recipe": self.recipe, "stage": "holdout", "correct": True,
                       "engine_success": True, "quality_pass": True, "resource_pass": True,
                       "context_size": 4096, "gpu_budget_mib": 10000, "gpu_total_mib": 16303.0,
                       "baseline_fit_target_mib": 6303,
                       "command": _command(self.engine, self.model, self.recipe, 8081, 4096, 6303)}
        self.report = {"run": {"_id": "run", "recommendation": self.recommendation,
                               "context_hash": context_hash,
                               "context": {"fixed": self.fixed, "model": {"identity": self.identity}}},
                       "experiments": [{"_id": "run:validated", "run_id": "run", "context_hash": context_hash,
                                        "status": "completed", "result": self.result}]}
        self.path = self.root / "recommendation.json"
        self.write_records()
        for name, replacement in (
            ("inspect_gguf", Mock(return_value={"path": str(self.model), "identity": self.identity})),
            ("probe_hardware", Mock(return_value=self.hardware)),
        ):
            patcher = patch.object(serve, name, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(serve.subprocess, "run", return_value=Mock(stdout="pinned build\n", stderr=""))
        self.version = patcher.start()
        self.addCleanup(patcher.stop)

    def write_records(self):
        self.path.write_text(json.dumps(self.recommendation))
        self.path.with_name("report.json").write_text(json.dumps(self.report))

    def test_saved_baseline_budget_and_fit_target_survive_new_port(self):
        command, _ = serve._validated_command(self.path, 8090)
        self.assertEqual(command[command.index("--port") + 1], "8090")
        self.assertEqual(command[command.index("--fit-target") + 1], "6303")
        self.assertEqual(command[command.index("--ctx-size") + 1], "4096")

    def test_model_mtime_change_is_rejected_even_with_same_header(self):
        stat = self.model.stat()
        os.utime(self.model, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(ValueError, "modification time"):
            serve._validated_command(self.path, 8081)

    def test_changed_engine_launcher_is_rejected_before_execution(self):
        self.engine.write_text("changed launcher")
        with self.assertRaisesRegex(ValueError, "launcher changed"):
            serve._validated_command(self.path, 8081)
        self.version.assert_not_called()

    def test_changed_engine_version_is_rejected(self):
        self.version.return_value.stdout = "another build"
        with self.assertRaisesRegex(ValueError, "version changed"):
            serve._validated_command(self.path, 8081)

    def test_changed_gpu_driver_is_rejected(self):
        self.hardware["gpu"]["driver"] = "different driver"
        with self.assertRaisesRegex(ValueError, "GPU identity"):
            serve._validated_command(self.path, 8081)

    def test_failed_holdout_cannot_be_served(self):
        self.result["quality_pass"] = False
        self.write_records()
        with self.assertRaisesRegex(ValueError, "successful matching holdout"):
            serve._validated_command(self.path, 8081)

    def test_recommendation_cannot_silently_expand_memory_budget(self):
        self.recommendation["gpu_budget_mib"] = 16303
        self.write_records()
        with self.assertRaisesRegex(ValueError, "gpu_budget_mib differs"):
            serve._validated_command(self.path, 8081)

    def test_recorded_baseline_fit_target_is_checked(self):
        self.result["baseline_fit_target_mib"] = 1024
        self.write_records()
        with self.assertRaisesRegex(ValueError, "baseline fit target"):
            serve._validated_command(self.path, 8081)

    def test_recorded_launch_settings_cannot_drift(self):
        command = self.result["command"]
        command[command.index("--threads") + 1] = "1"
        self.write_records()
        with self.assertRaisesRegex(ValueError, "launch command differs"):
            serve._validated_command(self.path, 8081)


if __name__ == "__main__":
    unittest.main()
