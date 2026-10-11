"""Local GGUF inspection and planning boundaries; no GPU or model loading required."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from gguf import GGUFWriter

from atlas_gpu_harness.model_probe import inspect_gguf, plan_model


class ModelProbeTests(unittest.TestCase):
    def write_model(self, path, architecture="qwen35moe", experts=256):
        writer = GGUFWriter(path, architecture)
        writer.add_name("Tiny metadata fixture")
        writer.add_block_count(40)
        writer.add_expert_count(experts)
        writer.add_expert_used_count(8)
        writer.add_context_length(8192)
        writer.add_file_type(0)
        writer.add_tensor("fixture.weight", np.zeros(32, dtype=np.float32))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

    def test_real_gguf_metadata_and_explicit_partial_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "snapshots" / ("a" * 40) / "model.gguf"
            path.parent.mkdir(parents=True)
            self.write_model(path)
            model = inspect_gguf(path)
            self.assertEqual(model["architecture"], "qwen35moe")
            self.assertEqual(model["expert_count"], 256)
            self.assertEqual(model["identity"]["hf_revision"], "a" * 40)
            self.assertIn("not_weight_checksum", model["identity"]["kind"])
            self.assertLess(model["identity"]["header_bytes"], model["file_bytes"])
            original_identity = model["identity"]
            # Changing a weight must not be presented as detected by a header fingerprint.
            with path.open("r+b") as file:
                file.seek(-1, 2)
                file.write(b"\x01")
            self.assertEqual(inspect_gguf(path)["identity"], original_identity)

    def test_missing_or_invalid_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "missing.gguf"
            with self.assertRaises(ValueError):
                inspect_gguf(path)
            path.write_bytes(b"not a gguf file")
            with self.assertRaises(ValueError):
                inspect_gguf(path)

    def test_offload_estimate_preserves_real_baseline_and_cannot_promise_fit(self):
        model = {"architecture": "qwen35moe", "expert_count": 256,
                 "block_count": 40, "file_bytes": 20 * 1024 ** 3}
        hardware = {"gpu": {"total_mib": 16303, "free_mib": 14500}}
        plan = plan_model(model, hardware, 12000)
        self.assertTrue(plan["estimate_only"])
        self.assertGreater(plan["initial_n_cpu_moe"], 0)
        self.assertLessEqual(plan["initial_n_cpu_moe"], 40)
        self.assertIsNone(plan["candidates"][0]["n_cpu_moe"])
        self.assertEqual(plan["hardware"], hardware)
        for budget in (0, 2048, 20000, float("nan"), True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                plan_model(model, hardware, budget)
        for bad in ({**model, "expert_count": 0}, {**model, "architecture": "llama"}):
            with self.assertRaises(ValueError):
                plan_model(bad, hardware, 12000)


if __name__ == "__main__":
    unittest.main()
