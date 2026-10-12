"""Real CUDA validation of the original packed-key candidate."""

import random
import unittest

try:
    import torch
except ImportError:  # optional extra: uv sync --extra torch
    torch = None

from atlas_gpu_harness.evaluator import _operator, _oracle


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "PyTorch with a CUDA GPU required")
class PackedKeyGpuTests(unittest.TestCase):
    def test_exact_indices_across_ties_negatives_and_signed_zero(self):
        rng = random.Random(20261011)
        values_pool = [-3.0, -1.0, -0.0, 0.0, 0.5, 1.0, 3.0]
        for n in (1, 2, 4, 7, 16, 33, 257):
            for _ in range(8):
                values = [rng.choice(values_pool) for _ in range(n)]
                k = rng.randint(1, n)
                actual = _operator(torch.tensor(values, device="cuda"), k, "packed_topk").cpu().tolist()
                self.assertEqual(actual, _oracle(values, k), (n, k, values))


if __name__ == "__main__":
    unittest.main()
