"""Memory planning regressions for the 16 GiB Qwen3 GPU runs."""
import unittest
from types import SimpleNamespace

from atlas_tune import agent, models


class MemoryPlanningTests(unittest.TestCase):
    def artifact(self, size=8.11, format='gguf'):
        cfg = dict(num_hidden_layers=36, num_key_value_heads=8,
                   num_attention_heads=32, head_dim=128, hidden_size=2560)
        info = SimpleNamespace(id='Qwen/Qwen3-4B', siblings=[
            SimpleNamespace(rfilename='model.safetensors', size=int(size * 2**30))])
        art = models._hf_artifact(info, cfg)
        art.engine_format = format
        return art

    def test_explicit_head_dim_and_kv_heads_determine_cache_size(self):
        self.assertEqual(self.artifact().kv_bytes_per_token, 147456)

    def test_default_candidates_budget_full_context_for_every_slot(self):
        for name, knob in [('llamacpp', 'parallel'), ('vllm', 'max_num_seqs'),
                           ('sglang', 'max_running_requests')]:
            art = self.artifact(format='gguf' if name == 'llamacpp' else 'hf')
            hw = {'gpu': {'vram_gib': 16}, 'cpu_count': 8}
            for context in (4096, 8192, 32768):
                with self.subTest(engine=name, context=context):
                    candidates = agent.defaults([name], [art], hw, 16, 'throughput', context=context)
                    self.assertTrue(candidates)
                    for c in candidates:
                        self.assertGreaterEqual(c.config[knob], 1)
                        total = art.size_gib + c.config[knob] * context * 147456 / 2**30
                        self.assertLessEqual(total, 16 * .9)

    def test_shortlist_requires_room_for_one_full_context(self):
        art = self.artifact(size=13, format='hf')
        self.assertEqual(models.shortlist([art], 16, 64, context=16384), [])
        self.assertEqual(models.shortlist([art], 16, 64, context=8192), [art])

    def test_missing_architecture_defaults_to_one_slot(self):
        art = models.Artifact('unknown', 'hf', 'unknown', 8)
        candidates = agent.defaults(['vllm'], [art], {'gpu': {'vram_gib': 16}}, 16, 'throughput')
        self.assertTrue(all(c.config['max_num_seqs'] == 1 for c in candidates))


if __name__ == '__main__':
    unittest.main()
