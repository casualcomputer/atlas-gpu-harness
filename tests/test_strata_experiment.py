import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mongomock

from atlas_gpu_harness import strata_experiment as suite
from atlas_gpu_harness.store import AtlasStore


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return list(text.encode())


class StrataExperimentTests(unittest.TestCase):
    def test_native_eos_is_removed_but_other_trailing_tokens_are_preserved(self):
        class Tokenizer:
            def encode(self, text, **kwargs):
                return [1000 if text == '<|im_end|>' else 1001]
            def decode(self, ids):
                return ''.join(chr(i) for i in ids)
        tok = Tokenizer()
        body = list(b'{"answer": 1}')
        self.assertEqual(suite.answer_text(tok, body + [1000]), '{"answer": 1}')
        self.assertEqual(suite.answer_text(tok, body + list(b'bad') + [1000]), '{"answer": 1}bad')
        self.assertEqual(body, list(b'{"answer": 1}'))

    def test_real_input_budget_and_overrides_are_deterministic(self):
        a = suite.workload(CharacterTokenizer(), 32768, 42)
        b = suite.workload(CharacterTokenizer(), 32768, 42)
        self.assertEqual(a, b)
        self.assertEqual(len(a['input_ids']), 32768)
        self.assertEqual(len(a['entry_token_positions']), 10)
        text = bytes(a['input_ids']).decode()
        for key, value in a['expected']['settings'].items():
            self.assertIn(f'{key} = {value}', text)
        self.assertNotEqual(a['expected'], suite.workload(CharacterTokenizer(), 32768, 43)['expected'])
        with self.assertRaises(ValueError):
            suite.workload(CharacterTokenizer(), 4096, 42)

    def test_quality_gate_rejects_stale_value_and_boolean_integer(self):
        expected = suite.workload(CharacterTokenizer(), 32768, 42)['expected']
        answer = {**expected, 'review': 'The later amendments supersede prior settings.'}
        self.assertTrue(suite.score(json.dumps(answer), expected)['passed'])
        answer = json.loads(json.dumps(answer))
        key = list(answer['settings'])[1]
        answer['settings'][key] -= 7
        self.assertFalse(suite.score(json.dumps(answer), expected)['passed'])
        self.assertFalse(suite.score('[]', expected)['passed'])
        self.assertFalse(suite.score('{', expected)['passed'])

    def test_no_fast_failure_or_holdout_leakage_in_selection(self):
        rows = []
        for policy, rate, passed in [('default', 40, True), ('cpu_4', 45, True), ('adaptive_160', 90, False)]:
            rows += [{'result': {'stage': 'tuning', 'policy': policy, 'decode_tps': rate, 'correct': passed}}] * 3
        rows += [{'result': {'stage': 'holdout', 'policy': 'no_pcie', 'decode_tps': 200, 'correct': True}}] * 3
        self.assertEqual(suite.select_policy(rows, 3)[0], 'cpu_4')
        rows[3]['result']['correct'] = False
        self.assertEqual(suite.select_policy(rows, 3)[0], 'default')

    def test_confirm_requires_every_pair_and_matching_length(self):
        rows = [dict(stage='holdout', block=b, policy=p, prompt_tokens=65536,
                     decode_tps=rate, correct=True)
                for b in range(3) for p, rate in [('default', 40), ('cpu_4', 44)]]
        self.assertTrue(suite.paired_summary(rows, 'cpu_4', 65536, 3)['supported'])
        self.assertFalse(suite.paired_summary(rows, 'cpu_4', 131072, 3)['supported'])
        rows[-1]['correct'] = False
        self.assertFalse(suite.paired_summary(rows, 'cpu_4', 65536, 3)['supported'])

    def test_atlas_resume_reuses_trials_and_exports_resolved_evidence(self):
        store = AtlasStore.__new__(AtlasStore)
        db = mongomock.MongoClient().db
        store.runs, store.experiments, store.decisions = db.runs, db.experiments, db.decisions
        run_id = store.create_run({}, 'ctx', 21)
        store.runs.update_one({'_id': run_id}, {'$set': {'repeats': 3}})
        run = store.run(run_id)
        calls = []

        def evaluate(cfg, a, b, c, tok, prompt, policy, log):
            calls.append(policy)
            return dict(policy=policy, prompt_tokens=prompt['prompt_tokens'], correct=True,
                        decode_tps=45 if policy == 'cpu_4' else 40)

        with tempfile.TemporaryDirectory() as tmp, patch.object(suite, 'evaluate', evaluate):
            directory = Path(tmp)
            with self.assertRaises(suite.CheckpointStop):
                suite.execute(store, run, {}, CharacterTokenizer(), (None, None, None), directory,
                              directory/'latest.json', stop_after=1)
            self.assertEqual(len(calls), 1)
            suite.execute(store, run, {}, CharacterTokenizer(), (None, None, None), directory, directory/'latest.json')
            self.assertEqual(len(calls), 21)
            suite.execute(store, run, {}, CharacterTokenizer(), (None, None, None), directory, directory/'latest.json')
            self.assertEqual(len(calls), 21)
            exported = json.loads((directory/'latest.json').read_text())
            self.assertEqual(exported['atlas_audit']['dangling_references'], 0)
            self.assertEqual(exported['atlas_audit']['attempts'], [1] * 21)
            self.assertEqual(exported['run']['recommendation'], 'cpu_4')


if __name__ == '__main__':
    unittest.main()
