import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from atlas_tune import agent, issues, tuner
from atlas_tune.store import Store
from test_tune import ARTIFACT, HW, FakeRunner, FAST


class HistoryRecoveryTests(unittest.TestCase):
    def test_attempts_survive_cache_replacement_and_reopen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'db'
            store = Store(path, mirror=False)
            trial = dict(key='k', run_id='r', model='m', machine='gpu', engine='vllm', config={}, status='failed')
            store.record_attempt(trial, 'trial')
            store.put_trial(trial)
            trial['status'] = 'ok'
            store.record_attempt(trial, 'trial')
            store.put_trial(trial)
            store.db.close()
            reopened = Store(path, mirror=False)
            self.assertEqual([a['payload']['status'] for a in reopened.attempts()], ['failed', 'ok'])
            self.assertEqual(reopened.trial('k')['status'], 'ok')
            reopened.db.close()

    def test_legacy_history_imported_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'db'
            store = Store(path, mirror=False)
            store.put_trial(dict(key='k', run_id='r', model='m', machine='gpu', status='failed'))
            store.db.execute('DROP TABLE attempts')
            store.db.commit()
            store.db.close()
            for _ in range(2):
                store = Store(path, mirror=False)
                self.assertEqual(len(store.attempts()), 1)
                self.assertEqual(store.attempts()[0]['phase'], 'legacy')
                store.db.close()

    def test_failure_categories_and_transient_policy(self):
        for error, category, retryable in [
            ('CUDA out of memory', 'oom', False),
            ('Free memory on device 1 less than desired GPU memory utilization', 'memory_reservation', False),
            ('TimeoutExpired: download', 'timeout', True),
            ('no space left on device', 'disk_full', True),
            ('port is already allocated', 'port_conflict', True),
            ('401 Unauthorized', 'authentication', True),
            ('unrecognized arguments --wrong', 'unsupported', False),
            ('SSL certificate failure', 'network', True),
            ('something else', 'unknown', False),
        ]:
            with self.subTest(error=error):
                described = issues.describe({'status':'failed', 'error':error})
                self.assertEqual((described['category'], described['retryable']), (category, retryable))

    def test_transient_cache_retries_and_explicit_retry_overrides_oom(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / 'db', mirror=False)
            runner = FakeRunner()
            t = tuner.Tuner(tuner.Options(model='org/m', confirm=0), store=store, runner=runner, log=lambda *_:None)
            candidate = agent.Candidate('vllm', ARTIFACT.id, {})
            with mock.patch('atlas_tune.hardware.gpus',return_value=[]), mock.patch.dict('atlas_tune.bench.WORKLOADS', {'chat':FAST}):
                first = t.trial(HW, [ARTIFACT], candidate)
                first.update(status='failed',error='TimeoutExpired')
                store.put_trial(first)
                self.assertFalse(t.trial(HW,[ARTIFACT],candidate).get('reused'))
                first.update(error='CUDA out of memory')
                store.put_trial(first)
                self.assertTrue(t.trial(HW,[ARTIFACT],candidate)['reused'])
                t.opts.retry_failed=True
                self.assertFalse(t.trial(HW,[ARTIFACT],candidate).get('reused'))
            self.assertEqual(len(runner.started),3)

    def test_agent_receives_serving_context_and_reused_oom_generates_fix(self):
        t=tuner.Tuner(tuner.Options(model='m',context=32768,llm='test'),store=mock.Mock(),runner=mock.Mock())
        with mock.patch('atlas_tune.agent.propose',return_value=[]) as propose:
            t._agent(HW,[ARTIFACT],['vllm'],[],1)
        self.assertEqual(propose.call_args.kwargs['workload']['context'],32768)
        t.opts.llm=None
        t.store.attempts.return_value=[]
        failed={'engine':'vllm','artifact':ARTIFACT.to_dict(),'config':{'max_num_seqs':8},'status':'failed',
                'error':'CUDA out of memory','reused':True}
        self.assertTrue(t.next_round(HW,[ARTIFACT],['vllm'],[failed],2))

    def test_interrupted_trials_are_preserved_and_not_cached_as_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'db',mirror=False)
            runner=mock.Mock()
            runner.image_id.return_value='image@sha256:test'
            runner.start.side_effect=KeyboardInterrupt
            t=tuner.Tuner(tuner.Options(model='org/m'),store=store,runner=runner,log=lambda *_:None)
            with mock.patch('atlas_tune.hardware.gpus',return_value=[]), self.assertRaises(KeyboardInterrupt):
                t.trial(HW,[ARTIFACT],agent.Candidate('vllm',ARTIFACT.id,{}))
            rows=store.attempts()
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['payload']['status'],'interrupted')
            self.assertEqual(rows[0]['payload']['phase'],'trial')
            self.assertIn('source_commit',rows[0]['payload'])
            self.assertEqual(store.trials(),[])
