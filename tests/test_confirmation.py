import tempfile
import unittest
from pathlib import Path
from unittest import mock

from atlas_tune import bench, tuner
from atlas_tune.store import Store
from test_tune import ARTIFACT, HW


def measured(key='k', rate=100):
    return dict(key=key,run_id='old',model='org/m',machine='gpu',engine='vllm',artifact=ARTIFACT.to_dict(),
                context=8192,config={},status='ok',quality={'pass':True,'passed':8,'total':8},
                metrics={'levels':[{'concurrency':1,'ok':8,'requests':8,'output_tok_s':rate}]},
                launch=['docker','run','vllm@sha256:test'],hypothesis='test')


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store=Store(Path(self.tmp.name)/'db',mirror=False)
        self.t=tuner.Tuner(tuner.Options(model='org/m',out_dir=Path(self.tmp.name)/'out'),
                           store=self.store,runner=mock.Mock(),log=lambda *_:None)
        self.store.start_run(self.t.run_id,'org/m','throughput','chat',HW)

    def test_one_candidate_is_confirmed_and_full_count_required(self):
        trial=measured()
        with mock.patch.object(self.t,'_measure',return_value=101) as measure:
            self.t.confirm([trial])
        self.assertEqual(measure.call_count,2)
        self.assertEqual(trial['metrics']['confirm_scores'],[101,101])
        with mock.patch('atlas_tune.deploy.write_all'):
            self.assertIsNotNone(self.t.finish(HW,[trial])['winner'])

    def test_incomplete_or_disabled_confirmations_do_not_export(self):
        for scores, requested in [([101],2),([],0),([None,101],2)]:
            with self.subTest(scores=scores):
                trial=measured()
                trial['metrics']['confirm_scores']=scores
                trial['confirmation_run_id']=self.t.run_id
                self.t.opts.confirm=requested
                with mock.patch('atlas_tune.deploy.write_all') as deploy:
                    summary=self.t.finish(HW,[trial])
                self.assertIsNone(summary['winner'])
                deploy.assert_not_called()
                self.assertEqual(self.store.recipes(),[])

    def test_unconfirmed_faster_candidate_cannot_win(self):
        a,b=measured('confirmed',100),measured('unconfirmed',110)
        a['metrics']['confirm_scores']=[90,95]
        a['confirmation_run_id']=self.t.run_id
        with mock.patch('atlas_tune.deploy.write_all'):
            summary=self.t.finish(HW,[a,b])
        self.assertEqual(summary['winner']['score'],95)

    def test_relaunch_checks_quality_and_preserves_failed_attempt(self):
        self.t.runner.start.return_value=mock.Mock(base_url='http://test',model='m')
        with mock.patch('atlas_tune.bench.quality',return_value={'pass':True,'passed':3,'total':8}), \
             mock.patch('atlas_tune.bench.load',return_value=measured()['metrics']), \
             mock.patch('atlas_tune.hardware.gpus',return_value=[]):
            self.assertIsNone(self.t._measure(measured(),8))
        attempt=self.store.attempts()[0]
        self.assertEqual(attempt['phase'],'confirmation')
        self.assertEqual(attempt['payload']['status'],'quality_failed')
        self.assertEqual(attempt['payload']['quality']['passed'],3)

    def test_latency_ranks_response_time_not_decode_rate(self):
        slow={'levels':[{'concurrency':1,'ok':8,'requests':8,'per_user_tok_s':200,'response_p99_ms':2000}]}
        fast={'levels':[{'concurrency':1,'ok':8,'requests':8,'per_user_tok_s':100,'response_p99_ms':1000}]}
        self.assertGreater(bench.score(fast,'latency'),bench.score(slow,'latency'))

    def test_long_response_latency_keeps_score_precision(self):
        trial=measured()
        trial['metrics']['levels'][0]['response_p99_ms']=60000
        trial['metrics']['confirm_scores']=[.017,.018]
        self.assertEqual(tuner.final_score(trial,'latency'),.017)
