import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from atlas_tune import bench, issues, tuner
from atlas_tune.store import Store
from test_tune import fake_server, HW


class WorkloadGateTests(unittest.TestCase):
    def test_custom_wrong_answer_cannot_hide_among_passing_smoke_checks(self):
        httpd=fake_server(.001)
        try:
            q=bench.quality(f'http://127.0.0.1:{httpd.server_port}','fake',extra_tasks=[{'prompt':'new question','expected':95}])
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(q['passed'],8)
        self.assertEqual(q['total'],9)
        self.assertFalse(q['pass'])

    def test_custom_prompt_messages_are_preserved(self):
        client=mock.MagicMock()
        client.__enter__.return_value=client
        client.post.return_value.json.return_value={'choices':[{'message':{'content':'95'}}]}
        with mock.patch('atlas_tune.bench.httpx.Client',return_value=client):
            bench.quality('http://fake','fake',extra_tasks=[{'prompt':'user only','expected':95},
                          {'prompt':'with system','system':'brief','expected':95}])
        bodies=[c.kwargs['json'] for c in client.post.call_args_list]
        self.assertEqual(bodies[-2]['messages'],[{'role':'user','content':'user only'}])
        self.assertEqual(bodies[-1]['messages'][0],{'role':'system','content':'brief'})

    def test_every_level_must_meet_ttft_limit(self):
        metrics={'levels':[{'concurrency':1,'ok':8,'requests':8,'output_tok_s':100,'ttft_p99_ms':50},
                           {'concurrency':4,'ok':8,'requests':8,'output_tok_s':300,'ttft_p99_ms':500}],
                 'max_ttft_ms':100}
        self.assertIsNone(bench.score(metrics,'throughput'))
        self.assertEqual(issues.describe({'status':'ok','metrics':metrics})['category'],'latency_limit')
        metrics['max_ttft_ms']=600
        self.assertEqual(bench.score(metrics,'throughput'),300)

    def test_quality_file_validates_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'quality.json'
            for invalid in [{},[],[{'prompt':'x'}],[{'prompt':'x','expected':{},'system':3}]]:
                path.write_text(json.dumps(invalid))
                with self.assertRaises(ValueError): bench.read_quality_cases(path)
            path.write_text('[{"prompt":"x","expected":95}]')
            self.assertEqual(bench.read_quality_cases(path)[0]['expected'],95)

    def test_agent_and_loader_receive_required_workload(self):
        opts=tuner.Options(model='m',llm='test',context=16384,concurrency=(1,3),max_ttft_ms=200,
                           quality_cases=[{'prompt':'x','expected':1}])
        t=tuner.Tuner(opts,store=mock.Mock(),runner=mock.Mock(),log=lambda *_:None)
        with mock.patch('atlas_tune.agent.propose',return_value=[]) as propose:
            t._agent(HW,[],['vllm'],[],1)
        brief=propose.call_args.kwargs['workload']
        self.assertEqual(brief['concurrency'],(1,3))
        self.assertEqual(brief['context'],16384)
        self.assertEqual(brief['max_ttft_ms'],200)
        with mock.patch('atlas_tune.bench.load',return_value={'levels':[]}) as load:
            metrics=t._load(mock.Mock())
        self.assertEqual(load.call_args.args[2].concurrency,(1,3))
        self.assertEqual(metrics['max_ttft_ms'],200)

    def test_load_failure_is_reported_as_failure_not_ok(self):
        from atlas_tune import agent
        from test_tune import ARTIFACT
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(Path(tmp)/'db',mirror=False)
            runner=mock.Mock()
            runner.image_id.return_value='image@sha256:test'
            t=tuner.Tuner(tuner.Options(model='org/m'),store=store,runner=runner,log=lambda *_:None)
            metrics={'levels':[{'concurrency':1,'ok':7,'requests':8,'output_tok_s':100}]}
            with mock.patch.object(t,'_quality',return_value={'pass':True,'passed':8,'total':8}), \
                 mock.patch.object(t,'_load',return_value=metrics), mock.patch('atlas_tune.hardware.gpus',return_value=[]):
                trial=t.trial(HW,[ARTIFACT],agent.Candidate('vllm',ARTIFACT.id,{}))
            self.assertEqual(trial['status'],'load_failed')
            self.assertEqual(trial['issue']['category'],'load_failure')
