import unittest
from unittest import mock

from atlas_tune import agent, engines, models, tuner
from test_tune import HW


class OffloadTests(unittest.TestCase):
    def artifacts(self):
        return [models.Artifact('gguf:org/m:Q8_0','gguf','org/m',20,quant='Q8_0',layers=40,
                                kv_bytes_per_token=147456,revision='abc',file='m.gguf'),
                models.Artifact('hf:org/m','hf','org/m',20,layers=40,kv_bytes_per_token=147456,revision='abc')]

    def test_dense_offload_requires_opt_in_and_enough_host_ram(self):
        arts=self.artifacts()
        self.assertEqual(models.shortlist(arts,16,60),[])
        self.assertEqual(len(models.shortlist(arts,16,60,allow_cpu_offload=True)),2)
        self.assertEqual(models.shortlist(arts,16,8,allow_cpu_offload=True),[])

    def test_offload_baselines_reserve_kv_and_start_one_sequence(self):
        rows=agent.defaults(['llamacpp','vllm','sglang'],self.artifacts(),HW,16,'throughput')
        bases={c.engine:c.config for c in rows if c.origin=='baseline'}
        llama=bases['llamacpp']
        self.assertLess(llama['n_gpu_layers'],40)
        self.assertEqual(llama['parallel'],1)
        kv_gib=8192*147456/2**30
        self.assertLessEqual(20*llama['n_gpu_layers']/40+kv_gib,16*.8)
        self.assertGreater(bases['vllm']['cpu_offload_gb'],0)
        self.assertEqual(bases['vllm']['max_num_seqs'],1)
        self.assertEqual(bases['sglang']['max_running_requests'],1)

    def test_unsupported_offload_is_rejected_not_silently_removed(self):
        t=tuner.Tuner(tuner.Options(model='m'),store=mock.Mock(),runner=mock.Mock(),log=lambda *_:None)
        t.allowed['vllm']={}
        self.assertIsNone(t._fit(agent.Candidate('vllm','hf:org/m',{'cpu_offload_gb':8})))

    def test_revisions_are_pinned_in_every_engine_launch(self):
        gguf,hf=self.artifacts()
        for name in ('vllm','sglang'):
            args=engines.ENGINES[name].args(hf,{},8192)
            self.assertEqual(args[args.index('--revision')+1],'abc')
        args=engines.ENGINES['llamacpp'].args(gguf,{},8192)
        self.assertIn('https://huggingface.co/org/m/resolve/abc/m.gguf',args)
        self.assertIn('/root/.cache/llama.cpp/org_m_abc_m.gguf',args)

    def test_resolution_records_revision_and_explicit_head_dim(self):
        from types import SimpleNamespace
        info=SimpleNamespace(id='org/m',sha='abc',siblings=[SimpleNamespace(rfilename='model.safetensors',size=8*2**30)])
        a=models._hf_artifact(info,{'num_hidden_layers':36,'num_key_value_heads':8,'head_dim':128})
        self.assertEqual(a.revision,'abc')
        self.assertEqual(a.kv_bytes_per_token,147456)
