import unittest
from atlas_tune import agent
from atlas_tune.models import Artifact


class MemoryRecoveryTests(unittest.TestCase):
    def test_vllm_startup_reservation_failure_lowers_claim(self):
        art=Artifact('hf:qwen','hf','Qwen/Qwen3-4B',7.5)
        trial={'status':'failed','engine':'vllm','artifact':art.to_dict(),
               'config':{'gpu_memory_utilization':.94,'max_num_seqs':4},
               'logs':'ValueError: Free memory on device cuda:0 (14.66/15.92 GiB) on startup is less than desired GPU memory utilization (0.94, 14.97 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.'}
        fixes=agent.rule_fixes(trial,art)
        self.assertEqual(len(fixes),1)
        self.assertEqual(fixes[0].config,{'gpu_memory_utilization':.89,'max_num_seqs':4})
        self.assertEqual(fixes[0].origin,'fix')

    def test_generic_oom_never_increases_small_sequence_limit(self):
        art=Artifact('a','hf','r',4)
        for count in (1,2,4):
            trial={'status':'failed','engine':'vllm','artifact':art.to_dict(),
                   'config':{'max_num_seqs':count},'logs':'torch.OutOfMemoryError: CUDA out of memory'}
            for fix in agent.rule_fixes(trial,art):
                self.assertLessEqual(fix.config['max_num_seqs'],count)

    def test_unrelated_startup_failure_has_no_memory_fix(self):
        art=Artifact('a','hf','r',4)
        self.assertEqual(agent.rule_fixes({'status':'failed','engine':'vllm','artifact':art.to_dict(),
                                          'config':{},'logs':'Model repository not found'},art),[])
