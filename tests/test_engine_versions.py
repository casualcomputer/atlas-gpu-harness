import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from atlas_tune.engines import ENGINES
from atlas_tune.runner import DockerRunner
from atlas_tune.store import Store
from atlas_tune.tuner import Options, Tuner


class EngineVersionTests(unittest.TestCase):
    def test_package_version_wins_over_nvidia_banner(self):
        with patch.object(DockerRunner, '_inside', side_effect=[
                '--max-running-requests N', 'CUDA Version 13.0.3\n\n0.5.21\n']):
            self.assertEqual(DockerRunner().describe(ENGINES['sglang'])['version'],'0.5.21')

    def test_named_llamacpp_version_is_preserved(self):
        version='version: 0.6.0-dev (build 11552, commit 23b0202a1)'
        with patch.object(DockerRunner, '_inside', side_effect=['-np N',version]):
            self.assertEqual(DockerRunner().describe(ENGINES['llamacpp'])['version'],version)

    def test_cached_cuda_banner_is_refreshed(self):
        with tempfile.TemporaryDirectory() as d:
            store=Store(Path(d)/'atlas.db',mirror=False)
            store.put_engine_info('digest','sglang','CUDA Version 13.0.3','--max-running-requests N')
            runner=Mock()
            runner.has_image.return_value=True
            runner.image_id.return_value='digest'
            runner.describe.return_value={'version':'0.5.21','help':'--max-running-requests N'}
            t=Tuner(Options(model='m',engines=['sglang']),store=store,runner=runner,log=lambda *_:None)
            t.inspect_engines(['sglang'])
            runner.describe.assert_called_once()
            self.assertEqual(store.engine_info('digest')['version'],'0.5.21')
