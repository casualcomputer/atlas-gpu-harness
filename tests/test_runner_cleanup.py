"""A failed docker run must not leave a container outside Tuner's finally block."""
import subprocess
import unittest
from unittest import mock

from atlas_tune.engines import ENGINES
from atlas_tune.models import Artifact
from atlas_tune.runner import DockerRunner, EngineFailed


class StartCleanupTests(unittest.TestCase):
    def test_timeout_removes_container_by_reserved_name(self):
        failure = subprocess.TimeoutExpired(['docker', 'run'], 120)
        with mock.patch('atlas_tune.runner.subprocess.run', side_effect=[failure, mock.Mock()]) as run:
            with self.assertRaises(subprocess.TimeoutExpired):
                DockerRunner().start(ENGINES['vllm'], Artifact('a', 'hf', 'repo', 4), {}, 8192)
        launch = run.call_args_list[0].args[0]
        name = launch[launch.index('--name') + 1]
        self.assertEqual(run.call_args_list[1].args[0], ['docker', 'rm', '-f', name])

    def test_nonzero_run_removes_created_container_and_preserves_error(self):
        with mock.patch('atlas_tune.runner.subprocess.run', side_effect=[
                mock.Mock(returncode=125, stderr='port is already allocated'), mock.Mock()]) as run:
            with self.assertRaisesRegex(EngineFailed, 'port is already allocated'):
                DockerRunner().start(ENGINES['vllm'], Artifact('a', 'hf', 'repo', 4), {}, 8192)
        launch = run.call_args_list[0].args[0]
        self.assertEqual(run.call_args_list[1].args[0],
                         ['docker', 'rm', '-f', launch[launch.index('--name') + 1]])

    def test_cleanup_failure_does_not_mask_start_timeout(self):
        failure = subprocess.TimeoutExpired(['docker', 'run'], 120)
        with mock.patch('atlas_tune.runner.subprocess.run', side_effect=[failure, OSError('daemon unavailable')]) as run:
            with self.assertRaises(subprocess.TimeoutExpired) as raised:
                DockerRunner().start(ENGINES['vllm'], Artifact('a', 'hf', 'repo', 4), {}, 8192)
        self.assertIs(raised.exception, failure)
        self.assertEqual(run.call_count, 2)
