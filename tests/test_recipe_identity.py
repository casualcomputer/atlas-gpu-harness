import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

from atlas_tune import cli, hardware
from test_tune import HW


class RecipeIdentityTests(unittest.TestCase):
    def test_cpu_and_ram_change_cache_identity(self):
        first={**HW,'cpu_model':'CPU A'}
        for change in ({'cpu_model':'CPU B'},{'cpu_count':16},{'ram_total_gib':128}):
            self.assertNotEqual(hardware.fingerprint(first),hardware.fingerprint({**first,**change}))
        self.assertEqual(hardware.fingerprint(first),hardware.fingerprint({**first,'ram_available_gib':30}))

    def test_legacy_recipe_prints_measured_digest(self):
        store=mock.Mock()
        store.recipes.return_value=[dict(engine='vllm',workload='chat',score=100,updated='now',goal='throughput',
                                         trial_key='k',launch=['docker','run','vllm/vllm-openai:latest','--model','org/m'])]
        store.trial.return_value={'image':'vllm/vllm-openai@sha256:abc'}
        out=io.StringIO()
        with mock.patch('atlas_tune.store.Store',return_value=store), \
             mock.patch('atlas_tune.hardware.detect',return_value=HW), contextlib.redirect_stdout(out):
            self.assertEqual(cli._recipe(SimpleNamespace(model='org/m',goal='throughput')),0)
        self.assertIn('vllm/vllm-openai@sha256:abc',out.getvalue())
        self.assertNotIn(':latest',out.getvalue())

    def test_registry_port_is_not_mistaken_for_an_image_tag(self):
        store=mock.Mock()
        store.recipes.return_value=[dict(engine='vllm',workload='chat',score=100,updated='now',goal='throughput',
                                         trial_key='k',launch=['docker','run','registry:5000/engine:latest'])]
        store.trial.return_value={'image':'registry:5000/engine@sha256:abc'}
        out=io.StringIO()
        with mock.patch('atlas_tune.store.Store',return_value=store), \
             mock.patch('atlas_tune.hardware.detect',return_value=HW), contextlib.redirect_stdout(out):
            cli._recipe(SimpleNamespace(model='org/m',goal='throughput'))
        self.assertIn('registry:5000/engine@sha256:abc',out.getvalue())
