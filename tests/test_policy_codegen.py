"""Generated policies must remain pure, bounded placement decisions."""

import hashlib
import io
import json
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from atlas_gpu_harness.policy_codegen import PolicyError, evaluate_policy, propose_policy


CONTEXT = {"model": {"block_count": 48}, "hardware": {"gpu_memory_mib": 16384},
           "previous_trials": []}


class PolicyCodegenTests(unittest.TestCase):
    def test_context_conditioned_policy_revises_after_failure(self):
        code = """def choose(context):
    layers = 12
    if len(context['previous_trials']) > 0:
        previous = context['previous_trials'][-1]
        layers = previous['n_cpu_moe']
        if previous['status'] == 'oom':
            layers = layers + 8
    return {'n_cpu_moe': min(context['model']['block_count'], layers)}
"""
        self.assertEqual(evaluate_policy(code, CONTEXT), {"n_cpu_moe": 12})
        revised = dict(CONTEXT, previous_trials=[{"n_cpu_moe": 12, "status": "oom"}])
        self.assertEqual(evaluate_policy(code, revised), {"n_cpu_moe": 20})

    def test_malicious_or_unbounded_syntax_is_rejected(self):
        bodies = [
            "import os\n    return {'n_cpu_moe': 1}",
            "open('/tmp/atlas-policy-should-never-exist', 'w')\n    return {'n_cpu_moe': 1}",
            "return {'n_cpu_moe': context.__class__}",
            "while True:\n        pass\n    return {'n_cpu_moe': 1}",
            "return {'n_cpu_moe': len([x for x in context])}",
            "return {'n_cpu_moe': __import__('os').getpid()}",
            "context['model']['block_count'] = 999\n    return {'n_cpu_moe': 1}",
            "return choose(context)",
        ]
        for body in bodies:
            with self.subTest(body=body), self.assertRaises(PolicyError):
                evaluate_policy("def choose(context):\n    " + body, CONTEXT)

    def test_bounds_and_output_contract(self):
        for returned in ("{'n_cpu_moe': -1}", "{'n_cpu_moe': 49}", "{'n_cpu_moe': True}",
                         "{'n_cpu_moe': 1.0}", "{'n_cpu_moe': 1, 'extra': 2}", "5"):
            with self.subTest(returned=returned), self.assertRaises(PolicyError):
                evaluate_policy("def choose(context):\n    return " + returned, CONTEXT)

    def test_resource_exhaustion_is_bounded(self):
        code = "def choose(context):\n    value = 'x' * 1000000000\n    return {'n_cpu_moe': len(value)}"
        with self.assertRaises(PolicyError):
            evaluate_policy(code, CONTEXT)

    def test_context_rejects_credentials(self):
        with self.assertRaises(PolicyError):
            evaluate_policy("def choose(context):\n    return {'n_cpu_moe': 1}",
                            dict(CONTEXT, hardware={"api_key": "do-not-send"}))

    def test_real_proposal_protocol_and_evidence_without_network(self):
        code = "def choose(context):\n    return {'n_cpu_moe': int(context['model']['block_count'] / 3)}"
        answer = {"model": "gemma3:12b", "response": json.dumps({"hypothesis": "Keep one third on CPU", "code": code}),
                  "eval_count": 80, "done_reason": "stop"}
        with tempfile.TemporaryDirectory() as directory, patch(
                "atlas_gpu_harness.policy_codegen.urllib.request.OpenerDirector.open",
                return_value=io.BytesIO(json.dumps(answer).encode())) as request:
            result = propose_policy(CONTEXT, directory)
            payload = json.loads(request.call_args.args[0].data)
            self.assertEqual(payload["keep_alive"], 0)
            self.assertEqual(payload["options"], {"temperature": 0, "num_predict": 600})
            self.assertFalse(payload["stream"])
            self.assertEqual(result["n_cpu_moe"], 16)
            self.assertEqual(result["source_hash"], hashlib.sha256(code.encode()).hexdigest())
            self.assertEqual(Path(result["path"]).read_text(), code)
            self.assertEqual(json.loads(Path(result["path"]).with_suffix(".json").read_text())["validation_status"], "passed")

    def test_invalid_generated_code_is_saved_and_never_replaced(self):
        code = "def choose(context):\n    import os\n    return {'n_cpu_moe': 16}"
        answer = {"response": json.dumps({"hypothesis": "Invalid policy", "code": code})}
        with tempfile.TemporaryDirectory() as directory, patch(
                "atlas_gpu_harness.policy_codegen.urllib.request.OpenerDirector.open",
                return_value=io.BytesIO(json.dumps(answer).encode())):
            with self.assertRaises(PolicyError):
                propose_policy(CONTEXT, directory)
            receipts = list(Path(directory).glob("*.json"))
            self.assertEqual(len(receipts), 1)
            evidence = json.loads(receipts[0].read_text())
            self.assertEqual(evidence["validation_status"], "failed")
            self.assertEqual(evidence["code"], code)

    def test_null_baseline_error_preserves_code_and_actual_exception(self):
        code = "def choose(context):\n    return {'n_cpu_moe': context['previous_trials'][0]['n_cpu_moe'] + 1}"
        answer = {"response": json.dumps({"hypothesis": "An invalid numeric baseline assumption", "code": code})}
        context = dict(CONTEXT, previous_trials=[{"n_cpu_moe": None}])
        with tempfile.TemporaryDirectory() as directory, patch(
                "atlas_gpu_harness.policy_codegen.urllib.request.OpenerDirector.open",
                return_value=io.BytesIO(json.dumps(answer).encode())):
            with self.assertRaises(PolicyError) as failure:
                propose_policy(context, directory)
            self.assertEqual(failure.exception.evidence["code"], code)
            self.assertIn("TypeError", failure.exception.evidence["execution_error"])
            self.assertIn("NoneType", failure.exception.evidence["execution_error"])
            self.assertLessEqual(len(failure.exception.evidence["execution_error"]), 800)

    def test_local_proposer_does_not_use_proxy_environment(self):
        code = "def choose(context):\n    return {'n_cpu_moe': 16}"
        answer = {"response": json.dumps({"hypothesis": "Test", "code": code})}
        with tempfile.TemporaryDirectory() as directory, patch.dict(
                "os.environ", {"http_proxy": "http://untrusted-proxy.invalid:9999", "no_proxy": ""}), patch(
                "atlas_gpu_harness.policy_codegen.urllib.request.build_opener", wraps=urllib.request.build_opener) as opener, patch(
                "atlas_gpu_harness.policy_codegen.urllib.request.OpenerDirector.open",
                return_value=io.BytesIO(json.dumps(answer).encode())):
            propose_policy(CONTEXT, directory)
            self.assertEqual(opener.call_args.args[0].proxies, {})


if __name__ == "__main__":
    unittest.main()
