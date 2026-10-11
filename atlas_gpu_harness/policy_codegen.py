"""Ask a local model to write a small, testable MoE placement policy."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ENDPOINT = "http://127.0.0.1:11434/api/generate"
SAFE_CALLS = {"min", "max", "int", "len"}
CONTEXT_KEYS = {"hardware", "model", "previous_trials", "previous_policy", "feedback"}
ALLOWED_NODES = {
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign, ast.If,
    ast.Return, ast.Name, ast.Load, ast.Store, ast.Constant, ast.Dict,
    ast.Subscript, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.UAdd,
    ast.USub, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE,
}


class PolicyError(ValueError):
    """The proposer or its generated policy did not meet the contract."""

    def __init__(self, message: str, evidence: dict | None = None):
        super().__init__(message)
        self.evidence = evidence or {}


def _context(context: dict) -> dict:
    if not isinstance(context, dict) or set(context) - CONTEXT_KEYS:
        raise PolicyError("Context accepts only hardware, model, previous_trials, previous_policy and feedback")
    try:
        encoded = json.dumps(context, allow_nan=False)
        clean = json.loads(encoded)
        blocks = clean["model"]["block_count"]
    except (ValueError, TypeError, KeyError) as exc:
        raise PolicyError("Context must be JSON with model.block_count") from exc
    if type(blocks) is not int or not 1 <= blocks <= 10000 or len(encoded) > 32768:
        raise PolicyError("Invalid model.block_count or context exceeds 32 KiB")

    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {"password", "secret", "api_key", "access_token", "mongodb_uri", "credential"}:
                    raise PolicyError("Secret fields must not enter proposer context")
                check(item)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, str) and ("mongodb://" in value or "mongodb+srv://" in value):
            raise PolicyError("Database credentials must not enter proposer context")

    check(clean)
    return clean


def _validate_code(code: str) -> None:
    if not isinstance(code, str) or not code.strip() or len(code) > 8000:
        raise PolicyError("Policy code must contain 1 to 8000 characters")
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as exc:
        raise PolicyError("Policy is not valid Python") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise PolicyError("Policy must contain exactly one function: choose(context)")
    function = tree.body[0]
    args = function.args
    if (function.name != "choose" or function.decorator_list or function.returns
            or args.posonlyargs or args.kwonlyargs or args.defaults or args.kw_defaults
            or args.vararg or args.kwarg or len(args.args) != 1
            or args.args[0].arg != "context" or args.args[0].annotation):
        raise PolicyError("Policy signature must be undecorated choose(context) with no annotations or defaults")
    nodes = list(ast.walk(tree))
    if len(nodes) > 400:
        raise PolicyError("Policy exceeds 400 syntax nodes")
    for node in nodes:
        if type(node) not in ALLOWED_NODES or (isinstance(node, ast.FunctionDef) and node is not function):
            raise PolicyError(f"Policy syntax is not allowed: {type(node).__name__}")
        if isinstance(node, ast.Name) and "__" in node.id:
            raise PolicyError("Dunder names are not allowed")
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, float, str, bool, type(None))
                                             or len(str(node.value)) > 1024 or "__" in str(node.value)):
            raise PolicyError("Policy constant is not allowed")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name)
                                          or node.func.id not in SAFE_CALLS or node.keywords):
            raise PolicyError("Only min, max, int and len calls without keyword arguments are allowed")
        if isinstance(node, ast.Assign) and any(not isinstance(target, ast.Name)
                                               or target.id in SAFE_CALLS | {"context", "choose"}
                                               for target in node.targets):
            raise PolicyError("Assignments may only create local variables")
        if isinstance(node, ast.Dict) and any(key is None for key in node.keys):
            raise PolicyError("Dictionary unpacking is not allowed")


# Trusted runner only: generated code cannot import, access attributes, or call IO.
# This is deliberately a constrained policy language, not a sandbox for arbitrary Python.
_RUNNER = r'''
import json, sys
if sys.platform.startswith("linux"):
    import resource
    for limit, value in ((resource.RLIMIT_CPU, 1), (resource.RLIMIT_AS, 256 * 1024 * 1024),
                         (resource.RLIMIT_FSIZE, 0), (resource.RLIMIT_CORE, 0), (resource.RLIMIT_NOFILE, 16)):
        resource.setrlimit(limit, (value, value))
payload = json.load(sys.stdin)
namespace = {"__builtins__": {"min": min, "max": max, "int": int, "len": len}}
exec(compile(payload["code"], "<generated-policy>", "exec"), namespace)
result = namespace["choose"](payload["context"])
if (type(result) is not dict or set(result) != {"n_cpu_moe"}
        or type(result["n_cpu_moe"]) is not int
        or not 0 <= result["n_cpu_moe"] <= payload["context"]["model"]["block_count"]):
    raise ValueError("Policy must return one integer n_cpu_moe within model block count")
print(json.dumps(result))
'''


def evaluate_policy(code: str, context: dict) -> dict:
    """Validate and evaluate generated placement code in an isolated, bounded process."""
    clean = _context(context)
    _validate_code(code)
    try:
        with tempfile.TemporaryDirectory(prefix="atlas-policy-") as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _RUNNER],
                input=json.dumps({"code": code, "context": clean}), text=True,
                capture_output=True, timeout=3, env={}, cwd=directory,
            )
    except subprocess.TimeoutExpired as exc:
        raise PolicyError("Policy exceeded its three-second time limit") from exc
    if result.returncode:
        detail = result.stderr.strip().splitlines()[-1][:800] if result.stderr.strip() else "Worker terminated without an exception message"
        raise PolicyError(f"Policy evaluation failed: {detail}",
                          {"execution_error": detail, "worker_returncode": result.returncode})
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise PolicyError("Policy worker returned invalid JSON") from exc


def propose_policy(context: dict, output_dir: str | Path, model: str = "gemma3:12b") -> dict:
    """Generate actual code through local Ollama, preserve evidence, and evaluate it."""
    clean = _context(context)
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"policy-{uuid.uuid4().hex[:12]}"
    path = directory / f"{stem}.py"
    receipt = directory / f"{stem}.json"
    prompt = (
        "Write a Python MoE placement policy for llama.cpp. Return JSON with exactly hypothesis and code, "
        "both strings. Code must define only def choose(context): and return {'n_cpu_moe': integer}. "
        "n_cpu_moe is the number of initial layers whose experts stay in CPU RAM, from 0 through "
        "context['model']['block_count']. More CPU experts save GPU memory but can slow inference. "
        "Use hardware, model and previous_trials evidence to choose or revise placement. "
        "A previous trial with n_cpu_moe=null is the engine's automatic baseline, not a numeric placement. "
        "Never do arithmetic on null; use feedback['initial_estimate_n_cpu_moe'] as a numeric starting point. "
        "Use == None or != None for null checks; Python 'is' is not allowed. "
        "If previous_policy contains rejected code, repair it using feedback['previous_errors']. "
        "The same model weights, context length and evaluation stay fixed. Never claim improvement before testing. "
        "Allowed Python: local variable assignment, if/else, return, arithmetic + - * / // %, comparisons, "
        "and/or/not, dictionary literals, indexing and calls to min, max, int, len only. "
        "No imports, attributes, loops, comprehensions, annotations, defaults, decorators, dunder names, "
        "or other function calls. Do not mutate context. No markdown fences. Keep code below 20 lines. "
        "Context follows as JSON data:\n" + json.dumps(clean, sort_keys=True)
    )
    request = urllib.request.Request(ENDPOINT, data=json.dumps({
        "model": model, "prompt": prompt, "format": "json", "stream": False,
        "keep_alive": 0, "options": {"temperature": 0, "num_predict": 600},
    }).encode(), headers={"Content-Type": "application/json"}, method="POST")
    evidence = {"context": clean, "proposer": {"engine": "ollama", "endpoint": ENDPOINT,
                                                "requested_model": model, "temperature": 0,
                                                "num_predict": 600, "keep_alive": 0}}
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=180) as response:
            answer = json.load(response)
        evidence["proposer"].update({key: answer[key] for key in (
            "model", "created_at", "done_reason", "total_duration", "load_duration",
            "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration") if key in answer})
        evidence["raw_response"] = answer.get("response")
        proposal = json.loads(answer["response"])
        if (not isinstance(proposal, dict) or set(proposal) != {"hypothesis", "code"}
                or not all(isinstance(proposal[key], str) for key in proposal)):
            raise PolicyError("Ollama must return JSON strings hypothesis and code")
        evidence.update(proposal)
        path.write_text(proposal["code"], encoding="utf-8")
        evidence.update({"path": str(path), "source_hash": hashlib.sha256(proposal["code"].encode()).hexdigest()})
        evidence.update(evaluate_policy(proposal["code"], clean))
        evidence["validation_status"] = "passed"
    except (PolicyError, ValueError, TypeError, KeyError, urllib.error.URLError, TimeoutError) as exc:
        if isinstance(exc, PolicyError):
            evidence.update(exc.evidence)
        evidence.update({"validation_status": "failed", "error": str(exc)[:1000], "receipt_path": str(receipt)})
        receipt.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        raise PolicyError(f"Policy proposal failed; evidence saved at {receipt}: {exc}", evidence) from exc
    receipt.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence
