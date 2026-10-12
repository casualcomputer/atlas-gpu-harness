"""Quality floor and streaming load test against any OpenAI-compatible server.

The same client measures every engine, so numbers are comparable across engines.
Throughput = output tokens / wall time across all requests at a concurrency level.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import statistics
import time
from dataclasses import dataclass

import httpx

# Small exact-answer checks: a smoke test that the model, quantization and chat template
# still produce correct answers. Not a general quality eval.
QUALITY_TASKS = (
    ("add", "What is 37 + 58? Return the integer answer.", 95),
    ("multiply", "What is 14 * 6? Return the integer answer.", 84),
    ("subtract", "What is 91 - 46? Return the integer answer.", 45),
    ("extract_city", "Record: name=Mira; city=Oslo; role=engineer. Return the city.", "Oslo"),
    ("extract_id", 'Record: {"id":"AX-204","status":"ready"}. Return the id.', "AX-204"),
    ("code_sum", "Python code: values = [2, 4, 6]; result = sum(values). What is result?", 12),
    ("code_filter", "Python code: result = len([x for x in [1, 2, 3, 4] if x % 2 == 0]). What is result?", 2),
    ("code_index", 'Python code: letters = ["a", "b", "c"]; result = letters[1]. What is result?', "b"),
)
SYSTEM = 'Return only one JSON object with exactly one key, "answer". No explanation or Markdown.'
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


@dataclass(frozen=True)
class Workload:
    name: str
    input_tokens: int
    output_tokens: int
    concurrency: tuple[int, ...]

    def requests_at(self, level: int) -> int:
        return max(2 * level, 8)


WORKLOADS = {
    "smoke": Workload("smoke", 64, 32, (1, 2)),       # minutes, not hours: check a setup works
    "chat": Workload("chat", 256, 256, (1, 4, 16)),
    "rag": Workload("rag", 4000, 256, (1, 4)),
    "batch": Workload("batch", 512, 512, (16, 64)),
}
WORKLOAD_VERSION = "v2"  # bump whenever prompts, checks or scoring change: memory keys on it


def score_answer(content: str, expected) -> bool:
    """Correct answer, as {"answer": ...} JSON or as a bare value. Formatting is not judged:
    the point is to catch a config that breaks the model (bad template, quantization, KV)."""
    content = re.sub(r"<think>.*?</think>", "", content or "", flags=re.DOTALL).strip()
    content = re.sub(r"^```\w*|```$", "", content.strip()).strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    value = content
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "answer" in parsed:
                value = parsed["answer"]
        except (ValueError, TypeError):
            pass
    text = str(value).strip().strip('"\'` .').strip()
    return text == str(expected) or text.lower() == str(expected).lower()


def quality(base_url: str, model: str, timeout: float = 120) -> dict:
    rows = []
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        for name, prompt, expected in QUALITY_TASKS:
            body = {"model": model, "temperature": 0, "max_tokens": 256, **NO_THINK,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
            try:
                response = client.post(base_url + "/v1/chat/completions", json=body)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"].get("content") or ""
                rows.append({"task": name, "passed": score_answer(content, expected), "output": content[:300]})
            except (httpx.HTTPError, ValueError, KeyError, IndexError) as error:
                rows.append({"task": name, "passed": False, "error": f"{type(error).__name__}: {error}"[:300]})
    passed = sum(r["passed"] for r in rows)
    # "pass" here only means the server answers sensibly at all. Whether a config degraded the
    # model is judged against the best config of the same model (see quality_ok).
    return {"passed": passed, "total": len(rows), "pass": passed >= 2, "tasks": rows}


def quality_ok(trial: dict, reference_passed: int, tolerance: int = 1) -> bool:
    """A config may win only if it answers about as well as the best config of this model."""
    q = trial.get("quality") or {}
    return bool(q.get("pass")) and q.get("passed", 0) >= reference_passed - tolerance


def _prompt(tokens: int, seed: int) -> str:
    rng = random.Random(seed)
    words = ["system", "latency", "kernel", "memory", "cache", "request", "router", "expert", "batch",
             "token", "queue", "server", "region", "replica", "budget", "context", "stream", "shard"]
    filler = " ".join(rng.choice(words) for _ in range(int(tokens * 0.75)))
    return (f"Notes:\n{filler}\n\nWrite a long, detailed numbered list of operational recommendations "
            "based on these notes. Keep going until you run out of space.")


async def _one(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int) -> dict:
    """One streamed request. A transport error before any byte arrives is retried once and
    counted: that is the client/network, not the engine. Errors after streaming starts count."""
    result = await _attempt(client, url, model, prompt, max_tokens)
    if not result["ok"] and result.get("before_first_byte"):
        result = await _attempt(client, url, model, prompt, max_tokens)
        result["retried"] = True
    return result


async def _attempt(client: httpx.AsyncClient, url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = {"model": model, "temperature": 0.7, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, **NO_THINK,
            "messages": [{"role": "user", "content": prompt}]}
    start = time.perf_counter()
    first = None
    chunks = 0
    usage_tokens = None
    try:
        async with client.stream("POST", url, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:") or line.strip() == "data: [DONE]":
                    continue
                event = json.loads(line[5:])
                if event.get("usage") and event["usage"].get("completion_tokens"):
                    usage_tokens = event["usage"]["completion_tokens"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    if delta.get("content") or delta.get("reasoning_content"):
                        chunks += 1
                        first = first or time.perf_counter()
    except (httpx.HTTPError, ValueError) as error:
        transport = isinstance(error, (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError))
        return {"ok": False, "error": f"{type(error).__name__}: {error}"[:200],
                "before_first_byte": transport and first is None}
    end = time.perf_counter()
    tokens = usage_tokens or chunks
    if not first or tokens < 2:
        return {"ok": False, "error": "no streamed output"}
    return {"ok": True, "ttft": first - start, "total": end - start, "tokens": tokens,
            "tpot": (end - first) / (tokens - 1)}


def _pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


async def _level(base_url: str, model: str, workload: Workload, level: int, timeout: float) -> dict:
    url = base_url + "/v1/chat/completions"
    count = workload.requests_at(level)
    gate = asyncio.Semaphore(level)
    # No connection reuse: with queued requests, reusing a keep-alive connection the server is
    # closing made ~10% of requests fail against real llama.cpp. Localhost connects are cheap.
    limits = httpx.Limits(max_connections=level, max_keepalive_connections=0)
    async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
        async def run(i):
            async with gate:
                return await _one(client, url, model, _prompt(workload.input_tokens, 1000 + i), workload.output_tokens)

        start = time.perf_counter()
        results = await asyncio.gather(*(run(i) for i in range(count)))
        wall = time.perf_counter() - start
    ok = [r for r in results if r["ok"]]
    if not ok:
        return {"concurrency": level, "requests": count, "ok": 0, "errors": [r["error"] for r in results][:3]}
    ttft = [r["ttft"] for r in ok]
    tpot = [r["tpot"] for r in ok]
    return {"concurrency": level, "requests": count, "ok": len(ok),
            "output_tok_s": round(sum(r["tokens"] for r in ok) / wall, 1),
            "requests_s": round(len(ok) / wall, 3),
            "ttft_p50_ms": round(statistics.median(ttft) * 1000), "ttft_p99_ms": round(_pct(ttft, 0.99) * 1000),
            "tpot_p50_ms": round(statistics.median(tpot) * 1000, 1),
            "per_user_tok_s": round(1 / statistics.median(tpot), 1),
            "retries": sum(1 for r in results if r.get("retried")),
            "errors": [r["error"] for r in results if not r["ok"]][:3]}


def load(base_url: str, model: str, workload: Workload, timeout: float = 600) -> dict:
    asyncio.run(_level(base_url, model, Workload("warmup", 32, 16, (1,)), 1, timeout))
    levels = [asyncio.run(_level(base_url, model, workload, c, timeout)) for c in workload.concurrency]
    return {"workload": workload.name, "levels": levels}


def score(metrics: dict | None, goal: str) -> float | None:
    """One number to rank trials: throughput = best total tok/s at any level where every
    request succeeded; latency = single-user decode tok/s at the lowest concurrency."""
    if not metrics:
        return None
    clean = [lv for lv in metrics.get("levels", []) if lv.get("ok") == lv.get("requests") and lv.get("ok")]
    if not clean:
        return None
    if goal == "latency":
        return min(clean, key=lambda lv: lv["concurrency"])["per_user_tok_s"]
    return max(lv["output_tok_s"] for lv in clean)
