"""Research tools the agent can call: web search, GitHub issues, page fetch, HF model card.

Everything returned here is untrusted text. It only ever becomes a *candidate* config,
which must pass the engine allowlist and then be measured before it can win.
"""

from __future__ import annotations

import html
import os
import re
from urllib.parse import urlsplit

import httpx

ENGINE_REPOS = {"vllm": "vllm-project/vllm", "sglang": "sgl-project/sglang", "llamacpp": "ggml-org/llama.cpp"}
_client = None


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=20, follow_redirects=True,
                               headers={"User-Agent": "atlas-tune/0.3 (inference config research)"})
    return _client


def available() -> dict:
    """GitHub, page fetch and model cards need no key; web search is the only optional one."""
    return {"web_search": bool(os.environ.get("TAVILY_API_KEY") or os.environ.get("BRAVE_API_KEY")),
            "github": True, "fetch": True, "model_card": True}


ERROR_LINE = re.compile(r"(out of memory|OutOfMemory|CUDA error|RuntimeError|ValueError|AssertionError|"
                        r"NotImplementedError|not supported|unsupported|error:|Error:|failed to)", re.IGNORECASE)


def error_signature(logs: str | None, fallback: str = "") -> str:
    """The most specific error line in a log tail, cleaned of paths and numbers, for issue search."""
    lines = [ln.strip() for ln in (logs or "").splitlines() if ERROR_LINE.search(ln)]
    if not lines:
        return ""  # our own messages ("not ready after …") are not engine bugs worth searching
    line = lines[-1]
    line = re.sub(r"(/[\w.\-]+)+|0x[0-9a-f]+|\b\d+(\.\d+)?\s*(GiB|MiB|MB|GB)?\b", " ", line)
    line = re.sub(r"^[^A-Za-z]+|[\[\]():'\"`]", " ", line)
    words = [w for w in line.split() if len(w) > 2][:8]
    return " ".join(words)


def related_issues(engine: str, logs: str | None, error: str = "", limit: int = 3) -> list[dict]:
    """Best-effort: GitHub issues matching a failed trial's error. Never raises."""
    signature = error_signature(logs, error)
    if not signature:
        return []
    try:
        return [{"title": i["title"], "url": i["url"], "state": i["state"]}
                for i in search_github_issues(engine, f"is:issue {signature}", max_results=limit)]
    except Exception:  # noqa: BLE001 - rate limits or offline must not affect the run
        return []


def search_web(query: str, max_results: int = 6) -> list[dict]:
    """Tavily if TAVILY_API_KEY is set, else Brave if BRAVE_API_KEY is set."""
    if key := os.environ.get("TAVILY_API_KEY"):
        response = _http().post("https://api.tavily.com/search",
                                json={"api_key": key, "query": query, "max_results": max_results})
        response.raise_for_status()
        return [{"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("content") or "")[:500]}
                for r in response.json().get("results", [])]
    if key := os.environ.get("BRAVE_API_KEY"):
        response = _http().get("https://api.search.brave.com/res/v1/web/search",
                               params={"q": query, "count": max_results},
                               headers={"X-Subscription-Token": key, "Accept": "application/json"})
        response.raise_for_status()
        return [{"title": r.get("title"), "url": r.get("url"), "snippet": (r.get("description") or "")[:500]}
                for r in response.json().get("web", {}).get("results", [])]
    raise RuntimeError("no web search key: set TAVILY_API_KEY or BRAVE_API_KEY")


def search_github_issues(engine: str, query: str, max_results: int = 6) -> list[dict]:
    """Issues and discussions in the engine's repo, e.g. a model name + 'OOM' or 'not supported'."""
    repo = ENGINE_REPOS.get(engine, engine)
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    response = _http().get("https://api.github.com/search/issues",
                           params={"q": f"repo:{repo} {query}", "per_page": max_results},
                           headers=headers)
    response.raise_for_status()
    return [{"title": i["title"], "url": i["html_url"], "state": i["state"],
             "snippet": re.sub(r"\s+", " ", i.get("body") or "")[:500]}
            for i in response.json().get("items", [])]


def fetch_page(url: str, max_chars: int = 6000) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.hostname in ("localhost", "127.0.0.1"):
        raise ValueError("only public http(s) URLs")
    response = _http().get(url)
    response.raise_for_status()
    text = response.text
    if "html" in response.headers.get("content-type", ""):
        text = re.sub(r"(?is)<(script|style|nav|header|footer).*?</\1>", " ", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def model_card(repo: str, max_chars: int = 8000) -> str:
    """Hugging Face model cards often include a recommended `vllm serve` / llama.cpp command."""
    return fetch_page(f"https://huggingface.co/{repo}/raw/main/README.md", max_chars)


TOOLS = {"search_web": search_web, "search_github_issues": search_github_issues,
         "fetch_page": fetch_page, "model_card": model_card}

SCHEMAS = [
    {"type": "function", "function": {"name": "search_web", "description":
        "Web search (blogs, forum threads, recipes) for serving configs of this model/GPU. Needs a search key.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "search_github_issues", "description":
        "Search issues in an engine's GitHub repo for known bugs, OOMs or required flags for this model.",
        "parameters": {"type": "object", "properties": {
            "engine": {"type": "string", "enum": list(ENGINE_REPOS)}, "query": {"type": "string"}},
            "required": ["engine", "query"]}}},
    {"type": "function", "function": {"name": "fetch_page", "description": "Read one public web page as text.",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}}},
    {"type": "function", "function": {"name": "model_card", "description":
        "Read a Hugging Face model card (README) for recommended serving commands.",
        "parameters": {"type": "object", "properties": {"repo": {"type": "string"}}, "required": ["repo"]}}},
]
