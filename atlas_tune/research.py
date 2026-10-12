"""Research tools the agent can call: GitHub issues, page fetch, HF model card.

Web search comes from the user's own LLM provider (Claude, Gemini, OpenAI, xAI, Perplexity
through LiteLLM), so there is no separate search account. Everything returned here is
untrusted text: it can only become a *candidate* config, which must pass the engine
allowlist and then be measured before it can win. These tools reach public internet
hosts only; they cannot read local files or private network addresses.
"""

from __future__ import annotations

import html
import ipaddress
import os
import re
import socket
from urllib.parse import urljoin, urlsplit

import httpx

ENGINE_REPOS = {"vllm": "vllm-project/vllm", "sglang": "sgl-project/sglang", "llamacpp": "ggml-org/llama.cpp"}
_client = None


def _http() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(timeout=20, follow_redirects=False,
                               headers={"User-Agent": "atlas-tune/0.3 (inference config research)"})
    return _client


def native_web_search(llm: str | None) -> bool:
    """True when the LLM provider offers built-in web search (used via the same API key)."""
    if not llm:
        return False
    try:
        import litellm

        return bool(litellm.supports_web_search(model=llm))
    except Exception:  # noqa: BLE001 - unknown model names just mean "no"
        return False


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


def _public(url: str) -> None:
    """Refuse anything but public internet hosts (no localhost, LAN, link-local or cloud metadata)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("only http(s) URLs")
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    except socket.gaierror as error:
        raise ValueError(f"cannot resolve {parts.hostname}") from error
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            raise ValueError(f"refusing non-public address {address} for {parts.hostname}")


def fetch_page(url: str, max_chars: int = 6000) -> str:
    for _ in range(4):  # follow redirects manually so every hop is checked
        _public(url)
        response = _http().get(url)
        if response.is_redirect:
            url = urljoin(url, response.headers.get("location", ""))
            continue
        break
    else:
        raise ValueError("too many redirects")
    response.raise_for_status()
    text = response.text
    if "html" in response.headers.get("content-type", ""):
        text = re.sub(r"(?is)<(script|style|nav|header|footer).*?</\1>", " ", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    return re.sub(r"\s+", " ", text).strip()[:max_chars]


def model_card(repo: str, max_chars: int = 8000) -> str:
    """Hugging Face model cards often include a recommended `vllm serve` / llama.cpp command."""
    if not re.fullmatch(r"[\w.\-]+/[\w.\-]+", repo):
        raise ValueError("repo must look like org/name")
    return fetch_page(f"https://huggingface.co/{repo}/raw/main/README.md", max_chars)


TOOLS = {"search_github_issues": search_github_issues,
         "fetch_page": fetch_page, "model_card": model_card}

SCHEMAS = [
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
