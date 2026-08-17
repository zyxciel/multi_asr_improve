"""OpenAI-compatible chat client (vLLM / vLLM-Ascend / MindIE HTTP).

Uses stdlib only — no CUDA/NPU binding in-process. Point --llm-base-url at an
Ascend 910B serving endpoint (e.g. vLLM-Ascend) that exposes /v1/chat/completions.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


def normalize_openai_base_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/v1"):
        return url
    return url + "/v1"


def chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 1024,
    api_key: str | None = None,
    timeout_s: float = 300.0,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> str:
    """POST /v1/chat/completions; return assistant message content."""
    root = normalize_openai_base_url(base_url)
    endpoint = f"{root}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"OpenAI-compat HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"OpenAI-compat connection failed ({endpoint}): {e}") from e

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError(f"OpenAI-compat empty choices: {str(body)[:300]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if content is None:
        raise RuntimeError(f"OpenAI-compat missing message.content: {str(body)[:300]}")
    if isinstance(content, list):
        # Some servers return content parts
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def chat_completion_many(
    jobs: list[dict[str, Any]],
    *,
    max_workers: int = 8,
) -> list[str | BaseException]:
    """
    Run many chat_completion calls concurrently (server-side continuous batching).

    Each job is kwargs for chat_completion. Returns results in the same order as jobs;
    failed items are Exception instances.
    """
    if not jobs:
        return []
    workers = max(1, min(int(max_workers), len(jobs)))
    results: list[str | BaseException] = [RuntimeError("unset")] * len(jobs)

    def _one(idx: int, kwargs: dict[str, Any]) -> tuple[int, str | BaseException]:
        try:
            return idx, chat_completion(**kwargs)
        except BaseException as exc:  # noqa: BLE001
            return idx, exc

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_one, i, job) for i, job in enumerate(jobs)]
        for fut in as_completed(futs):
            idx, value = fut.result()
            results[idx] = value
    return results
