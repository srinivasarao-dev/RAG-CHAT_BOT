"""Shared helpers for tools: HTTP client, TTL cache, output trimming."""
from __future__ import annotations

import functools
import json
import threading
import time
from typing import Any, Callable

import httpx

from chatbot.config import settings

# Identify the bot honestly; Wikipedia and others require a descriptive UA with contact info
USER_AGENT = settings.http_user_agent
MAX_TOOL_OUTPUT = 6000  # characters; keeps the LLM context small and fast

_client = httpx.Client(
    timeout=httpx.Timeout(15.0, connect=5.0),
    headers={"User-Agent": USER_AGENT},
    follow_redirects=True,
)


def http_get(url: str, **kwargs: Any) -> httpx.Response:
    resp = _client.get(url, **kwargs)
    resp.raise_for_status()
    return resp


def http_post(url: str, **kwargs: Any) -> httpx.Response:
    resp = _client.post(url, **kwargs)
    resp.raise_for_status()
    return resp


def trim(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "\n...[truncated]"


def to_text(data: Any) -> str:
    if isinstance(data, str):
        return trim(data)
    return trim(json.dumps(data, ensure_ascii=False, indent=1, default=str))


_cache: dict[str, tuple[float, str]] = {}
_cache_lock = threading.Lock()


def cached(ttl: int | None = None) -> Callable:
    """Cache a tool function's string result for `ttl` seconds, keyed by its arguments."""

    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> str:
            key = f"{fn.__module__}.{fn.__name__}:{json.dumps([args, kwargs], sort_keys=True, default=str)}"
            now = time.time()
            with _cache_lock:
                hit = _cache.get(key)
                if hit and hit[0] > now:
                    return hit[1]
            result = fn(*args, **kwargs)
            # Don't cache failures so the next call retries
            if not result.startswith("Error"):
                with _cache_lock:
                    _cache[key] = (now + (ttl if ttl is not None else settings.tool_cache_ttl), result)
            return result

        return wrapper

    return decorator


def safe(fn: Callable[..., str]) -> Callable[..., str]:
    """Turn exceptions into an error string so one failing source never crashes the agent."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            return f"Error from {fn.__name__}: {type(e).__name__}: {e}"

    return wrapper
