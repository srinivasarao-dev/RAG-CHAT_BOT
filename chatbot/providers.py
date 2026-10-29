"""LLM providers: Groq (default), Anthropic Claude and OpenAI (GPT / Codex).

Keys come from the server's .env or are supplied per request by the user (bring your own key).
Model lists are fetched live from each provider, because providers retire and add models often.
A model is addressed as "<provider>:<model id>", e.g. "anthropic:claude-opus-5-5".
"""
from __future__ import annotations

import hashlib
import re
import time
from typing import Any

from langchain_core.language_models import BaseChatModel

from chatbot.config import GROQ_MODELS, fetch_groq_models, settings

PROVIDERS = {
    "groq": {"name": "Groq", "hint": "Free tier · fast open models", "key_url": "https://console.groq.com/keys"},
    "anthropic": {"name": "Claude (Anthropic)", "hint": "Claude Opus / Sonnet / Haiku", "key_url": "https://console.anthropic.com/settings/keys"},
    "openai": {"name": "OpenAI (GPT & Codex)", "hint": "GPT and Codex coding models", "key_url": "https://platform.openai.com/api-keys"},
}

# Preferred order when the provider returns these models (anything else the key can use is listed after)
_PREFERRED = {
    "anthropic": ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-fable-5-1"],
    "openai": [],  # ordered by _openai_rank instead: Codex models first, then newest GPT
}
# Claude models whose API accepts the server-side refusal fallback ("default" routing)
_CLAUDE_FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
# OpenAI model ids that aren't chat/coding models
_OPENAI_SKIP = ("audio", "realtime", "tts", "transcribe", "whisper", "embedding", "moderation", "dall-e",
                "image", "search", "davinci", "babbage", "instruct", "computer-use", "deep-research")


def server_key(provider: str) -> str:
    return {"groq": settings.groq_api_key, "anthropic": settings.anthropic_api_key, "openai": settings.openai_api_key}.get(provider, "")


def resolve_key(provider: str, user_keys: dict[str, str] | None) -> str:
    """The server's key wins; otherwise the key the user entered in Settings."""
    return server_key(provider) or ((user_keys or {}).get(provider) or "").strip()


def split_model(value: str, default_provider: str = "groq") -> tuple[str, str]:
    provider, _, model = value.partition(":")
    if provider in PROVIDERS and model:
        return provider, model
    return default_provider, value  # legacy ids without a provider prefix are Groq models


# ---------------------------------------------------------------- model lists

_model_cache: dict[str, tuple[float, list[str]]] = {}


def _openai_rank(model_id: str) -> tuple:
    codex = "codex" in model_id
    version = [int(x) for x in re.findall(r"\d+", model_id)[:2]] or [0]
    small = any(s in model_id for s in ("mini", "nano"))
    return (not codex, small, [-v for v in version], model_id)


def list_models(provider: str, api_key: str) -> list[str]:
    """Models this key can use, best first. Cached for 10 minutes per key."""
    if provider == "groq":
        return fetch_groq_models(api_key) if api_key else GROQ_MODELS
    if not api_key:
        return []
    cache_key = provider + hashlib.sha256(api_key.encode()).hexdigest()
    hit = _model_cache.get(cache_key)
    if hit and hit[0] > time.time():
        return hit[1]

    if provider == "anthropic":
        import anthropic

        ids = [m.id for m in anthropic.Anthropic(api_key=api_key, max_retries=1).models.list(limit=100)]
        preferred = [m for m in _PREFERRED["anthropic"] if m in ids]
        models = preferred + [m for m in ids if m not in preferred]
    elif provider == "openai":
        import openai

        ids = [m.id for m in openai.OpenAI(api_key=api_key, max_retries=1).models.list()]
        chat = [m for m in ids if (m.startswith(("gpt-", "o1", "o3", "o4", "o5", "chatgpt")) or "codex" in m)
                and not any(s in m for s in _OPENAI_SKIP)
                and not re.search(r"-\d{4}-\d{2}-\d{2}$", m)]  # skip dated snapshots of the same model
        models = sorted(chat, key=_openai_rank)
    else:
        raise ValueError(f"Unknown provider: {provider}")

    _model_cache[cache_key] = (time.time() + 600, models)
    return models


# ---------------------------------------------------------------- chat models

def build_llm(provider: str, model: str, api_key: str, mode: str = "chat") -> BaseChatModel:
    """A streaming chat model with sensible per-provider settings. mode: "chat" or "code"."""
    if provider == "groq":
        from langchain_groq import ChatGroq

        kwargs: dict[str, Any] = {}
        if model.startswith("qwen/") or "deepseek-r1" in model:
            kwargs["reasoning_format"] = "hidden"  # don't stream <think> blocks to the user
        if model.startswith("openai/gpt-oss"):
            # fewer hidden reasoning tokens: faster and kinder to free-tier limits; code needs a bit more
            kwargs["reasoning_effort"] = "medium" if mode == "code" else "low"
        # No client-side retries: on a rate limit the server switches to another Groq model instead
        return ChatGroq(api_key=api_key, model=model, temperature=0.2, streaming=True, max_retries=0, **kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {}
        # Models before the 4.6 generation don't support adaptive thinking / effort
        legacy = re.search(r"claude-(3|haiku-4|sonnet-4(-[015])?\b|opus-4(-[015])?\b)", model)
        if not legacy:
            # Current Claude models: adaptive thinking, depth controlled by effort. No temperature:
            # sampling parameters are rejected on current models. Coding responds well to higher effort.
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": "high" if mode == "code" else "medium"}
        if model in _CLAUDE_FALLBACK_MODELS:
            # If a safety classifier declines, the API reroutes the request to a suitable model in the same call
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["model_kwargs"] = {"fallbacks": "default"}
        return ChatAnthropic(api_key=api_key, model=model, max_tokens=64000, streaming=True, max_retries=2, **kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        reasoning = "codex" in model or model.startswith(("gpt-5", "o1", "o3", "o4", "o5"))
        kwargs = {}
        if reasoning:
            # Codex models are served through the Responses API; reasoning models take effort, not temperature
            kwargs["use_responses_api"] = True
            kwargs["reasoning"] = {"effort": "medium" if mode == "code" else "low"}
        else:
            kwargs["temperature"] = 0.2
        return ChatOpenAI(api_key=api_key, model=model, streaming=True, max_retries=2, **kwargs)

    raise ValueError(f"Unknown provider: {provider}")


def default_model(provider: str, models: list[str]) -> str:
    if provider == "groq" and settings.groq_model in models:
        return settings.groq_model
    return models[0] if models else ""
