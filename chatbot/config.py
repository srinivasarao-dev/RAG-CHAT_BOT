"""Central configuration, read from environment variables / .env (and Streamlit secrets if present)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value.strip()
    # Streamlit Cloud stores secrets in st.secrets rather than env vars
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return default


# Groq models known to handle tool calling well. Any other Groq model id can be typed in the UI.
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "moonshotai/kimi-k2-instruct",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
]


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: _get("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _get("GROQ_MODEL", GROQ_MODELS[0]))

    # Storage: Supabase if configured, otherwise a local SQLite file
    supabase_url: str = field(default_factory=lambda: _get("SUPABASE_URL"))
    supabase_key: str = field(default_factory=lambda: _get("SUPABASE_KEY"))
    sqlite_path: str = field(default_factory=lambda: _get("SQLITE_PATH", "chat_history.db"))

    # Optional API keys: each tool falls back to a keyless API or library when its key is missing
    tavily_api_key: str = field(default_factory=lambda: _get("TAVILY_API_KEY"))
    reddit_client_id: str = field(default_factory=lambda: _get("REDDIT_CLIENT_ID"))
    reddit_client_secret: str = field(default_factory=lambda: _get("REDDIT_CLIENT_SECRET"))
    reddit_user_agent: str = field(default_factory=lambda: _get("REDDIT_USER_AGENT", "realtime-chatbot/1.0"))
    http_user_agent: str = field(
        default_factory=lambda: _get("HTTP_USER_AGENT", "RealtimeChatbot/1.0 (https://github.com/realtime-chatbot/realtime-chatbot) httpx")
    )
    newsapi_key: str = field(default_factory=lambda: _get("NEWSAPI_KEY"))
    brave_api_key: str = field(default_factory=lambda: _get("BRAVE_API_KEY"))
    serper_api_key: str = field(default_factory=lambda: _get("SERPER_API_KEY"))
    x_bearer_token: str = field(default_factory=lambda: _get("X_BEARER_TOKEN"))
    bluesky_handle: str = field(default_factory=lambda: _get("BLUESKY_HANDLE"))
    bluesky_app_password: str = field(default_factory=lambda: _get("BLUESKY_APP_PASSWORD"))
    finnhub_api_key: str = field(default_factory=lambda: _get("FINNHUB_API_KEY"))
    github_token: str = field(default_factory=lambda: _get("GITHUB_TOKEN"))

    # How many past messages are sent to the model as conversation memory
    history_window: int = field(default_factory=lambda: int(_get("HISTORY_WINDOW", "20")))
    # Seconds a tool result is reused before fetching again (keeps answers fresh but avoids rate limits)
    tool_cache_ttl: int = field(default_factory=lambda: int(_get("TOOL_CACHE_TTL", "120")))

    @property
    def use_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)


settings = Settings()
