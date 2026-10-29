"""Central configuration, read from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value else default


# Fallback list, in order of preference, used when Groq's live model list can't be fetched.
# Groq retires models regularly, so the server asks Groq which models are available at startup.
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

# Groq models that aren't general chat models (speech, safety classifiers, etc.)
_NON_CHAT = ("whisper", "guard", "safeguard", "orpheus", "tts", "allam", "playai")


def fetch_groq_models(api_key: str) -> list[str]:
    """Chat models currently available to this key, best first. Falls back to GROQ_MODELS."""
    if not api_key:
        return GROQ_MODELS
    try:
        import httpx

        data = httpx.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        ).json()["data"]
        ids = [m["id"] for m in data if m.get("active", True) and not any(x in m["id"] for x in _NON_CHAT)]
        preferred = [m for m in GROQ_MODELS if m in ids]
        return preferred + sorted(m for m in ids if m not in preferred) or GROQ_MODELS
    except Exception:
        return GROQ_MODELS


@dataclass
class Settings:
    groq_api_key: str = field(default_factory=lambda: _get("GROQ_API_KEY"))
    groq_model: str = field(default_factory=lambda: _get("GROQ_MODEL", GROQ_MODELS[0]))
    # Optional extra providers. Users can also enter their own keys in the web UI's Settings.
    anthropic_api_key: str = field(default_factory=lambda: _get("ANTHROPIC_API_KEY"))
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY"))

    # Code mode: run Python snippets in a subprocess on THIS machine. Defaults to on only when the
    # server listens on localhost; never enable it on a public deployment.
    code_execution: bool = field(
        default_factory=lambda: _get("CODE_EXECUTION", "auto").lower() in ("1", "true", "on", "yes")
        or (_get("CODE_EXECUTION", "auto").lower() == "auto" and _get("HOST", "127.0.0.1") in ("127.0.0.1", "localhost"))
    )

    # Storage: Supabase if configured, otherwise a local SQLite file
    supabase_url: str = field(default_factory=lambda: _get("SUPABASE_URL"))
    supabase_key: str = field(default_factory=lambda: _get("SUPABASE_KEY"))
    # Public "anon"/publishable key: used by the browser for sign-in. Setting it turns on login.
    supabase_anon_key: str = field(default_factory=lambda: _get("SUPABASE_ANON_KEY"))
    # AUTH: auto (on when SUPABASE_ANON_KEY is set) | on | off
    auth_mode: str = field(default_factory=lambda: _get("AUTH", "auto").lower())
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

    def __post_init__(self) -> None:
        # Accept a bare project ref ("abcd1234...") as well as the full https://<ref>.supabase.co URL
        url = self.supabase_url.rstrip("/")
        if url and not url.startswith("http"):
            url = f"https://{url}" if "." in url else f"https://{url}.supabase.co"
        self.supabase_url = url

    @property
    def auth_required(self) -> bool:
        if self.auth_mode in ("off", "0", "false", "no"):
            return False
        has_keys = bool(self.supabase_url and self.supabase_anon_key)
        return has_keys if self.auth_mode == "auto" else has_keys and self.auth_mode in ("on", "1", "true", "yes")

    @property
    def use_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_key)


settings = Settings()
