"""Optional login with Supabase Auth.

When SUPABASE_ANON_KEY is set (and AUTH isn't "off"), every API call must carry the user's Supabase access
token ("Authorization: Bearer <jwt>"). The token is verified by asking Supabase Auth, which works with
every Supabase signing-key setup, and the result is cached for a few minutes. The verified user id then
owns that user's chats and files, so people can't read each other's data.

Without login (local use), the "user id" is the name typed in Settings, which is fine on your own computer.
"""
from __future__ import annotations

import hashlib
import threading
import time

from fastapi import HTTPException, Request

from chatbot.config import settings

_cache: dict[str, tuple[float, dict]] = {}
_lock = threading.Lock()
_client = None
CACHE_SECONDS = 300


def auth_enabled() -> bool:
    return settings.auth_required


def _supabase():
    global _client
    if _client is None:
        from supabase import create_client

        _client = create_client(settings.supabase_url, settings.supabase_anon_key or settings.supabase_key)
    return _client


def verify_token(token: str) -> dict:
    """{'id', 'email'} for a valid Supabase access token, else raises HTTPException(401)."""
    key = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    try:
        user = _supabase().auth.get_user(token).user
    except Exception:
        user = None
    if not user:
        raise HTTPException(401, "Your session has expired. Please sign in again.")
    info = {"id": user.id, "email": user.email or ""}
    with _lock:
        if len(_cache) > 5000:
            _cache.clear()
        _cache[key] = (now + CACHE_SECONDS, info)
    return info


def current_user_id(request: Request, claimed: str | None = None) -> str:
    """The user id for this request: the verified Supabase user when login is on, else the claimed name."""
    if not auth_enabled():
        return (claimed or "default").strip()[:120] or "default"
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise HTTPException(401, "Please sign in.")
    return verify_token(header[7:].strip())["id"]
