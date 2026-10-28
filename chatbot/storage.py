"""Chat persistence: Supabase (Postgres) when configured, otherwise a local SQLite file.

Both stores expose the same methods so the app doesn't care which one is active.
Run supabase/schema.sql in the Supabase SQL editor before first use.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

from chatbot.config import Settings, settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore(Protocol):
    backend: str

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]: ...
    def create_conversation(self, user_id: str, title: str) -> str: ...
    def rename_conversation(self, conversation_id: str, title: str) -> None: ...
    def delete_conversation(self, conversation_id: str) -> None: ...
    def get_messages(self, conversation_id: str) -> list[dict]: ...
    def add_message(self, conversation_id: str, role: str, content: str, metadata: dict | None = None) -> None: ...
    def log_tool_call(self, conversation_id: str, tool: str, args: dict, output: str, latency_ms: int) -> None: ...


class SupabaseStore:
    backend = "supabase"

    def __init__(self, url: str, key: str):
        from supabase import create_client

        self.db = create_client(url, key)

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]:
        res = (
            self.db.table("conversations")
            .select("id,title,updated_at")
            .eq("user_id", user_id)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return res.data or []

    def create_conversation(self, user_id: str, title: str) -> str:
        res = self.db.table("conversations").insert({"user_id": user_id, "title": title[:120]}).execute()
        return res.data[0]["id"]

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        self.db.table("conversations").update({"title": title[:120]}).eq("id", conversation_id).execute()

    def delete_conversation(self, conversation_id: str) -> None:
        self.db.table("conversations").delete().eq("id", conversation_id).execute()

    def get_messages(self, conversation_id: str) -> list[dict]:
        res = (
            self.db.table("messages")
            .select("role,content,metadata,created_at")
            .eq("conversation_id", conversation_id)
            .order("id")
            .execute()
        )
        return res.data or []

    def add_message(self, conversation_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        self.db.table("messages").insert(
            {"conversation_id": conversation_id, "role": role, "content": content, "metadata": metadata or {}}
        ).execute()
        self.db.table("conversations").update({"updated_at": _now()}).eq("id", conversation_id).execute()

    def log_tool_call(self, conversation_id: str, tool: str, args: dict, output: str, latency_ms: int) -> None:
        self.db.table("tool_calls").insert(
            {
                "conversation_id": conversation_id,
                "tool": tool,
                "args": args,
                "output_preview": output[:2000],
                "latency_ms": latency_ms,
            }
        ).execute()


class SQLiteStore:
    backend = "sqlite"

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, title TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS tool_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
                    tool TEXT NOT NULL, args TEXT, output_preview TEXT, latency_ms INTEGER, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);
                PRAGMA foreign_keys = ON;
                """
            )

    def _exec(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.fetchall()

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]:
        rows = self._exec(
            "SELECT id, title, updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        return [dict(r) for r in rows]

    def create_conversation(self, user_id: str, title: str) -> str:
        cid, now = str(uuid.uuid4()), _now()
        self._exec("INSERT INTO conversations VALUES (?,?,?,?,?)", (cid, user_id, title[:120], now, now))
        return cid

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        self._exec("UPDATE conversations SET title=? WHERE id=?", (title[:120], conversation_id))

    def delete_conversation(self, conversation_id: str) -> None:
        self._exec("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        self._exec("DELETE FROM tool_calls WHERE conversation_id=?", (conversation_id,))
        self._exec("DELETE FROM conversations WHERE id=?", (conversation_id,))

    def get_messages(self, conversation_id: str) -> list[dict]:
        rows = self._exec(
            "SELECT role, content, metadata, created_at FROM messages WHERE conversation_id=? ORDER BY id",
            (conversation_id,),
        )
        return [{**dict(r), "metadata": json.loads(r["metadata"] or "{}")} for r in rows]

    def add_message(self, conversation_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        now = _now()
        self._exec(
            "INSERT INTO messages (conversation_id, role, content, metadata, created_at) VALUES (?,?,?,?,?)",
            (conversation_id, role, content, json.dumps(metadata or {}), now),
        )
        self._exec("UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id))

    def log_tool_call(self, conversation_id: str, tool: str, args: dict, output: str, latency_ms: int) -> None:
        self._exec(
            "INSERT INTO tool_calls (conversation_id, tool, args, output_preview, latency_ms, created_at) VALUES (?,?,?,?,?,?)",
            (conversation_id, tool, json.dumps(args), output[:2000], latency_ms, _now()),
        )


def get_store(cfg: Settings = settings) -> tuple[Any, str | None]:
    """Return (store, warning). Falls back to SQLite if Supabase is missing or unreachable."""
    if cfg.use_supabase:
        try:
            store = SupabaseStore(cfg.supabase_url, cfg.supabase_key)
            store.db.table("conversations").select("id").limit(1).execute()  # verify schema exists
            return store, None
        except Exception as e:  # noqa: BLE001
            return SQLiteStore(cfg.sqlite_path), f"Supabase unavailable ({e}); using local SQLite instead."
    return SQLiteStore(cfg.sqlite_path), None
