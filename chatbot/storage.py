"""Chat persistence: Supabase (Postgres + pgvector) when configured, otherwise a local SQLite file.

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

import numpy as np

from chatbot.config import Settings, settings
from chatbot.documents import top_k


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChatStore(Protocol):
    backend: str

    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]: ...
    def conversation_owner(self, conversation_id: str) -> str | None: ...
    def create_conversation(self, user_id: str, title: str) -> str: ...
    def rename_conversation(self, conversation_id: str, title: str) -> None: ...
    def delete_conversation(self, conversation_id: str) -> None: ...
    def get_messages(self, conversation_id: str) -> list[dict]: ...
    def add_message(self, conversation_id: str, role: str, content: str, metadata: dict | None = None) -> None: ...
    def log_tool_call(self, conversation_id: str, tool: str, args: dict, output: str, latency_ms: int) -> None: ...
    def add_file(self, user_id: str, name: str, mime: str, size: int, text: str, chunks: list[str], embeddings: np.ndarray) -> dict: ...
    def attach_files(self, file_ids: list[str], conversation_id: str, user_id: str) -> None: ...
    def list_files(self, conversation_id: str) -> list[dict]: ...
    def get_file_texts(self, file_ids: list[str]) -> dict[str, str]: ...
    def search_chunks(self, file_ids: list[str], query_vec: np.ndarray, k: int = 6) -> list[dict]: ...
    def delete_file(self, file_id: str, user_id: str) -> None: ...


class SupabaseStore:
    backend = "supabase"

    def __init__(self, url: str, key: str):
        from supabase import create_client

        self.db = create_client(url, key)

    # ---- conversations
    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]:
        res = (
            self.db.table("conversations").select("id,title,updated_at")
            .eq("user_id", user_id).order("updated_at", desc=True).limit(limit).execute()
        )
        return res.data or []

    def conversation_owner(self, conversation_id: str) -> str | None:
        res = self.db.table("conversations").select("user_id").eq("id", conversation_id).limit(1).execute()
        return res.data[0]["user_id"] if res.data else None

    def create_conversation(self, user_id: str, title: str) -> str:
        res = self.db.table("conversations").insert({"user_id": user_id, "title": title[:120]}).execute()
        return res.data[0]["id"]

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        self.db.table("conversations").update({"title": title[:120]}).eq("id", conversation_id).execute()

    def delete_conversation(self, conversation_id: str) -> None:
        # messages, tool_calls, files and chunks are removed by ON DELETE CASCADE
        self.db.table("conversations").delete().eq("id", conversation_id).execute()

    def get_messages(self, conversation_id: str) -> list[dict]:
        res = (
            self.db.table("messages").select("role,content,metadata,created_at")
            .eq("conversation_id", conversation_id).order("id").execute()
        )
        return res.data or []

    def add_message(self, conversation_id: str, role: str, content: str, metadata: dict | None = None) -> None:
        self.db.table("messages").insert(
            {"conversation_id": conversation_id, "role": role, "content": content, "metadata": metadata or {}}
        ).execute()
        self.db.table("conversations").update({"updated_at": _now()}).eq("id", conversation_id).execute()

    def log_tool_call(self, conversation_id: str, tool: str, args: dict, output: str, latency_ms: int) -> None:
        self.db.table("tool_calls").insert(
            {"conversation_id": conversation_id, "tool": tool, "args": args,
             "output_preview": output[:2000], "latency_ms": latency_ms}
        ).execute()

    # ---- files
    def add_file(self, user_id, name, mime, size, text, chunks, embeddings) -> dict:
        row = self.db.table("files").insert(
            {"user_id": user_id, "name": name[:255], "mime": mime, "size": size, "content": text, "n_chunks": len(chunks)}
        ).execute().data[0]
        batch = [
            {"file_id": row["id"], "chunk_index": i, "content": c, "embedding": "[" + ",".join(f"{x:.6f}" for x in e) + "]"}
            for i, (c, e) in enumerate(zip(chunks, embeddings))
        ]
        for start in range(0, len(batch), 200):
            self.db.table("file_chunks").insert(batch[start:start + 200]).execute()
        return {k: row[k] for k in ("id", "name", "mime", "size", "n_chunks", "created_at")}

    def attach_files(self, file_ids, conversation_id, user_id) -> None:
        if file_ids:
            self.db.table("files").update({"conversation_id": conversation_id}).in_("id", file_ids).eq("user_id", user_id).execute()

    def list_files(self, conversation_id) -> list[dict]:
        res = (
            self.db.table("files").select("id,name,mime,size,n_chunks,created_at")
            .eq("conversation_id", conversation_id).order("created_at").execute()
        )
        return res.data or []

    def get_file_texts(self, file_ids) -> dict[str, str]:
        if not file_ids:
            return {}
        res = self.db.table("files").select("id,content").in_("id", file_ids).execute()
        return {r["id"]: r["content"] for r in res.data or []}

    def search_chunks(self, file_ids, query_vec, k: int = 6) -> list[dict]:
        if not file_ids:
            return []
        res = self.db.rpc(
            "match_file_chunks",
            {"query_embedding": "[" + ",".join(f"{x:.6f}" for x in query_vec) + "]", "file_ids": file_ids, "match_count": k},
        ).execute()
        return res.data or []

    def delete_file(self, file_id, user_id) -> None:
        self.db.table("files").delete().eq("id", file_id).eq("user_id", user_id).execute()


class SQLiteStore:
    backend = "sqlite"

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.executescript(
                """
                PRAGMA foreign_keys = ON;
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
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conversation_id TEXT,
                    name TEXT NOT NULL, mime TEXT, size INTEGER, content TEXT, n_chunks INTEGER, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS file_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL, content TEXT NOT NULL, embedding BLOB NOT NULL);
                CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id, updated_at);
                CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);
                CREATE INDEX IF NOT EXISTS idx_files_conv ON files(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_file ON file_chunks(file_id);
                """
            )

    def _exec(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self.lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur.fetchall()

    # ---- conversations
    def list_conversations(self, user_id: str, limit: int = 50) -> list[dict]:
        rows = self._exec(
            "SELECT id, title, updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        return [dict(r) for r in rows]

    def conversation_owner(self, conversation_id: str) -> str | None:
        rows = self._exec("SELECT user_id FROM conversations WHERE id=?", (conversation_id,))
        return rows[0]["user_id"] if rows else None

    def create_conversation(self, user_id: str, title: str) -> str:
        cid, now = str(uuid.uuid4()), _now()
        self._exec("INSERT INTO conversations VALUES (?,?,?,?,?)", (cid, user_id, title[:120], now, now))
        return cid

    def rename_conversation(self, conversation_id: str, title: str) -> None:
        self._exec("UPDATE conversations SET title=? WHERE id=?", (title[:120], conversation_id))

    def delete_conversation(self, conversation_id: str) -> None:
        self._exec("DELETE FROM file_chunks WHERE file_id IN (SELECT id FROM files WHERE conversation_id=?)", (conversation_id,))
        self._exec("DELETE FROM files WHERE conversation_id=?", (conversation_id,))
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

    # ---- files
    def add_file(self, user_id, name, mime, size, text, chunks, embeddings) -> dict:
        fid, now = str(uuid.uuid4()), _now()
        with self.lock:
            self.conn.execute(
                "INSERT INTO files (id, user_id, conversation_id, name, mime, size, content, n_chunks, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (fid, user_id, None, name[:255], mime, size, text, len(chunks), now),
            )
            self.conn.executemany(
                "INSERT INTO file_chunks (file_id, chunk_index, content, embedding) VALUES (?,?,?,?)",
                [(fid, i, c, np.asarray(e, dtype=np.float32).tobytes()) for i, (c, e) in enumerate(zip(chunks, embeddings))],
            )
            self.conn.commit()
        return {"id": fid, "name": name, "mime": mime, "size": size, "n_chunks": len(chunks), "created_at": now}

    def attach_files(self, file_ids, conversation_id, user_id) -> None:
        for fid in file_ids:
            self._exec("UPDATE files SET conversation_id=? WHERE id=? AND user_id=?", (conversation_id, fid, user_id))

    def list_files(self, conversation_id) -> list[dict]:
        rows = self._exec(
            "SELECT id, name, mime, size, n_chunks, created_at FROM files WHERE conversation_id=? ORDER BY created_at",
            (conversation_id,),
        )
        return [dict(r) for r in rows]

    def get_file_texts(self, file_ids) -> dict[str, str]:
        if not file_ids:
            return {}
        marks = ",".join("?" * len(file_ids))
        return {r["id"]: r["content"] for r in self._exec(f"SELECT id, content FROM files WHERE id IN ({marks})", tuple(file_ids))}

    def search_chunks(self, file_ids, query_vec, k: int = 6) -> list[dict]:
        if not file_ids:
            return []
        marks = ",".join("?" * len(file_ids))
        rows = self._exec(
            f"SELECT c.file_id, c.chunk_index, c.content, c.embedding, f.name FROM file_chunks c "
            f"JOIN files f ON f.id = c.file_id WHERE c.file_id IN ({marks})",
            tuple(file_ids),
        )
        items = [
            {"file_id": r["file_id"], "file_name": r["name"], "chunk_index": r["chunk_index"], "content": r["content"],
             "embedding": np.frombuffer(r["embedding"], dtype=np.float32)}
            for r in rows
        ]
        return top_k(query_vec, items, k)

    def delete_file(self, file_id, user_id) -> None:
        self._exec("DELETE FROM file_chunks WHERE file_id IN (SELECT id FROM files WHERE id=? AND user_id=?)", (file_id, user_id))
        self._exec("DELETE FROM files WHERE id=? AND user_id=?", (file_id, user_id))


def get_store(cfg: Settings = settings) -> tuple[Any, str | None]:
    """Return (store, warning). Falls back to SQLite if Supabase is missing or unreachable."""
    if cfg.use_supabase:
        try:
            store = SupabaseStore(cfg.supabase_url, cfg.supabase_key)
            store.db.table("conversations").select("id").limit(1).execute()  # verify schema exists
            store.db.table("files").select("id").limit(1).execute()
            return store, None
        except Exception as e:  # noqa: BLE001
            reason = str(e)
            if "PGRST205" in reason or "schema cache" in reason:
                reason = "tables not found: run supabase/schema.sql in the Supabase SQL Editor"
            return SQLiteStore(cfg.sqlite_path), f"Supabase unavailable ({reason}); using local SQLite instead."
    return SQLiteStore(cfg.sqlite_path), None
