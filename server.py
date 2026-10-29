"""Real-time chatbot web server: FastAPI API + streaming chat (Server-Sent Events) + static web UI.

Run:  python server.py        (then open http://localhost:8000)
"""
import json
import queue
import threading
import time
from pathlib import Path
from typing import Iterator, Literal

from anyio import to_thread
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from chatbot import documents
from chatbot.agent import build_agent, stream_answer
from chatbot.auth import auth_enabled, current_user_id
from chatbot.config import settings
from chatbot.file_context import build_file_context
from chatbot.providers import PROVIDERS, build_llm, default_model, list_models, resolve_key, server_key, split_model
from chatbot.storage import get_store
from chatbot.tools import ALL_TOOLS, CODE_TOOLS, TOOL_LABELS
from chatbot.tools.code import execute_python
from chatbot.tools.router import route_tools

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Pulse AI")
store, store_warning = get_store(settings)
documents.warm_up()  # load the embedding model in the background
EXTRA_LABELS = {"search_files": "📎 Your files"}

# Models for providers whose key is in .env, fetched once at startup
SERVER_MODELS: dict[str, list[str]] = {}
for _p in PROVIDERS:
    try:
        SERVER_MODELS[_p] = list_models(_p, server_key(_p)) if server_key(_p) else []
    except Exception as e:  # noqa: BLE001
        print(f"  ! Couldn't list {_p} models: {e}")
        SERVER_MODELS[_p] = []
DEFAULT_PROVIDER = "groq" if SERVER_MODELS["groq"] else next((p for p in PROVIDERS if SERVER_MODELS[p]), "groq")
DEFAULT_MODEL = f"{DEFAULT_PROVIDER}:{default_model(DEFAULT_PROVIDER, SERVER_MODELS[DEFAULT_PROVIDER] or list_models('groq', ''))}"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=50000)
    conversation_id: str | None = None
    user_id: str = "default"
    model: str = ""  # "<provider>:<model id>"
    mode: Literal["chat", "code"] = "chat"
    tools: list[str] | None = None
    user_tz: str = "UTC"
    api_keys: dict[str, str] = {}  # keys the user entered in Settings (used when the server has none)
    api_key: str | None = None  # legacy: a Groq key
    file_ids: list[str] = Field(default=[], max_length=20)  # uploaded files to attach to this conversation
    images: list[dict] = Field(default=[], max_length=4)  # [{"name", "data_url"}] for vision models


class ModelsRequest(BaseModel):
    provider: str
    api_key: str = ""


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class RunRequest(BaseModel):
    code: str = Field(min_length=1, max_length=50000)


@app.get("/api/config")
def get_config():
    return {
        "providers": [
            {"id": p, **meta, "server_key": bool(server_key(p)), "models": SERVER_MODELS[p]}
            for p, meta in PROVIDERS.items()
        ],
        "default_model": DEFAULT_MODEL,
        "tools": [{"name": n, "label": TOOL_LABELS.get(n, n)} for n in ALL_TOOLS],
        "code_tools": CODE_TOOLS,
        "code_execution": settings.code_execution,
        "storage": store.backend,
        "storage_warning": store_warning,
        "server_has_key": bool(server_key("groq")),
        # Login: the browser signs in with Supabase Auth using the public anon key
        "auth": {"enabled": auth_enabled(), "supabase_url": settings.supabase_url if auth_enabled() else "",
                 "anon_key": settings.supabase_anon_key if auth_enabled() else ""},
        "upload": {"max_mb": documents.MAX_FILE_BYTES // (1024 * 1024)},
    }


def _own_conversation(request: Request, conversation_id: str, claimed: str | None) -> str:
    """The caller's user id, after checking they own the conversation (404 otherwise)."""
    user_id = current_user_id(request, claimed)
    if store.conversation_owner(conversation_id) != user_id:
        raise HTTPException(404, "Conversation not found.")
    return user_id


@app.post("/api/files")
def upload_file(request: Request, file: UploadFile = File(...), user_id: str | None = Form(None)):
    """Upload a document for chat: extract text, chunk, embed and store it. Returns the file's metadata."""
    uid = current_user_id(request, user_id or request.query_params.get("user_id"))
    data = file.file.read(documents.MAX_FILE_BYTES + 1)
    try:
        processed = documents.process_file(file.filename or "file", data)
    except documents.UnsupportedFile as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Couldn't read this file: {e}"[:300])
    meta = store.add_file(uid, file.filename or "file", file.content_type or "", len(data),
                          processed["text"], processed["chunks"], processed["embeddings"])
    return {**meta, "chars": len(processed["text"])}


@app.get("/api/conversations/{conversation_id}/files")
def conversation_files(request: Request, conversation_id: str, user_id: str = "default"):
    _own_conversation(request, conversation_id, user_id)
    return store.list_files(conversation_id)


@app.delete("/api/files/{file_id}")
def delete_file(request: Request, file_id: str, user_id: str = "default"):
    store.delete_file(file_id, current_user_id(request, user_id))
    return {"ok": True}


@app.post("/api/models")
def get_models(req: ModelsRequest):
    """Models available to a key the user typed in Settings."""
    if req.provider not in PROVIDERS:
        raise HTTPException(400, "Unknown provider")
    key = resolve_key(req.provider, {req.provider: req.api_key})
    if not key:
        raise HTTPException(400, "No API key")
    try:
        return {"models": list_models(req.provider, key)}
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "401" in msg or "auth" in msg.lower() or "invalid" in msg.lower():
            msg = "That API key was rejected. Check it and try again."
        raise HTTPException(400, msg[:300])


@app.post("/api/run")
def run_code(request: Request, req: RunRequest):
    """Run a Python code block from the chat (the ▶ Run button). Local use only."""
    if not settings.code_execution:
        raise HTTPException(403, "Code execution is disabled on this server (CODE_EXECUTION=on enables it).")
    current_user_id(request)  # signed-in users only, when login is on
    return execute_python(req.code)


@app.get("/api/me")
def me(request: Request, user_id: str = "default"):
    return {"user_id": current_user_id(request, user_id), "auth": auth_enabled()}


@app.get("/api/conversations")
def list_conversations(request: Request, user_id: str = "default"):
    return store.list_conversations(current_user_id(request, user_id))


@app.get("/api/conversations/{conversation_id}/messages")
def get_messages(request: Request, conversation_id: str, user_id: str = "default"):
    _own_conversation(request, conversation_id, user_id)
    return store.get_messages(conversation_id)


@app.patch("/api/conversations/{conversation_id}")
def rename_conversation(request: Request, conversation_id: str, body: RenameRequest, user_id: str = "default"):
    _own_conversation(request, conversation_id, user_id)
    store.rename_conversation(conversation_id, body.title)
    return {"ok": True}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(request: Request, conversation_id: str, user_id: str = "default"):
    _own_conversation(request, conversation_id, user_id)
    store.delete_conversation(conversation_id)
    return {"ok": True}


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


class _Stopped(Exception):
    """The browser disconnected (Stop button / closed tab)."""


def _is_rate_limit(msg: str) -> bool:
    return "rate_limit" in msg or "429" in msg or "Rate limit" in msg or "RateLimit" in msg


def _friendly_error(msg: str, provider: str, model: str) -> str:
    name = PROVIDERS[provider]["name"]
    if _is_rate_limit(msg):
        if provider == "groq":
            return "All Groq models are rate-limited right now (free tier: 8,000 tokens/min per model). Wait about a minute and try again."
        return f"{name} rate limit reached. Wait a moment and try again."
    if "model_not_found" in msg or "does not exist" in msg or "not_found_error" in msg:
        return f"The model '{model}' isn't available on {name}. Pick another model at the top of the page."
    if "invalid_api_key" in msg or "authentication" in msg.lower() or "401" in msg:
        return f"Invalid {name} API key. Check it in Settings."
    if "credit balance" in msg or "insufficient_quota" in msg or "billing" in msg.lower():
        return f"Your {name} account has no credit left. Add billing on the provider's website, or switch provider."
    return msg


def _chat_events(req: ChatRequest, user_id: str, provider: str, model: str, api_key: str, stop: threading.Event) -> Iterator[str]:
    cid = req.conversation_id
    history = store.get_messages(cid) if cid else []
    if not cid:
        cid = store.create_conversation(user_id, req.message[:60])
    store.attach_files(req.file_ids, cid, user_id)
    files = store.list_files(cid)
    attached = [{"id": f["id"], "name": f["name"]} for f in files if f["id"] in req.file_ids]
    attached += [{"name": im.get("name", "image"), "image": True} for im in req.images]
    store.add_message(cid, "user", req.message, {"files": attached} if attached else None)
    yield _sse("meta", {"conversation_id": cid})

    answer, tools_used, pending_args, t0 = "", [], {}, time.perf_counter()
    status = "complete"
    images = [im["data_url"] for im in req.images if str(im.get("data_url", "")).startswith("data:image/")]
    # Uploaded documents: full text if small, otherwise retrieval (Groq's free tier has a small token budget)
    file_context, file_tools = "", []
    if files:
        yield _sse("tool_start", {"id": "files", "name": "search_files", "label": EXTRA_LABELS["search_files"],
                                  "args": {"files": f"{len(files)} file{'s' if len(files) > 1 else ''}"}})
        f0 = time.perf_counter()
        file_context, file_tools = build_file_context(store, files, req.message, 6000 if provider == "groq" else 60000)
        yield _sse("tool_end", {"id": "files", "name": "search_files", "latency_ms": int((time.perf_counter() - f0) * 1000), "ok": True})
    if req.mode == "code":
        tool_names = [t for t in CODE_TOOLS if t != "run_python" or settings.code_execution]
    elif provider == "groq":
        # Only send the tools this question needs: keeps each call small enough for Groq's free tier
        recent = [m["content"] for m in history if m["role"] == "user"][-2:]
        tool_names = route_tools(req.message, recent, req.tools)
    else:
        tool_names = req.tools  # paid providers have room for every enabled tool
    max_steps = 10 if req.mode == "code" else 6
    # Groq's free models each have their own rate limit, so a busy one falls back to the next.
    # Paid providers are never silently switched to a different model.
    candidates = [model]
    if provider == "groq":
        candidates += [m for m in list_models("groq", api_key) if m != model][:2]
    if images and provider == "groq":
        status = "error"
        yield _sse("error", {"message": "Groq's models can't read images. Pick a Claude or OpenAI model at the top of the page."})
        candidates = []
    try:
        for attempt, model in enumerate(candidates):
            agent = build_agent(build_llm(provider, model, api_key, req.mode), tool_names, file_tools)
            try:
                for ev in stream_answer(agent, req.message, history, user_tz=req.user_tz, max_steps=max_steps,
                                        mode=req.mode, context=file_context, images=images):
                    if stop.is_set():
                        raise _Stopped
                    kind = ev["type"]
                    if kind == "token":
                        answer += ev["text"]
                        yield _sse("token", {"text": ev["text"]})
                    elif kind == "reset":
                        answer = ""
                        yield _sse("reset", {})
                    elif kind == "tool_start":
                        pending_args[ev["id"]] = ev["args"]
                        label = TOOL_LABELS.get(ev["name"]) or EXTRA_LABELS.get(ev["name"], ev["name"])
                        yield _sse("tool_start", {"id": ev["id"], "name": ev["name"], "label": label, "args": ev["args"]})
                    elif kind == "tool_end":
                        args = pending_args.pop(ev["id"], {})
                        tools_used.append({"name": ev["name"], "args": args, "latency_ms": ev["latency_ms"]})
                        yield _sse("tool_end", {"id": ev["id"], "name": ev["name"], "latency_ms": ev["latency_ms"], "ok": not ev["output"].startswith("Error")})
                        try:
                            store.log_tool_call(cid, ev["name"], args, ev["output"], ev["latency_ms"])
                        except Exception:
                            pass  # analytics logging must never break the chat
                break
            except _Stopped:
                raise
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                retryable = _is_rate_limit(msg) or "tool_use_failed" in msg or "Failed to call a function" in msg
                if retryable and not answer and attempt + 1 < len(candidates):
                    nxt = candidates[attempt + 1]
                    why = "is busy (rate limit)" if _is_rate_limit(msg) else "made a bad tool call"
                    yield _sse("notice", {"message": f"{model} {why}, switching to {nxt}"})
                    yield _sse("reset", {})
                    continue
                raise
    except _Stopped:  # the user pressed Stop / closed the tab: keep what was generated so far
        status = "stopped"
    except Exception as e:  # noqa: BLE001
        status = "error"
        yield _sse("error", {"message": _friendly_error(str(e), provider, model)[:500]})
    metadata = {"tools": tools_used, "model": f"{provider}:{model}", "mode": req.mode,
                "total_ms": int((time.perf_counter() - t0) * 1000), "status": status}
    if answer or status != "error":
        store.add_message(cid, "assistant", answer, metadata)
    yield _sse("done", metadata)


async def _relay(req: ChatRequest, user_id: str, provider: str, model: str, api_key: str):
    """Run the (blocking) agent in a worker thread and relay its events to the browser.

    If the browser disconnects, `stop` tells the worker to finish early, so the partial
    answer is still saved instead of the generator being abandoned mid-way.
    """
    q: queue.Queue = queue.Queue()
    stop = threading.Event()

    def produce():
        try:
            for chunk in _chat_events(req, user_id, provider, model, api_key, stop):
                q.put(chunk)
        except Exception as e:  # noqa: BLE001
            q.put(_sse("error", {"message": f"Server error: {e}"[:500]}))
        finally:
            q.put(None)

    threading.Thread(target=produce, daemon=True).start()
    try:
        while (chunk := await to_thread.run_sync(q.get, abandon_on_cancel=True)) is not None:
            yield chunk
    finally:
        stop.set()


@app.post("/api/chat")
def chat(request: Request, req: ChatRequest):
    user_id = current_user_id(request, req.user_id)
    if req.conversation_id and store.conversation_owner(req.conversation_id) != user_id:
        raise HTTPException(404, "Conversation not found.")
    provider, model = split_model(req.model or DEFAULT_MODEL)
    keys = dict(req.api_keys)
    if req.api_key and "groq" not in keys:
        keys["groq"] = req.api_key
    api_key = resolve_key(provider, keys)
    if not api_key:
        name = PROVIDERS[provider]["name"]
        raise HTTPException(400, f"No {name} API key. Add it in Settings, or pick a model from another provider.")
    if not model:
        model = default_model(provider, list_models(provider, api_key))
    return StreamingResponse(
        _relay(req, user_id, provider, model, api_key),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    print(f"\n  Pulse AI running at http://localhost:{port}")
    print(f"  Code execution: {'ON (local only)' if settings.code_execution else 'off'}\n")
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=port)
