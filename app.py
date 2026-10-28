"""Real-time chatbot: Streamlit UI + Groq tool-calling agent + live data APIs + Supabase history.

Run:  streamlit run app.py
"""
import json
import time

import streamlit as st

from chatbot.agent import build_agent, build_llm, stream_answer
from chatbot.config import GROQ_MODELS, settings
from chatbot.storage import get_store
from chatbot.tools import ALL_TOOLS, TOOL_LABELS

st.set_page_config(page_title="Real-time Chatbot", page_icon="⚡", layout="wide")

STARTERS = [
    "Why is NVDA stock moving today?",
    "How are Nifty, Sensex, gold and oil doing right now?",
    "What are AI researchers posting about on social media today?",
    "Weather and air quality in Hyderabad this week",
    "Live Premier League scores",
    "Trending new Python repos on GitHub this week",
]


@st.cache_resource
def _store():
    return get_store(settings)


store, store_warning = _store()


# ---------- session state ----------
ss = st.session_state
ss.setdefault("conversation_id", None)
ss.setdefault("messages", [])
ss.setdefault("pending_prompt", None)


def open_conversation(cid: str | None) -> None:
    ss.conversation_id = cid
    ss.messages = store.get_messages(cid) if cid else []


# ---------- sidebar ----------
with st.sidebar:
    st.title("⚡ Real-time Chatbot")

    if st.button("➕ New chat", use_container_width=True):
        open_conversation(None)
        st.rerun()

    with st.expander("⚙️ Settings", expanded=not settings.groq_api_key):
        api_key = st.text_input(
            "Groq API key",
            value=settings.groq_api_key,
            type="password",
            help="Get one free at https://console.groq.com/keys. Set GROQ_API_KEY in .env to skip this.",
        )
        model_choice = st.selectbox(
            "Model",
            GROQ_MODELS + ["Custom…"],
            index=GROQ_MODELS.index(settings.groq_model) if settings.groq_model in GROQ_MODELS else 0,
        )
        model = st.text_input("Custom Groq model id") if model_choice == "Custom…" else model_choice
        user_id = st.text_input("Your name / user id", value="default", help="Chats are saved per user id.")
        user_tz = st.text_input("Your timezone", value="Asia/Kolkata", help="IANA name, e.g. Europe/London")
        enabled_tools = st.multiselect(
            "Data sources",
            options=list(ALL_TOOLS),
            default=list(ALL_TOOLS),
            format_func=lambda n: TOOL_LABELS.get(n, n),
        )

    st.caption(f"💾 Storage: **{store.backend}**")
    if store_warning:
        st.warning(store_warning)

    st.subheader("History")
    try:
        conversations = store.list_conversations(user_id or "default")
    except Exception as e:  # noqa: BLE001
        conversations = []
        st.error(f"Couldn't load history: {e}")
    for conv in conversations:
        col1, col2 = st.columns([5, 1])
        active = conv["id"] == ss.conversation_id
        if col1.button(("▶ " if active else "") + conv["title"], key=f"open_{conv['id']}", use_container_width=True):
            open_conversation(conv["id"])
            st.rerun()
        if col2.button("🗑", key=f"del_{conv['id']}", help="Delete chat"):
            store.delete_conversation(conv["id"])
            if active:
                open_conversation(None)
            st.rerun()


# ---------- helpers ----------
def render_sources(metadata: dict) -> None:
    tools_used = metadata.get("tools") or []
    if not tools_used:
        return
    with st.expander(f"🔎 Used {len(tools_used)} live source{'s' if len(tools_used) != 1 else ''}", expanded=False):
        for t in tools_used:
            label = TOOL_LABELS.get(t["name"], t["name"])
            st.markdown(f"**{label}** · `{json.dumps(t.get('args', {}), ensure_ascii=False)}` · {t.get('latency_ms', '?')} ms")
    if metadata.get("model"):
        st.caption(f"{metadata['model']} · {metadata.get('total_ms', 0) / 1000:.1f}s")


def run_turn(prompt: str) -> None:
    history = list(ss.messages)
    if ss.conversation_id is None:
        ss.conversation_id = store.create_conversation(user_id or "default", prompt[:60])
    store.add_message(ss.conversation_id, "user", prompt)
    ss.messages.append({"role": "user", "content": prompt, "metadata": {}})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Thinking…", expanded=False)
        placeholder = st.empty()
        answer, tools_used, pending_args, failed = "", [], {}, False
        t0 = time.perf_counter()

        agent = build_agent(build_llm(api_key, model), enabled_tools)
        for attempt in range(2):  # retry once: models occasionally emit a malformed tool call
            try:
                for event in stream_answer(agent, prompt, history, user_tz=user_tz):
                    kind = event["type"]
                    if kind == "token":
                        answer += event["text"]
                        placeholder.markdown(answer + "▌")
                    elif kind == "reset":
                        answer = ""
                        placeholder.empty()
                    elif kind == "tool_start":
                        pending_args[event["id"]] = event["args"]
                        label = TOOL_LABELS.get(event["name"], event["name"])
                        status.update(label=f"{label}…", state="running")
                        status.write(f"{label}: `{json.dumps(event['args'], ensure_ascii=False)}`")
                    elif kind == "tool_end":
                        args = pending_args.pop(event["id"], {})
                        tools_used.append({"name": event["name"], "args": args, "latency_ms": event["latency_ms"]})
                        try:
                            store.log_tool_call(ss.conversation_id, event["name"], args, event["output"], event["latency_ms"])
                        except Exception:
                            pass  # analytics logging must never break the chat
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if attempt == 0 and not answer and ("tool_use_failed" in msg or "Failed to call a function" in msg):
                    status.write("Retrying…")
                    continue
                failed = True
                status.update(label="Error", state="error")
                answer = answer or f"⚠️ Sorry, something went wrong: `{msg[:500]}`"
                break

        total_ms = int((time.perf_counter() - t0) * 1000)
        if not failed:
            status.update(
                label=f"Used {len(tools_used)} source{'s' if len(tools_used) != 1 else ''}" if tools_used else "Answered directly",
                state="complete",
            )
        placeholder.markdown(answer or "_(no answer)_")

    metadata = {"tools": tools_used, "model": model, "total_ms": total_ms}
    store.add_message(ss.conversation_id, "assistant", answer, metadata)
    ss.messages.append({"role": "assistant", "content": answer, "metadata": metadata})


# ---------- main chat ----------
if not ss.messages:
    st.markdown("## What do you want to know right now?")
    st.caption(
        f"{len(ALL_TOOLS)} live sources: web, news, X/Twitter, Reddit, YouTube, GitHub, Yahoo Finance, "
        "crypto, sports, weather and more."
    )
    cols = st.columns(3)
    for i, s in enumerate(STARTERS):
        if cols[i % 3].button(s, key=f"starter_{i}", use_container_width=True):
            ss.pending_prompt = s
            st.rerun()

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            render_sources(m.get("metadata") or {})

prompt = st.chat_input("Ask about anything happening right now…")
if ss.pending_prompt:
    prompt, ss.pending_prompt = ss.pending_prompt, None
if prompt:
    if not api_key:
        st.error("Add your Groq API key in ⚙️ Settings (sidebar) to start chatting.")
    else:
        run_turn(prompt)
        st.rerun()  # refresh the sidebar history with the new/updated chat
