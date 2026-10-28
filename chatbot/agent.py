"""Tool-calling agent (Groq LLM + live data tools) that streams events for the UI."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from chatbot.config import settings
from chatbot.tools import get_tools

SYSTEM_PROMPT = """You are a real-time research assistant with live access to the internet and data APIs.

Current date and time: {now} (UTC). The user's timezone: {user_tz}.
Your training data is out of date. Anything that changes over time (news, prices, weather, scores,
releases, who holds a role, recent events, "latest"/"current"/"today" questions) MUST be answered
with tools, never from memory.

How to work:
- Pick the most specific tool:
  * Markets: stock_quote (price now), stock_history (performance over time), stock_news (why it moved),
    market_overview (indices/gold/oil/FX), mutual_fund_nav (Indian MFs), crypto_price, currency_convert.
  * People & opinions: social_posts (X/Twitter, Bluesky, Mastodon; use from_user for a specific
    expert or company account), reddit_search, hacker_news.
  * News & web: news_search (current events), web_search (everything else), rss_feed (a blog or
    publication), read_webpage (open a result when snippets aren't enough).
  * Tech: github_search (repos, trending projects), stackoverflow_search, arxiv_search (papers).
  * World: weather (includes air quality), sports_scores (live games), earthquakes, current_datetime.
  * Background facts: wikipedia. Videos: youtube_search.
- Use calculator for any arithmetic (percent changes, totals, conversions); don't do math in your head.
- Call several tools at once when a question needs several sources (e.g. stock_quote + stock_news
  for "why is X stock down today?").
- If a tool fails or returns nothing useful, try another tool or rephrase the query before giving up.
- Never invent numbers, dates, quotes or URLs. If live data couldn't be fetched, say so plainly.

How to answer:
- Lead with the direct answer, then supporting detail. Use short paragraphs, bullet points or tables.
- Mention how fresh the data is (e.g. "as of 14:05 UTC") for prices, weather and news.
- Cite sources inline as markdown links [source name](url) using URLs returned by tools.
- For casual conversation or questions that don't depend on current facts, answer directly without tools.
"""


def build_llm(api_key: str, model: str, temperature: float = 0.2) -> ChatGroq:
    kwargs: dict[str, Any] = {}
    if model.startswith("qwen/") or "deepseek-r1" in model:
        kwargs["reasoning_format"] = "hidden"  # don't stream <think> blocks to the user
    return ChatGroq(api_key=api_key, model=model, temperature=temperature, streaming=True, max_retries=2, **kwargs)


def build_agent(llm: ChatGroq, tool_names: list[str] | None = None):
    tools = get_tools(tool_names)
    try:  # LangChain >= 1.0
        from langchain.agents import create_agent

        return create_agent(llm, tools)
    except ImportError:
        from langgraph.prebuilt import create_react_agent

        return create_react_agent(llm, tools)


def history_to_messages(history: list[dict], window: int = settings.history_window) -> list[BaseMessage]:
    msgs: list[BaseMessage] = []
    for m in history[-window:]:
        if m["role"] == "user":
            msgs.append(HumanMessage(m["content"]))
        elif m["role"] == "assistant" and m["content"]:
            msgs.append(AIMessage(m["content"]))
    return msgs


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return ""


def stream_answer(
    agent,
    question: str,
    history: list[dict],
    user_tz: str = "UTC",
    max_steps: int = 12,
) -> Iterator[dict]:
    """Run the agent and yield UI events:
    {"type": "token", "text"} | {"type": "reset"} | {"type": "tool_start", "id", "name", "args"} |
    {"type": "tool_end", "id", "name", "output", "latency_ms"}
    """
    now = datetime.now(timezone.utc).strftime("%A, %d %B %Y, %H:%M")
    messages = [
        SystemMessage(SYSTEM_PROMPT.format(now=now, user_tz=user_tz)),
        *history_to_messages(history),
        HumanMessage(question),
    ]
    started: dict[str, tuple[str, float]] = {}

    for mode, chunk in agent.stream(
        {"messages": messages},
        config={"recursion_limit": max_steps * 2 + 1},
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            msg, meta = chunk
            if isinstance(msg, ToolMessage) or meta.get("langgraph_node") == "tools":
                continue
            text = _text_of(msg.content)
            if text:
                yield {"type": "token", "text": text}
        elif mode == "updates":
            for node_update in chunk.values():
                if not isinstance(node_update, dict):
                    continue
                for msg in node_update.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        # Any text streamed before a tool call is preamble ("Let me search..."), not the answer
                        yield {"type": "reset"}
                        for tc in msg.tool_calls:
                            started[tc["id"]] = (tc["name"], time.perf_counter())
                            yield {"type": "tool_start", "id": tc["id"], "name": tc["name"], "args": tc["args"]}
                    elif isinstance(msg, ToolMessage):
                        name, t0 = started.get(msg.tool_call_id, (msg.name or "tool", time.perf_counter()))
                        yield {
                            "type": "tool_end",
                            "id": msg.tool_call_id,
                            "name": name,
                            "output": _text_of(msg.content),
                            "latency_ms": int((time.perf_counter() - t0) * 1000),
                        }
