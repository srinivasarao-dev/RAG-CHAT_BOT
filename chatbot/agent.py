"""Tool-calling agent (any provider's LLM + live data tools) that streams events for the UI."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Iterator

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

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
- Be efficient: most questions need 1-3 tool calls. Make independent calls in parallel in ONE step,
  then answer. Only use read_webpage when the snippets truly lack the answer, and at most once.
- If a tool fails or returns nothing useful, try one alternative, then answer with what you have.
- Never invent numbers, dates, quotes or URLs. If live data couldn't be fetched, say so plainly.

How to answer:
- Lead with the direct answer, then supporting detail. Use short paragraphs, bullet points or tables.
- Mention how fresh the data is (e.g. "as of 14:05 UTC") for prices, weather and news.
- Cite sources inline as markdown links [source name](url) using URLs returned by tools.
  Never use bracket citation markers like 【1】 or 【weather】. If a tool returned no URL (prices,
  weather, conversions), name the source in plain text (e.g. "CoinGecko", "Open-Meteo") without a link.
- For casual conversation or questions that don't depend on current facts, answer directly without tools.
"""

CODE_PROMPT = """You are an expert software engineer and pair programmer, working in a chat interface.

Current date and time: {now} (UTC). The user's timezone: {user_tz}.
Libraries, APIs and best practices change after your training data. When versions, release notes,
deprecations or current APIs matter, check them with tools instead of relying on memory.

Your tools:
- package_info: the latest version and release date of a PyPI or npm package. Check it before
  recommending a version, an install command, or an API that may have changed.
- github_file: read real source code from public GitHub repos (list a folder with path="", then open files).
- read_webpage / web_search: official docs, changelogs, migration guides, error messages.
- stackoverflow_search: known issues and fixes for specific errors.
- github_search: find libraries and example projects.
- run_python (when available): run Python to test code you wrote, reproduce a bug, or check an output.
  It runs in a fresh process each time with a 20 second limit, no stdin and no persisted state, so
  include every import and print the results you need.

How to work:
- Understand the task first. If something essential is ambiguous, state your assumption and proceed.
- Prefer complete, runnable code over fragments: include imports, and show how to run it.
- For non-trivial Python, test it with run_python before presenting it when that's practical; if a run
  fails, fix the code and run again. Say briefly what you verified.
- When debugging, find the root cause, explain it in one or two sentences, then show the fix.
- When reviewing code, list concrete issues by severity with the fix for each.
- Keep explanations short and specific. Don't repeat the user's code back unless you changed it.

Formatting:
- Fenced code blocks with a language tag (```python, ```bash, ```ts ...). One block per file, with the
  file name in a line above it when there are several files.
- Cite docs or sources you relied on as markdown links. Never use bracket citation markers like 【1】.
"""


def build_agent(llm: BaseChatModel, tool_names: list[str] | None = None, extra_tools: list | None = None):
    tools = get_tools(tool_names) + list(extra_tools or [])
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
    if isinstance(content, list):  # content blocks: keep only visible text, never thinking/reasoning blocks
        return "".join(
            b.get("text", "") if isinstance(b, dict) and b.get("type", "text") == "text"
            else (b if isinstance(b, str) else "")
            for b in content
        )
    return ""


def stream_answer(
    agent,
    question: str,
    history: list[dict],
    user_tz: str = "UTC",
    max_steps: int = 6,
    mode: str = "chat",
    context: str = "",
    images: list[str] | None = None,
) -> Iterator[dict]:
    """Run the agent and yield UI events:
    {"type": "token", "text"} | {"type": "reset"} | {"type": "tool_start", "id", "name", "args"} |
    {"type": "tool_end", "id", "name", "output", "latency_ms"}
    """
    now = datetime.now(timezone.utc).strftime("%A, %d %B %Y, %H:%M")
    prompt = (CODE_PROMPT if mode == "code" else SYSTEM_PROMPT).format(now=now, user_tz=user_tz)
    if context:
        prompt += "\n\n" + context
    # Images (data URLs) go to the model as vision content blocks next to the question
    user_content: Any = question
    if images:
        user_content = [{"type": "text", "text": question}] + [
            {"type": "image_url", "image_url": {"url": url}} for url in images
        ]
    messages = [
        SystemMessage(prompt),
        *history_to_messages(history),
        HumanMessage(user_content),
    ]
    started: dict[str, tuple[str, float]] = {}
    try:
        yield from _stream(agent, messages, max_steps, started)
    except GraphRecursionError:
        yield {"type": "token", "text": "\n\n_I ran out of research steps. Ask a narrower question for a more complete answer._"}


def _stream(agent, messages: list[BaseMessage], max_steps: int, started: dict) -> Iterator[dict]:
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
                    if isinstance(msg, AIMessage) and (msg.response_metadata or {}).get("stop_reason") == "refusal":
                        yield {"type": "token", "text": "\n\n_The model declined to answer this request._"}
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
