# ⚡ Real-time Chatbot

A chatbot that answers with **live data**. A Groq-hosted LLM decides which of **24 live sources** to call (web, news, X/Twitter,
Bluesky, Mastodon, Reddit, Hacker News, YouTube, GitHub, Yahoo Finance, Finnhub, crypto, sports, weather and more), streams the answer token by token,
cites sources, and saves every conversation to **Supabase** (or a local SQLite file).

## How it works

```
User ──► Streamlit UI (app.py) ──► Agent (chatbot/agent.py, LangChain + Groq tool calling)
              ▲                         │  picks 0..n tools, can call several in parallel
              │ streamed tokens         ▼
              │                    Tools (chatbot/tools/)  ──► APIs, with library fallbacks
              │                         │
              └──── Supabase / SQLite ◄─┘  conversations, messages, tool_calls (chatbot/storage.py)
```

### Data sources and fallbacks

Every tool works **without any API key**. Keys only upgrade quality or limits.

| Category | Tool | Primary (optional key) | Fallback (free, no key) |
|---|---|---|---|
| Web | `web_search` | Tavily → Serper (Google) → Brave | DuckDuckGo (`ddgs`) |
| | `news_search` | NewsAPI | DuckDuckGo News → Google News RSS |
| | `read_webpage` | `trafilatura` | BeautifulSoup |
| | `rss_feed` | any RSS/Atom feed, Substack, blogs (auto-discovers feeds) | n/a |
| Social | `social_posts` | X/Twitter API v2 (`X_BEARER_TOKEN`), Bluesky search (app password) | Bluesky author feeds + Mastodon → web results on x.com |
| | `reddit_search` | Reddit API via PRAW | Reddit public JSON → web results on reddit.com |
| | `hacker_news` | Algolia HN API | n/a |
| | `youtube_search` | `yt-dlp` | n/a |
| Dev | `github_search` | GitHub API (`GITHUB_TOKEN` raises limits) | GitHub API unauthenticated |
| | `stackoverflow_search` | Stack Exchange API | n/a |
| Markets | `stock_quote` | Finnhub real-time (`FINNHUB_API_KEY`) | Yahoo Finance (`yfinance`) |
| | `stock_history` | Yahoo Finance | n/a |
| | `stock_news` | Finnhub company news | news search for the ticker |
| | `market_overview` | Yahoo Finance: US/India/Europe/Asia indices, gold, oil, FX, crypto | n/a |
| | `mutual_fund_nav` | mfapi.in (Indian MF NAVs, AMFI) | n/a |
| | `crypto_price` | CoinGecko | n/a |
| | `currency_convert` | Frankfurter (ECB) | open.er-api.com |
| World | `weather` | Open-Meteo forecast + air quality (AQI, PM2.5) | n/a |
| | `sports_scores` | ESPN scoreboard: NBA, NFL, MLB, NHL, EPL, La Liga, UCL, F1, tennis, UFC… | n/a |
| | `earthquakes` | USGS live feed | n/a |
| | `current_datetime` | Python `zoneinfo` | n/a |
| | `calculator` | safe local math evaluator | n/a |
| Reference | `wikipedia` | MediaWiki API | `wikipedia` library → DuckDuckGo |
| | `arxiv_search` | arXiv export API | `arxiv` library |

Tool results are cached for `TOOL_CACHE_TTL` seconds (60s for prices) so repeated questions are instant and
rate limits aren't hit, while answers stay fresh. A failing source returns an error message to the model
instead of crashing, so the model can try another tool.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

Edit `.env` and set at least `GROQ_API_KEY` (free at https://console.groq.com/keys). Then run:

```bash
streamlit run app.py
```

### Connect Supabase (optional, recommended)

1. Create a project at https://supabase.com.
2. Open **SQL Editor**, paste [`supabase/schema.sql`](supabase/schema.sql), and click **Run**.
3. Under **Project Settings → API**, copy the project URL and the **service_role** key into `.env`:
   ```
   SUPABASE_URL=https://xxxx.supabase.co
   SUPABASE_KEY=eyJ...
   ```
4. Restart the app. The sidebar shows `Storage: supabase`.

If Supabase isn't configured or can't be reached, the app keeps working with a local `chat_history.db`
and shows a warning in the sidebar.

Tables:
- `conversations`: one row per chat (per `user_id`)
- `messages`: user and assistant messages, plus `metadata` (tools used, model, response time)
- `tool_calls`: every external API call with arguments, output preview and latency (for analytics and debugging)

## Project layout

```
app.py                  Streamlit UI: streaming chat, live tool status, chat history sidebar
chatbot/
  config.py             settings from .env / Streamlit secrets
  agent.py              system prompt, Groq LLM, agent, event streaming
  storage.py            SupabaseStore + SQLiteStore (same interface)
  tools/
    __init__.py         tool registry (add new tools here)
    common.py           HTTP client, TTL cache, error handling
    web.py              web search, news, webpage reader, RSS feeds
    social.py           X/Bluesky/Mastodon, Reddit, Hacker News, YouTube, GitHub, Stack Overflow
    finance.py          stocks, history, company news, market overview, mutual funds, crypto, currency
    data.py             weather + air quality, sports, earthquakes, date/time, calculator
    knowledge.py        Wikipedia, arXiv
supabase/schema.sql     database schema
```

## Adding a new data source

```python
# chatbot/tools/data.py (or any tools module)
@tool
@safe
@cached(ttl=300)
def my_source(query: str) -> str:
    """Describe exactly when the model should use this tool. The model reads this docstring."""
    return to_text(http_get("https://api.example.com/search", params={"q": query}).json())
```

Then add `my_source` to `ALL_TOOLS` and `TOOL_LABELS` in `chatbot/tools/__init__.py`.

## Notes

- **Model choice:** `llama-3.3-70b-versatile` (default) and `openai/gpt-oss-120b` are the most reliable at
  tool calling on Groq. Small models like `llama-3.1-8b-instant` are faster but pick tools less accurately.
- **X / Twitter:** the X API's search endpoint requires a paid plan (Basic or above). Without
  `X_BEARER_TOKEN`, `social_posts` reads Bluesky and Mastodon, where many researchers and journalists
  now post, and falls back to web-search results from x.com. Ask e.g. *"latest posts from karpathy"*.
- **Groq rate limits:** 24 tool definitions add ~2.4k tokens per model call. On Groq's free tier, untick
  sources you don't need in ⚙️ Settings → *Data sources* to save tokens.
- **Reddit:** Reddit blocks most anonymous API traffic. For real post scores and comments, create a free
  "script" app at https://www.reddit.com/prefs/apps and set `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`.
- **User agent:** set `HTTP_USER_AGENT` to identify your bot with your own contact, e.g.
  `MyChatbot/1.0 (https://yoursite.com)`. Wikipedia requires this and may throttle generic clients.
- **Deploying to Streamlit Cloud:** put the same keys in the app's *Secrets*. `config.py` reads them automatically.
