# ⚡ Pulse AI: Real-time Chatbot

A chatbot that answers with **live data**. A Groq-hosted LLM decides which of **24 live sources** to call (web, news, X/Twitter,
Bluesky, Mastodon, Reddit, Hacker News, YouTube, GitHub, Yahoo Finance, Finnhub, crypto, sports, weather and more), streams the answer token by token in a ChatGPT-style web UI,
cites sources, and saves every conversation to **Supabase** (or a local SQLite file).

## How it works

```
Browser UI (static/) ──SSE──► FastAPI (server.py) ──► Agent (chatbot/agent.py, LangChain + Groq tool calling)
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
python server.py
```

and open **http://localhost:8000**.

### The web UI

- ChatGPT-style layout: chat history sidebar (grouped by date, searchable, rename/delete), model picker, light/dark theme
- Answers stream word by word; live "source chips" show which APIs are being called and how long each took
- Markdown answers with tables, syntax-highlighted code (with copy button) and clickable source links
- Stop generating (button or `Esc`), regenerate, copy answer, `Ctrl+K` for a new chat
- Each chat has its own URL (`/#c=<id>`), so refresh keeps your place
- Settings: your name (chats are saved per name), timezone, which data sources to use, and a Groq key if the server has none
- Works on phones (collapsible sidebar)

### AI providers: Groq, Claude, OpenAI (GPT & Codex)

Groq is the default (free tier). Users can also bring their own keys for **Claude** (Anthropic) and
**OpenAI** (GPT and Codex models) in **Settings → AI providers**. Keys entered there stay in the browser
and are sent with each message; they are never stored on the server. To make a provider available to
everyone using your server, set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in `.env` instead.

Each provider's model list is fetched live from its API, so new models appear automatically. Claude runs
with adaptive thinking (effort `medium` in Chat, `high` in Code mode); Codex models use OpenAI's
Responses API.

### Code mode

Switch the composer from **Chat** to **Code** for programming help:

- A coding-focused assistant that writes complete, runnable code, debugs errors and reviews code
- `package_info`: live latest versions from PyPI and npm (so install commands aren't outdated)
- `github_file`: reads real source code from public GitHub repositories
- `run_python`: the assistant runs Python to test its own code before answering
- A **▶ Run** button on every Python code block, with the output shown underneath

Code execution runs Python in a separate process on the server machine, with a 20-second limit and without
your `.env` secrets. It is **not** a security sandbox, so it's enabled automatically only when the server
listens on localhost (`CODE_EXECUTION=auto`). Never enable it on a public deployment.

### Chat with your files

Click the 📎 button (or drag files onto the page, or paste a screenshot) to attach files to a message:

- **Documents**: PDF, Word (.docx), text, Markdown, CSV, JSON, HTML, Jupyter notebooks and source code (up to 20 MB).
  Text is extracted, split into sections and embedded locally with `fastembed` (BAAI/bge-small-en-v1.5,
  no API key). Small files are given to the model in full; large ones are searched, with the best passages
  included automatically and a `search_files` tool for follow-up lookups.
- **Images**: sent directly to vision models (Claude or OpenAI). Groq's models can't read images.
- Files stay attached to the conversation, so follow-up questions keep using them.

Embeddings are stored in Supabase with **pgvector** (`file_chunks` table + `match_file_chunks` function in
`supabase/schema.sql`), or in SQLite locally.

### Login (Supabase Auth)

Set `SUPABASE_ANON_KEY` (the **publishable** key, `sb_publishable_…`) and the app shows an email + password
sign-in screen. Every API call then requires the user's Supabase token, and each user only sees their own
chats and files. In Supabase, under **Authentication → URL Configuration**, set the *Site URL* to your app's
address so email-confirmation links come back to it. Without the key, the app runs without login (fine on
your own computer).

### Deploy online

The repo includes a `Dockerfile` (with the embedding model baked in) and a `render.yaml` blueprint.

1. Run `supabase/schema.sql` in your Supabase project (container disks are temporary, so data must live there).
2. Push the repo to GitHub, then in [Render](https://render.com): **New → Blueprint** → pick the repo.
3. Fill in the secrets it asks for: `GROQ_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY` (secret key),
   `SUPABASE_ANON_KEY` (publishable key), plus any optional API keys.
4. After the first deploy, set Supabase's *Site URL* to the Render URL.

Any Docker host works the same way (`docker build -t pulse-ai . && docker run -p 8000:8000 --env-file .env pulse-ai`).
In production `CODE_EXECUTION` is off and login is on.

### Connect Supabase (optional, recommended)

1. Create a project at https://supabase.com.
2. Open **SQL Editor**, paste [`supabase/schema.sql`](supabase/schema.sql), and click **Run**.
3. Under **Project Settings → API**, copy the project URL (or just the project ref) and the **secret** key (`sb_secret_…`, or the legacy service_role key) into `.env`:
   ```
   SUPABASE_URL=https://xxxx.supabase.co
   SUPABASE_KEY=sb_secret_...
   ```
4. Restart the server. The sidebar shows **Synced to Supabase** with a green dot.

If Supabase isn't configured or can't be reached, the app keeps working with a local `chat_history.db`
and shows a warning in the sidebar.

Tables:
- `conversations`: one row per chat (per `user_id`)
- `messages`: user and assistant messages, plus `metadata` (tools used, model, response time)
- `tool_calls`: every external API call with arguments, output preview and latency (for analytics and debugging)

## Project layout

```
server.py               FastAPI server: REST API for chats + streaming /api/chat (Server-Sent Events)
static/
  index.html            page layout
  app.css               styles (light + dark themes, mobile layout)
  app.js                chat logic: streaming, markdown, history, settings
chatbot/
  config.py             settings from .env; fetches the live Groq model list
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

- **Models:** at startup the server asks Groq which models your key can use, so retired models never show up.
  `openai/gpt-oss-120b` is the default; `openai/gpt-oss-20b` is faster.
- **X / Twitter:** the X API's search endpoint requires a paid plan (Basic or above). Without
  `X_BEARER_TOKEN`, `social_posts` reads Bluesky and Mastodon, where many researchers and journalists
  now post, and falls back to web-search results from x.com. Ask e.g. *"latest posts from karpathy"*.
- **Groq rate limits:** 24 tool definitions add ~2.4k tokens per model call. On Groq's free tier, untick
  sources you don't need in **Settings → Live data sources** to save tokens.
- **Reddit:** Reddit blocks most anonymous API traffic. For real post scores and comments, create a free
  "script" app at https://www.reddit.com/prefs/apps and set `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET`.
- **User agent:** set `HTTP_USER_AGENT` to identify your bot with your own contact, e.g.
  `MyChatbot/1.0 (https://yoursite.com)`. Wikipedia requires this and may throttle generic clients.
- **Deploying:** any Python host works (Render, Railway, Fly.io, a VPS). Start command:
  `python server.py` with `HOST=0.0.0.0` and the host's `PORT`, and add your `.env` values as environment
  variables. The UI has no login: anyone with the URL can chat using your server's Groq key, so keep the
  URL private or add authentication before sharing it publicly.
