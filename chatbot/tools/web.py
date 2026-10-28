"""Web search, news and page reading.

Fallback chains:
  web_search:    Tavily -> Serper (Google results) -> Brave (each if key) -> DuckDuckGo (ddgs library)
  news_search:   NewsAPI (if key)     -> DuckDuckGo News -> Google News RSS (feedparser)
  read_webpage:  trafilatura          -> BeautifulSoup
  rss_feed:      any RSS/Atom feed (feedparser); auto-discovers the feed from a site URL
"""
import urllib.parse

from langchain_core.tools import tool

from chatbot.config import settings
from chatbot.tools.common import cached, http_get, http_post, safe, to_text, trim


def _ddgs():
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS
    return DDGS()


def _tavily(query: str, max_results: int, topic: str = "general") -> list[dict]:
    resp = http_post(
        "https://api.tavily.com/search",
        json={
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "topic": topic,
            "include_answer": True,
        },
    )
    data = resp.json()
    results = [{"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content")} for r in data.get("results", [])]
    if data.get("answer"):
        results.insert(0, {"summary": data["answer"]})
    return results


def _serper(query: str, max_results: int) -> list[dict]:
    data = http_post(
        "https://google.serper.dev/search",
        json={"q": query, "num": max_results},
        headers={"X-API-KEY": settings.serper_api_key},
    ).json()
    results = [{"title": r.get("title"), "url": r.get("link"), "snippet": r.get("snippet"), "date": r.get("date")}
               for r in data.get("organic", [])]
    box = data.get("answerBox") or data.get("knowledgeGraph")
    if box:
        results.insert(0, {"summary": box.get("answer") or box.get("snippet") or box.get("description")})
    return results


def _brave(query: str, max_results: int) -> list[dict]:
    data = http_get(
        "https://api.search.brave.com/res/v1/web/search",
        params={"q": query, "count": max_results},
        headers={"X-Subscription-Token": settings.brave_api_key, "Accept": "application/json"},
    ).json()
    return [{"title": r.get("title"), "url": r.get("url"), "snippet": r.get("description"), "date": r.get("age")}
            for r in (data.get("web") or {}).get("results", [])]


@tool
@safe
@cached()
def web_search(query: str, max_results: int = 6) -> str:
    """Search the live web for up-to-date information. Use for general questions, recent events,
    facts that may have changed, products, people, prices and anything not covered by a more specific tool."""
    max_results = max(1, min(max_results, 10))
    providers = [
        ("tavily", settings.tavily_api_key, lambda: _tavily(query, max_results)),
        ("google_serper", settings.serper_api_key, lambda: _serper(query, max_results)),
        ("brave", settings.brave_api_key, lambda: _brave(query, max_results)),
    ]
    for name, key, fn in providers:
        if not key:
            continue
        try:
            results = fn()
            if results:
                return to_text({"source": name, "results": results})
        except Exception:
            pass  # try the next provider
    hits = _ddgs().text(query, max_results=max_results)
    results = [{"title": h.get("title"), "url": h.get("href"), "snippet": h.get("body")} for h in hits]
    return to_text({"source": "duckduckgo", "results": results})


@tool
@safe
@cached()
def news_search(query: str, max_results: int = 8) -> str:
    """Get the latest news articles about a topic, with publish dates and sources.
    Use for breaking news, current events, 'what happened today/this week' questions."""
    max_results = max(1, min(max_results, 15))

    if settings.newsapi_key:
        try:
            data = http_get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "sortBy": "publishedAt", "pageSize": max_results, "language": "en"},
                headers={"X-Api-Key": settings.newsapi_key},
            ).json()
            articles = [
                {
                    "title": a.get("title"),
                    "source": (a.get("source") or {}).get("name"),
                    "published": a.get("publishedAt"),
                    "url": a.get("url"),
                    "snippet": a.get("description"),
                }
                for a in data.get("articles", [])
            ]
            if articles:
                return to_text({"source": "newsapi", "articles": articles})
        except Exception:
            pass

    try:
        hits = _ddgs().news(query, max_results=max_results)
        articles = [
            {
                "title": h.get("title"),
                "source": h.get("source"),
                "published": h.get("date"),
                "url": h.get("url"),
                "snippet": h.get("body"),
            }
            for h in hits
        ]
        if articles:
            return to_text({"source": "duckduckgo_news", "articles": articles})
    except Exception:
        pass

    import feedparser

    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    )
    feed = feedparser.parse(http_get(url).content)
    articles = [
        {
            "title": e.get("title"),
            "source": (e.get("source") or {}).get("title"),
            "published": e.get("published"),
            "url": e.get("link"),
        }
        for e in feed.entries[:max_results]
    ]
    return to_text({"source": "google_news_rss", "articles": articles})


@tool
@safe
@cached(ttl=600)
def read_webpage(url: str) -> str:
    """Fetch a web page and return its main readable text. Use after a search to read
    the full article/page when snippets are not enough."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    html = http_get(url).text
    try:
        import trafilatura

        text = trafilatura.extract(html, include_comments=False, include_tables=True)
        if text:
            return trim(f"URL: {url}\n\n{text}")
    except Exception:
        pass
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
    return trim(f"URL: {url}\n\n{text}")


@tool
@safe
@cached(ttl=300)
def rss_feed(url: str, limit: int = 10) -> str:
    """Read the latest entries of any RSS/Atom feed or blog, e.g. a company engineering blog, a professional's
    Substack ('https://name.substack.com'), a YouTube channel feed, or a site's homepage (its feed is auto-discovered)."""
    import feedparser

    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    resp = http_get(url)
    feed = feedparser.parse(resp.content)
    if not feed.entries and "html" in resp.headers.get("content-type", ""):
        from bs4 import BeautifulSoup

        link = BeautifulSoup(resp.text, "html.parser").find(
            "link", type=lambda t: t in ("application/rss+xml", "application/atom+xml")
        )
        if link and link.get("href"):
            feed_url = urllib.parse.urljoin(url, link["href"])
            feed = feedparser.parse(http_get(feed_url).content)
        elif "substack.com" in url:
            feed = feedparser.parse(http_get(url.rstrip("/") + "/feed").content)
    if not feed.entries:
        return f"Error: no RSS/Atom feed found at {url}."
    entries = [
        {
            "title": e.get("title"),
            "published": e.get("published") or e.get("updated"),
            "summary": trim(_strip_tags(e.get("summary", "")), 300),
            "url": e.get("link"),
        }
        for e in feed.entries[: max(1, min(limit, 25))]
    ]
    return to_text({"feed": feed.feed.get("title"), "entries": entries})


def _strip_tags(html: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "html.parser").get_text(" ").strip()
