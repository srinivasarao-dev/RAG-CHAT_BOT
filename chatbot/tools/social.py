"""Community / social sources.

Fallback chains:
  reddit_search:   PRAW (official API, if keys) -> Reddit public JSON -> DuckDuckGo results on reddit.com
  hacker_news:     Algolia HN API (no key)
  youtube_search:  yt-dlp library (no key)
  social_posts:    X API v2 (X_BEARER_TOKEN) + Bluesky (public author feeds; search with app password)
                   + Mastodon (public) -> web search results on x.com
  github_search:   GitHub REST API (GITHUB_TOKEN optional, raises the rate limit)
  stackoverflow_search: Stack Exchange API (no key)
"""
import re
import time
from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool

from chatbot.config import settings
from chatbot.tools.common import cached, http_get, http_post, safe, to_text

_VALID_SORTS = {"relevance", "hot", "top", "new", "comments"}
_VALID_TIMES = {"hour", "day", "week", "month", "year", "all"}


def _reddit_praw(query: str, subreddit: str, sort: str, time_filter: str, limit: int) -> list[dict]:
    import praw

    reddit = praw.Reddit(
        client_id=settings.reddit_client_id,
        client_secret=settings.reddit_client_secret,
        user_agent=settings.reddit_user_agent,
        check_for_async=False,
    )
    sub = reddit.subreddit(subreddit or "all")
    if query:
        posts = sub.search(query, sort=sort, time_filter=time_filter, limit=limit)
    else:
        posts = getattr(sub, sort if sort in {"hot", "new", "top"} else "hot")(limit=limit)
    return [
        {
            "title": p.title,
            "subreddit": p.subreddit.display_name,
            "score": p.score,
            "comments": p.num_comments,
            "created_utc": int(p.created_utc),
            "url": f"https://www.reddit.com{p.permalink}",
            "text": (p.selftext or "")[:400],
        }
        for p in posts
    ]


def _reddit_public(query: str, subreddit: str, sort: str, time_filter: str, limit: int) -> list[dict]:
    base = f"https://www.reddit.com/r/{subreddit}" if subreddit else "https://www.reddit.com"
    if query:
        url = f"{base}/search.json"
        params = {"q": query, "sort": sort, "t": time_filter, "limit": limit, "restrict_sr": bool(subreddit)}
    else:
        url = f"{base}/{sort if sort in {'hot', 'new', 'top'} else 'hot'}.json"
        params = {"limit": limit, "t": time_filter}
    data = http_get(url, params=params).json()
    return [
        {
            "title": c["data"].get("title"),
            "subreddit": c["data"].get("subreddit"),
            "score": c["data"].get("score"),
            "comments": c["data"].get("num_comments"),
            "created_utc": int(c["data"].get("created_utc", 0)),
            "url": "https://www.reddit.com" + c["data"].get("permalink", ""),
            "text": (c["data"].get("selftext") or "")[:400],
        }
        for c in data.get("data", {}).get("children", [])
    ]


def _reddit_via_search(query: str, subreddit: str, limit: int) -> list[dict]:
    """Last resort when Reddit's API isn't configured or blocks anonymous requests."""
    from chatbot.tools.web import _ddgs

    q = " ".join(filter(None, ["reddit", f"r/{subreddit}" if subreddit else "", query]))
    hits = _ddgs().text(q, max_results=limit * 3)
    posts = [
        {"title": h.get("title"), "url": h.get("href"), "text": h.get("body")}
        for h in hits
        if "reddit.com" in (h.get("href") or "")
    ]
    return posts[:limit]


@tool
@safe
@cached()
def reddit_search(query: str = "", subreddit: str = "", sort: str = "relevance", time_filter: str = "week", limit: int = 8) -> str:
    """Search Reddit posts and discussions to learn what people are saying right now.
    Args: query (leave empty to browse a subreddit), subreddit (e.g. 'technology', without r/),
    sort: relevance|hot|top|new|comments, time_filter: hour|day|week|month|year|all."""
    subreddit = subreddit.removeprefix("r/").strip()
    sort = sort if sort in _VALID_SORTS else "relevance"
    time_filter = time_filter if time_filter in _VALID_TIMES else "week"
    limit = max(1, min(limit, 25))
    if settings.reddit_client_id and settings.reddit_client_secret:
        try:
            return to_text({"source": "reddit_api", "posts": _reddit_praw(query, subreddit, sort, time_filter, limit)})
        except Exception:
            pass
    try:
        return to_text({"source": "reddit_public_json", "posts": _reddit_public(query, subreddit, sort, time_filter, limit)})
    except Exception:
        posts = _reddit_via_search(query, subreddit, limit)
        if not posts:
            return (
                "Error: Reddit blocks anonymous access and web search found no Reddit threads. "
                "Set REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET for full Reddit access, or try web_search."
            )
        return to_text({"source": "web_search_reddit", "note": "search snippets, not live scores", "posts": posts})


@tool
@safe
@cached()
def hacker_news(query: str = "", limit: int = 10) -> str:
    """Get Hacker News stories: the current front page (empty query) or the most recent stories
    matching a query. Good for tech/startup/programming news."""
    limit = max(1, min(limit, 30))
    if query:
        url = "https://hn.algolia.com/api/v1/search_by_date"
        params = {"query": query, "tags": "story", "hitsPerPage": limit}
    else:
        url = "https://hn.algolia.com/api/v1/search"
        params = {"tags": "front_page", "hitsPerPage": limit}
    hits = http_get(url, params=params).json().get("hits", [])
    return to_text(
        [
            {
                "title": h.get("title"),
                "points": h.get("points"),
                "comments": h.get("num_comments"),
                "created_at": h.get("created_at"),
                "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                "discussion": f"https://news.ycombinator.com/item?id={h.get('objectID')}",
            }
            for h in hits
        ]
    )


@tool
@safe
@cached(ttl=600)
def youtube_search(query: str, limit: int = 5) -> str:
    """Search YouTube videos. Returns titles, channels, view counts, durations and links."""
    import yt_dlp

    limit = max(1, min(limit, 15))
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    return to_text(
        [
            {
                "title": e.get("title"),
                "channel": e.get("channel") or e.get("uploader"),
                "views": e.get("view_count"),
                "duration_s": e.get("duration"),
                "url": e.get("url") or f"https://www.youtube.com/watch?v={e.get('id')}",
            }
            for e in (info or {}).get("entries", [])
        ]
    )


# ---------------------------------------------------------------- social posts (X / Bluesky / Mastodon)

_bsky_session: dict = {}


def _strip_html(html: str) -> str:
    from bs4 import BeautifulSoup

    return BeautifulSoup(html or "", "html.parser").get_text(" ").strip()


def _x_posts(query: str, from_user: str, limit: int) -> list[dict]:
    q = " ".join(filter(None, [f"from:{from_user.lstrip('@')}" if from_user else "", query, "-is:retweet"]))
    data = http_get(
        "https://api.x.com/2/tweets/search/recent",
        params={
            "query": q,
            "max_results": max(10, min(limit, 100)),
            "tweet.fields": "created_at,public_metrics,author_id",
            "expansions": "author_id",
            "user.fields": "username,name,verified",
        },
        headers={"Authorization": f"Bearer {settings.x_bearer_token}"},
    ).json()
    users = {u["id"]: u for u in data.get("includes", {}).get("users", [])}
    posts = []
    for t in data.get("data", []):
        u = users.get(t.get("author_id"), {})
        m = t.get("public_metrics", {})
        posts.append(
            {
                "platform": "x",
                "author": f"{u.get('name')} (@{u.get('username')})" + (" ✓" if u.get("verified") else ""),
                "text": t.get("text"),
                "created_at": t.get("created_at"),
                "likes": m.get("like_count"),
                "reposts": m.get("retweet_count"),
                "url": f"https://x.com/{u.get('username', 'i')}/status/{t['id']}",
            }
        )
    return posts[:limit]


def _bsky_token() -> str | None:
    if not (settings.bluesky_handle and settings.bluesky_app_password):
        return None
    if _bsky_session.get("expires", 0) > time.time():
        return _bsky_session["jwt"]
    resp = http_post(
        "https://bsky.social/xrpc/com.atproto.server.createSession",
        json={"identifier": settings.bluesky_handle, "password": settings.bluesky_app_password},
    ).json()
    _bsky_session.update(jwt=resp["accessJwt"], expires=time.time() + 90 * 60)
    return resp["accessJwt"]


def _bsky_post(p: dict) -> dict:
    handle = p["author"]["handle"]
    return {
        "platform": "bluesky",
        "author": f"{p['author'].get('displayName') or handle} (@{handle})",
        "text": p.get("record", {}).get("text"),
        "created_at": p.get("record", {}).get("createdAt"),
        "likes": p.get("likeCount"),
        "reposts": p.get("repostCount"),
        "url": f"https://bsky.app/profile/{handle}/post/{p['uri'].rsplit('/', 1)[-1]}",
    }


def _bluesky_posts(query: str, from_user: str, limit: int) -> list[dict]:
    user = from_user.lstrip("@")
    if "." in user and "@" not in user:
        # Author feeds are public: no login needed
        feed = http_get(
            "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed",
            params={"actor": user, "limit": min(limit * 2, 50), "filter": "posts_no_replies"},
        ).json().get("feed", [])
        posts = [_bsky_post(item["post"]) for item in feed if "reason" not in item]  # skip reposts
        if query:
            words = [w.lower() for w in query.split()]
            posts = [p for p in posts if any(w in (p["text"] or "").lower() for w in words)] or posts
        return posts[:limit]
    token = _bsky_token()
    if not token or not query:
        return []
    data = http_get(
        "https://bsky.social/xrpc/app.bsky.feed.searchPosts",
        params={"q": query, "limit": min(limit, 50), "sort": "latest"},
        headers={"Authorization": f"Bearer {token}"},
    ).json()
    return [_bsky_post(p) for p in data.get("posts", [])][:limit]


def _mastodon_posts(query: str, from_user: str, limit: int) -> list[dict]:
    base = "https://mastodon.social"
    user = from_user.lstrip("@")
    if user.count("@") == 1:  # user@instance
        acct = http_get(f"{base}/api/v1/accounts/lookup", params={"acct": user}).json()
        statuses = http_get(
            f"{base}/api/v1/accounts/{acct['id']}/statuses", params={"limit": limit, "exclude_reblogs": "true"}
        ).json()
    elif query and not user:
        # Public hashtag timeline: "machine learning" -> #machinelearning
        tag = re.sub(r"\W", "", query if len(query) <= 30 else query.split()[0])
        statuses = http_get(f"{base}/api/v1/timelines/tag/{tag}", params={"limit": limit}).json()
    else:
        return []
    return [
        {
            "platform": "mastodon",
            "author": f"{s['account'].get('display_name') or s['account']['acct']} (@{s['account']['acct']})",
            "text": _strip_html(s.get("content"))[:500],
            "created_at": s.get("created_at"),
            "likes": s.get("favourites_count"),
            "reposts": s.get("reblogs_count"),
            "url": s.get("url"),
        }
        for s in statuses
    ][:limit]


@tool
@safe
@cached()
def social_posts(query: str = "", from_user: str = "", limit: int = 10) -> str:
    """Latest public posts from X (Twitter), Bluesky and Mastodon: what experts, founders, companies,
    journalists and people are posting right now. query: topic or keywords. from_user: one account to read,
    e.g. 'sundarpichai' (X), 'nytimes.com' (Bluesky handle), 'Gargron@mastodon.social' (Mastodon)."""
    limit = max(1, min(limit, 25))
    posts, notes = [], []
    sources = [
        ("x", _x_posts if settings.x_bearer_token else None),
        ("bluesky", _bluesky_posts),
        ("mastodon", _mastodon_posts),
    ]
    for name, fn in sources:
        if fn is None:
            notes.append(f"{name}: no API token configured")
            continue
        try:
            posts.extend(fn(query, from_user, limit))
        except Exception as e:  # noqa: BLE001
            notes.append(f"{name}: {type(e).__name__}")

    if not posts:
        # No API returned posts: search-engine results for posts on X/Twitter
        from chatbot.tools.web import _ddgs

        q = " ".join(filter(None, [from_user, query, "twitter"]))
        hits = _ddgs().text(q, max_results=limit * 2)
        posts = [
            {"platform": "x (via web search)", "title": h.get("title"), "text": h.get("body"), "url": h.get("href")}
            for h in hits
            if any(d in (h.get("href") or "") for d in ("x.com/", "twitter.com/"))
        ][:limit]

    posts.sort(key=lambda p: p.get("created_at") or "", reverse=True)
    if not posts:
        return f"No recent posts found. {'; '.join(notes)}"
    return to_text({"posts": posts[: limit * 2], "notes": notes})


# ---------------------------------------------------------------- developer communities


@tool
@safe
@cached(ttl=600)
def github_search(query: str = "", created_within_days: int = 0, language: str = "", limit: int = 8) -> str:
    """Search GitHub repositories, sorted by stars. Use created_within_days (e.g. 7) to find trending new
    projects and language to filter (e.g. 'python'). Leave query empty to get this week's trending repos."""
    parts = [query] if query else []
    if created_within_days or not query:
        since = (datetime.now(timezone.utc) - timedelta(days=created_within_days or 7)).date()
        parts.append(f"created:>{since}")
    if language:
        parts.append(f"language:{language}")
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    data = http_get(
        "https://api.github.com/search/repositories",
        params={"q": " ".join(parts), "sort": "stars", "order": "desc", "per_page": max(1, min(limit, 20))},
        headers=headers,
    ).json()
    return to_text(
        [
            {
                "repo": r["full_name"],
                "description": (r.get("description") or "")[:200],
                "stars": r.get("stargazers_count"),
                "language": r.get("language"),
                "created": (r.get("created_at") or "")[:10],
                "last_push": (r.get("pushed_at") or "")[:10],
                "url": r.get("html_url"),
            }
            for r in data.get("items", [])
        ]
    )


@tool
@safe
@cached(ttl=900)
def stackoverflow_search(query: str, limit: int = 5) -> str:
    """Search Stack Overflow for programming questions that have answers."""
    data = http_get(
        "https://api.stackexchange.com/2.3/search/advanced",
        params={
            "q": query,
            "site": "stackoverflow",
            "order": "desc",
            "sort": "relevance",
            "answers": 1,
            "pagesize": max(1, min(limit, 10)),
        },
    ).json()
    return to_text(
        [
            {
                "title": _strip_html(q.get("title")),
                "score": q.get("score"),
                "answers": q.get("answer_count"),
                "has_accepted_answer": q.get("accepted_answer_id") is not None,
                "tags": q.get("tags"),
                "last_activity": datetime.fromtimestamp(q.get("last_activity_date", 0), timezone.utc).date().isoformat(),
                "url": q.get("link"),
            }
            for q in data.get("items", [])
        ]
    )
