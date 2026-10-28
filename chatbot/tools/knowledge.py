"""Reference knowledge: Wikipedia (REST API -> wikipedia library) and arXiv (export API -> arxiv library)."""
from langchain_core.tools import tool

from chatbot.tools.common import cached, http_get, safe, to_text


@tool
@safe
@cached(ttl=3600)
def wikipedia(query: str) -> str:
    """Look up encyclopedic background on a person, place, concept or event on Wikipedia.
    Not for breaking news. Use web_search or news_search for that."""
    try:  # one request: search + intro extracts + URLs
        data = http_get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": 2,
                "prop": "extracts|info",
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "format": "json",
            },
        ).json()
        pages = sorted((data.get("query") or {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
        if pages:
            return to_text([{"title": p.get("title"), "summary": p.get("extract"), "url": p.get("fullurl")} for p in pages])
    except Exception:
        pass
    try:
        import wikipedia as wiki_lib

        title = wiki_lib.search(query, results=1)
        if title:
            page = wiki_lib.page(title[0], auto_suggest=False)
            return to_text({"title": page.title, "summary": page.summary, "url": page.url})
    except Exception:
        pass

    # Wikipedia rate-limited or down: use search-engine snippets of Wikipedia pages instead
    from chatbot.tools.web import _ddgs

    hits = [h for h in _ddgs().text(f"{query} wikipedia", max_results=6) if "wikipedia.org" in (h.get("href") or "")]
    if not hits:
        return f"No Wikipedia article found for '{query}'."
    return to_text([{"title": h.get("title"), "summary": h.get("body"), "url": h.get("href")} for h in hits[:3]])


@tool
@safe
@cached(ttl=3600)
def arxiv_search(query: str, max_results: int = 4) -> str:
    """Search arXiv for scientific papers (newest first). Returns titles, authors, dates, abstracts and links."""
    max_results = max(1, min(max_results, 10))
    try:  # official export API directly: much faster than the library's built-in delays
        import feedparser

        feed = feedparser.parse(
            http_get(
                "https://export.arxiv.org/api/query",
                params={"search_query": f"all:{query}", "max_results": max_results, "sortBy": "relevance"},
            ).content
        )
        papers = [
            {
                "title": " ".join(e.title.split()),
                "authors": ", ".join(a.get("name", "") for a in e.get("authors", [])[:5]),
                "published": e.get("published", "")[:10],
                "abstract": " ".join(e.get("summary", "").split())[:600],
                "url": e.get("id"),
            }
            for e in feed.entries
        ]
        if papers:
            papers.sort(key=lambda p: p["published"], reverse=True)
            return to_text(papers)
    except Exception:
        pass

    import arxiv

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    papers = [
        {
            "title": r.title,
            "authors": ", ".join(a.name for a in r.authors[:5]),
            "published": r.published.date().isoformat(),
            "abstract": r.summary[:600],
            "url": r.entry_id,
        }
        for r in arxiv.Client().results(search)
    ]
    papers.sort(key=lambda p: p["published"], reverse=True)
    return to_text(papers or f"No arXiv papers found for '{query}'.")
