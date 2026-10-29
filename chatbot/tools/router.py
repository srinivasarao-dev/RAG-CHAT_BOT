"""Pick only the tools a question is likely to need.

Every tool definition is sent to the LLM on every call (~100 tokens each). Sending all 24 costs ~2.4k
tokens per step, which quickly exhausts Groq's free-tier per-minute limit. A cheap keyword router
cuts that to a handful of tools. General tools are always included, so nothing becomes unreachable:
the model can still fall back to web/news search for anything the router didn't anticipate.
"""
import re

ALWAYS = ["web_search", "news_search", "read_webpage", "current_datetime", "calculator"]

ROUTES: dict[str, tuple[list[str], list[str]]] = {
    "markets": (
        ["stock_quote", "stock_history", "stock_news", "market_overview"],
        ["stock", "share", "shares", "market", "markets", "nifty", "sensex", "nasdaq", "dow", "s&p", "index",
         "indices", "ticker", "earnings", "ipo", "invest", "portfolio", "dividend", "nse", "bse", "nyse",
         "gold", "silver", "oil", "crude", "brent", "bond", "yield", "vix", "trading", "rally", "crash",
         "bull", "bear", "valuation", "market cap", "price target", "52 week", "returns"],
    ),
    "funds": (["mutual_fund_nav"], ["mutual fund", "fund", "nav", "sip", "elss", "amc", "flexi cap", "large cap", "mid cap", "small cap"]),
    "crypto": (["crypto_price"], ["crypto", "bitcoin", "btc", "ethereum", "eth", "solana", "sol", "dogecoin", "doge",
                                  "xrp", "coin", "token", "binance", "usdt", "altcoin", "memecoin"]),
    "currency": (["currency_convert"], ["convert", "conversion", "exchange rate", "forex", "fx", "rupee", "inr", "usd",
                                        "eur", "gbp", "jpy", "aed", "dollar", "euro", "pound", "yen", "dirham", "currency"]),
    "social": (["social_posts", "reddit_search", "hacker_news"],
               ["twitter", "tweet", "tweets", "x.com", "post", "posts", "posting", "posted", "bluesky", "mastodon",
                "reddit", "subreddit", "opinion", "opinions", "saying", "people think", "reaction", "reactions",
                "trending", "viral", "sentiment", "hacker news", "hn", "community", "experts", "founder", "ceo"]),
    "video": (["youtube_search"], ["youtube", "video", "videos", "watch", "highlights", "trailer", "tutorial", "podcast"]),
    "dev": (["github_search", "github_file", "package_info", "stackoverflow_search", "hacker_news"],
            ["github", "repo", "repository", "open source", "open-source", "library", "framework", "stackoverflow",
             "stack overflow", "error", "exception", "bug", "code", "python", "javascript", "api", "npm", "pip",
             "package", "version", "release", "install", "upgrade", "sdk", "typescript", "react", "django", "fastapi"]),
    "weather": (["weather"], ["weather", "rain", "raining", "temperature", "forecast", "humid", "humidity", "aqi",
                              "air quality", "pollution", "sunny", "storm", "cyclone", "monsoon", "snow", "uv", "sunrise", "sunset", "hot", "cold"]),
    "sports": (["sports_scores"], ["score", "scores", "match", "matches", "game", "games", "fixture", "fixtures", "nba",
                                   "nfl", "mlb", "nhl", "premier league", "epl", "la liga", "champions league", "football",
                                   "soccer", "basketball", "f1", "formula 1", "grand prix", "tennis", "atp", "ufc", "league", "standings"]),
    "quakes": (["earthquakes"], ["earthquake", "earthquakes", "quake", "tremor", "seismic", "tsunami"]),
    "reference": (["wikipedia"], ["who is", "who was", "what is", "history of", "biography", "born", "founded", "wikipedia", "definition", "explain"]),
    "research": (["arxiv_search"], ["paper", "papers", "arxiv", "research", "study", "studies", "preprint", "journal"]),
    "feeds": (["rss_feed"], ["blog", "substack", "rss", "feed", "newsletter"]),
}


def _matches(text: str, keyword: str) -> bool:
    if " " in keyword or not keyword.isalnum():
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def route_tools(question: str, recent_user_messages: list[str] | None = None, enabled: list[str] | None = None) -> list[str]:
    """Tool names for this question. Recent user messages help follow-ups like "and in USD?"."""
    text = " ".join([question, *(recent_user_messages or [])]).lower()
    # $AAPL / TSLA-style tickers suggest market questions
    if re.search(r"\$[a-z]{1,5}\b|\b[A-Z]{2,5}\.(NS|BO)\b", " ".join([question, *(recent_user_messages or [])])):
        text += " stock"
    names = list(ALWAYS)
    for tools, keywords in ROUTES.values():
        if any(_matches(text, k) for k in keywords):
            names += [t for t in tools if t not in names]
    if enabled is not None:
        names = [n for n in names if n in enabled]
    return names
