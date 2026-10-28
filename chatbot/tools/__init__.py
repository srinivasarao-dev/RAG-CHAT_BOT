"""Tool registry. Add a new data source by writing a @tool function and listing it here."""
from langchain_core.tools import BaseTool

from chatbot.tools.data import calculator, current_datetime, earthquakes, sports_scores, weather
from chatbot.tools.finance import (
    crypto_price,
    currency_convert,
    market_overview,
    mutual_fund_nav,
    stock_history,
    stock_news,
    stock_quote,
)
from chatbot.tools.knowledge import arxiv_search, wikipedia
from chatbot.tools.social import (
    github_search,
    hacker_news,
    reddit_search,
    social_posts,
    stackoverflow_search,
    youtube_search,
)
from chatbot.tools.web import news_search, read_webpage, rss_feed, web_search

# label shown in the UI, grouped by category
TOOL_LABELS = {
    # web & news
    "web_search": "🌐 Web search",
    "news_search": "📰 News",
    "read_webpage": "📄 Read webpage",
    "rss_feed": "📡 RSS / blogs",
    # social & communities
    "social_posts": "🐦 X / Bluesky / Mastodon",
    "reddit_search": "👾 Reddit",
    "hacker_news": "🟧 Hacker News",
    "youtube_search": "🎥 YouTube",
    "github_search": "🐙 GitHub",
    "stackoverflow_search": "💬 Stack Overflow",
    # markets
    "stock_quote": "📈 Stock quote",
    "stock_history": "📊 Stock history",
    "stock_news": "🗞️ Company news",
    "market_overview": "🌍 Markets overview",
    "mutual_fund_nav": "🏦 Mutual funds (India)",
    "crypto_price": "🪙 Crypto",
    "currency_convert": "💱 Currency",
    # world data
    "weather": "🌦️ Weather & air quality",
    "sports_scores": "🏟️ Sports scores",
    "earthquakes": "🌋 Earthquakes",
    "current_datetime": "🕒 Date & time",
    "calculator": "🧮 Calculator",
    # reference
    "wikipedia": "📚 Wikipedia",
    "arxiv_search": "🔬 arXiv papers",
}

ALL_TOOLS: dict[str, BaseTool] = {
    t.name: t
    for t in [
        web_search, news_search, read_webpage, rss_feed,
        social_posts, reddit_search, hacker_news, youtube_search, github_search, stackoverflow_search,
        stock_quote, stock_history, stock_news, market_overview, mutual_fund_nav, crypto_price, currency_convert,
        weather, sports_scores, earthquakes, current_datetime, calculator,
        wikipedia, arxiv_search,
    ]
}

assert set(ALL_TOOLS) == set(TOOL_LABELS), "every tool needs a UI label"


def get_tools(enabled: list[str] | None = None) -> list[BaseTool]:
    if enabled is None:
        return list(ALL_TOOLS.values())
    return [ALL_TOOLS[name] for name in enabled if name in ALL_TOOLS]
