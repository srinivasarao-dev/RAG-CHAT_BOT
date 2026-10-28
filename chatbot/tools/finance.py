"""Markets and money.

  stock_quote:       Finnhub real-time quote (FINNHUB_API_KEY) -> Yahoo Finance (yfinance)
  stock_history:     Yahoo Finance (yfinance)
  stock_news:        Finnhub company news (key) -> news search for the ticker
  market_overview:   Yahoo Finance batch quotes for indices, commodities, FX, crypto
  mutual_fund_nav:   mfapi.in (Indian mutual funds, AMFI data, no key)
  crypto_price:      CoinGecko API (no key)
  currency_convert:  Frankfurter API (ECB rates, no key) -> open.er-api.com
"""
import warnings
from datetime import date, datetime, timedelta, timezone

from langchain_core.tools import tool

from chatbot.config import settings
from chatbot.tools.common import cached, http_get, safe, to_text

warnings.filterwarnings("ignore", module="yfinance")

_MARKETS = {
    "us": {"^GSPC": "S&P 500", "^IXIC": "Nasdaq", "^DJI": "Dow Jones", "^VIX": "VIX volatility", "^TNX": "US 10Y yield"},
    "india": {"^NSEI": "Nifty 50", "^BSESN": "Sensex", "^NSEBANK": "Nifty Bank", "INR=X": "USD/INR"},
    "europe": {"^FTSE": "FTSE 100", "^GDAXI": "DAX", "^FCHI": "CAC 40", "EURUSD=X": "EUR/USD"},
    "asia": {"^N225": "Nikkei 225", "^HSI": "Hang Seng", "000001.SS": "Shanghai Composite", "JPY=X": "USD/JPY"},
    "commodities": {"GC=F": "Gold", "SI=F": "Silver", "CL=F": "WTI crude oil", "BZ=F": "Brent crude", "NG=F": "Natural gas"},
    "crypto": {"BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "SOL-USD": "Solana"},
}


def _r(x):
    return round(float(x), 4) if x is not None else None


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


@tool
@safe
@cached(ttl=30)
def stock_quote(symbol: str) -> str:
    """Get the latest real-time stock / ETF / index quote for a ticker symbol
    (e.g. AAPL, TSLA, ^GSPC, RELIANCE.NS for NSE India, TCS.BO for BSE). If you only know the
    company name, find the ticker with web_search first."""
    sym = symbol.strip().upper()

    # Finnhub: true real-time for US listings on the free tier
    if settings.finnhub_api_key and "." not in sym and not sym.startswith("^"):
        try:
            q = http_get("https://finnhub.io/api/v1/quote", params={"symbol": sym, "token": settings.finnhub_api_key}).json()
            if q.get("c"):
                return to_text(
                    {
                        "symbol": sym, "source": "finnhub", "price": q["c"], "change": q.get("d"),
                        "change_pct": q.get("dp"), "open": q.get("o"), "day_high": q.get("h"), "day_low": q.get("l"),
                        "previous_close": q.get("pc"),
                        "as_of": datetime.fromtimestamp(q["t"], timezone.utc).isoformat() if q.get("t") else None,
                    }
                )
        except Exception:
            pass

    import yfinance as yf

    fi = yf.Ticker(sym).fast_info
    price, prev = fi.get("lastPrice"), fi.get("previousClose")
    if not price:
        return f"Error: no market data for '{symbol}'. Check the ticker (use web_search to find it)."
    change = price - prev if prev else None
    return to_text(
        {
            "symbol": sym,
            "source": "yahoo_finance",
            "currency": fi.get("currency"),
            "price": _r(price),
            "previous_close": _r(prev),
            "change": _r(change),
            "change_pct": round(change / prev * 100, 2) if prev else None,
            "day_high": _r(fi.get("dayHigh")),
            "day_low": _r(fi.get("dayLow")),
            "volume": fi.get("lastVolume"),
            "market_cap": fi.get("marketCap"),
            "52w_high": _r(fi.get("yearHigh")),
            "52w_low": _r(fi.get("yearLow")),
            "exchange_timezone": fi.get("timezone"),
            "fetched_at": _now_utc(),
            "note": "Real-time during market hours (may be delayed up to 15 min on some exchanges), otherwise last close.",
        }
    )


@tool
@safe
@cached(ttl=300)
def stock_history(symbol: str, period: str = "1mo") -> str:
    """Get price performance of a stock/index/crypto ticker over a period: return %, high, low and
    recent closing prices. period: 5d|1mo|3mo|6mo|ytd|1y|5y|max."""
    import yfinance as yf

    period = period if period in {"5d", "1mo", "3mo", "6mo", "ytd", "1y", "2y", "5y", "max"} else "1mo"
    interval = "1d" if period in {"5d", "1mo", "3mo", "6mo", "ytd"} else "1wk"
    h = yf.Ticker(symbol.strip().upper()).history(period=period, interval=interval)
    if h.empty:
        return f"Error: no price history for '{symbol}'."
    closes = h["Close"]
    first, last = float(closes.iloc[0]), float(closes.iloc[-1])
    step = max(1, len(closes) // 12)  # ~12 evenly spaced points keeps the output small
    return to_text(
        {
            "symbol": symbol.upper(),
            "period": period,
            "start": str(h.index[0].date()),
            "end": str(h.index[-1].date()),
            "start_price": _r(first),
            "end_price": _r(last),
            "return_pct": round((last / first - 1) * 100, 2),
            "period_high": _r(h["High"].max()),
            "period_low": _r(h["Low"].min()),
            "closes": {str(i.date()): _r(v) for i, v in closes.iloc[::step].items()},
        }
    )


@tool
@safe
@cached(ttl=600)
def stock_news(symbol: str, days: int = 7) -> str:
    """Get the latest news headlines about a specific company / ticker (e.g. earnings, deals, analyst moves)."""
    sym = symbol.strip().upper()
    if settings.finnhub_api_key:
        try:
            to_d = date.today()
            items = http_get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": sym.split(".")[0], "from": str(to_d - timedelta(days=max(1, days))), "to": str(to_d), "token": settings.finnhub_api_key},
            ).json()
            if items:
                return to_text(
                    {
                        "source": "finnhub",
                        "articles": [
                            {
                                "headline": a.get("headline"),
                                "source": a.get("source"),
                                "published": datetime.fromtimestamp(a["datetime"], timezone.utc).isoformat() if a.get("datetime") else None,
                                "summary": (a.get("summary") or "")[:300],
                                "url": a.get("url"),
                            }
                            for a in items[:10]
                        ],
                    }
                )
        except Exception:
            pass
    from chatbot.tools.web import news_search

    return news_search.invoke({"query": f"{sym} stock", "max_results": 8})


@tool
@safe
@cached(ttl=60)
def market_overview(region: str = "us") -> str:
    """Snapshot of major markets right now: indices, yields, FX, commodities or crypto with daily % change.
    region: us|india|europe|asia|commodities|crypto|all."""
    import yfinance as yf

    region = region.lower().strip()
    groups = _MARKETS if region == "all" else {region: _MARKETS.get(region, _MARKETS["us"])}
    names = {sym: name for g in groups.values() for sym, name in g.items()}
    df = yf.download(list(names), period="5d", interval="1d", progress=False, auto_adjust=True)["Close"]
    out = {}
    for group, syms in groups.items():
        rows = []
        for sym in syms:
            if sym not in df:
                continue
            s = df[sym].dropna()  # markets that haven't opened today have NaN for today
            if s.empty:
                continue
            last, prev = float(s.iloc[-1]), float(s.iloc[-2]) if len(s) > 1 else None
            rows.append(
                {
                    "name": names[sym], "symbol": sym, "last": _r(last),
                    "change_pct": round((last / prev - 1) * 100, 2) if prev else None,
                    "date": str(s.index[-1].date()),
                }
            )
        out[group] = rows
    return to_text({"fetched_at": _now_utc(), "markets": out})


@tool
@safe
@cached(ttl=3600)
def mutual_fund_nav(fund_name: str) -> str:
    """Get the latest NAV (net asset value) and recent returns of an Indian mutual fund by name
    (e.g. 'Parag Parikh Flexi Cap Direct Growth')."""
    found = http_get("https://api.mfapi.in/mf/search", params={"q": fund_name}).json()
    if not found:
        return f"Error: no Indian mutual fund matching '{fund_name}'."
    # Prefer Direct-Growth plans unless the user asked otherwise
    wanted = fund_name.lower()
    def score(f):
        n = f["schemeName"].lower()
        return ("direct" in n) == ("regular" not in wanted) and "growth" in n
    scheme = sorted(found, key=score, reverse=True)[0]
    data = http_get(f"https://api.mfapi.in/mf/{scheme['schemeCode']}").json()
    navs = data.get("data", [])
    if not navs:
        return f"Error: no NAV data for {scheme['schemeName']}."
    latest = float(navs[0]["nav"])

    def ret(days: int):
        if len(navs) <= days:
            return None
        return round((latest / float(navs[days]["nav"]) - 1) * 100, 2)

    return to_text(
        {
            "scheme": data.get("meta", {}).get("scheme_name"),
            "fund_house": data.get("meta", {}).get("fund_house"),
            "category": data.get("meta", {}).get("scheme_category"),
            "nav": latest,
            "nav_date": navs[0]["date"],
            "return_1m_pct": ret(21),
            "return_1y_pct": ret(250),
            "return_3y_pct": ret(750),
            "other_matches": [f["schemeName"] for f in found[:5] if f is not scheme],
        }
    )


@tool
@safe
@cached(ttl=60)
def crypto_price(coin: str, vs_currency: str = "usd") -> str:
    """Get the live price, 24h change, market cap and volume of a cryptocurrency
    (name or symbol, e.g. 'bitcoin', 'ETH', 'solana')."""
    found = http_get("https://api.coingecko.com/api/v3/search", params={"query": coin}).json().get("coins", [])
    if not found:
        return f"Error: cryptocurrency '{coin}' not found."
    coin_id = found[0]["id"]
    vs = vs_currency.lower()
    data = http_get(
        "https://api.coingecko.com/api/v3/simple/price",
        params={
            "ids": coin_id,
            "vs_currencies": vs,
            "include_24hr_change": "true",
            "include_market_cap": "true",
            "include_24hr_vol": "true",
            "include_last_updated_at": "true",
        },
    ).json().get(coin_id, {})
    return to_text(
        {
            "coin": found[0].get("name"),
            "symbol": found[0].get("symbol"),
            "currency": vs,
            "price": data.get(vs),
            "change_24h_pct": data.get(f"{vs}_24h_change"),
            "market_cap": data.get(f"{vs}_market_cap"),
            "volume_24h": data.get(f"{vs}_24h_vol"),
            "last_updated": datetime.fromtimestamp(data["last_updated_at"], timezone.utc).isoformat() if data.get("last_updated_at") else None,
        }
    )


@tool
@safe
@cached(ttl=600)
def currency_convert(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between fiat currencies using the latest exchange rates (e.g. 100 USD to INR)."""
    src, dst = from_currency.upper().strip(), to_currency.upper().strip()
    try:
        data = http_get("https://api.frankfurter.app/latest", params={"from": src, "to": dst}).json()
        rate, as_of = data["rates"][dst], data.get("date")
    except Exception:
        data = http_get(f"https://open.er-api.com/v6/latest/{src}").json()
        rate, as_of = data["rates"][dst], data.get("time_last_update_utc")
    return f"{amount:,.2f} {src} = {amount * rate:,.4f} {dst} (rate {rate}, as of {as_of})"
