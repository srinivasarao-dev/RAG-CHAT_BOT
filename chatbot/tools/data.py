"""Structured real-time data: time, weather, air quality, earthquakes, sports, math.

  current_datetime:  Python stdlib (zoneinfo)
  weather:           Open-Meteo forecast + air-quality APIs (no key)
  earthquakes:       USGS real-time feed (no key)
  sports_scores:     ESPN public scoreboard API (no key)
  calculator:        safe local arithmetic (LLMs are unreliable at exact math)
"""
import ast
import math
import operator
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

from chatbot.tools.common import cached, http_get, safe, to_text

_WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "dense drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "heavy freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "rain showers", 81: "heavy rain showers", 82: "violent rain showers",
    85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm", 96: "thunderstorm with hail",
    99: "severe thunderstorm with hail",
}


def _aqi_label(aqi):
    if aqi is None:
        return None
    for limit, label in [(50, "good"), (100, "moderate"), (150, "unhealthy for sensitive groups"),
                         (200, "unhealthy"), (300, "very unhealthy")]:
        if aqi <= limit:
            return label
    return "hazardous"


@tool
@safe
def current_datetime(timezone: str = "UTC") -> str:
    """Get the current date and time in a timezone (IANA name, e.g. 'Asia/Kolkata', 'America/New_York')."""
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        tz, timezone = ZoneInfo("UTC"), "UTC"
    now = datetime.now(tz)
    return f"{now:%A, %d %B %Y, %H:%M:%S} ({timezone}, UTC{now:%z})"


@tool
@safe
@cached(ttl=600)
def weather(location: str, days: int = 3) -> str:
    """Get current weather, air quality (AQI, PM2.5) and a daily forecast (up to 7 days) for a city or place."""
    geo = http_get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": location, "count": 1, "language": "en", "format": "json"},
    ).json()
    if not geo.get("results"):
        return f"Error: location '{location}' not found."
    place = geo["results"][0]
    lat, lon = place["latitude"], place["longitude"]
    data = http_get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,wind_speed_10m,precipitation,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,uv_index_max,sunrise,sunset",
            "timezone": "auto",
            "forecast_days": max(1, min(days, 7)),
        },
    ).json()
    air = {}
    try:
        air = http_get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon, "current": "us_aqi,pm2_5,pm10"},
        ).json().get("current", {})
    except Exception:
        pass
    cur = data.get("current", {})
    daily = data.get("daily", {})
    forecast = [
        {
            "date": d,
            "conditions": _WEATHER_CODES.get(daily["weather_code"][i], "unknown"),
            "max_c": daily["temperature_2m_max"][i],
            "min_c": daily["temperature_2m_min"][i],
            "rain_chance_pct": daily["precipitation_probability_max"][i],
            "uv_index": daily["uv_index_max"][i],
            "sunrise": daily["sunrise"][i][-5:],
            "sunset": daily["sunset"][i][-5:],
        }
        for i, d in enumerate(daily.get("time", []))
    ]
    return to_text(
        {
            "source": "Open-Meteo",
            "location": f"{place.get('name')}, {place.get('admin1', '')}, {place.get('country', '')}",
            "local_time": cur.get("time"),
            "now": {
                "conditions": _WEATHER_CODES.get(cur.get("weather_code"), "unknown"),
                "temp_c": cur.get("temperature_2m"),
                "feels_like_c": cur.get("apparent_temperature"),
                "humidity_pct": cur.get("relative_humidity_2m"),
                "wind_kmh": cur.get("wind_speed_10m"),
                "precip_mm": cur.get("precipitation"),
            },
            "air_quality": {
                "us_aqi": air.get("us_aqi"),
                "level": _aqi_label(air.get("us_aqi")),
                "pm2_5": air.get("pm2_5"),
                "pm10": air.get("pm10"),
            },
            "forecast": forecast,
        }
    )


@tool
@safe
@cached(ttl=300)
def earthquakes(min_magnitude: float = 4.5, period: str = "day", region: str = "") -> str:
    """Get recent earthquakes worldwide from the USGS live feed. period: hour|day|week.
    min_magnitude: 1.0|2.5|4.5 (or 'significant' quakes when >= 6). region filters by place text (e.g. 'Japan')."""
    period = period if period in {"hour", "day", "week"} else "day"
    level = "significant" if min_magnitude >= 6 else "4.5" if min_magnitude >= 4.5 else "2.5" if min_magnitude >= 2.5 else "1.0"
    feed = http_get(f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{level}_{period}.geojson").json()
    quakes = []
    for f in feed.get("features", []):
        p = f["properties"]
        if region and region.lower() not in (p.get("place") or "").lower():
            continue
        quakes.append(
            {
                "magnitude": p.get("mag"),
                "place": p.get("place"),
                "time_utc": datetime.fromtimestamp(p["time"] / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M"),
                "tsunami_warning": bool(p.get("tsunami")),
                "depth_km": f["geometry"]["coordinates"][2],
                "url": p.get("url"),
            }
        )
    quakes.sort(key=lambda q: q["magnitude"] or 0, reverse=True)
    return to_text({"count": len(quakes), "earthquakes": quakes[:15]} if quakes else f"No M{level}+ earthquakes in the past {period}{' near ' + region if region else ''}.")


_LEAGUES = {
    "nba": "basketball/nba", "wnba": "basketball/wnba", "ncaab": "basketball/mens-college-basketball",
    "nfl": "football/nfl", "ncaaf": "football/college-football", "mlb": "baseball/mlb", "nhl": "hockey/nhl",
    "premier league": "soccer/eng.1", "epl": "soccer/eng.1", "la liga": "soccer/esp.1", "bundesliga": "soccer/ger.1",
    "serie a": "soccer/ita.1", "ligue 1": "soccer/fra.1", "champions league": "soccer/uefa.champions",
    "europa league": "soccer/uefa.europa", "mls": "soccer/usa.1", "isl": "soccer/ind.1",
    "f1": "racing/f1", "atp": "tennis/atp", "wta": "tennis/wta", "ufc": "mma/ufc", "pga": "golf/pga",
}


@tool
@safe
@cached(ttl=60)
def sports_scores(league: str, team: str = "") -> str:
    """Live scores, today's results and upcoming fixtures. league: nba|nfl|mlb|nhl|wnba|ncaab|ncaaf|
    premier league|la liga|bundesliga|serie a|ligue 1|champions league|europa league|mls|isl|f1|atp|wta|ufc|pga.
    Optionally filter by team name. For cricket (IPL, internationals) use news_search / web_search instead."""
    path = _LEAGUES.get(league.lower().strip())
    if not path:
        return f"Error: unsupported league '{league}'. Supported: {', '.join(_LEAGUES)}. Use web_search for others."
    data = http_get(f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard").json()
    games = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        teams = [
            {"name": c.get("team", {}).get("displayName") or c.get("athlete", {}).get("displayName"),
             "score": c.get("score"), "home": c.get("homeAway") == "home", "winner": c.get("winner")}
            for c in comp.get("competitors", [])
        ]
        if team and not any(team.lower() in (t["name"] or "").lower() for t in teams) and team.lower() not in ev.get("name", "").lower():
            continue
        status = ev.get("status", {}).get("type", {})
        games.append(
            {
                "event": ev.get("name"),
                "start_utc": ev.get("date"),
                "status": status.get("detail") or status.get("description"),
                "teams": teams,
                "venue": (comp.get("venue") or {}).get("fullName"),
            }
        )
    league_name = (data.get("leagues") or [{}])[0].get("name", league)
    return to_text({"league": league_name, "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "games": games[:20]}
                   if games else f"No {league_name} games on the current scoreboard{' for ' + team if team else ''}.")


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {name: getattr(math, name) for name in
          ["sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan", "floor", "ceil", "factorial", "radians", "degrees"]}
_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")


@tool
@safe
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression exactly, e.g. '(1520.5 - 1210) / 1210 * 100' or 'sqrt(2) * 10**3'.
    Use it for percentages, conversions and any math on fetched numbers instead of computing mentally."""
    expr = expression.replace("^", "**")
    result = _eval(ast.parse(expr, mode="eval").body)
    return f"{expression} = {round(result, 10) if isinstance(result, float) else result}"
