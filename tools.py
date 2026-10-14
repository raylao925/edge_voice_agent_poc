#!/usr/bin/env python3
"""Voice agent tools: web search, news search, weather, page reader.

All free / no API key required (POC-friendly):
  - web_search / news_search : DuckDuckGo via `ddgs`
  - get_weather              : Open-Meteo (geocoding + forecast)
  - fetch_page               : Jina Reader (clean text extraction)
"""
import json
import urllib.parse
import urllib.request

_TIMEOUT = 20
_UA = {"User-Agent": "voice-agent-poc/0.1"}


def _http_get(url: str, timeout: int = _TIMEOUT) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _ddg_retry(fn, *args, **kwargs):
    """ddgs rotates backends; retry once on timeout before giving up."""
    import time
    from ddgs.exceptions import DDGSException
    last = None
    for i in range(3):
        try:
            return fn(*args, **kwargs)
        except DDGSException as e:
            last = e
            time.sleep(1)
    raise last


# ---------------------------------------------------------------- web search
def web_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo text search. Returns a compact text digest."""
    from ddgs import DDGS
    with DDGS() as ddgs:
        hits = list(_ddg_retry(ddgs.text, query, max_results=max_results))
    if not hits:
        return "没有搜到相关结果。"
    lines = []
    for i, h in enumerate(hits, 1):
        title = (h.get("title") or "").strip()
        body = (h.get("body") or "").strip()
        lines.append(f"{i}. {title}\n   {body[:200]}")
    return "\n".join(lines)


def news_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo news search. Returns a compact text digest."""
    from ddgs import DDGS
    with DDGS() as ddgs:
        hits = list(_ddg_retry(ddgs.news, query, max_results=max_results))
    if not hits:
        return "没有搜到相关新闻。"
    lines = []
    for i, h in enumerate(hits, 1):
        title = (h.get("title") or "").strip()
        date = (h.get("date") or "").strip()
        body = (h.get("body") or "").strip()
        lines.append(f"{i}. {title}（{date}）\n   {body[:200]}")
    return "\n".join(lines)


# ---------------------------------------------------------------- weather
_WMO = {
    0: "晴", 1: "大致晴", 2: "多云", 3: "阴",
    45: "雾", 48: "雾凇",
    51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨",
    56: "冻毛毛雨", 57: "冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "雷阵雨伴冰雹",
}


def get_weather(city: str) -> str:
    """Current weather + today/tomorrow forecast via Open-Meteo (no key)."""
    q = urllib.parse.quote(city)
    geo = json.loads(_http_get(
        f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1"
        f"&language=zh&format=json"))
    results = geo.get("results") or []
    if not results:
        return f"找不到城市「{city}」，换个名字试试。"
    g = results[0]
    lat, lon, name = g["latitude"], g["longitude"], g.get("name", city)
    country = g.get("country", "")
    fc = json.loads(_http_get(
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
        "&daily=weather_code,temperature_2m_max,temperature_2m_min"
        "&timezone=auto&forecast_days=2"))
    cur = fc.get("current", {})
    daily = fc.get("daily", {})
    w_now = _WMO.get(cur.get("weather_code"), "未知")
    out = [f"{name}（{country}）现在：{w_now}，{cur.get('temperature_2m')}°C，"
           f"湿度 {cur.get('relative_humidity_2m')}%，风速 {cur.get('wind_speed_10m')} km/h。"]
    dates, codes = daily.get("time", []), daily.get("weather_code", [])
    tmax, tmin = daily.get("temperature_2m_max", []), daily.get("temperature_2m_min", [])
    labels = ["今天", "明天"]
    for i in range(min(2, len(dates))):
        out.append(f"{labels[i]}（{dates[i]}）：{_WMO.get(codes[i] if i < len(codes) else -1, '未知')}，"
                   f"{tmin[i] if i < len(tmin) else '?'}~{tmax[i] if i < len(tmax) else '?'}°C。")
    return "\n".join(out)


# ---------------------------------------------------------------- page reader
def fetch_page(url: str, max_chars: int = 4000) -> str:
    """Extract clean article text via Jina Reader (no key)."""
    rurl = "https://r.jina.ai/" + url
    text = _http_get(rurl, timeout=30)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text[:max_chars] if len(text) > max_chars else text


TOOLS = {
    "web_search": web_search,
    "news_search": news_search,
    "get_weather": get_weather,
    "fetch_page": fetch_page,
}
