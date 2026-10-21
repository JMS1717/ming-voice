"""Tools the assistant can call while answering: web search, weather, calculator, clock.

The thinker (Ming-3.0's Bailing LLM) was trained for function calling: given the function signatures in its system
prompt it emits <tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value></tool_call> instead of an answer, and
continues once the results come back in an OBSERVATION turn (see Thinker.stream). The basics need no API key:
DuckDuckGo (via the ddgs package) for search (Brave Search too when .brave_key is set), Open-Meteo for weather.
Only the query text leaves the machine.
"""
from __future__ import annotations

import ast
import datetime as _dt
import html
import json
import math
import operator
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "ming-voice/1.0 (+https://github.com/JMS1717/ming-voice)"}


def _get_json(url: str, timeout: float = 3.5, tries: int = 2):
    """GET a JSON API; a slow, dropped or 5xx answer is tried once more (a 4xx is not)."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code < 500 or i == tries - 1:
                raise
        except (OSError, ValueError):  # timeout, connection reset, half a body
            if i == tries - 1:
                raise
        time.sleep(0.2)


# --------------------------------------------------------------------------- web search

_NEWSY = re.compile(r"\b(news|latest|headlines?|today'?s|this week|announce[ds]?|breaking)\b", re.I)
BRAVE_KEY_FILE = Path(__file__).resolve().parent / ".brave_key"


def brave_key() -> str:
    """A Brave Search API key (optional; https://brave.com/search/api), from MING_BRAVE_KEY or the .brave_key file."""
    if os.environ.get("MING_BRAVE_KEY"):
        return os.environ["MING_BRAVE_KEY"].strip()
    try:
        return BRAVE_KEY_FILE.read_text().strip()
    except OSError:
        return ""


def _brave(query: str, news: bool, n: int) -> list[dict]:
    """Brave Search (web or news) in the same shape as ddgs results; [] without a key or on any error."""
    key = brave_key()
    if not key:
        return []
    q = {"q": query, "count": n, "extra_snippets": "true", "safesearch": "moderate"}
    if news:
        q["freshness"] = "pw"  # past week: otherwise evergreen "All NASA News" pages come first
    url = f"https://api.search.brave.com/res/v1/{'news' if news else 'web'}/search?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={**UA, "Accept": "application/json", "X-Subscription-Token": key})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            d = json.load(r)
    except Exception:
        return []
    items = d.get("results", []) if news else d.get("web", {}).get("results", [])
    untag = lambda s: re.sub(r"<[^>]+>", "", html.unescape(s or ""))  # noqa: E731 (<strong> around matches)
    return [{"title": untag(h.get("title")), "href": h.get("url", ""),
             "body": " ".join(untag(x) for x in [h.get("description")] + (h.get("extra_snippets") or [])[:2]),
             "date": h.get("age", "") if news else (h.get("page_age") or "")[:10],
             "source": (h.get("meta_url") or {}).get("hostname", "") if news else ""}
            for h in items if h.get("title") or h.get("description")]


def web_search(query: str, max_results: int = 5) -> tuple[str, dict]:
    from ddgs import DDGS
    query = (query or "").strip()
    if not query:
        return "Error: empty query.", {}
    hits = []
    if _NEWSY.search(query):  # dated headlines beat landing pages for "latest news about X"
        try:  # (DuckDuckGo's news ranks better than Brave's, which mixes in evergreen pages or minor blogs)
            hits = [{"title": h.get("title", ""), "body": h.get("body", ""), "href": h.get("url", ""),
                     "date": (h.get("date") or "")[:10], "source": h.get("source", "")}
                    for h in DDGS().news(query, max_results=max_results)]
        except Exception:
            hits = []
        if len(hits) < 3:
            hits += _brave(query, True, max_results)
    if len(hits) < 3:  # web results: Brave when a key is set (fresher, more reliable), else / then DuckDuckGo
        hits += _brave(query, False, max_results)
    if len(hits) < 3:
        for backend in ("auto", "duckduckgo", "brave"):
            try:
                hits += [h for h in DDGS().text(query, max_results=max_results, backend=backend)
                         if h.get("title") or h.get("body")]
            except Exception:
                continue
            if len(hits) >= 3:
                break
    seen, results = set(), []
    for h in hits:
        url = h.get("href", "")
        if url in seen:
            continue
        seen.add(url)
        results.append(h)
        if len(results) >= max_results:
            break
    if not results:
        return "No results found.", {"query": query, "sources": []}
    lines = []
    for i, h in enumerate(results, 1):
        meta = " · ".join(x for x in (h.get("source"), h.get("date")) if x)
        body = re.sub(r"\s+", " ", h.get("body", "")).strip()[:300]
        lines.append(f"[{i}] {h.get('title', '').strip()}{f' ({meta})' if meta else ''}: {body}")
    sources = [{"title": h.get("title", "").strip() or h.get("href", ""), "url": h.get("href", "")} for h in results]
    return "\n".join(lines), {"query": query, "sources": sources}


# --------------------------------------------------------------------------- weather

_WMO = {0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
        51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "freezing drizzle", 57: "freezing drizzle",
        61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain", 67: "freezing rain",
        71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains", 80: "light showers", 81: "showers",
        82: "violent showers", 85: "snow showers", 86: "heavy snow showers", 95: "thunderstorm",
        96: "thunderstorm with hail", 99: "thunderstorm with heavy hail"}
_IMPERIAL = {"US", "LR", "MM"}


_PLACES: dict[str, list] = {}  # geocoding results by lower-case place name


def get_weather(location: str) -> tuple[str, dict]:
    location = (location or "").strip()
    if not location:
        return "Error: no location given. Ask the user where.", {}
    # "Paris, France" -> search "Paris", prefer the matching country/admin area
    name, _, hint = location.partition(",")
    key = name.strip().lower()
    if key not in _PLACES:  # (places don't move: one lookup per name saves a round trip on every later ask)
        _PLACES[key] = _get_json("https://geocoding-api.open-meteo.com/v1/search?" + urllib.parse.urlencode(
            {"name": name.strip(), "count": 10, "language": "en"})).get("results") or []
    places = list(_PLACES[key])
    if not places:
        return f"Error: could not find a place called {location!r}.", {}
    hint = hint.strip().lower()
    if hint:
        places = sorted(places, key=lambda p: not any(hint in str(p.get(k, "")).lower()
                                                      for k in ("country", "country_code", "admin1", "admin2")))
    p = places[0]
    imperial = p.get("country_code") in _IMPERIAL
    unit, wind = ("fahrenheit", "mph") if imperial else ("celsius", "kmh")
    w = _get_json("https://api.open-meteo.com/v1/forecast?" + urllib.parse.urlencode({
        "latitude": p["latitude"], "longitude": p["longitude"], "timezone": "auto", "forecast_days": 3,
        "temperature_unit": unit, "wind_speed_unit": wind,
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,is_day",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"}))
    deg = "°F" if imperial else "°C"
    c, d = w["current"], w["daily"]
    where = ", ".join(x for x in (p.get("name"), p.get("admin1"), p.get("country")) if x)
    lines = [f"Weather for {where} (local time {c['time'].replace('T', ' ')}):",
             f"Now: {_WMO.get(c['weather_code'], 'unknown')}, {c['temperature_2m']:.0f}{deg} "
             f"(feels like {c['apparent_temperature']:.0f}{deg}), humidity {c['relative_humidity_2m']}%, "
             f"wind {c['wind_speed_10m']:.0f} {wind.replace('kmh', 'km/h')}."]
    for i, day in enumerate(d["time"]):
        label = ("Today", "Tomorrow")[i] if i < 2 else _dt.date.fromisoformat(day).strftime("%A")
        rain = d["precipitation_probability_max"][i]
        lines.append(f"{label}: {_WMO.get(d['weather_code'][i], 'unknown')}, high {d['temperature_2m_max'][i]:.0f}{deg}, "
                     f"low {d['temperature_2m_min'][i]:.0f}{deg}" + (f", {rain}% chance of rain" if rain is not None else "") + ".")
    return "\n".join(lines), {"place": where, "sources": [{"title": "Open-Meteo", "url": "https://open-meteo.com/"}]}


# --------------------------------------------------------------------------- calculator

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {k: getattr(math, k) for k in ("sqrt", "sin", "cos", "tan", "asin", "acos", "atan", "log", "log10", "log2",
                                        "exp", "floor", "ceil", "factorial", "radians", "degrees")}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_CONSTS = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        a, b = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(b) > 1000:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](a, b)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError("unsupported expression")


def calculate(expression: str) -> tuple[str, dict]:
    expr = (expression or "").replace("^", "**").replace("×", "*").replace("÷", "/")
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expr)  # thousands separators (not argument commas)
    expr = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*", r"\1/100*", expr, flags=re.I)  # "15% of 80"
    try:
        v = _eval(ast.parse(expr, mode="eval"))
    except Exception as e:
        return f"Error: could not evaluate {expression!r} ({e}).", {}
    if isinstance(v, float):
        v = int(v) if v.is_integer() and abs(v) < 1e15 else float(f"{v:.10g}")
    return f"{expression} = {v}", {}


# --------------------------------------------------------------------------- clock

def get_current_time(timezone: str = "") -> tuple[str, dict]:
    now = _dt.datetime.now().astimezone()
    tz = (timezone or "").strip()
    if tz:
        try:
            from zoneinfo import ZoneInfo
            now = _dt.datetime.now(ZoneInfo(tz))
        except Exception:
            pass
    return now.strftime("%A, %B %d, %Y, %I:%M %p %Z").replace(" 0", " "), {}


# --------------------------------------------------------------------------- timers and reminders

TIMERS_FILE = Path(__file__).resolve().parent / ".timers.json"
_ONES = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                       "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_UNIT_S = {"h": 3600, "m": 60, "s": 1}
_DUR_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(h(?:ou)?rs?|h|min(?:ute)?s?|m|sec(?:ond)?s?|s)\b")
_CLOCK_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)|\b(\d{1,2}):(\d{2})\b|\b(noon|midday|midnight)\b")
_DAYS = "monday tuesday wednesday thursday friday saturday sunday".split()


def _words_to_digits(s: str) -> str:
    """'twenty five minutes and a half' -> '25 minutes and a half' (spelled-out numbers up to 99)"""
    out, acc = [], None
    for tok in re.split(r"(\s+|-)", s):
        w = tok.strip().lower()
        if w in _TENS or w in _ONES:
            acc = (acc or 0) + (_TENS.get(w) if w in _TENS else _ONES[w])
            continue
        if not w and acc is not None:  # whitespace or hyphen between number words
            continue
        if acc is not None:
            out.append(f"{acc} ")
            acc = None
        out.append(tok)
    if acc is not None:
        out.append(str(acc))
    return re.sub(r"\s+", " ", "".join(out)).strip()


def parse_when(when: str, now: _dt.datetime | None = None) -> tuple[_dt.datetime, bool] | None:
    """'10 minutes', 'an hour and a half', '90 seconds', 'at 5:30 pm', 'tomorrow at 9am', 'friday at noon'
    -> (due time, True if it was a duration), or None if it can't be understood."""
    now = now or _dt.datetime.now().astimezone()
    s = " " + (when or "").lower().replace(",", " ") + " "
    s = re.sub(r"\bhalf an? (hour|minute)\b", r"0.5 \1", s)
    s = re.sub(r"\ban? (hour|minute|second)\b", r"1 \1", s)
    s = re.sub(r"\ba (couple of|couple|few)\b", lambda m: "2" if "couple" in m.group(1) else "3", s)
    s = _words_to_digits(s)
    s = re.sub(r"(\d+(?:\.\d+)?) (?:and a half|and 1 half) (\w+)", lambda m: f"{float(m.group(1)) + 0.5} {m.group(2)}", s)
    s = re.sub(r"(\d+(?:\.\d+)?) (\w+?)s? and a half\b", lambda m: f"{float(m.group(1)) + 0.5} {m.group(2)}", s)
    c = _CLOCK_RE.search(s)
    if c:
        if c.group(6):
            h, mi = (12 if c.group(6) in ("noon", "midday") else 0), 0
        elif c.group(1):
            h, mi = int(c.group(1)) % 12 + (12 if c.group(3).startswith("p") else 0), int(c.group(2) or 0)
        else:
            h, mi = int(c.group(4)), int(c.group(5))
        if h > 23 or mi > 59:
            return None
        due = now.replace(hour=h, minute=mi, second=0, microsecond=0)
        day = next((i for i, d in enumerate(_DAYS) if d in s), None)
        if "tomorrow" in s:
            due += _dt.timedelta(days=1)
        elif day is not None:
            due += _dt.timedelta(days=(day - now.weekday()) % 7 or (7 if due <= now else 0))
        elif due <= now:  # "at 7" said in the evening: the next 7 o'clock
            due += _dt.timedelta(days=1)
            if not c.group(3) and not c.group(6) and h < 12 and (due - _dt.timedelta(hours=12)) > now:
                due -= _dt.timedelta(hours=12)  # "at 7:30" (no am/pm) at 5 pm: 7:30 pm
        return due, False
    secs = sum(float(n) * _UNIT_S[u[0]] for n, u in _DUR_RE.findall(s))
    if not secs:
        bare = re.fullmatch(r"\s*(?:in |for |after )?(\d+(?:\.\d+)?)\s*", s)
        secs = float(bare.group(1)) * 60 if bare else 0  # "set a timer for 10": minutes
    if not 0 < secs <= 7 * 86400:
        return None
    return now + _dt.timedelta(seconds=round(secs)), True


def reminder_text(message: str) -> tuple[str, str]:
    """What the model passed as the reminder -> (label, what to say when it goes off); ("", "") for a plain timer.
    'Remind me to stretch' -> ('stretch', 'Time to stretch.'); 'A 10-minute timer is set.' -> ('', '')."""
    m = re.sub(r"\s+", " ", (message or "").strip().rstrip(".!")).strip()
    verb = re.match(r"(?:(?:please )?remind (?:me|the user|him|her|them) (?:to|that)|reminder(?: to|:)|to) (.+)", m, re.I)
    label = verb.group(1).strip() if verb else m
    already = re.match(r"(?:it'?s )?time to (.+)", label, re.I)  # ('Reminder: time to stretch': not "Time to time to")
    if already:
        label, verb = already.group(1).strip(), already
    content =set(re.findall(r"[a-z]+", _words_to_digits(label.lower()))) - {
        "a", "an", "the", "my", "your", "timer", "reminder", "is", "set", "for", "minute", "minutes", "second",
        "seconds", "hour", "hours", "in", "and", "half", "done", "up", "time", "s"}
    if not content:
        return "", ""
    return label, (f"Time to {label}." if verb else f"Reminder: {label}.")


def spoken_duration(secs: float) -> str:
    secs = int(round(secs))
    h, m, s = secs // 3600, secs % 3600 // 60, secs % 60
    parts = [f"{v} {u}{'' if v == 1 else 's'}" for v, u in ((h, "hour"), (m, "minute"), (s, "second")) if v]
    return " and ".join(parts[:2]) if parts else "0 seconds"


def spoken_time(t: _dt.datetime, now: _dt.datetime | None = None) -> str:
    now = now or _dt.datetime.now().astimezone()
    clock = t.strftime("%I:%M %p").lstrip("0").replace(":00 ", " ")
    days = (t.date() - now.date()).days
    return clock if days == 0 else f"tomorrow at {clock}" if days == 1 else f"{t.strftime('%A')} at {clock}"


class Timers:
    """Timers and reminders, kept in .timers.json so they survive a restart. After start(on_fire), on_fire(entry,
    late_s) is called from a background thread when one goes off (also for ones that went off while the server was
    down, up to 12 h late)."""

    def __init__(self, path: Path = TIMERS_FILE):
        self.path, self.on_fire = path, None
        self.cv = threading.Condition()
        try:
            self.items = [t for t in json.loads(path.read_text()) if t["due"] > time.time() - 12 * 3600]
        except (OSError, ValueError, KeyError, TypeError):
            self.items = []

    def start(self, on_fire):
        self.on_fire = on_fire
        threading.Thread(target=self._loop, daemon=True).start()

    def _save(self):
        try:
            self.path.write_text(json.dumps(self.items))
        except OSError:
            pass

    def add(self, due: float, label: str, duration: float | None, say: str = "") -> dict:
        with self.cv:
            t = {"id": max([int(time.time() * 1000)] + [x["id"] + 1 for x in self.items]),  # (unique, and newest last)
                 "due": due, "label": label, "duration": duration, "say": say}
            self.items.append(t)
            self.items.sort(key=lambda x: x["due"])
            self._save()
            self.cv.notify()
        return t

    def remove(self, ts: list[dict]):
        with self.cv:
            ids = {t["id"] for t in ts}
            self.items = [t for t in self.items if t["id"] not in ids]
            self._save()
            self.cv.notify()

    def pending(self) -> list[dict]:
        with self.cv:
            return list(self.items)

    def _loop(self):
        while True:
            with self.cv:
                while not self.items or self.items[0]["due"] > time.time():
                    wait = self.items[0]["due"] - time.time() if self.items else None
                    self.cv.wait(timeout=None if wait is None else max(0.05, min(wait, 30)))
                t = self.items.pop(0)
                self._save()
            try:
                self.on_fire(t, max(0.0, time.time() - t["due"]))
            except Exception as e:
                print(f"timer callback failed: {e!r}", flush=True)


# --------------------------------------------------------------------------- Home Assistant

HA_TOKEN_FILE = Path(__file__).resolve().parent / ".ha_token"
HA_URL_FILE = Path(__file__).resolve().parent / ".ha_url"


def _ha_url() -> str:
    """MING_HA_URL, else the .ha_url file, else http://homeassistant.local:8123"""
    url = os.environ.get("MING_HA_URL", "")
    if not url:
        try:
            url = HA_URL_FILE.read_text().strip()
        except OSError:
            url = "http://homeassistant.local:8123"
    return url.rstrip("/")


def ha_token() -> str:
    """A long-lived access token (HA: your profile -> Security), from MING_HA_TOKEN or the .ha_token file."""
    if os.environ.get("MING_HA_TOKEN"):
        return os.environ["MING_HA_TOKEN"].strip()
    try:
        return HA_TOKEN_FILE.read_text().strip()
    except OSError:
        return ""


# Direct control sees every device the token can (not only those exposed to Assist); MING_HA_DIRECT=0: Assist only.
HA_DIRECT = os.environ.get("MING_HA_DIRECT", "1") == "1"


def home_assistant(command: str) -> tuple[str, dict]:
    """Do a spoken-style command or answer a question about the user's devices: directly from HA's device list
    (ha.py: lights, plugs, locks, thermostat, ...), else through HA's own intent handling (Assist)."""
    command = (command or "").strip()
    if not command:
        return "Error: empty command.", {}
    if not HA_DIRECT:
        return assist(command)
    import ha
    try:
        done = ha.handle(command, _ha_url(), ha_token())
    except Exception as e:  # (the websocket API unreachable, ...: Assist may still manage)
        print(f"home_assistant: direct control failed ({e!r}); trying Assist", flush=True)
        done = None
    if done is not None:
        return done
    text, info = assist(command)
    if info.get("response_type") == "error":
        try:  # it didn't know what was meant: tell the model what there is
            text += f" Devices in the home: {ha.names_by_area(ha.devices(_ha_url(), ha_token()))}."
        except Exception:
            pass
    return text, info


def assist(command: str) -> tuple[str, dict]:
    """Hand the command to Home Assistant's own intent handling (Assist), which resolves device/area names and
    aliases and only touches entities exposed to voice assistants in HA."""
    req = urllib.request.Request(f"{_ha_url()}/api/conversation/process", method="POST",
                                 data=json.dumps({"text": command, "language": "en",
                                                  "agent_id": "conversation.home_assistant"}).encode(),
                                 headers={**UA, "Authorization": f"Bearer {ha_token()}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=8) as r:
        resp = json.load(r).get("response", {})
    speech = (resp.get("speech", {}).get("plain", {}) or {}).get("speech", "").strip()
    data = resp.get("data", {}) or {}
    kind = resp.get("response_type", "")
    done = [t.get("name") or t.get("id") for t in data.get("success", [])]
    failed = [t.get("name") or t.get("id") for t in data.get("failed", [])]
    if kind == "error":
        text = f"Home Assistant could not do that ({data.get('code', 'error')}): {speech}"
    else:
        text = speech or "Done."
        if failed:
            text += f" Failed for: {', '.join(map(str, failed))}."
    return text, {"targets": done, "failed": failed, "response_type": kind}


# --------------------------------------------------------------------------- Jellyfin

JELLYFIN_FILE = Path(__file__).resolve().parent / ".jellyfin.json"


def jellyfin_config() -> dict | None:
    """{"url", "key"} of a Jellyfin server (MING_JELLYFIN_URL/MING_JELLYFIN_KEY or the .jellyfin.json file)."""
    if os.environ.get("MING_JELLYFIN_URL"):
        return {"url": os.environ["MING_JELLYFIN_URL"], "key": os.environ.get("MING_JELLYFIN_KEY", "")}
    try:
        cfg = json.loads(JELLYFIN_FILE.read_text())
        return cfg if cfg.get("url") and cfg.get("key") else None
    except (OSError, ValueError):
        return None


def _jf(method: str, path: str, params: dict | None = None):
    cfg = jellyfin_config() or {}
    url = cfg.get("url", "").rstrip("/") + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None,
                                 headers={**UA, "Authorization": f'MediaBrowser Token="{cfg.get("key", "")}"'})
    with urllib.request.urlopen(req, timeout=8) as r:
        body = r.read()
    return json.loads(body) if body else None


_JF_TYPES = "Movie,Series,Episode,Audio,MusicAlbum,MusicArtist,Playlist"


def _jf_user() -> str:
    users = _jf("GET", "/Users")
    admins = [u for u in users if u.get("Policy", {}).get("IsAdministrator")]
    return (admins or users)[0]["Id"]


def _jf_label(it: dict) -> str:
    t = it.get("Type", "")
    if t == "Episode":
        return f"{it.get('SeriesName', '')} S{it.get('ParentIndexNumber', 0)}E{it.get('IndexNumber', 0)} \"{it.get('Name', '')}\""
    if t in ("Audio", "MusicAlbum"):
        return f"{it.get('Name', '')} by {', '.join(it.get('Artists') or [it.get('AlbumArtist', '')])}"
    year = f" ({it['ProductionYear']})" if it.get("ProductionYear") else ""
    return f"{it.get('Name', '')}{year} [{t.lower()}]"


def _jf_sessions(device: str = "") -> list[dict]:
    """Controllable players, most recently active first, filtered by device/client name when given."""
    s = [x for x in _jf("GET", "/Sessions") if x.get("SupportsRemoteControl")]
    s.sort(key=lambda x: x.get("LastActivityDate", ""), reverse=True)
    if device:
        d = device.lower()
        s = [x for x in s if d in (x.get("DeviceName", "") + " " + x.get("Client", "")).lower()] or []
    return s


def jellyfin(action: str, query: str = "", device: str = "") -> tuple[str, dict]:
    action = (action or "").strip().lower().replace(" ", "_")
    if action in ("search", "play"):
        if not query:
            return "Error: say what to search for or play.", {}
        user = _jf_user()
        items = _jf("GET", f"/Users/{user}/Items", {"searchTerm": query, "Recursive": "true",
                                                     "IncludeItemTypes": _JF_TYPES, "Limit": 6}).get("Items", [])
        if not items:
            return f"Nothing in the Jellyfin library matches {query!r}.", {}
        if action == "search":
            return "In the library: " + "; ".join(_jf_label(i) for i in items) + ".", {}
        it = items[0]
        if it["Type"] == "Series":  # continue the show: next unwatched episode
            nxt = _jf("GET", "/Shows/NextUp", {"userId": user, "seriesId": it["Id"], "Limit": 1}).get("Items", [])
            if not nxt:
                nxt = _jf("GET", f"/Shows/{it['Id']}/Episodes", {"userId": user, "Limit": 1}).get("Items", [])
            it = nxt[0] if nxt else it
        sessions = _jf_sessions(device)
        if not sessions:
            return ("Error: no Jellyfin player is open" + (f" matching {device!r}" if device else "")
                    + ". The Jellyfin app must be open on the TV, phone or computer to play there."), {}
        s = sessions[0]
        _jf("POST", f"/Sessions/{s['Id']}/Playing", {"playCommand": "PlayNow", "itemIds": it["Id"]})
        return f"Playing {_jf_label(it)} on {s.get('DeviceName')}.", {"targets": [s.get("DeviceName")]}
    if action in ("pause", "resume", "unpause", "stop", "next", "previous", "play_pause"):
        sessions = [x for x in _jf_sessions(device) if x.get("NowPlayingItem")] or _jf_sessions(device)
        if not sessions:
            return "Error: nothing is playing on Jellyfin.", {}
        s = sessions[0]
        cmd = {"pause": "Pause", "resume": "Unpause", "unpause": "Unpause", "stop": "Stop", "next": "NextTrack",
               "previous": "PreviousTrack", "play_pause": "PlayPause"}[action]
        _jf("POST", f"/Sessions/{s['Id']}/Playing/{cmd}")
        return f"{cmd} sent to {s.get('DeviceName')}.", {"targets": [s.get("DeviceName")]}
    if action in ("now_playing", "status"):
        playing = [x for x in _jf("GET", "/Sessions") if x.get("NowPlayingItem")]
        if not playing:
            return "Nothing is playing on Jellyfin right now.", {}
        return " ".join(f"{_jf_label(x['NowPlayingItem'])} is {'paused' if x.get('PlayState', {}).get('IsPaused') else 'playing'} "
                        f"on {x.get('DeviceName')}." for x in playing), {}
    if action == "devices":
        s = _jf_sessions()
        return ("Jellyfin players open now: " + ", ".join(f"{x.get('DeviceName')} ({x.get('Client')})" for x in s) + "."
                if s else "No Jellyfin players are open right now."), {}
    return f"Error: unknown action {action!r}.", {}


# --------------------------------------------------------------------------- Hermes (delegation)

HERMES_FILE = Path(__file__).resolve().parent / ".hermes.json"
HERMES_INSTRUCTIONS = (
    "This task comes from the user's voice assistant, which will read your final answer aloud. Do the task fully, "
    "then end with a short spoken-style result: two or three plain sentences, no markdown, lists, links or code.")


def hermes_config() -> dict | None:
    """{"url", "key"} of a Hermes Agent API server (MING_HERMES_URL/MING_HERMES_KEY or the .hermes.json file)."""
    if os.environ.get("MING_HERMES_URL"):
        return {"url": os.environ["MING_HERMES_URL"], "key": os.environ.get("MING_HERMES_KEY", "")}
    try:
        cfg = json.loads(HERMES_FILE.read_text())
        return cfg if cfg.get("url") else None
    except (OSError, ValueError):
        return None


def _hermes(method: str, path: str, body: dict | None = None, timeout: float = 10.0):
    cfg = hermes_config() or {}
    req = urllib.request.Request(cfg.get("url", "").rstrip("/") + path, method=method,
                                 data=None if body is None else json.dumps(body).encode(),
                                 headers={**UA, "Authorization": f"Bearer {cfg.get('key', '')}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# --------------------------------------------------------------------------- lists and memory

LISTS_FILE = Path(__file__).resolve().parent / ".lists.json"
MEMORY_FILE = Path(__file__).resolve().parent / ".memory.json"
MAX_MEMORY = 40  # facts (they're in the system prompt)


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _write_json(path: Path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    tmp.replace(path)


def _spoken_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _list_name(name: str) -> str:
    n = re.sub(r"\b(my|the|our|list|lists)\b", " ", str(name or "").lower())
    n = " ".join(n.replace("_", " ").split()).strip(" -")
    if re.fullmatch(r"to ?-? ?dos?|tasks?|chores?", n):
        return "to-do"
    if n in ("", "grocery", "groceries", "shopping", "food", "store"):
        return "shopping"
    return n


def _items(items) -> list[str]:
    if isinstance(items, str):  # (one string: "milk, eggs")
        items = re.split(r",|;|\n", items)
    out = []
    for i in items or []:
        i = " ".join(str(i).split()).strip(" .")
        if i and i.lower() not in (x.lower() for x in out):
            out.append(i)
    return out


def _same_item(a: str, b: str) -> bool:
    norm = lambda s: " ".join(re.sub(r"(?<=\w)(es|s)\b", "", w) for w in re.findall(r"[a-z0-9]+", s.lower())
                              if w not in ("a", "an", "the", "some"))
    a, b = norm(a), norm(b)
    return bool(a and b) and (a == b or f" {a} " in f" {b} " or f" {b} " in f" {a} ")


def lists_tool(action: str = "read", list: str = "shopping", items=None) -> tuple[str, dict]:
    """The user's own lists (.lists.json): shopping, to-do, or any other name."""
    lists, name, items = _read_json(LISTS_FILE, {}), _list_name(list), _items(items)
    have = lists.get(name, [])
    title = f"{name} list"
    if action == "add":
        if not items:
            return "Error: say what to add.", {}
        new = [i for i in items if not any(_same_item(i, h) for h in have)]
        lists[name] = have + new
        _write_json(LISTS_FILE, lists)
        dup = [i for i in items if i not in new]
        say = (f"Added {_spoken_list(new)} to your {title}." if new else "") + \
              (f" {_spoken_list(dup).capitalize()} {'was' if len(dup) == 1 else 'were'} already on it." if dup else "")
        return say.strip(), {"say": say.strip(), "list": name}
    if action == "remove":
        gone = [h for h in have if any(_same_item(i, h) for i in items)]
        missing = [i for i in items if not any(_same_item(i, h) for h in gone)]
        lists[name] = [h for h in have if h not in gone]
        _write_json(LISTS_FILE, lists)
        say = (f"Removed {_spoken_list(gone)} from your {title}." if gone else "") + \
              (f" {_spoken_list(missing).capitalize()} {'isn' if len(missing) == 1 else 'aren'}'t on it." if missing else "")
        return say.strip() or "Error: say what to remove.", {"say": say.strip(), "list": name}
    if action == "clear":
        lists[name] = []
        _write_json(LISTS_FILE, lists)
        say = f"Your {title} is cleared." if have else f"Your {title} was already empty."
        return say, {"say": say, "list": name}
    if action in ("names", "lists"):
        # (reads go back to the model, not straight to the user: it often reads a list before changing it)
        names = [f"a {n} list with {len(v)} {'item' if len(v) == 1 else 'items'}" for n, v in lists.items() if v]
        return f"The user has {_spoken_list(names)}." if names else "All the user's lists are empty.", {}
    if action == "read":
        return (f"The user's {title} is empty." if not have else
                f"The user's {title} ({len(have)} items): " + "; ".join(have) + "."), {"list": name}
    return f"Error: unknown action {action!r}.", {}


def memories() -> list[str]:
    return _read_json(MEMORY_FILE, [])


def memory_tool(action: str = "remember", fact: str = "") -> tuple[str, dict]:
    """Facts the user asked to be remembered across conversations (.memory.json); they go into the system prompt."""
    fact = " ".join(str(fact or "").split()).strip()
    facts = memories()
    if not fact:
        return "Error: say what to remember or forget.", {}
    if action == "remember":
        facts = [f for f in facts if f.lower() != fact.lower()] + [fact]
        _write_json(MEMORY_FILE, facts[-MAX_MEMORY:])
        return "Okay, I'll remember that.", {"say": "Okay, I'll remember that."}
    if action == "forget":
        stop = {"the", "a", "an", "my", "user", "s", "is", "that", "to", "of", "i", "me", "where", "what", "when"}
        # (crudely stemmed: "parking" and "parked" both match "park")
        words = lambda s: {re.sub(r"(ing|ed|es|s)$", "", w) if len(w) > 4 else w
                           for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in stop}
        want = words(fact)
        scored = sorted(((len(want & words(f)) / max(1, len(want)), f) for f in facts), reverse=True)
        if not scored or scored[0][0] < 0.5:
            return "There's nothing like that to forget.", {"say": "I don't have anything like that remembered."}
        gone = scored[0][1]
        _write_json(MEMORY_FILE, [f for f in facts if f != gone])
        return f"Forgot: {gone}", {"say": "Okay, I've forgotten that."}
    return f"Error: unknown action {action!r}.", {}


# --------------------------------------------------------------------------- registry

def _fn(name, description, params, required):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": params, "required": required}}}


SPECS = [
    _fn("web_search", "Search the web for live or recent information: news, prices, sports results, recent events, "
        "opening hours. Not for general knowledge.",
        {"query": {"type": "string", "description": "a short web search query"}}, ["query"]),
    _fn("get_weather", "Current weather and the 3-day forecast for a place.",
        {"location": {"type": "string", "description": "city name, optionally with region or country, e.g. 'Paris, France'"}},
        ["location"]),
    _fn("calculate", "Evaluate arithmetic exactly (+ - * / ** %, sqrt, sin, log, pi...). Use for any calculation "
        "beyond simple mental math.",
        {"expression": {"type": "string", "description": "e.g. '17*23', 'sqrt(2)' or '15% of 80'"}}, ["expression"]),
    _fn("get_current_time", "The current date and time in another time zone (the user's local date and time are "
        "already in your note).",
        {"timezone": {"type": "string", "description": "optional IANA time zone, e.g. 'Asia/Tokyo'"}}, []),
    _fn("timer", "Timers and reminders: set one ('set a timer for 10 minutes', 'remind me to call mom at 5 pm'), "
        "list them with the time left, or cancel one. When it goes off you will be told, and tell the user.",
        {"action": {"type": "string", "enum": ["set", "list", "cancel"]},
         "when": {"type": "string", "description": "for set: a duration ('10 minutes', 'an hour and a half') or a time "
                                                   "('5:30 pm', 'tomorrow at 9 am')"},
         "message": {"type": "string", "description": "for set: what to remind the user of (empty for a plain timer); "
                                                      "for cancel: which one ('the pasta timer', 'all')"}},
        ["action"]),
    _fn("lists", "Add items to, remove items from, read or clear the user's lists: shopping list, to-do list, or "
        "any list they name. Lists change between conversations, so call this every time the user asks what's on "
        "one.",
        {"action": {"type": "string", "enum": ["add", "remove", "read", "clear", "names"],
                    "description": "add: put the items on the list; remove: take them off (also when the user says "
                                   "they bought or did them); read: say what's on the list; clear: empty it; names: "
                                   "which lists exist"},
         "list": {"type": "string", "description": "which list, e.g. 'shopping' (the default) or 'to-do'"},
         "items": {"type": "string",
                   "description": "for add and remove: the items, separated by commas, e.g. 'milk, eggs, bread'"}},
        ["action"]),
    _fn("memory", "Remember something the user tells you to remember for later conversations ('remember that my "
        "sister's birthday is May 3rd', 'I parked on level 3'), or forget something when they say 'forget ...' "
        "(nothing is forgotten unless you call this). What you remember is listed in your instructions: answer "
        "questions from there.",
        {"action": {"type": "string", "enum": ["remember", "forget"]},
         "fact": {"type": "string", "description": "one short sentence, e.g. 'The user's sister's birthday is May 3.'"}},
        ["action", "fact"]),
    _fn("end_conversation", "End the conversation when the user says goodbye or asks you to end it or stop "
        "listening ('bye', 'that's all, thanks', 'goodnight', 'end the conversation'). You say a short goodbye.",
        {}, []),
]
HA_SPEC = _fn("home_assistant", "Control or ask about the user's smart home through Home Assistant: lights and "
              "lamps (on, off, brightness, colour), plugs, music on the speakers and the TV (pause, skip, volume), "
              "the thermostat, locks, the car (lock, climate, defrost), the vacuum, scenes and scripts (not "
              "computers or servers, not general questions). Call it for every such request or question, and pass "
              "the user's own words as the command, e.g. 'turn off the kitchen lights'.",
              {"command": {"type": "string", "description": "the request, in plain English"},
               "confirmed": {"type": "boolean", "description": "true only when you asked the user to confirm this "
                                                               "action and they just said yes"}}, ["command"])
# Actions that are hard to take back or open the house up: Ming asks first ("Do you want me to unlock the front
# door?") and only does it when the next turn says yes. MING_CONFIRM_RISKY=0 turns this off.
RISKY_RE = re.compile(r"\b(unlock|disarm)|\b(open|raise|lift)\b.*\b(garage|gate|door)|\b(garage|gate)\b.*\bopen"
                      r"|\b(alarm|security)\b.*\b(off|disable|deactivate)\b|\b(off|disable|deactivate)\b.*\b(alarm|security)\b",
                      re.I)
CONFIRM_WINDOW_S = 60


def ha_spec() -> dict:
    """HA_SPEC, naming the user's devices by room (so the model knows "the reading lamp" is one of them)."""
    spec = json.loads(json.dumps(HA_SPEC))
    if not HA_DIRECT:
        return spec
    try:
        import ha
        known = ha.names_by_area(ha.devices(_ha_url(), ha_token()))
    except Exception as e:
        print(f"home_assistant: can't list the devices ({e!r})", flush=True)
        known = ""
    if known:
        spec["function"]["description"] += f" The user's devices: {known}."
    return spec


def is_risky(command: str) -> bool:
    c = command.strip().lower()
    question = c.endswith("?") or re.match(r"(is|are|was|did|does|do|what|which|how|who|when|why)\b", c)
    return bool(RISKY_RE.search(c)) and not question
JELLYFIN_SPEC = _fn("jellyfin", "Control the user's Jellyfin media player for watching or listening: play a movie, "
                    "show or music they name on a device where the Jellyfin app is open (a show continues with the "
                    "next episode), pause/resume/stop/next/previous, what's playing, which devices are available, or "
                    "whether a title is in their library. Use it only when the user talks about watching or "
                    "listening to something.",
                    {"action": {"type": "string", "enum": ["search", "play", "pause", "resume", "stop", "next",
                                                           "previous", "now_playing", "devices"]},
                     "query": {"type": "string", "description": "title, show, artist or album (search and play)"},
                     "device": {"type": "string", "description": "optional device name, e.g. 'TV' or 'phone'"}},
                    ["action"])
HERMES_SPEC = _fn("ask_hermes", "Hand a task to Hermes, the user's separate AI agent on their home server. Use it "
                  "ONLY when the user says the name Hermes (e.g. 'ask Hermes to ...'); never on your own, and never "
                  "for questions you can answer yourself or with your other tools. It runs in the background "
                  "(seconds to minutes) and its answer is read out when ready, so don't wait for it.",
                  {"task": {"type": "string", "description": "the complete task, with all details from the conversation"}},
                  ["task"])
# The old behaviour (Ming hands anything its tools can't do to Hermes on its own): MING_HERMES_AUTO=1
HERMES_AUTO_SPEC = _fn("ask_hermes", "Delegate a task to Hermes, the user's more capable AI agent on their home "
                       "server (always use this when the user says to ask Hermes; it can also check and manage the "
                       "user's computers and servers), for anything your other tools can't do: in-depth research, "
                       "comparing options, reading web pages, the user's files, GitHub, email, calendar, messages, "
                       "Spotify, reminders and scheduled jobs, and multi-step work. It runs in the background (seconds "
                       "to minutes) and its answer is read out when ready, so don't wait for it. For simply switching "
                       "devices, use home_assistant instead.", HERMES_SPEC["function"]["parameters"]["properties"],
                       ["task"])
PLAY_RE = re.compile(r"\b(play|watch|put on|throw on|listen|start|resume|continue|queue|shuffle|next episode)", re.I)
MEDIA_RE = re.compile(PLAY_RE.pattern[:-1] + r"|jellyfin|movie|film|show|series|season|episode|song|music|album|"
                      r"artist|track|pause|stop|skip|next|previous|back|playing|library|tv|seen|have)", re.I)
# what ending the conversation sounds like (end_conversation only runs when the user said something like it)
END_RE = re.compile(r"\b(bye|goodbye|good ?night|that'?s (all|it)|that is (all|it)|end (the |this |our )?"
                    r"(conversation|chat|call|session)|stop (listening|the conversation)|hang up|we'?re (done|good)|"
                    r"i'?m (done|good)|talk (to you )?(later|soon|tomorrow)|see (you|ya)|later|go to sleep|"
                    r"sleep now|no more questions|nothing else|i'?m all set|over and out|exit|quit)\b", re.I)
GOODBYES = ["Okay, bye!", "Bye for now!", "Talk to you later!", "Okay, see you later!"]
# The butler persona (Jarvis preset): what the tools say themselves, in character
BUTLER_GOODBYES = ["Very good, sir. I'll be here if you need me.", "As you wish, sir. Until next time.",
                   "Of course, sir. Do call if anything comes up.", "Very good, sir. I shall keep an eye on things."]
BUTLER_OPENERS = ["Very good, sir.", "Right away, sir.", "Certainly, sir.", "Done, sir.", "Of course, sir."]


def in_character(say: str, persona: str, n: int = 0) -> str:
    """'Okay, turned off the lamp.' -> 'Certainly, sir. I've turned off the lamp.' (persona 'jarvis')."""
    if persona != "jarvis" or not say:
        return say
    if say.startswith("Just to be sure: do you want me to "):
        return "Just to confirm, sir: shall I " + say[len("Just to be sure: do you want me to "):]
    m = re.match(r"(okay|ok|alright|sure)[,.!]?\s+", say, re.I)
    if not m:  # a status: 'The lamp is on, sir.'
        return say if re.search(r"\bsir\b", say, re.I) else re.sub(r"\.(\s|$)", r", sir.\1", say, count=1)
    rest = say[m.end():]
    if re.match(r"(turned|set|switched|toggled|cancelled|added|removed|saved|asked|started|made|paused|resumed|"
                r"muted|unmuted)\b", rest):
        rest = "I've " + rest
    elif re.match(r"timer set\b", rest):
        rest = "I've set a timer" + rest[len("timer set"):]
    return f"{BUTLER_OPENERS[n % len(BUTLER_OPENERS)]} {rest[:1].upper()}{rest[1:]}"
HERMES_NAME_RE = re.compile(r"\b(h[eu]rm[eèéi]|erm[eè]s)", re.I)  # (and how ASR writes it: Hermès, Hermies, ...)


def hermes_auto() -> bool:
    return os.environ.get("MING_HERMES_AUTO", "0") == "1"
_IMPL = {"web_search": web_search, "get_weather": get_weather, "calculate": calculate,
         "get_current_time": get_current_time, "home_assistant": home_assistant, "jellyfin": jellyfin,
         "lists": lists_tool, "memory": memory_tool}
# Tools that change something: never run for a speculative reply (only once the user's turn has really ended)
SIDE_EFFECTS = {"home_assistant", "ask_hermes", "jellyfin", "timer", "lists", "memory"}
# What the assistant says while a slow tool runs (when it said nothing itself before calling it)
FILLERS = {"web_search": "Let me look that up.", "get_weather": "Let me check the weather."}
MAX_RESULT_CHARS = 1600  # ~400 tokens: results stay in the context for the rest of the chat


class Toolbox:
    """The tools offered to the model. run() never raises: errors go back to the model as text."""

    def __init__(self, specs=None, home: str = ""):
        if specs is None:  # the default set, plus Home Assistant / Hermes once configured
            specs = SPECS + ([ha_spec()] if ha_token() else []) + ([JELLYFIN_SPEC] if jellyfin_config() else [])
            hermes = hermes_config()
            if hermes:  # (optional "about" in .hermes.json: e.g. which servers it runs on)
                specs = specs + [json.loads(json.dumps(HERMES_AUTO_SPEC if hermes_auto() else HERMES_SPEC))]
                if hermes.get("about"):
                    specs[-1]["function"]["description"] += " " + hermes["about"]
        self.specs = list(specs)
        self.on_delegate_done = None  # callback(task, ok, text) when a Hermes run finishes
        self.runs: dict[str, str] = {}  # running Hermes run id -> task
        self._last_delegation = 0.0
        self.home = home  # the user's location: in the system prompt, and the default for the weather
        self.timers = Timers()  # (the server starts it: Timers.start(on_fire))
        self.turn = 0  # conversation turns (Thinker.stream): a confirmation has to come in a later turn
        self._confirm = None  # (command, turn, time) of a risky action waiting for the user's yes
        self.heard = None  # () -> what the user said this turn (transcribed on first use; Thinker.stream sets it)
        self.persona = ""  # "jarvis": what the tools say is said in character (in_character)

    def new_turn(self, heard=None):
        self.turn += 1
        self.heard = heard

    def signature(self) -> str:
        """What the system prompt depends on (a change means rebuilding the context)."""
        return json.dumps([self.specs, self.home, memories()], sort_keys=True)

    @property
    def names(self):
        return [s["function"]["name"] for s in self.specs]

    def run(self, name: str, args: dict, cancel: threading.Event | None = None) -> tuple[str, dict]:
        """-> (result text for the model, info for the UI: query/sources/elapsed/error)"""
        t0 = time.perf_counter()
        impl = ({**_IMPL, "ask_hermes": self._ask_hermes, "timer": self._timer, "jellyfin": self._jellyfin,
                 "home_assistant": self._home_assistant, "end_conversation": self._end_conversation}
                ).get(name) if name in self.names else None
        if impl is None:
            return f"Error: there is no tool named {name!r}.", {"error": "unknown tool"}
        args = {k: v for k, v in (args or {}).items() if isinstance(k, str)}
        if name == "get_weather" and self.home and not str(args.get("location", "")).strip():
            args["location"] = self.home
        try:
            text, info = impl(**args)
        except TypeError as e:
            text, info = f"Error: bad arguments for {name}: {e}.", {"error": str(e)}
        except Exception as e:  # network down, API changed, ...
            text, info = f"Error: {name} failed ({type(e).__name__}: {e}).", {"error": str(e)}
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS].rsplit("\n", 1)[0]
        if not (info or {}).get("say") and name != "lists":  # (said right before it answers, this works better than in the system prompt)
            text += "\n(Answer from this in one or two short spoken sentences: no lists, no URLs, no follow-up offers.)"
        info = dict(info or {}, elapsed_ms=round((time.perf_counter() - t0) * 1000))
        if info.get("say"):
            info["say"] = in_character(info["say"], self.persona, self.turn)
        return text, info


    # -- Home Assistant, with a spoken confirmation for risky actions --
    def _home_assistant(self, command: str = "", confirmed=False) -> tuple[str, dict]:
        command = str(command or "").strip()
        if HA_DIRECT and self.heard is not None:  # what the user said, when it's a command ha.py understands
            try:
                import ha
                command = ha.pick(command, self.heard() or "", ha.devices(_ha_url(), ha_token(), max_age=ha.STATE_S))
            except Exception as e:
                print(f"home_assistant: can't check the user's words ({e!r})", flush=True)
        if not is_risky(command) or os.environ.get("MING_CONFIRM_RISKY", "1") == "0":
            return home_assistant(command)
        words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower())) - {"the", "a", "my", "please", "can", "you"}
        p, self._confirm = self._confirm, None
        # The yes has to come from the user in a later turn (never the same reply that asked), within a minute,
        # for the same thing (an "unlock the back door" right after asking about the front door is asked again).
        # What runs is what the user was asked about.
        if (p is not None and p[1] < self.turn and time.time() - p[2] < CONFIRM_WINDOW_S
                and (confirmed in (True, "true", "True", "yes") or words(command) == words(p[0]))
                and words(command) <= words(p[0])):
            text, info = home_assistant(p[0])
            return text, dict(info, confirmed=True)
        self._confirm = (command, self.turn, time.time())
        q = command.rstrip(".?! ")
        say = f"Just to be sure: do you want me to {q[:1].lower() + q[1:]}?"
        return (f"Not done yet: this needs the user's confirmation, and you asked them: \"{say}\" If they say yes, "
                f"call home_assistant again with the same command and confirmed=true; otherwise don't.",
                {"say": say, "needs_confirmation": True})

    # -- the user said goodbye: the server ends the conversation once the goodbye has played --
    def _end_conversation(self, **_) -> tuple[str, dict]:
        said = self.heard() if self.heard is not None else ""
        if self.heard is not None and not END_RE.search(said or ""):
            return ("Not ended: the user didn't say goodbye. Carry on with the conversation: answer what they "
                    "said."), {"refused": True, "heard": said}
        butler = self.persona == "jarvis"
        say = (BUTLER_GOODBYES if butler else GOODBYES)[self.turn % len(GOODBYES)]
        if re.search(r"good ?night", said or "", re.I):
            say = "Good night, sir. Sleep well." if butler else "Good night!"
        return "Ended.", {"say": say, "end": True}

    # -- Jellyfin: playback only when the user asked for it ("tell me a joke" once tried to play "Joke & Dagger") --
    def _jellyfin(self, action: str = "", query: str = "", device: str = "") -> tuple[str, dict]:
        if self.heard is not None:
            said = self.heard() or ""
            if HA_DIRECT and "home_assistant" in self.names:  # "run movie mode" is a script, not a film
                try:
                    import ha
                    if ha.script_named(said, ha.devices(_ha_url(), ha_token(), max_age=ha.STATE_S)):
                        return self._home_assistant(said)
                except Exception:
                    pass
            play = str(action).strip().lower() == "play"
            if not (PLAY_RE if play else MEDIA_RE).search(said):
                return ("Not done: the user isn't asking about watching or listening to something. Don't use "
                        "jellyfin; answer what they asked yourself, right now."), {"refused": True, "heard": said}
        return jellyfin(action=action, query=query, device=device)

    # -- timers and reminders (they go off through Timers.on_fire) --
    def _timer(self, action: str = "set", when: str = "", message: str = "") -> tuple[str, dict]:
        action, when, message = (action or "set").strip().lower(), str(when or "").strip(), str(message or "").strip()
        now = _dt.datetime.now().astimezone()
        if action == "set":
            parsed = parse_when(when) or (parse_when(message) if not when else None)
            if parsed is None:
                return (f"Error: could not understand the time {when!r}. Ask the user when it should go off.",
                        {"error": "bad time"})
            due, is_duration = parsed
            secs = (due - now).total_seconds()
            message, alarm = reminder_text(message)
            self.timers.add(due.timestamp(), message, secs if is_duration else None, alarm)
            spoken = f"in {spoken_duration(secs)}" if is_duration else f"at {spoken_time(due, now)}"
            say = (f"Okay, timer set for {spoken_duration(secs)}." if is_duration and not message
                   else f"Okay, I'll remind you {spoken}.")
            return f"Set; it goes off {spoken}.", {"when": spoken, "message": message, "say": say}
        items = self.timers.pending()
        if action == "list":
            if not items:
                return "No timers or reminders are set.", {}
            return "; ".join(
                (f"{spoken_duration(t['duration'])} timer" if t["duration"] and not t["label"] else
                 f"reminder: {t['label'] or 'no message'}")
                + f", {spoken_duration(t['due'] - time.time())} left (goes off at "
                  f"{spoken_time(_dt.datetime.fromtimestamp(t['due']).astimezone(), now)})" for t in items), {}
        if action == "cancel":
            which = (message or when).lower()
            if not items:
                return "There was nothing to cancel.", {"say": "There's no timer or reminder to cancel."}
            if "all" in which.split() or "every" in which:
                hit = items
            else:
                words = set(re.findall(r"[a-z0-9]+", _words_to_digits(which))) - {"the", "timer", "reminder", "my", "a"}
                score = lambda t: len(words & set(re.findall(r"[a-z0-9]+", t["label"].lower() + " " + (  # noqa: E731
                    spoken_duration(t["duration"]).replace("minutes", "minute") if t["duration"] else ""))))
                best = max(items, key=score)
                newest = max(items, key=lambda t: t["id"])
                hit = [best] if score(best) or len(items) == 1 else [newest] if not words else []
            if not hit:
                return f"Error: no timer matches {which!r}; pending: " + self._timer("list")[0], {"error": "no match"}
            self.timers.remove(hit)
            return (f"Cancelled {len(hit)}.", {"message": which,
                    "say": f"Okay, cancelled all {len(hit)}." if len(hit) > 1 else "Okay, it's cancelled."})
        return f"Error: unknown action {action!r}.", {"error": "bad action"}

    # -- Hermes: start a background run, announce its result through on_delegate_done --
    def _ask_hermes(self, task: str) -> tuple[str, dict]:
        task = (task or "").strip()
        if not task:
            return "Error: empty task.", {}
        if not hermes_auto() and self.heard is not None:
            said = self.heard() or ""
            if not HERMES_NAME_RE.search(said):  # the model reached for Hermes on its own: not wanted
                return ("Not sent: the user did not ask for Hermes, so don't use it. Answer the user now yourself, "
                        "or with your other tools; if you can't, say so briefly."), {"refused": True, "heard": said}
        if time.time() - self._last_delegation < 20:  # the model sometimes calls it again right after
            return "Already passed to Hermes.", {"duplicate": True, "say": "Hermes is already on it."}
        self._last_delegation = time.time()
        run = _hermes("POST", "/v1/runs", {"input": task, "instructions": HERMES_INSTRUCTIONS})
        rid = run.get("run_id") or run.get("id")
        if not rid:
            return f"Error: Hermes did not start the task ({run}).", {"error": "no run id"}
        self.runs[rid] = task
        threading.Thread(target=self._watch_run, args=(rid, task), daemon=True).start()
        return ("Done: Hermes is now working on it in the background; its answer will be read out "
                "automatically when ready."), {"run_id": rid, "say": "Okay, I've asked Hermes. I'll tell you when it's done."}

    def _watch_run(self, rid: str, task: str, timeout_s: float = 1800):
        t_end, delay, ok, text, warned = time.time() + timeout_s, 1.0, False, "", False
        while time.time() < t_end:
            time.sleep(delay)
            delay = min(delay * 1.5, 5.0)
            try:
                st = _hermes("GET", f"/v1/runs/{rid}")
            except Exception as e:
                text = f"lost contact with Hermes ({type(e).__name__})"
                continue
            status = str(st.get("status", ""))
            if status in ("completed", "failed", "cancelled", "error", "expired"):
                ok = status == "completed"
                out = st.get("output")
                if isinstance(out, list):  # (Responses-style output items)
                    out = " ".join(c.get("text", "") for item in out if isinstance(item, dict)
                                   for c in item.get("content", []) if isinstance(c, dict))
                text = str(out or st.get("error") or status).strip()
                break
            if "approval" in status and not warned and self.on_delegate_done:
                warned = True
                self.on_delegate_done(task, None, "Hermes is waiting for the user's approval before it can "
                                                  "continue (they can approve it in Hermes, e.g. on Telegram).")
        else:
            text = text or "it is taking longer than 30 minutes; it may still finish later"
        self.runs.pop(rid, None)
        if self.on_delegate_done:
            self.on_delegate_done(task, ok, text[:MAX_RESULT_CHARS])

    def cancel_runs(self):
        for rid in list(self.runs):
            try:
                _hermes("POST", f"/v1/runs/{rid}/stop", {})
            except Exception:
                pass


def tool_guidance(home: str = "") -> str:
    """How to use the tools in a spoken conversation (added to the persona's system prompt)."""
    today = _dt.date.today().strftime("%A, %B %d, %Y").replace(" 0", " ")
    where = f" The user is in {home}." if home else ""
    return (f"Today is {today}.{where} You have tools, but most questions don't need one: answer general knowledge "
            "(geography, science, history, well-known facts), advice, explanations, stories, jokes and conversation "
            "yourself, without tools and without mentioning them (when asked for a joke, just tell one). Call a tool "
            "only when the answer depends on live or "
            "recent information (news, weather, prices, scores, recent events, what's best or newest this year: "
            "search when asked to research or compare current things), on the time in another place, or on exact "
            "arithmetic, or when the user asks you to do something a tool does (timers and reminders, their lists, "
            "remembering something, devices, media). You can always get the weather and look things up online, so never say you can't: call the "
            "tool. Requests and questions about the user's devices and music (which lights are on, is the car locked, "
            "skip this song, turn it up) need home_assistant every time: they change, so never answer them from "
            "earlier in the conversation or guess. "
            "Your private note at the start of each reply has the user's local date and time; use get_current_time "
            "only for the time somewhere else. After a tool returns, answer "
            "from its results in one or two short spoken sentences, never reading out URLs, citation numbers or lists. "
            "If the results don't answer the question, say so briefly." + remembered())


def remembered() -> str:
    facts = memories()
    return ("\nSaved memories (use them when relevant; only the memory tool can change them, so when the user asks "
            "you to forget one, call memory with action forget before you answer): " +
            " ".join(f if f.endswith((".", "!", "?")) else f + "." for f in facts)) if facts else ""


def system_tools_block(specs) -> str:
    """The function-calling section of the system prompt, exactly as the model's chat template renders it."""
    return ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
            + "".join("\n" + json.dumps(s, ensure_ascii=False) for s in specs) +
            "\n</tools>\n\nIf none of the functions can be used, point it out. If the given question lacks the "
            "parameters required by the function, also point it out.\nIf you need to use a function, for each "
            "function call, output the function name and arguments within the following XML format:\n"
            "<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n"
            "<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call>\n")


_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


def parse_tool_calls(text: str) -> list[tuple[str, dict]]:
    """'<tool_call>name\\n<arg_key>k</arg_key>\\n<arg_value>v</arg_value>\\n</tool_call>...' -> [(name, {k: v})]"""
    calls = []
    for body in _CALL_RE.findall(text):
        name = body.split("<arg_key>", 1)[0].strip()
        if not name:
            continue
        args = {}
        for k, v in _ARG_RE.findall(body):
            v = v.strip()
            try:  # the template writes non-string values as JSON
                v = json.loads(v) if v[:1] in "[{" or v in ("true", "false", "null") or re.fullmatch(r"-?\d+(\.\d+)?", v) else v
            except ValueError:
                pass
            args[k.strip()] = v
        calls.append((name, args))
    return calls

