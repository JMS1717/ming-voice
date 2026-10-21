"""Tools the assistant can call while answering: web search, weather, calculator, clock.

The thinker (Ming-3.0's Bailing LLM) was trained for function calling: given the function signatures in its system
prompt it emits <tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value></tool_call> instead of an answer, and
continues once the results come back in an OBSERVATION turn (see Thinker.stream). Everything here needs no API key:
DuckDuckGo (via the ddgs package) for search, Open-Meteo for weather. Only the query text leaves the machine.
"""
from __future__ import annotations

import ast
import datetime as _dt
import json
import math
import operator
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "ming-voice/1.0 (+https://github.com/JMS1717/ming-voice)"}


def _get_json(url: str, timeout: float = 6.0):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.load(r)


# --------------------------------------------------------------------------- web search

_NEWSY = re.compile(r"\b(news|latest|headlines?|today'?s|this week|announce[ds]?|breaking)\b", re.I)


def web_search(query: str, max_results: int = 5) -> tuple[str, dict]:
    from ddgs import DDGS
    query = (query or "").strip()
    if not query:
        return "Error: empty query.", {}
    hits = []
    if _NEWSY.search(query):  # dated headlines beat landing pages for "latest news about X"
        try:
            hits = [{"title": h.get("title", ""), "body": h.get("body", ""), "href": h.get("url", ""),
                     "date": (h.get("date") or "")[:10], "source": h.get("source", "")}
                    for h in DDGS().news(query, max_results=max_results)]
        except Exception:
            hits = []
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
        body = re.sub(r"\s+", " ", h.get("body", "")).strip()[:240]
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


def get_weather(location: str) -> tuple[str, dict]:
    location = (location or "").strip()
    if not location:
        return "Error: no location given. Ask the user where.", {}
    # "Paris, France" -> search "Paris", prefer the matching country/admin area
    name, _, hint = location.partition(",")
    found = _get_json("https://geocoding-api.open-meteo.com/v1/search?"
                      + urllib.parse.urlencode({"name": name.strip(), "count": 10, "language": "en"}))
    places = found.get("results") or []
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


def home_assistant(command: str) -> tuple[str, dict]:
    """Hand a spoken-style command or question to Home Assistant's own intent handling (Assist), which resolves
    device/area names and aliases and only touches entities exposed to voice assistants in HA."""
    command = (command or "").strip()
    if not command:
        return "Error: empty command.", {}
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
    _fn("get_current_time", "The current date and time (the user's local time unless a time zone is given).",
        {"timezone": {"type": "string", "description": "optional IANA time zone, e.g. 'Asia/Tokyo'"}}, []),
]
HA_SPEC = _fn("home_assistant", "Control or ask about devices in the user's home through Home Assistant: lights, "
              "switches, plugs, speakers and TV, thermostat, locks, the car, scenes and scripts (not computers or "
              "servers, not general questions). Pass the request as a "
              "short plain command or question naming the device or room, e.g. 'turn off the kitchen lights', "
              "'set the living room lamp to 30 percent', 'is the car locked?'.",
              {"command": {"type": "string", "description": "the request, in plain English"}}, ["command"])
JELLYFIN_SPEC = _fn("jellyfin", "The user's Jellyfin media server (movies, shows, music): search the library, play "
                    "something on a device where the Jellyfin app is open (a show continues with the next episode), "
                    "pause/resume/stop/next/previous, what's playing, which devices are available.",
                    {"action": {"type": "string", "enum": ["search", "play", "pause", "resume", "stop", "next",
                                                           "previous", "now_playing", "devices"]},
                     "query": {"type": "string", "description": "title, show, artist or album (search and play)"},
                     "device": {"type": "string", "description": "optional device name, e.g. 'TV' or 'phone'"}},
                    ["action"])
HERMES_SPEC = _fn("ask_hermes", "Delegate a task to Hermes, the user's more capable AI agent on their home server "
                  "(always use this when the user says to ask Hermes; it can also check and manage the user's "
                  "computers and servers), "
                  "for anything your other tools can't do: in-depth research, comparing options, reading web pages, "
                  "the user's files, GitHub, email, calendar, messages, Spotify, reminders and scheduled jobs, and "
                  "multi-step work. It runs in the background (seconds to minutes) and its answer is read out when "
                  "ready, so don't wait for it. For simply switching devices, use home_assistant instead.",
                  {"task": {"type": "string", "description": "the complete task, with all details from the conversation"}},
                  ["task"])
_IMPL = {"web_search": web_search, "get_weather": get_weather, "calculate": calculate,
         "get_current_time": get_current_time, "home_assistant": home_assistant, "jellyfin": jellyfin}
# Tools that change something: never run for a speculative reply (only once the user's turn has really ended)
SIDE_EFFECTS = {"home_assistant", "ask_hermes", "jellyfin"}
# What the assistant says while a slow tool runs (when it said nothing itself before calling it)
FILLERS = {"web_search": "Let me look that up.", "get_weather": "Let me check the weather."}
MAX_RESULT_CHARS = 1600  # ~400 tokens: results stay in the context for the rest of the chat


class Toolbox:
    """The tools offered to the model. run() never raises: errors go back to the model as text."""

    def __init__(self, specs=None, home: str = ""):
        if specs is None:  # the default set, plus Home Assistant / Hermes once configured
            specs = SPECS + ([HA_SPEC] if ha_token() else []) + ([JELLYFIN_SPEC] if jellyfin_config() else [])
            hermes = hermes_config()
            if hermes:  # (optional "about" in .hermes.json: e.g. which servers it runs on)
                specs = specs + [json.loads(json.dumps(HERMES_SPEC))]
                if hermes.get("about"):
                    specs[-1]["function"]["description"] += " " + hermes["about"]
        self.specs = list(specs)
        self.on_delegate_done = None  # callback(task, ok, text) when a Hermes run finishes
        self.runs: dict[str, str] = {}  # running Hermes run id -> task
        self._last_delegation = 0.0
        self.home = home  # the user's location: in the system prompt, and the default for the weather

    def signature(self) -> str:
        """What the system prompt depends on (a change means rebuilding the context)."""
        return json.dumps([self.specs, self.home], sort_keys=True)

    @property
    def names(self):
        return [s["function"]["name"] for s in self.specs]

    def run(self, name: str, args: dict, cancel: threading.Event | None = None) -> tuple[str, dict]:
        """-> (result text for the model, info for the UI: query/sources/elapsed/error)"""
        t0 = time.perf_counter()
        impl = ({**_IMPL, "ask_hermes": self._ask_hermes}).get(name) if name in self.names else None
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
        info = dict(info or {}, elapsed_ms=round((time.perf_counter() - t0) * 1000))
        return text, info


    # -- Hermes: start a background run, announce its result through on_delegate_done --
    def _ask_hermes(self, task: str) -> tuple[str, dict]:
        task = (task or "").strip()
        if not task:
            return "Error: empty task.", {}
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
            "yourself, without tools and without mentioning them. Call a tool only when the answer depends on live or "
            "recent information (news, weather, prices, scores, recent events), on the current time, or on exact "
            "arithmetic. You do not know the current time: use get_current_time for it. After a tool returns, answer "
            "from its results in one or two short spoken sentences, never reading out URLs, citation numbers or lists. "
            "If the results don't answer the question, say so briefly.")


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

