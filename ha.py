"""Home Assistant devices, controlled directly.

HA's own intent handling (Assist) only knows entities exposed to voice assistants, treats only the light domain
as lights (most lamps are smart plugs: switch.*) and answers "what lights are on?" with "not any". So Ming reads
the whole device list itself (states + the entity, device and area registries over the websocket API), keeps
what a person would call a device (no config/diagnostic entities like a plug's status LED, nothing hidden or
disabled), understands simple commands and questions about them, and calls HA's services directly. Anything it
doesn't understand goes to Assist as before.

The model's own wording of a command is often worse than the user's ("what's the temperature in the car?" came
through as "is the car locked?", copied from the tool's example), so pick() prefers what the user actually said
whenever that is a command this module understands.
"""
import difflib
import json
import re
import time
import urllib.request

CONTROLLABLE = {"light", "switch", "fan", "cover", "lock", "climate", "media_player", "scene", "script", "vacuum",
                "input_boolean", "humidifier", "water_heater", "valve", "siren"}
# switches people call lights: by name ("Kitchen Lights", "Dresser Lamp", "TV Stand LED")
LIGHTISH = re.compile(r"\b(lights?|lamps?|leds?|bulbs?|strips?|sconces?|chandeliers?|lanterns?|ambilight|"
                      r"nightlights?|night ?stands?|bedside)\b", re.I)
# ... and plugs that aren't lights, even among a room's "lights"
NOT_LIGHT = re.compile(r"\b(tv|television|screen|fan|heater|heat|defrost|charger|charging|pc|computer|desktop|"
                       r"server|printer|router|modem|ac|air ?con\w*|purifier|humidifier|dehumidifier|kettle|"
                       r"coffee|pump|filter|speaker|console|xbox|playstation|monitor|camera|sprinkler|"
                       r"irrigation|valve|usb)\b", re.I)
# cars (by the integration or the device's maker): "the car" means their devices
CAR_RE = re.compile(r"\b(kia|hyundai|genesis|tesla|teslemetry|tessie|toyota|lexus|bmw|mercedes|volvo|polestar|"
                    r"renault|nissan|ford|lincoln|mazda|subaru|honda|acura|audi|volkswagen|vw|skoda|seat|cupra|"
                    r"porsche|rivian|lucid|jaguar|land rover|chevrolet|gmc|cadillac|jeep|peugeot|citroen|opel|"
                    r"fiat|smartcar)", re.I)
KIND = {"vacuum": "vacuum", "lock": "lock", "climate": "thermostat", "fan": "fan", "media_player": "speaker"}
CACHE_S = 60  # (the device names in the tool description)
STATE_S = 2  # (commands and questions: states change)
FOLLOW_S = 300  # "turn it off", "and the bedroom too": about the last command, for this long

_cache = {"t": 0.0, "devices": None}
_last = {"ids": [], "action": None, "value": None, "step": None, "color": None, "lights": False, "t": 0.0}


# --------------------------------------------------------------------------- reading HA

def fetch(base: str, token: str) -> list[dict]:
    """Every controllable entity (see build), from one websocket connection."""
    from websockets.sync.client import connect
    with connect(re.sub(r"^http", "ws", base) + "/api/websocket", open_timeout=6, close_timeout=2,
                 max_size=2**25) as ws:
        json.loads(ws.recv())
        ws.send(json.dumps({"type": "auth", "access_token": token}))
        if json.loads(ws.recv()).get("type") != "auth_ok":
            raise PermissionError("Home Assistant rejected the token")
        for i, kind in enumerate(("get_states", "config/entity_registry/list", "config/device_registry/list",
                                  "config/area_registry/list"), 1):
            ws.send(json.dumps({"id": i, "type": kind}))
        out = {}
        while len(out) < 4:
            m = json.loads(ws.recv())
            if m.get("type") == "result":
                out[m["id"]] = m.get("result") or []
    return build(out[1], out[2], out[3], out[4])


def spoken(name: str) -> str:
    """A name as it's said: '[LG] webOS TV QNED99UPA' -> 'LG webOS TV' (no brackets, no model numbers)."""
    words = [w for w in re.sub(r"[\[\]()_]", " ", name).split() if not (re.search(r"\d", w) and len(w) >= 5)]
    return " ".join(words) or name


def build(states, entities, devices, areas) -> list[dict]:
    """-> [{id, domain, name, say, names, kind, area, area_names, state, attrs, light, car}] (no config/diagnostic,
    hidden or disabled entities)."""
    reg = {e["entity_id"]: e for e in entities}
    devs = {d["id"]: d for d in devices}
    area = {a["area_id"]: a for a in areas}
    out = []
    for s in states:
        eid = s["entity_id"]
        domain = eid.split(".", 1)[0]
        if domain not in CONTROLLABLE:
            continue
        e = reg.get(eid, {})
        if e.get("entity_category") or e.get("hidden_by") or e.get("disabled_by"):
            continue
        d = devs.get(e.get("device_id")) or {}
        a = area.get(e.get("area_id") or d.get("area_id")) or {}
        attrs = s.get("attributes") or {}
        slug = eid.split(".", 1)[1].replace("_", " ")
        name = str(attrs.get("friendly_name") or e.get("name") or e.get("original_name") or slug).strip()
        car = bool(CAR_RE.search(f"{e.get('platform') or ''} {d.get('manufacturer') or ''}".replace("_", " ")))
        kind = KIND.get(domain, "")
        if domain == "media_player" and (attrs.get("device_class") == "tv" or re.search(r"\b(tv|television)\b",
                                                                                        name, re.I)):
            kind = "tv"
        names = [name, str(d.get("name_by_user") or d.get("name") or ""), *map(str, e.get("aliases") or []),
                 slug if name == slug else ""]  # (an entity id isn't a name: "living_room_lights" is the Wall Lights)
        if kind and not car:
            names.append(kind)
        names = [n for n in dict.fromkeys(n.strip() for n in names) if n]
        # a plug is a light when its name says so ("TV Stand LED"), or when it's in a room and nothing says it's
        # something else (most plugs in rooms run lamps)
        light = domain == "light" or (domain == "switch" and not car and (
            bool(LIGHTISH.search(name)) or (not NOT_LIGHT.search(" ".join(names))
                                            and bool(LIGHTISH.search(" ".join(names)) or a.get("name")))))
        out.append({"id": eid, "domain": domain, "name": name, "say": spoken(name), "names": names, "kind": kind,
                    "area": a.get("name") or "", "area_names": [a["name"], *a.get("aliases", [])] if a.get("name") else [],
                    "state": s.get("state", ""), "attrs": attrs, "light": light, "car": car})
    return out


def devices(base: str, token: str, fresh: bool = False, max_age: float = CACHE_S) -> list[dict]:
    if fresh or _cache["devices"] is None or time.time() - _cache["t"] > max_age:
        _cache["devices"], _cache["t"] = fetch(base, token), time.time()
    return _cache["devices"]


def call_service(base: str, token: str, domain: str, service: str, data: dict):
    req = urllib.request.Request(f"{base}/api/services/{domain}/{service}", method="POST",
                                 data=json.dumps(data).encode(),
                                 headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.load(r)


# --------------------------------------------------------------------------- understanding a command

def _norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9%.]+", " ", s.lower().replace("'s", "")).strip()
    return re.sub(r"\b(\w\w+?)s\b", r"\1", s)  # (plural-blind: "lights" = "light")


QUESTION = re.compile(r"^(is|are|was|were|what|what's|which|who|how many|how much|how|tell me|any|do i have|"
                      r"did i leave|did i|check|whats)\b")
ACTIONS = [  # (pattern, action): the first match wins
    (QUESTION, "status"),
    (re.compile(r"\bunmute\b"), "unmute"), (re.compile(r"\bmute\b"), "mute"),
    (re.compile(r"\b(volume|louder|quieter|softer)\b"), "volume"),
    (re.compile(r"\bpause\b|\bstop\b.*\b(music|songs?|track|playing|playback|audio|podcast|spotify)\b"), "pause"),
    (re.compile(r"\b(resume|unpause)\b|^play$|^play (the )?(music|it|again)\b|\bkeep playing\b"), "play"),
    (re.compile(r"\b(skip|next (song|track))\b|^next\b"), "next"),
    (re.compile(r"\b(previous|last) (song|track)\b|^previous\b"), "previous"),
    (re.compile(r"\bunlock\b"), "unlock"), (re.compile(r"\block\b"), "lock"),
    (re.compile(r"\b(dock|go home|come home|back to (its|the) (dock|base|charger))\b"), "dock"),
    (re.compile(r"\b(dim|dimmer|darker|brighten|brighter|brightness|\d+ ?(%|percent))"), "brightness"),
    (re.compile(r"\btoggle\b"), "toggle"),
    (re.compile(r"\b(set|make|change)\b.*\b(temperature|thermostat|heat|cool|degrees?)\b|\bdegrees?\b"), "temperature"),
    (re.compile(r"\boff\b|\b(kill|disable|deactivate|stop)\b"), "off"),
    (re.compile(r"\bon\b|\b(enable|activate|start|run|launch|wake)\b"), "on"),
    (re.compile(r"\b(open|raise|lift)\b"), "open"), (re.compile(r"\b(close|lower|shut)\b"), "close"),
    (re.compile(r"\b(warm|heat|cool|preheat|precondition|defrost)\b"), "on"),  # "warm up the car"
]
VERBS = re.compile(r"\b(turn|switch|shut|power|flip|put|set|make|change|toggle|dim|dimmer|darker|brighten|brighter|"
                   r"brightness|kill|disable|deactivate|stop|enable|activate|start|run|launch|wake|unlock|lock|open|"
                   r"raise|lift|close|lower|on|off|to|up|down|percent|degrees?|temperature|still|currently|right now|"
                   r"now|pause|resume|unpause|play|playing|skip|next|previous|mute|unmute|volume|louder|quieter|"
                   r"softer|send|dock|go home|come home|back|color|colour)\b")
STOP = set(_norm("the a an my our all every everything please can could would you will in of and it its them that "
                  "this those these they one at for with be there locked unlocked opened closed running status "
                  "state offline online whole house home anything something which what any is are was were do does "
                  "did have has i me tell how many much who if or too also as well again mode scene script routine "
                  "uh um so just okay ok hey thanks thank actually maybe ming jarvis little bit more less some "
                  "music song track audio sound").split())
LIGHT_WORDS = {"light", "lamp", "lighting"}
CAR_WORDS = {"car", "vehicle", "truck"}
CLIMATE_WORDS = set(_norm("climate temperature temp heat heating heater ac air conditioning warm cool cold hot "
                          "preheat precondition").split())
MEDIA_ACTIONS = {"pause", "play", "next", "previous", "volume_up", "volume_down", "volume_set", "mute", "unmute"}
COLORS = ["warm white", "soft white", "cool white", "daylight", "red", "green", "blue", "purple", "pink", "orange",
          "yellow", "cyan", "magenta", "teal", "violet", "lavender", "turquoise", "gold", "indigo", "lime", "white"]
KELVIN = {"warm white": 2700, "soft white": 2700, "cool white": 5000, "daylight": 6000, "white": 4000}
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b")
SERVICE = {  # (action -> domain -> service); None = not possible; missing = turn_on / turn_off
    "on": {"cover": "open_cover", "lock": None, "vacuum": "start", "scene": "turn_on", "script": "turn_on",
           "valve": "open_valve"},
    "off": {"cover": "close_cover", "lock": None, "vacuum": "return_to_base", "scene": None, "script": None,
            "valve": "close_valve"},
    "open": {"cover": "open_cover", "valve": "open_valve", "lock": "unlock"},
    "close": {"cover": "close_cover", "valve": "close_valve", "lock": "lock"},
    "lock": {"lock": "lock"}, "unlock": {"lock": "unlock"}, "dock": {"vacuum": "return_to_base"},
    "toggle": {d: "toggle" for d in ("light", "switch", "fan", "input_boolean", "media_player", "cover")},
}
MEDIA_SERVICE = {"pause": "media_pause", "play": "media_play", "next": "media_next_track",
                 "previous": "media_previous_track", "volume_up": "volume_up", "volume_down": "volume_down",
                 "volume_set": "volume_set", "mute": "volume_mute", "unmute": "volume_mute"}
STATE_WORDS = {"unavailable": "offline", "heat": "heating", "cool": "cooling", "heat_cool": "on", "auto": "on"}
ON_STATES = {"on", "open", "opening", "playing", "unlocked", "cleaning", "heat", "cool", "heat_cool", "auto", "dry",
             "fan_only"}
WAKE = re.compile(r"^((hey|hi|ok|okay) (ming|jarvis)\b[,.!]?\s*|(hey|hi) \w+[,.!]\s*)")
# two commands in one breath ("turn off the lights and lock the car"): the model splits those, not pick()
MULTI = re.compile(r"\b(and|then|also)\b.*\b(turn|switch|set|lock|unlock|open|close|dim|start|stop|pause|play|make)\b")


def parse(command: str) -> dict:
    c = command.lower().strip().rstrip("?.! ")
    c = WAKE.sub("", c)
    c = re.sub(r"\b(left|still) (on|open|unlocked|running|playing)\b", r"\2", c)  # ("Left Shelf" is a name)
    follow = bool(re.match(r"(and|also|now)\b", c) or re.search(r"\b(too|as well|also)$", c))
    c = re.sub(r"^((and|also|now|so|um|uh|okay|ok|please|can you|could you|would you|will you|i want you to|"
               r"go ahead and)[, ]+)+", "", c).replace("what's", "what is").replace("whats", "what is")
    if re.match(r"(vacuum|clean|mop)\b(?! (is|are)\b)", c):  # "vacuum the house"
        c = "start the vacuum"
    action = next((a for rx, a in ACTIONS if rx.search(c)), None)
    num = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|degrees?|°)?", c)
    value = float(num.group(1)) if num else None
    step = None
    down = bool(re.search(r"\b(down|quieter|softer|lower|decrease|reduce|dimmer|darker|less)\b", c))
    up = bool(re.search(r"\b(up|louder|increase|raise|brighter|brighten|more)\b", c))
    lights = bool(re.search(r"\b(lights?|lamps?|lighting)\b", c))
    if action is None and lights and (up or down):  # "turn the lights down"
        action = "brightness"
    if action == "volume":
        action = "volume_set" if value is not None else "volume_down" if down else "volume_up" if up else None
    elif action == "brightness" and value is None:
        if down and "dim " not in f"{c} " or re.search(r"\b(dimmer|darker)\b", c):
            step = -25
        elif up:
            step = 25
        else:
            value = 30.0 if re.search(r"\bdim\b", c) else 100.0
    color = COLOR_RE.search(c)
    if color and action in (None, "on", "brightness") and (lights or re.search(r"\b(make|turn|set|change|colou?r)\b", c)) \
            and not (color.group(1) in ("warm", "cool") or re.search(r"\b(car|vehicle|thermostat)\b", c)):
        action = "color"
    target = QUESTION.sub(" ", c)
    target = re.sub(r"(\d+(?:\.\d+)?)\s*(%|percent|degrees?|°)?", " ", target)
    full = re.sub(r"\s+", " ", VERBS.sub(" ", target)).strip()  # (color words kept: "Party - Red")
    if action == "color":
        target = COLOR_RE.sub(" ", target)
    target = re.sub(r"\s+", " ", VERBS.sub(" ", target)).strip()
    return {"text": c, "action": action, "value": value, "step": step, "color": color and color.group(1),
            "target": target, "full": full, "lights": lights, "follow": follow,
            "ref": bool(re.search(r"\b(it|them|they|those|these|that one|this one)\b", c)),
            "every": bool(re.search(r"\b(all|every|everything|whole house|anything)\b", c))}


def _words(s: str) -> set[str]:
    return set(_norm(s).split()) - STOP


def _fresh() -> bool:
    return bool(_last["ids"]) and time.time() - _last["t"] < FOLLOW_S


def follow_up(p: dict) -> dict:
    """'and the bedroom too' after 'turn off the living room lights': the same action (and value)."""
    if p["action"] is None and p["follow"] and _fresh() and _last["action"] not in (None, "status"):
        p.update(action=_last["action"], value=p["value"] if p["value"] is not None else _last["value"],
                 step=_last["step"], color=_last["color"], lights=p["lights"] or _last["lights"])
    return p


def resolve(devs: list[dict], p: dict) -> tuple[list[dict], str]:
    """-> (the devices meant, the area named or ""). Sets p["named"] when they were named (not a room's or
    the home's lights)."""
    p["named"] = True
    t = p["target"]
    tw = _words(t)
    if p["ref"] and not tw - LIGHT_WORDS and _fresh():  # "turn it off", "set them to 40 percent"
        ids = set(_last["ids"])
        return [d for d in devs if d["id"] in ids], ""
    # a device called exactly what was said wins ("the bedroom lamp" is "Bedroom Lamp", not the bedroom's lamps;
    # "Party - Red" is a script, not a colour)
    def exact(words):
        if not words - LIGHT_WORDS or words & CAR_WORDS:  # ("the lights", "the car": not a name)
            return []
        return [d for d in devs if any(_words(n) == words for n in d["names"])]
    if p["action"] == "color":
        named = exact(_words(p["full"]))
        if named and all(d["domain"] in ("script", "scene") for d in named):
            p["action"] = "on"
            return named, ""
    named = exact(tw)
    if named:
        if p["action"] in MEDIA_ACTIONS:
            named = [d for d in named if d["domain"] == "media_player"] or named
        return named, ""
    nt = " " + _norm(t) + " "
    areas = {}  # an area named in it ("the living room lights", "everything in the bedroom")
    for d in devs:
        for an in d["area_names"]:
            if " " + _norm(an) + " " in nt:
                areas.setdefault(_norm(an), len(an))
    area = max(areas, key=areas.get) if areas else ""
    area_name = next((d["area"] for d in devs if area and any(_norm(a) == area for a in d["area_names"])), "")
    rest = tw - set(area.split())
    pool = [d for d in devs if not area or any(_norm(a) == area for a in d["area_names"])]
    if rest & CAR_WORDS:
        rest -= CAR_WORDS
        pool = [d for d in pool if d["car"]]
        climate = [d for d in pool if d["domain"] == "climate"]
        if climate and (rest & CLIMATE_WORDS or p["action"] == "temperature" or re.search(
                r"\b(warm|heat|cool|preheat|precondition|temperature|temp|climate|hot|cold|a/?c)\b", p["text"])):
            return climate, area_name
        if not rest:
            if p["action"] in ("lock", "unlock") or re.search(r"\block", p["text"]):
                return [d for d in pool if d["domain"] == "lock"], area_name
            if p["action"] == "status":
                return pool, area_name
    elif p["action"] == "temperature" and not rest - {"thermostat", "heat", "ac", "heating", "cooling"}:
        return [d for d in pool if d["domain"] == "climate" and not d["car"]], area_name
    if p["action"] == "dock":
        pool = [d for d in pool if d["domain"] == "vacuum"]
        rest -= {"vacuum", "robot", "it", "base", "charger", "home"}
        if not rest:
            return pool, area_name
    core = rest - LIGHT_WORDS
    if p["action"] == "status" and not core and re.search(r"\b(playing|song|music|track|listening)\b", p["text"]):
        media = [d for d in pool if d["domain"] == "media_player"]  # "what's playing?", "what song is this?"
        return [d for d in media if d["state"] == "playing"] or media, area_name
    if p["action"] in MEDIA_ACTIONS:  # music: the speakers (or TV) named, or the ones playing
        media = [d for d in pool if d["domain"] == "media_player"]
        named = core - {"speaker", "tv", "television"}
        if not named:
            if core & {"tv", "television"} or core & {"speaker"}:
                media = [d for d in media if d["kind"] == ("speaker" if "speaker" in core else "tv")]
            active = [d for d in media if d["state"] == "playing"] or (
                [d for d in media if d["state"] == "paused"] if p["action"] == "play" else [])
            if p["action"] in ("next", "previous") and active:  # (once: a group and its speakers skip together)
                active = [next((d for d in active if d["attrs"].get("media_title")), active[0])]
            p["named"] = bool(core)
            return active or media, area_name  # (none active: act() says nothing is playing)
        pool = media

    def score(d):  # named devices: "the dresser lamp", "TV stand", "the vacuum", "party lights" (a script)
        best = 0.0
        for n in d["names"]:
            nw = _words(n) - LIGHT_WORDS
            if not nw:
                continue
            if nw <= core:  # every word of a name said
                s = 1.0 + len(nw) * 0.01
            elif core <= nw:  # all of what was said is in the name
                s = 0.7 + 0.3 * len(core) / len(nw)
            else:
                s = max(len(nw & core) / len(nw | core),
                        0.9 * difflib.SequenceMatcher(None, " ".join(sorted(nw)), " ".join(sorted(core))).ratio())
            if d["light"] and not p["lights"] and s < 1.1:
                s -= 0.1  # "the TV" is the TV, not the "TV Lights" plug
            best = max(best, s)
        return best
    if core:
        scored = sorted(((score(d), d) for d in pool), key=lambda x: -x[0])
        if scored and scored[0][0] >= 0.65:
            top = scored[0][0]
            found = [d for s, d in scored if s >= top - 0.01]
            plural = {_norm(w) for w in re.findall(r"\b\w+s\b", t)} & core  # "the lamps": every lamp
            if plural:
                every = [d for s, d in scored if s >= 0.7 and plural <= _words(" ".join(d["names"]))]
                found = [d for d in every if d["domain"] not in ("script", "scene")] or every
            return found, area_name
        return [], area_name
    p["named"] = False
    if p["lights"]:
        lights = [d for d in pool if d["light"]]
        if "lamp" in rest:  # "the bedroom lamp": its lamps, if any are called that
            lamps = [d for d in lights if "lamp" in _words(" ".join(d["names"]))]
            p["named"], lights = bool(lamps), lamps or lights
        return lights, area_name
    if p["every"] or area or p["action"] == "status":
        kinds = ("light", "switch", "fan") + (("media_player",) if p["action"] == "status" else ())
        return [d for d in pool if d["domain"] in kinds and not d["car"]
                and (d["domain"] != "switch" or d["light"] or area)], area_name
    return [], area_name


def pick(command: str, heard: str, devs: list[dict]) -> str:
    """The command to carry out: what the user said, when that is one command this module understands (the model's
    rewording is often wrong), else the model's."""
    if not heard or MULTI.search(heard.lower()) or heard.strip().lower() == command.strip().lower():
        return command
    p = follow_up(parse(heard))
    if p["action"] is None:
        return command
    return heard if resolve(devs, p)[0] else command


def script_named(text: str, devs: list[dict]) -> bool:
    """Does `text` ask for a script or scene by name ("run movie mode")? (it isn't media, whatever it sounds like)"""
    if not re.search(r"\b(mode|scene|script|routine)\b", text.lower()):
        return False
    p = parse(text)
    found = resolve(devs, p)[0] if p["action"] in ("on", None) else []
    return bool(found) and all(d["domain"] in ("script", "scene") for d in found)


# --------------------------------------------------------------------------- answering

def _spoken(names: list[str]) -> str:
    names = list(dict.fromkeys(names))
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _is(names) -> str:
    return "is" if len(set(names)) == 1 and not names[0].lower().endswith("s") else "are"


def _state(d: dict) -> str:
    s, a = d["state"], d["attrs"]
    if d["domain"] == "climate":
        bits = [STATE_WORDS.get(s, s)]
        if a.get("current_temperature") is not None:
            bits.append(f"{a['current_temperature']:g} degrees now")
        if a.get("temperature") is not None and s != "off":
            bits.append(f"set to {a['temperature']:g}")
        return ", ".join(bits)
    if d["domain"] == "light" and s == "on" and a.get("brightness"):
        return f"on at {round(a['brightness'] / 2.55)} percent"
    if d["domain"] == "media_player" and s in ("playing", "paused") and a.get("media_title"):
        by = f" by {a['media_artist']}" if a.get("media_artist") else ""
        return f"{s} {a['media_title']}{by}"
    return STATE_WORDS.get(s, s.replace("_", " "))


def status(found: list[dict], p: dict, area: str) -> tuple[str, dict]:
    ids = [d["id"] for d in found]
    if found and all(d["domain"] == "media_player" for d in found) and re.search(
            r"\b(playing|song|music|track|listening)\b", p["text"]):
        playing = [f"{d['say']} is {_state(d)}" for d in found if d["state"] == "playing"]
        say = ". ".join(playing[:3]) + "." if playing else "Nothing is playing right now."
        return say, {"say": say, "targets": ids}
    if (p.get("named") and len(found) <= 5) or (len(found) <= 3 and not (p["lights"] and (p["every"] or area
                                                                                       or len(found) > 1))):
        lines = [f"{d['say']} {_is([d['say']])} {_state(d)}" for d in found]
        return "; ".join(lines) + ".", {"say": ". ".join(lines) + ".", "targets": ids}
    on = [d["say"] for d in found if d["state"] in ON_STATES]
    off = [d["say"] for d in found if d["state"] not in ON_STATES | {"unavailable", "unknown"}]
    gone = [d["say"] for d in found if d["state"] in ("unavailable", "unknown")]
    where = f" in the {area}" if area else ""
    what = "lights" if p["lights"] else "devices"
    text = ((f"On{where}: {_spoken(on)}." if on else f"No {what} are on{where}.") + (f" Off: {_spoken(off)}." if off else "")
            + (f" Offline: {_spoken(gone)}." if gone else ""))
    if on:
        say = (f"{_spoken(on)} {_is(on)} on{where}." if len(on) <= 5 else
               f"{len(on)} {what} are on{where}, including {_spoken(on[:3])}.")
    else:
        say = f"All the {what}{where} are off." if off else f"None of the {what}{where} are on."
    if gone and not on:  # (with some on, the offline ones are just noise)
        say += f" {_spoken(gone)} {_is(gone)} offline." if len(gone) <= 3 else f" {len(gone)} are offline."
    return text, {"say": say, "targets": ids}


def _service(d: dict, p: dict) -> tuple[str | None, dict]:
    """-> (the service for this device, its extra data), or (None, {}) when it can't do that."""
    action, value, dom, a = p["action"], p["value"], d["domain"], d["attrs"]
    if action in MEDIA_ACTIONS:
        if dom != "media_player":
            return None, {}
        extra = ({"volume_level": max(0, min(100, value or 0)) / 100} if action == "volume_set" else
                 {"is_volume_muted": action == "mute"} if action in ("mute", "unmute") else {})
        return MEDIA_SERVICE[action], extra
    if action == "brightness":
        if dom == "light":
            return "turn_on", ({"brightness_step_pct": p["step"]} if p["step"] else
                               {"brightness_pct": max(1, min(100, round(value)))})
        if dom == "switch" and d["light"] and (p["step"] or 0) > 0 and d["state"] == "off":
            return "turn_on", {}  # ("brighter": a lamp on a plug can at least come on)
        return None, {}
    if action == "color":
        modes = set(a.get("supported_color_modes") or [])
        colorful = bool(modes & {"hs", "xy", "rgb", "rgbw", "rgbww"})
        if dom != "light":
            return None, {}
        if p["color"] in KELVIN and (colorful or "color_temp" in modes):
            return "turn_on", {"color_temp_kelvin": KELVIN[p["color"]]}
        return ("turn_on", {"color_name": p["color"]}) if colorful else (None, {})
    if action == "temperature":
        return ("set_temperature", {"temperature": value}) if dom == "climate" and value is not None else (None, {})
    if dom == "script" and action == "off" and re.search(r"\b(off|stop|end)\b", d["name"], re.I):
        return "turn_on", {}  # ("turn off the party lights": run "Party - Lights off")
    if dom in ("script", "scene") and action == "toggle":
        return "turn_on", {}
    default = "turn_on" if action in ("on", "open") else "turn_off" if action in ("off", "close") else None
    return SERVICE.get(action, {}).get(dom, default if action not in ("toggle", "dock") else None), {}


def act(base: str, token: str, found: list[dict], p: dict) -> tuple[str, dict]:
    action, value = p["action"], p["value"]
    if action in ("pause", "next", "previous") and not any(d["state"] == "playing" for d in found):
        where = f" on the {_spoken([d['say'] for d in found])}" if 0 < len(found) <= 2 else ""
        return "Nothing is playing.", {"say": f"Nothing is playing{where}.", "targets": [d["id"] for d in found]}
    if action in MEDIA_ACTIONS and (not p.get("named") or len(found) > 2) and found and \
            all(d["domain"] == "media_player" for d in found):
        if action == "play" and not any(d["state"] in ("playing", "paused") for d in found):
            return "Nothing is paused.", {"say": "There's nothing paused to resume.", "targets": []}
        if action != "play" and not any(d["state"] == "playing" for d in found):
            return "Nothing is playing.", {"say": "Nothing is playing right now.", "targets": []}
    done, cant, gone, calls, already = [], [], [], {}, []
    for d in found:
        dom = d["domain"]
        if d["state"] == "unavailable":
            gone.append(d["say"])
            continue
        if action in ("on", "off") and dom in ("light", "switch", "fan", "input_boolean") and d["state"] == action:
            already.append(d["say"])  # (only what changes is said)
            continue
        svc, extra = _service(d, p)
        if svc is None:
            cant.append(d["say"])
            continue
        calls.setdefault((dom, svc, json.dumps(extra)), []).append(d)
    for (dom, svc, extra), ds in calls.items():
        call_service(base, token, dom, svc, dict(json.loads(extra), entity_id=[d["id"] for d in ds]))
        done += [d["say"] for d in ds]
    doms = {dom for dom, _, _ in calls}
    the = f"the {_spoken(done)}" if done else ""
    if len(done) > 4 and all(d["domain"] == "media_player" for d in found):
        the = f"{len(done)} speakers"
    if doms and doms <= {"script", "scene"}:
        say = f"Okay, started {_spoken(done)}."
    elif doms == {"vacuum"} and action in ("on", "off", "dock"):
        say = f"Okay, {the} {_is(done)} " + ("starting." if action == "on" else "heading back to its dock.")
    elif action in ("on", "off") and len(done) > 4:
        say = f"Okay, turned {action} {len(done)} {'lights' if p['lights'] or all(d['light'] for d in found) else 'devices'}."
    else:
        say = {"on": f"Okay, turned on {the}.", "off": f"Okay, turned off {the}.", "toggle": f"Okay, toggled {the}.",
               "lock": f"Okay, {the} {_is(done)} locked.", "unlock": f"Okay, {the} {_is(done)} unlocked.",
               "open": f"Okay, {the} {_is(done)} open.", "close": f"Okay, {the} {_is(done)} closed.",
               "brightness": (f"Okay, made {the} {'brighter' if (p['step'] or 0) > 0 else 'dimmer'}." if p["step"]
                              else f"Okay, {the} {_is(done)} set to {round(value or 0)} percent."),
               "temperature": f"Okay, {the} {_is(done)} set to {value or 0:g} degrees.",
               "color": f"Okay, {the} {_is(done)} {p['color']} now.",
               "pause": f"Okay, paused {the}.", "play": f"Okay, resumed {the}.",
               "next": "Okay, skipping to the next track.", "previous": "Okay, going back a track.",
               "volume_up": f"Okay, turned up {the}.", "volume_down": f"Okay, turned down {the}.",
               "volume_set": f"Okay, {the} volume is set to {round(value or 0)} percent.",
               "mute": f"Okay, muted {the}.", "unmute": f"Okay, unmuted {the}.",
               "dock": f"Okay, {the} {_is(done)} heading back to its dock."}[action]
    if not done:
        say = "" if not already else (f"The {_spoken(already)} {_is(already)} already {action}." if len(already) <= 2
                                      else f"They're already {action}.")
    if gone and (not done or len(found) <= 2):  # (in a group that worked, the offline ones are just noise)
        say += f" {_spoken(gone)} {_is(gone)} offline." if len(gone) <= 3 else f" {len(gone)} are offline."
    if cant:
        why = {"brightness": "can't be dimmed", "color": "can't change colour"}.get(action)
        say += f" {_spoken(cant)} {why}." if why else f" I can't do that with {_spoken(cant)}."
    text = ((f"Done ({action}): {_spoken(done)}." if done else "Nothing was changed.")
            + (f" Already {action}: {_spoken(already)}." if already else "")
            + (f" Offline: {_spoken(gone)}." if gone else "") + (f" Not possible for: {_spoken(cant)}." if cant else ""))
    return text, {"say": say.strip(), "targets": [d["id"] for d in found], "failed": gone + cant}


def handle(command: str, base: str, token: str) -> tuple[str, dict] | None:
    """Do or answer `command` from the device list; None when it isn't understood (-> Assist)."""
    p = follow_up(parse(command))
    if p["action"] is None:
        return None
    found, area = resolve(devices(base, token, max_age=STATE_S), p)
    if not found:
        return None
    _last.update(ids=[d["id"] for d in found], action=p["action"], value=p["value"], step=p["step"],
                 color=p["color"], lights=p["lights"], t=time.time())
    if p["action"] == "status":
        return status(found, p, area)
    if p["named"] and len({d["domain"] for d in found}) > 1:  # (a car: its lock, not its climate, for "lock")
        found = [d for d in found if _service(d, p)[0] or d["state"] == "unavailable"] or found
    _cache["t"] = 0.0  # (states change)
    return act(base, token, found, p)


def names_by_area(devs: list[dict], limit: int = 600) -> str:
    """'Living Room: Lamp, Corner Light; Bedroom: ...' for the tool description."""
    by = {}
    for d in devs:
        if d["domain"] in ("light", "switch", "fan", "cover", "lock", "climate", "vacuum", "scene", "script"):
            by.setdefault(d["area"] or "other", []).append(d["say"])
    out = "; ".join(f"{a}: {', '.join(dict.fromkeys(ns))}" for a, ns in sorted(by.items(), key=lambda x: x[0] == "other"))
    return out if len(out) <= limit else out[:limit].rsplit(",", 1)[0]
