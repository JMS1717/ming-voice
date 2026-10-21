"""Home Assistant devices, controlled directly.

HA's own intent handling (Assist) only knows entities exposed to voice assistants, treats only the light domain
as lights (most lamps are smart plugs: switch.*) and answers "what lights are on?" with "not any". So Ming reads
the whole device list itself (states + the entity, device and area registries over the websocket API), keeps
what a person would call a device (no config/diagnostic entities like a plug's status LED, nothing hidden or
disabled), understands simple commands and questions about them, and calls HA's services directly. Anything it
doesn't understand goes to Assist as before.
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
CACHE_S = 60

_cache = {"t": 0.0, "devices": None}


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


def build(states, entities, devices, areas) -> list[dict]:
    """-> [{id, domain, name, names, area, area_names, state, attrs, light, car}] (no config/diagnostic, hidden or
    disabled entities)."""
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
        names = [name, str(d.get("name_by_user") or d.get("name") or ""), *map(str, e.get("aliases") or []), slug]
        if domain in KIND and not car:
            names.append(KIND[domain])
        names = [n for n in dict.fromkeys(n.strip() for n in names) if n]
        # a plug is a light when its name says so ("TV Stand LED"), or when it's in a room and nothing says it's
        # something else (most plugs in rooms run lamps)
        light = domain == "light" or (domain == "switch" and not car and (
            bool(LIGHTISH.search(name)) or (not NOT_LIGHT.search(" ".join(names))
                                            and bool(LIGHTISH.search(" ".join(names)) or a.get("name")))))
        out.append({"id": eid, "domain": domain, "name": name, "names": names, "area": a.get("name") or "",
                    "area_names": [a["name"], *a.get("aliases", [])] if a.get("name") else [],
                    "state": s.get("state", ""), "attrs": attrs, "light": light, "car": car})
    return out


def devices(base: str, token: str, fresh: bool = False) -> list[dict]:
    if fresh or _cache["devices"] is None or time.time() - _cache["t"] > CACHE_S:
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
    (re.compile(r"\bunlock\b"), "unlock"), (re.compile(r"\block\b"), "lock"),
    (re.compile(r"\b(dim|brighten|brightness|\d+ ?(%|percent))"), "brightness"),
    (re.compile(r"\btoggle\b"), "toggle"),
    (re.compile(r"\b(set|make|change)\b.*\b(temperature|thermostat|heat|cool|degrees?)\b|\bdegrees?\b"), "temperature"),
    (re.compile(r"\boff\b|\b(kill|disable|deactivate|stop)\b"), "off"),
    (re.compile(r"\bon\b|\b(enable|activate|start|run|launch)\b"), "on"),
    (re.compile(r"\b(open|raise|lift)\b"), "open"), (re.compile(r"\b(close|lower|shut)\b"), "close"),
]
VERBS = re.compile(r"\b(turn|switch|shut|power|flip|put|set|make|change|toggle|dim|brighten|brightness|kill|disable|"
                   r"deactivate|stop|enable|activate|start|run|launch|unlock|lock|open|raise|lift|close|lower|on|"
                   r"off|to|up|down|percent|degrees?|temperature|still|currently|right now|now)\b")
STOP = set(_norm("the a an my our all every everything please can could would you will in of and it its them that this "
                  "those these at for with be there left locked unlocked opened closed running status state offline "
                  "online whole house home anything something which what any is are was were do does did have has i "
                  "me tell how many much who if or").split())
LIGHT_WORDS = {"light", "lamp", "lighting"}
SERVICE = {  # (action -> domain -> service); None = not possible; missing = turn_on / turn_off
    "on": {"cover": "open_cover", "lock": None, "vacuum": "start", "scene": "turn_on", "script": "turn_on",
           "valve": "open_valve"},
    "off": {"cover": "close_cover", "lock": None, "vacuum": "return_to_base", "scene": None,
            "valve": "close_valve"},
    "open": {"cover": "open_cover", "valve": "open_valve", "lock": "unlock"},
    "close": {"cover": "close_cover", "valve": "close_valve", "lock": "lock"},
    "lock": {"lock": "lock"}, "unlock": {"lock": "unlock"},
    "toggle": {d: "toggle" for d in ("light", "switch", "fan", "input_boolean", "media_player", "cover")},
}
STATE_WORDS = {"unavailable": "offline", "heat": "heating", "cool": "cooling", "heat_cool": "on", "auto": "on"}
ON_STATES = {"on", "open", "opening", "playing", "unlocked", "cleaning", "heat", "cool", "heat_cool", "auto", "dry",
             "fan_only"}


def parse(command: str) -> dict:
    c = command.lower().strip().rstrip("?.! ")
    c = re.sub(r"^((hey |ok |okay )?ming,? |please |can you |could you |would you |will you |i want you to |"
               r"go ahead and )+", "", c).replace("what's", "what is").replace("whats", "what is")
    action = next((a for rx, a in ACTIONS if rx.search(c)), None)
    num = re.search(r"(\d+(?:\.\d+)?)\s*(%|percent|degrees?|°)?", c)
    value = float(num.group(1)) if num else None
    if action == "brightness" and value is None:
        value = 30.0 if "dim" in c else 100.0
    target = QUESTION.sub(" ", c)
    target = VERBS.sub(" ", re.sub(r"(\d+(?:\.\d+)?)\s*(%|percent|degrees?|°)?", " ", target))
    return {"text": c, "action": action, "value": value, "target": re.sub(r"\s+", " ", target).strip(),
            "lights": bool(re.search(r"\b(lights?|lamps?|lighting)\b", c)),
            "every": bool(re.search(r"\b(all|every|everything|whole house|anything)\b", c))}


def _words(s: str) -> set[str]:
    return set(_norm(s).split()) - STOP


def resolve(devs: list[dict], p: dict) -> tuple[list[dict], str]:
    """-> (the devices meant, the area named or "")."""
    t = p["target"]
    tw = _words(t)
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
    if rest & {"car", "vehicle", "truck"}:
        rest -= {"car", "vehicle", "truck"}
        pool = [d for d in pool if d["car"]]
        if not rest:
            if p["action"] in ("lock", "unlock") or re.search(r"\block", p["text"]):
                return [d for d in pool if d["domain"] == "lock"], area_name
            if p["action"] == "status":
                return pool, area_name
    elif p["action"] == "temperature" and not rest - {"thermostat", "heat", "ac", "heating", "cooling"}:
        return [d for d in pool if d["domain"] == "climate" and not d["car"]], area_name

    def score(d):  # named devices: "the dresser lamp", "TV stand", "the vacuum"
        best = 0.0
        for n in d["names"]:
            nw = _words(n)
            if not nw:
                continue
            if nw <= rest:  # every word of a name said
                s = 1.0 + len(nw) * 0.01
            elif rest <= nw:  # all of what was said is in the name
                s = 0.7 + 0.3 * len(rest) / len(nw)
            else:
                s = max(len(nw & rest) / len(nw | rest),
                        0.9 * difflib.SequenceMatcher(None, " ".join(sorted(nw)), " ".join(sorted(rest))).ratio())
            if d["light"] and not p["lights"] and s < 1:
                s -= 0.1  # "the TV" is the TV, not the "TV Lights" plug
            best = max(best, s)
        return best
    core = rest - LIGHT_WORDS
    if core:
        scored = sorted(((score(d), d) for d in pool), key=lambda x: -x[0])
        if scored and scored[0][0] >= 0.65:
            top = scored[0][0]
            return [d for s, d in scored if s >= top - 0.01], area_name
        return [], area_name
    if p["lights"]:
        lights = [d for d in pool if d["light"]]
        if "lamp" in rest:  # "the bedroom lamp": its lamps, if any are called that
            lights = [d for d in lights if "lamp" in _words(" ".join(d["names"]))] or lights
        return lights, area_name
    if p["every"] or area or p["action"] == "status":
        kinds = ("light", "switch", "fan") + (("media_player",) if p["action"] == "status" else ())
        return [d for d in pool if d["domain"] in kinds and not d["car"]
                and (d["domain"] != "switch" or d["light"] or area)], area_name
    return [], area_name


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
    return STATE_WORDS.get(s, s.replace("_", " "))


def status(found: list[dict], p: dict, area: str) -> tuple[str, dict]:
    ids = [d["id"] for d in found]
    if len(found) <= 3 and not (p["lights"] and (p["every"] or area or len(found) > 1)):
        lines = [f"{d['name']} {_is([d['name']])} {_state(d)}" for d in found]
        return "; ".join(lines) + ".", {"say": ". ".join(lines) + ".", "targets": ids}
    on = [d["name"] for d in found if d["state"] in ON_STATES]
    off = [d["name"] for d in found if d["state"] not in ON_STATES | {"unavailable", "unknown"}]
    gone = [d["name"] for d in found if d["state"] in ("unavailable", "unknown")]
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


def act(base: str, token: str, found: list[dict], p: dict) -> tuple[str, dict]:
    action, value = p["action"], p["value"]
    done, cant, gone, calls, already = [], [], [], {}, []
    for d in found:
        dom = d["domain"]
        if d["state"] == "unavailable":
            gone.append(d["name"])
            continue
        if action in ("on", "off") and dom in ("light", "switch", "fan", "input_boolean") and d["state"] == action:
            already.append(d["name"])  # (only what changes is said)
            continue
        extra = {}
        if action == "brightness":
            svc = "turn_on" if dom == "light" else None
            extra = {"brightness_pct": max(1, min(100, round(value)))}
        elif action == "temperature":
            svc = "set_temperature" if dom == "climate" and value is not None else None
            extra = {"temperature": value}
        else:
            default = "turn_on" if action in ("on", "open") else "turn_off" if action in ("off", "close") else None
            svc = SERVICE.get(action, {}).get(dom, default if action != "toggle" else None)
        if svc is None:
            cant.append(d["name"])
            continue
        calls.setdefault((dom, svc, json.dumps(extra)), []).append(d)
    for (dom, svc, extra), ds in calls.items():
        call_service(base, token, dom, svc, dict(json.loads(extra), entity_id=[d["id"] for d in ds]))
        done += [d["name"] for d in ds]
    verb = {"on": "turned on", "off": "turned off", "toggle": "switched", "lock": "locked", "unlock": "unlocked",
            "open": "opened", "close": "closed", "brightness": f"set to {round(value or 0)} percent",
            "temperature": f"set to {value or 0:g} degrees"}[action]
    if calls and all(dom == "vacuum" for dom, _, _ in calls):
        verb = {"on": "starting", "off": "heading back to its dock"}.get(action, verb)
    if not done:
        say = "" if not already else (f"The {_spoken(already)} {_is(already)} already {action}." if len(already) <= 2
                                      else f"They're already {action}.")
    elif len(done) <= 4:
        say = (f"Okay, {verb} the {_spoken(done)}." if action in ("on", "off") and verb.startswith("turned") else
               f"Okay, the {_spoken(done)} {_is(done)} {verb}.")
    else:
        say = f"Okay, {verb} {len(done)} {'lights' if p['lights'] else 'devices'}."
    if gone and (not done or len(found) <= 2):  # (in a group that worked, the offline ones are just noise)
        say += f" {_spoken(gone)} {_is(gone)} offline." if len(gone) <= 3 else f" {len(gone)} are offline."
    if cant:
        say += (f" {_spoken(cant)} can't be dimmed." if action == "brightness" else
                f" I can't do that with {_spoken(cant)}.")
    text = ((f"Done ({verb}): {_spoken(done)}." if done else "Nothing was changed.")
            + (f" Offline: {_spoken(gone)}." if gone else "") + (f" Not possible for: {_spoken(cant)}." if cant else ""))
    return text, {"say": say.strip(), "targets": [d["id"] for d in found], "failed": gone + cant}


def handle(command: str, base: str, token: str) -> tuple[str, dict] | None:
    """Do or answer `command` from the device list; None when it isn't understood (-> Assist)."""
    p = parse(command)
    if p["action"] is None:
        return None
    found, area = resolve(devices(base, token), p)
    if not found:
        return None
    if p["action"] == "status":
        return status(found, p, area)
    _cache["t"] = 0.0  # (states change)
    return act(base, token, found, p)


def names_by_area(devs: list[dict], limit: int = 600) -> str:
    """'Living Room: Lamp, Corner Light; Bedroom: ...' for the tool description."""
    by = {}
    for d in devs:
        if d["domain"] in ("light", "switch", "fan", "cover", "lock", "climate", "vacuum", "scene", "script"):
            by.setdefault(d["area"] or "other", []).append(d["name"])
    out = "; ".join(f"{a}: {', '.join(dict.fromkeys(ns))}" for a, ns in sorted(by.items(), key=lambda x: x[0] == "other"))
    return out if len(out) <= limit else out[:limit].rsplit(",", 1)[0]
