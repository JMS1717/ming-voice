"""Tool calling: offline unit checks, then (if a server is running) typed conversations over the WebSocket.

    .venv/bin/python dev/test_tools.py            # unit checks + live checks against localhost:8000
    .venv/bin/python dev/test_tools.py --offline  # unit checks only (no server, no network)
    .venv/bin/python dev/test_tools.py --home     # only Home Assistant / Jellyfin (what's configured)
    .venv/bin/python dev/test_tools.py --hermes   # only the Hermes delegation (sends Hermes a real task)

Live checks: the model calls a tool when the question needs one (news, weather, arithmetic, time) and not for
general knowledge or small talk, answers from the results, says something while a slow tool runs, keeps the
conversation coherent afterwards (follow-up question), and nothing is spoken that looks like a URL or citation.
"""
import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import Toolbox, calculate, parse_tool_calls  # noqa: E402

URL = os.environ.get("MING_URL", "localhost:8000")
fails = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)


# --------------------------------------------------------------------------- offline
calls = parse_tool_calls("<tool_call>web_search\n<arg_key>query</arg_key>\n<arg_value>bitcoin price</arg_value>\n</tool_call>"
                         "<tool_call>calculate\n<arg_key>expression</arg_key>\n<arg_value>2+2</arg_value>\n</tool_call>")
check(calls == [("web_search", {"query": "bitcoin price"}), ("calculate", {"expression": "2+2"})], "parse two tool calls")
check(parse_tool_calls("<tool_call>get_current_time\n</tool_call>") == [("get_current_time", {})], "parse a call without arguments")
check(parse_tool_calls("<tool_call>web_search\n<arg_key>query</arg_key>\n<arg_value>cut off") == [("web_search", {})],
      "a truncated call keeps its name")
for expr, want in (("17*23", "391"), ("2^10", "1024"), ("15% of 80", "12"), ("sqrt(16)+1", "5"), ("1,234 * 2", "2468"),
                   ("max(3, 9)", "9"), ("10/4", "2.5")):
    check(calculate(expr)[0].endswith("= " + want), f"calculate {expr} = {want}")
check(calculate("__import__('os')")[0].startswith("Error"), "calculator refuses code")
check(calculate("9**9**9")[0].startswith("Error"), "calculator refuses huge powers")
import datetime as dt  # noqa: E402
from tools import parse_when, spoken_duration  # noqa: E402
now = dt.datetime(2026, 10, 21, 17, 0).astimezone()  # a Wednesday, 5 pm
for when, want in (("10 minutes", "17:10"), ("an hour and a half", "18:30"), ("half an hour", "17:30"),
                   ("90 seconds", "17:01:30"), ("twenty five minutes", "17:25"), ("1 hour 15 minutes", "18:15"),
                   ("2 and a half minutes", "17:02:30"), ("10", "17:10"), ("at 5:30 pm", "17:30"),
                   ("at 9 am", "Thu 09:00"), ("tomorrow at 9am", "Thu 09:00"), ("7:30", "19:30"),
                   ("noon", "Thu 12:00"), ("friday at 8 p.m.", "Fri 20:00"), ("in a few minutes", "17:03")):
    got = parse_when(when, now)
    fmt = "%a %H:%M" if " " in want else "%H:%M:%S" if want.count(":") == 2 else "%H:%M"
    check(got is not None and got[0].strftime(fmt) == want, f"parse_when {when!r} -> {want} (got {got and got[0].strftime(fmt)})")
check(parse_when("whenever", now) is None, "parse_when rejects nonsense")
check(spoken_duration(5400) == "1 hour and 30 minutes", "spoken duration")
from tools import reminder_text  # noqa: E402
for msg, want in (("Remind me to stretch", ("stretch", "Time to stretch.")), ("A 10-minute timer is set.", ("", "")),
                  ("pasta", ("pasta", "Reminder: pasta.")), ("", ("", "")), ("timer for 5 minutes", ("", "")),
                  ("Reminder: time to stretch", ("stretch", "Time to stretch.")),
                  ("Time to take the bread out", ("take the bread out", "Time to take the bread out."))):
    check(reminder_text(msg) == want, f"reminder_text {msg!r} -> {want} (got {reminder_text(msg)})")
tb = Toolbox()
tb.timers.path = tb.timers.path.with_name(".timers.test.json")  # (don't touch real timers)
tb.timers.items = []
r = tb.run("timer", {"action": "set", "when": "10 minutes"})
check(r[1].get("say") == "Okay, timer set for 10 minutes.", f"set a timer: {r}")
r = tb.run("timer", {"action": "set", "when": "at 11:59 pm", "message": "take out the trash"})
check("remind you at 11:59 PM" in r[1].get("say", ""), f"set a reminder: {r}")
check("take out the trash" in tb.run("timer", {"action": "list"})[0], "list timers")
check(tb.run("timer", {"action": "cancel", "message": "the trash reminder"})[0] == "Cancelled 1." and
      len(tb.timers.items) == 1 and tb.timers.items[0]["duration"], "cancel by name keeps the other one")
check(tb.run("timer", {"action": "cancel", "message": "all"})[0] == "Cancelled 1." and not tb.timers.items, "cancel all")
tb.timers.path.unlink(missing_ok=True)
import tools  # noqa: E402  risky Home Assistant actions need a yes in a later turn
sent = []
tools.home_assistant = lambda command, room="": (sent.append(command) or "Done.", {"targets": [command]})
hb = Toolbox(specs=tools.SPECS + [tools.HA_SPEC])
ha = lambda cmd, **kw: hb.run("home_assistant", {"command": cmd, **kw})
hb.new_turn()
check(ha("turn off the lamp")[0].startswith("Done.") and sent == ["turn off the lamp"], "safe action runs at once")
r = ha("unlock the front door")
check(r[1].get("needs_confirmation") and "unlock the front door?" in r[1]["say"] and len(sent) == 1, f"risky asks: {r}")
check(ha("unlock the front door", confirmed=True)[1].get("needs_confirmation"), "no yes in the same turn")
hb.new_turn()
check(ha("unlock the back door", confirmed=True)[1].get("needs_confirmation"), "a different door is asked again")
hb.new_turn()
r = ha("unlock back door", confirmed=True)
check(r[1].get("confirmed") and sent[-1] == "unlock the back door", f"confirmed next turn runs what was asked: {r}")
check(ha("open the garage door")[1].get("needs_confirmation"), "garage asks")
for cmd in ("is the garage open?", "Is the garage door open", "lock the front door", "turn on the porch light"):
    check(not tools.is_risky(cmd), f"not risky: {cmd}")
for cmd in ("disarm the alarm", "turn off the security system", "raise the garage door", "open the gate"):
    check(tools.is_risky(cmd), f"risky: {cmd}")
import ha as hamod  # noqa: E402  the device list and commands (a made-up home: nothing is called)
st = lambda eid, state, name, **a: {"entity_id": eid, "state": state, "attributes": {"friendly_name": name, **a}}  # noqa: E731
home = hamod.build(
    [st("light.ceiling", "on", "Ceiling Light", brightness=128, supported_color_modes=["hs", "color_temp"]), st("switch.reading_lamp", "off", "Reading Lamp"),
     st("switch.plug_1", "on", "Corner"), st("switch.reading_lamp_led", "on", "Reading Lamp LED"),
     st("switch.coffee", "off", "Coffee Maker"), st("switch.strip", "unavailable", "Bed Strip"),
     st("media_player.tv", "off", "Den TV"), st("switch.tv_lights", "off", "TV Lights"),
     st("lock.car_door", "locked", "Car Door Lock"), st("climate.car", "off", "Car Climate"),
     st("climate.hall", "heat", "Hallway", current_temperature=68, temperature=70), st("vacuum.robo", "docked", "Robo"),
     st("input_boolean.helper", "on", "Lamp Automation"), st("sensor.temp", "20", "Temperature"),
     st("media_player.kitchen", "playing", "Kitchen Speaker", media_title="Blue Song", media_artist="A Band"),
     st("script.movie_night", "off", "Movie Night"), st("light.left_shelf", "off", "Left Shelf")],
    [{"entity_id": "switch.reading_lamp_led", "entity_category": "config"},
     {"entity_id": "switch.plug_1", "device_id": "d1"}, {"entity_id": "switch.strip", "area_id": "bed"},
     {"entity_id": "light.ceiling", "area_id": "den"}, {"entity_id": "switch.reading_lamp", "area_id": "den"},
     {"entity_id": "switch.coffee", "area_id": "kitchen"}, {"entity_id": "media_player.tv", "area_id": "den"},
     {"entity_id": "switch.tv_lights", "area_id": "den"}, {"entity_id": "media_player.kitchen", "area_id": "kitchen"},
     {"entity_id": "lock.car_door", "device_id": "car", "platform": "some_cloud"},
     {"entity_id": "climate.car", "device_id": "car", "platform": "some_cloud"}],
    [{"id": "d1", "name": "Smart Plug", "area_id": "den"}, {"id": "car", "name": "My Car", "manufacturer": "Kia"}],
    [{"area_id": "den", "name": "Den", "aliases": ["family room"]}, {"area_id": "bed", "name": "Bedroom"},
     {"area_id": "kitchen", "name": "Kitchen"}])
byid = {d["id"]: d for d in home}
check("switch.reading_lamp_led" not in byid and "sensor.temp" not in byid, "config entities and sensors are left out")
check([d["id"] for d in home if d["light"]] == ["light.ceiling", "switch.reading_lamp", "switch.plug_1",
                                                 "switch.strip", "switch.tv_lights", "light.left_shelf"],
      "lamps on plugs count as lights")
check(byid["lock.car_door"]["car"] and not byid["climate.hall"]["car"], "the car's devices")
for cmd, action, want in (
        ("turn off the den lights", "off", ["light.ceiling", "switch.reading_lamp", "switch.plug_1", "switch.tv_lights"]),
        ("Turn on the family room lights please", "on", ["light.ceiling", "switch.reading_lamp", "switch.plug_1",
                                                          "switch.tv_lights"]),
        ("which lights are on?", "status", ["light.ceiling", "switch.reading_lamp", "switch.plug_1", "switch.strip",
                                            "switch.tv_lights", "light.left_shelf"]),
        ("did I leave the left shelf on?", "status", ["light.left_shelf"]),
        ("what's the temperature in the car?", "status", ["climate.car"]), ("warm up the car", "on", ["climate.car"]),
        ("turn the ceiling light blue", "color", ["light.ceiling"]), ("pause the music", "pause", ["media_player.kitchen"]),
        ("skip this song", "next", ["media_player.kitchen"]),
        ("turn up the volume in the kitchen", "volume_up", ["media_player.kitchen"]),
        ("run movie night", "on", ["script.movie_night"]), ("send the vacuum back to its dock", "dock", ["vacuum.robo"]),
        ("turn on the reading lamp", "on", ["switch.reading_lamp"]), ("turn off the TV", "off", ["media_player.tv"]),
        ("is the car locked?", "status", ["lock.car_door"]), ("lock the car", "lock", ["lock.car_door"]),
        ("set the thermostat to 72", "temperature", ["climate.hall"]), ("start the vacuum", "on", ["vacuum.robo"]),
        ("dim the ceiling light to 20%", "brightness", ["light.ceiling"]),
        ("what's the weather like", "status", []), ("turn on the jacuzzi", "on", [])):
    p = hamod.parse(cmd)
    got = [d["id"] for d in hamod.resolve(home, p)[0]]
    check(p["action"] == action and got == want, f"ha {cmd!r} -> {action} {want} (got {p['action']} {got})")
called = []
hamod.call_service = lambda b, t, dom, svc, data: called.append((dom, svc, data))
hamod.fetch = lambda b, t: home  # (states are re-read every couple of seconds)
hamod._cache.update(t=time.time(), devices=home)
r = hamod.handle("turn off the den lights", "", "")
check(called == [("light", "turn_off", {"entity_id": ["light.ceiling"]}),
                 ("switch", "turn_off", {"entity_id": ["switch.plug_1"]})]
      and r[1]["say"] == "Okay, turned off the Ceiling Light and Corner.", f"only what's on/off changes? {called} {r}")
called.clear()
hamod._cache.update(t=time.time(), devices=home)
r = hamod.handle("which lights are on in the bedroom", "", "")
check(not called and r[1]["say"] == "None of the lights in the Bedroom are on. Bed Strip is offline.",
      f"status question changes nothing: {r}")
r = hamod.handle("dim the den lights to 40 percent", "", "")
check(called == [("light", "turn_on", {"brightness_pct": 40, "entity_id": ["light.ceiling"]})]
      and "can't be dimmed" in r[1]["say"], f"dimming plugs: {r}")
called.clear()
r = hamod.handle("make the den lights warm white", "", "")
check(called == [("light", "turn_on", {"color_temp_kelvin": 2700, "entity_id": ["light.ceiling"]})]
      and r[1]["say"].startswith("Okay, the Ceiling Light is warm white now.") and "can't change colour" in r[1]["say"],
      f"colour temperature (plugs can't): {r}")
called.clear()
hamod._cache.update(t=time.time(), devices=home)
hamod.handle("turn off the reading lamp", "", "")
r = hamod.handle("now turn it on", "", "")  # (about the last command)
check(called == [("switch", "turn_on", {"entity_id": ["switch.reading_lamp"]})], f"'it' is the last device: {called} {r}")
called.clear()
r = hamod.handle("set the volume to 30 percent", "", "")
check(called == [("media_player", "volume_set", {"volume_level": 0.3, "entity_id": ["media_player.kitchen"]})],
      f"volume of what's playing: {called} {r}")
called.clear()
quiet = [dict(d, state="idle") if d["domain"] == "media_player" else d for d in home]
r = (hamod.act("", "", hamod.resolve(quiet, hamod.parse("turn the volume down"))[0], hamod.parse("turn the volume down")),
     hamod.act("", "", hamod.resolve(quiet, hamod.parse("play it again"))[0], hamod.parse("play it again")))
check(not called and r[0][1]["say"] == "Nothing is playing right now." and "nothing paused" in r[1][1]["say"],
      f"no music: not every speaker in the house: {called} {r}")
heard = "what's the temperature in the car?"  # the model's rewording vs what the user said
check(hamod.pick("is the car locked?", heard, home) == heard and hamod.pick("turn on the den lights", "uh, yes", home)
      == "turn on the den lights" and hamod.pick("turn off the lamp", "turn off the lamp and tell me a joke", home)
      == "turn off the lamp", "pick: the user's own words when they're a command")
both = "turn off the reading lamp and lock the car"
check(hamod.pick("turn off the lamp", both, home) == both
      and [(p["action"], [d["id"] for d in f]) for p, f, _ in hamod.split(both, home)]
      == [("off", ["switch.reading_lamp"]), ("lock", ["lock.car_door"])], f"two commands: {hamod.split(both, home)}")
p = hamod.parse("turn off the lights")
p["room"] = "family room"
check([d["id"] for d in hamod.resolve(home, p)[0]] == ["light.ceiling", "switch.reading_lamp", "switch.plug_1",
                                                       "switch.tv_lights"], "the lights of the room the user is in")
p = hamod.parse("turn off all the lights")
p["room"] = "Den"
check(len(hamod.resolve(home, p)[0]) == 6, "all the lights: the whole home, wherever the user is")
for said, ok in (("hey Jarvis, turn off the den lights", True), ("lock the car and dim the ceiling light", True),
                 ("please start the vacuum", True), ("tell me a story about a lamp", False),
                 ("is the car locked?", False), ("set a timer for 10 minutes", False), ("stop", False),
                 ("play some jazz", False), ("turn on the jacuzzi", False)):
    check(hamod.confident(said, home) == ok, f"a plain device command? {said!r} -> {ok}")
mass = home + hamod.build([st("media_player.den_speaker", "idle", "Den Speaker", mass_player_type="player")],
                          [{"entity_id": "media_player.den_speaker", "area_id": "den"}], [],
                          [{"area_id": "den", "name": "Den"}])
p = hamod.parse("play some Miles Davis in the den")
check(p["action"] == "play_media" and p["music"] == "miles davis"
      and [d["id"] for d in hamod.resolve(mass, p)[0]] == ["media_player.den_speaker"]
      and not hamod.resolve(home, hamod.parse("play some jazz"))[0], f"music by name (Music Assistant only): {p}")
check(hamod.script_named("run movie night mode", home) and not hamod.script_named("play a movie", home),
      "a script by name isn't a film")
night = time.mktime(dt.datetime(2026, 10, 21, 23, 0).timetuple())  # heads-ups: once, and only what's news
seen, car = {}, [dict(byid["lock.car_door"], state="unlocked")]
first = hamod.alerts(car, seen, now=night)
later = hamod.alerts(car, seen, now=night + 700)
again = hamod.alerts(car, seen, now=night + 1400)
ns = {}  # (in the daytime, an unlocked door is nothing to mention)
noon = hamod.alerts(car, ns, now=night - 11 * 3600) + hamod.alerts(car, ns, now=night - 11 * 3600 + 700)
vac = [dict(byid["vacuum.robo"], state="cleaning")]
vs = {}
hamod.alerts(vac, vs, now=night)
done = hamod.alerts([dict(vac[0], state="returning")], vs, now=night + 1200)
docked = hamod.alerts([dict(vac[0], state="docked")], vs, now=night + 1300)
check(not first and later == ["Heads up: the Car Door Lock is still unlocked."] and not again and not noon
      and done == ["The Robo has finished cleaning."] and not docked, f"heads-ups: {first} {later} {again} {done} {docked}")
lt = Toolbox(specs=tools.SPECS + [tools.HA_SPEC])  # device commands for later, and the wrong tool for one
lt.timers = tools.Timers(path=Path(tempfile.mkdtemp()) / "timers.json")
hamod._cache.update(t=time.time(), devices=home)
lt.new_turn(lambda: "turn off the den lights in 20 minutes")
r = lt.run("home_assistant", {"command": "turn off the den lights"})
lt.new_turn(lambda: "warm up the car at 7 am")
r2 = lt.run("timer", {"action": "set", "when": "7 am", "message": "warm up the car"})
check(r[1].get("say") == "Okay, I'll turn off the den lights in 20 minutes."
      and re.search(r"warm up the car (at|tomorrow at) 7 AM\.", r2[1].get("say", ""))
      and [t.get("ha") for t in lt.timers.pending()] == ["turn off the den lights", "warm up the car"],
      f"for later: {r} {r2} {lt.timers.pending()}")
lt.new_turn(lambda: "unlock the car in 5 minutes")
check("rather not" in lt.run("home_assistant", {"command": "unlock the car"})[1].get("say", ""), "risky: never on a timer")
lt.new_turn(lambda: "turn on the reading lamp")
lt.run("web_search", {"query": "reading lamp on"})
check(sent[-1] == "turn on the reading lamp", f"a device command, not a search: {sent}")
# (the speculative-reply gate in Thinker.stream checks the tool that really runs: here a side effect)
check(lt.resolve("web_search", {"query": "reading lamp on"})[0] in tools.SIDE_EFFECTS
      and lt.resolve("calculate", {"expression": "2+2"}) == ("home_assistant", {"command": "turn on the reading lamp"})
      and Toolbox(specs=tools.SPECS).resolve("web_search", {"query": "x"}) == ("web_search", {"query": "x"}),
      "resolve: the tool that really runs")
# "what's the weather?" without a place: the client's home setting, else Home Assistant's home coordinates
got = []
saved_json, saved_home = tools._get_json, tools.ha_home
tools._get_json = lambda url, **kw: got.append(url) or {
    "results": [{"name": "Lyon", "country": "France", "country_code": "FR", "latitude": 45.7, "longitude": 4.8}],
    "current": {"time": "2026-01-01T12:00", "temperature_2m": 5, "apparent_temperature": 3, "relative_humidity_2m": 80,
                "weather_code": 3, "wind_speed_10m": 10, "is_day": 1},
    "daily": {"time": ["2026-01-01"], "weather_code": [3], "temperature_2m_max": [6], "temperature_2m_min": [1],
              "precipitation_probability_max": [20]}}
tools.ha_home = lambda: {"latitude": 1.5, "longitude": 2.5, "name": "Home", "imperial": True}
wb = Toolbox(specs=tools.SPECS)
w1 = wb.run("get_weather", {"location": "your location"})
wb.home = "Lyon"
w2 = wb.run("get_weather", {"location": "here"})
w3 = tools.get_weather("Lyon")
tools._get_json, tools.ha_home = saved_json, saved_home
check(w1[0].startswith("Weather for Home") and "°F" in w1[0] and "latitude=1.5" in got[0]
      and w2[0].startswith("Weather for Lyon, France") and "°C" in w2[0] and w2[0].startswith(w3[0])
      and tools._HERE.fullmatch("") and tools._HERE.fullmatch("the user's location")
      and not tools._HERE.fullmatch("Homer"), f"weather here: {w1[0][:40]!r} {w2[0][:40]!r} {got}")
eb = Toolbox(specs=tools.SPECS)  # ending the conversation only when the user said goodbye
for said, ok in (("okay thanks, bye", True), ("that's all for now", True), ("end the conversation", True),
                 ("goodnight ming", True), ("what's the weather tomorrow", False), ("stop", False)):
    eb.new_turn(lambda s=said: s)
    r = eb.run("end_conversation", {})
    check(bool(r[1].get("end")) == ok, f"end_conversation for {said!r}: {r}")
for say, want in (("Okay, turned off the lamp.", "Very good, sir. I've turned off the lamp."),  # the butler's words
                  ("The lamp is on. The fan is off.", "The lamp is on, sir. The fan is off."),
                  ("Goodbye, sir.", "Goodbye, sir."), ("Okay, the car is locked.", "The car is secured, sir.")):
    check(tools.in_character(say, "jarvis") == want and tools.in_character(say, "") == say,
          f"jarvis: {say!r} -> {tools.in_character(say, 'jarvis')!r}")
started = []  # Hermes only when the user says its name
tools._hermes = lambda method, path, body=None, timeout=10.0: (started.append(body) or {"run_id": "r1"}) \
    if method == "POST" and path == "/v1/runs" else {"status": "completed", "output": "ok"}
xb = Toolbox(specs=tools.SPECS + [tools.HERMES_SPEC])
for said, ok in (("hey ming what is the capital of france", False), ("check my email", False),
                 ("Ask Hermes how long its server has been up.", True), ("can you ask hermès to check the backups", True),
                 ("have Hermies look into it", True), ("what's the weather in Hermosa Beach", False)):
    xb.new_turn(lambda s=said: s)
    xb._last_delegation = 0
    r = xb.run("ask_hermes", {"task": "x"})
    check(bool(r[1].get("run_id")) == ok and (ok or r[1].get("refused")), f"Hermes {'used' if ok else 'refused'}: {said!r}")
check(len(started) == 3, "only the asked-for tasks reached Hermes")
os.environ["MING_HERMES_AUTO"] = "1"
xb.new_turn(lambda: "what's on my calendar")
xb._last_delegation = 0
check(bool(xb.run("ask_hermes", {"task": "x"})[1].get("run_id")), "MING_HERMES_AUTO=1: Hermes on its own again")
del os.environ["MING_HERMES_AUTO"]
played = []  # Jellyfin plays only when asked to
tools.jellyfin = lambda action, query="", device="": (played.append((action, query)) or "ok", {})
jb = Toolbox(specs=tools.SPECS + [tools.JELLYFIN_SPEC])
for said, action, ok in (("tell me a joke", "play", False), ("play the office", "play", True),
                         ("put on some jazz", "play", True), ("do I have the office", "search", True),
                         ("tell me a joke", "search", False)):
    jb.new_turn(lambda s=said: s)
    r = jb.run("jellyfin", {"action": action, "query": "x"})
    check(bool(r[1].get("refused")) != ok, f"Jellyfin {action} {'runs' if ok else 'refused'}: {said!r}")
check(len(played) == 3, "only the asked-for Jellyfin actions ran")
check(tb.run("rm_rf", {})[0].startswith("Error"), "unknown tool -> error text, no exception")
check(tb.run("calculate", {"nope": 1})[0].startswith("Error"), "bad arguments -> error text, no exception")
import tempfile  # noqa: E402  (lists and memory, in a scratch folder)
from pathlib import Path  # noqa: E402
_tmp = Path(tempfile.mkdtemp())
tools.LISTS_FILE, tools.MEMORY_FILE = _tmp / "lists.json", _tmp / "memory.json"
say = lambda *a, **k: tb.run(*a, **k)[1].get("say")
check(say("lists", {"action": "add", "items": ["milk", "eggs"]}) == "Added milk and eggs to your shopping list.",
      "lists: add to the shopping list (the default)")
check(say("lists", {"action": "add", "list": "groceries", "items": "bread, Eggs"}) ==
      "Added bread to your shopping list. Eggs was already on it.", "lists: groceries = shopping; no duplicates")
check(say("lists", {"action": "add", "list": "my To Do list", "items": ["call the plumber"]}) ==
      "Added call the plumber to your to-do list.", "lists: to-do list name")
check(say("lists", {"action": "remove", "items": ["egg", "cheese"]}) ==
      "Removed eggs from your shopping list. Cheese isn't on it.", "lists: remove (singular matches plural)")
text = lambda *a, **k: tb.run(*a, **k)[0]
check(text("lists", {"action": "read"}) == "The user's shopping list (2 items): milk; bread.", "lists: read")
check(text("lists", {"action": "names"}) == "The user has a shopping list with 2 items and a to-do list with 1 item.",
      "lists: which lists")
check(say("lists", {"action": "clear", "list": "todo"}) == "Your to-do list is cleared.", "lists: clear")
check(say("memory", {"action": "remember", "fact": "The user's sister's birthday is May 3."}) ==
      "Okay, I'll remember that.", "memory: remember")
sig = tb.signature()
check("sister's birthday is May 3." in tools.tool_guidance(), "memory: in the system prompt")
check(say("memory", {"action": "forget", "fact": "my sister's birthday"}) == "Okay, I've forgotten that." and
      not tools.memories() and tb.signature() != sig, "memory: forget (and the prompt changes)")

if "--offline" in sys.argv:
    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)

# --------------------------------------------------------------------------- live
import websockets  # noqa: E402

URLISH = re.compile(r"https?://|www\.|\.com\b|\[\d+\]")


async def ask(ws, text, timeout=60):
    """Send a typed message (None: just wait for the next reply), collect the reply: text, tool events, audio."""
    if text is not None:
        await ws.send(json.dumps({"type": "text", "text": text}))
    out = {"text": "", "tools": [], "audio_s": 0.0, "t0": time.perf_counter(), "t_first_audio": None}
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            m = await asyncio.wait_for(ws.recv(), timeout=deadline - time.perf_counter())
        except asyncio.TimeoutError:
            break
        if isinstance(m, bytes):
            out["audio_s"] += len(m) / 2 / 44100
            out["t_first_audio"] = out["t_first_audio"] or time.perf_counter()
            continue
        ev = json.loads(m)
        if ev["type"] == "text":
            out["text"] += ev["delta"]
        elif ev["type"] == "tool" and ev["status"] == "done":
            out["tools"].append(ev)
        elif ev["type"] == "done":
            break
    return out


async def main():
    offered = Toolbox().names  # (Home Assistant / Hermes only when configured on this machine)
    async with websockets.connect(f"ws://{URL}/ws", max_size=2**24) as ws:
        json.loads(await ws.recv())  # hello
        await ws.send(json.dumps({"type": "config", "tools": True, "home": "Chicago, IL", "voice": "US_Stella"}))
        await ws.send(json.dumps({"type": "reset"}))
        await asyncio.sleep(1.0)
        cases = [  # (question, tool expected or None, words the answer should contain (any))
            ("What is the capital of France?", None, ["paris"]),
            ("What will the weather be like tomorrow?", "get_weather", ["tomorrow", "°", "degree"]),  # home: Chicago
            ("What's the weather like in Chicago right now?", "get_weather", ["°", "degree", "chicago"]),
            ("What is 1234 times 5678?", "calculate", ["7006652", "7,006,652", "seven million"]),
            ("What's the latest news about NASA?", "web_search", []),
            ("Tell me a short joke.", None, []),
            ("What time is it?", None, [time.strftime("%I:%M", time.localtime(t)).lstrip("0")  # (from the time note)
                                        for t in (time.time(), time.time() + 60)]),
            ("What time is it in Tokyo right now?", "get_current_time", []),
            ("Set a timer for 10 minutes.", "timer", ["10 minutes", "ten minutes"]),
            ("Actually, cancel that timer.", "timer", ["cancel"]),
        ]
        n_general = len(cases)
        if "home_assistant" in offered:  # read-only: nothing gets switched
            cases.append(("Is the kitchen light on?", "home_assistant", ["kitchen", "off", " on"]))
        if "jellyfin" in offered:  # read-only: a library search
            cases.append(("Do I have The Office on Jellyfin?", "jellyfin", ["office"]))
        if "--hermes" in sys.argv:  # only the delegation check
            cases = []
        if "--home" in sys.argv:  # only the Home Assistant / Jellyfin checks (then Hermes)
            cases = cases[n_general:]
        for q, want_tool, words in cases:
            r = await ask(ws, q)
            names = [t["name"] for t in r["tools"]]
            ans = r["text"].strip()
            first = (r["t_first_audio"] - r["t0"]) * 1000 if r["t_first_audio"] else None
            print(f"\nQ: {q}\n   tools: {[(t['name'], t.get('args'), t.get('elapsed_ms')) for t in r['tools']]}\n"
                  f"   A: {ans!r}\n   audio {r['audio_s']:.1f} s, first audio after {first and round(first)} ms")
            if want_tool:
                check(want_tool in names, f"{q!r} calls {want_tool}")
            else:
                check(not names, f"{q!r} needs no tool")
            if words:
                check(any(w in ans.lower() for w in words), f"{q!r} answer mentions one of {words}")
            check(r["audio_s"] > 0.5, f"{q!r} was spoken")
            check(not URLISH.search(ans), f"{q!r} answer has no URLs or citation numbers")
        if "--home" not in sys.argv and "--hermes" not in sys.argv:  # a timer that really goes off
            r = await ask(ws, "Remind me to stretch in 5 seconds.")
            print(f"\nQ: 5 s reminder\n   tools: {[(t['name'], t.get('args')) for t in r['tools']]}\n   A: {r['text'].strip()!r}")
            check("timer" in [t["name"] for t in r["tools"]], "the reminder calls timer")
            t0 = time.perf_counter()
            r = await ask(ws, None, timeout=30)
            names = [t["name"] for t in r["tools"]]
            print(f"   announced after {time.perf_counter() - t0:.0f} s: {r['text'].strip()!r} {names}")
            check("timer_done" in names and "stretch" in r["text"].lower() and r["audio_s"] > 0.5,
                  "the reminder goes off and is spoken")
        if "ask_hermes" in offered and "--hermes" in sys.argv:  # a real (harmless) Hermes task, only when asked for
            await asyncio.sleep(1.0)
            r = await ask(ws, "Can you ask Hermes how long its server has been up?")
            names = [t["name"] for t in r["tools"]]
            print(f"\nQ: delegate to Hermes\n   tools: {names}\n   A: {r['text'].strip()!r}")
            check("ask_hermes" in names, "delegation calls ask_hermes")
            t0 = time.perf_counter()
            r = await ask(ws, None, timeout=240)
            names = [t["name"] for t in r["tools"]]
            print(f"   announced after {time.perf_counter() - t0:.0f} s: {r['text'].strip()!r} {names}")
            check("hermes_result" in names and r["audio_s"] > 0.5, "Hermes result announced and spoken")
            check(any(w in r["text"].lower() for w in ("day", "hour", "minute", "week")), "announcement has the uptime")
    print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")
    sys.exit(1 if fails else 0)


asyncio.run(main())
