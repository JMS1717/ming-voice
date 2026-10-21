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
import time

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
                  ("pasta", ("pasta", "Reminder: pasta.")), ("", ("", "")), ("timer for 5 minutes", ("", ""))):
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
tools.home_assistant = lambda command: (sent.append(command) or "Done.", {"targets": [command]})
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
        await ws.send(json.dumps({"type": "config", "tools": True, "home": "Seattle, WA", "voice": "US_Stella"}))
        await ws.send(json.dumps({"type": "reset"}))
        await asyncio.sleep(1.0)
        cases = [  # (question, tool expected or None, words the answer should contain (any))
            ("What is the capital of France?", None, ["paris"]),
            ("What will the weather be like tomorrow?", "get_weather", ["tomorrow", "°", "degree"]),  # home: Seattle
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
