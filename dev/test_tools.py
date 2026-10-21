"""Tool calling: offline unit checks, then (if a server is running) typed conversations over the WebSocket.

    .venv/bin/python dev/test_tools.py            # unit checks + live checks against localhost:8000
    .venv/bin/python dev/test_tools.py --offline  # unit checks only (no server, no network)

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
tb = Toolbox()
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
            ("What time is it?", "get_current_time", []),
        ]
        if "home_assistant" in offered:  # read-only: nothing gets switched
            cases.append(("Is the kitchen light on?", "home_assistant", ["kitchen", "off", " on"]))
        if "--hermes" in sys.argv:  # only the delegation check
            cases = []
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
        if "ask_hermes" in offered:  # a real (harmless) task for the Hermes agent; its answer arrives on its own
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
