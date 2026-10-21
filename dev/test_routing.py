"""Tool choice for spoken questions, against the running server: general knowledge and small talk use no tool (also
after "Hey Ming"), the clock only when asked the time, and Hermes never unless the user names it. The questions are
spoken by the server's own TTS and sent push-to-talk style, all in one conversation (earlier turns can prime it).

    .venv/bin/python dev/test_routing.py
"""
import asyncio
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import soundfile as sf
import torch
import torchaudio
import websockets

URL = os.environ.get("MING_URL", "localhost:8000")
ANY = "*"  # any tool (or none) is fine, as long as it isn't one of the unwanted ones
CASES = [  # (spoken question, tools wanted (empty: none), tools never wanted)
    ("Hey Ming, what is the capital of France?", [], []),
    ("And what about Germany?", [], []),
    ("Hey Ming, how far away is the moon?", [], []),
    ("Hey Ming, good morning!", [], []),
    ("Tell me a fun fact about octopuses.", [], []),
    ("Who wrote Pride and Prejudice?", [], []),
    ("Hey Ming, how do I boil an egg?", [], []),
    ("What's today's date?", ANY, ["ask_hermes"]),
    ("Hey Ming, what time is it?", ANY, ["ask_hermes"]),  # (answered from the time note: checked below)
    ("Tell me a joke.", [], []),
    ("What's on my calendar tomorrow?", ANY, ["ask_hermes", "get_current_time"]),
    ("Can you check my email?", ANY, ["ask_hermes", "get_current_time"]),
    ("Hey Ming, research the best electric cars this year and compare them.", ["web_search"],
     ["ask_hermes", "get_current_time"]),
    ("How much disk space is left on my server?", ANY, ["ask_hermes", "get_current_time"]),
    ("Hey Ming, what's the weather like tomorrow?", ["get_weather"], ["ask_hermes", "get_current_time"]),
    ("What is two hundred forty times thirty seven?", ANY, ["ask_hermes", "get_current_time", "web_search"]),
    ("Thanks, that's all.", [], []),
]


def speech(text, voice="US_James"):
    wav = urllib.request.urlopen(f"http://{URL}/api/tts?" + urllib.parse.urlencode({"text": text, "voice": voice})).read()
    a, sr = sf.read(io.BytesIO(wav), dtype="float32")
    return torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()


async def ask(ws, audio, timeout=60):
    await ws.send(json.dumps({"type": "ptt", "state": "down"}))
    pcm = (np.clip(np.pad(audio, (0, 4000)), -1, 1) * 32767).astype("<i2")
    for i in range(0, len(pcm), 512):
        await ws.send(pcm[i:i + 512].tobytes())
    await ws.send(json.dumps({"type": "ptt", "state": "up"}))
    out, t0 = {"text": "", "tools": [], "first_audio": None}, time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            m = await asyncio.wait_for(ws.recv(), timeout=timeout)
        except asyncio.TimeoutError:
            break
        if isinstance(m, bytes):
            out["first_audio"] = out["first_audio"] or round((time.perf_counter() - t0) * 1000)
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
    fails = []
    audio = [speech(q) for q, _, _ in CASES]  # (before connecting: TTS while the session waits would be slow)
    async with websockets.connect(f"ws://{URL}/ws", max_size=2**24) as ws:
        json.loads(await ws.recv())  # hello
        await ws.send(json.dumps({"type": "config", "mode": "ptt", "tools": True, "home": "Chicago, IL",
                                  "voice": "US_Stella"}))
        await ws.send(json.dumps({"type": "reset"}))
        await asyncio.sleep(1.0)
        for (q, want, never), a in zip(CASES, audio):
            r = await ask(ws, a)
            names = [t["name"] for t in r["tools"] if not t.get("refused")]  # (a refused call did nothing)
            ok = (want == ANY or set(want) <= set(names)) and not set(names) & set(never) \
                and (want or want == ANY or not names)
            if "time is it" in q:  # the right time (this minute or the one before)
                now = time.time()
                ok = ok and any(time.strftime("%I:%M", time.localtime(t)).lstrip("0") in r["text"]
                                for t in (now, now - 60))
            if "date" in q:  # the date, not the time
                ok = ok and time.strftime("%B") in r["text"]
            refused = [t["name"] for t in r["tools"] if t.get("refused")]
            calls = [(t["name"], t.get("args")) for t in r["tools"] if not t.get("refused")]
            print(f"{'PASS' if ok else 'FAIL'} {q}\n     tools {calls}{f' + refused {refused}' if refused else ''}, "
                  f"first audio {r['first_audio']} ms: {r['text'].strip()[:140]!r}")
            if not ok:
                fails.append(q)
            await asyncio.sleep(0.5)
    print("\nALL PASS" if not fails else f"\n{len(fails)}/{len(CASES)} FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    asyncio.run(main())
