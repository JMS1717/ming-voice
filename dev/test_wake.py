"""Wake-word mode against the running server: speech without the wake word is ignored, "Hey Ming, ..." is answered,
follow-ups inside the window need no wake word, it goes back to sleep afterwards, and "Hey Ming" on its own wakes it
(chime) so the next sentence is answered. The "microphone" is the server's own TTS, streamed in real time.

    .venv/bin/python dev/test_wake.py
"""
import asyncio
import io
import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import soundfile as sf
import torch
import torchaudio
import websockets

URL = os.environ.get("MING_URL", "localhost:8000")
FRAME = 512
FOLLOW_UP_S = 4.0
fails = []


def check(ok, what):
    print(("PASS " if ok else "FAIL ") + what)
    if not ok:
        fails.append(what)


def speech(text, voice="US_James"):
    wav = urllib.request.urlopen(f"http://{URL}/api/tts?" + urllib.parse.urlencode({"text": text, "voice": voice})).read()
    a, sr = sf.read(io.BytesIO(wav), dtype="float32")
    return torchaudio.functional.resample(torch.from_numpy(a), sr, 16000).numpy()


async def main():
    async with websockets.connect(f"ws://{URL}/ws", max_size=2**24) as ws:
        json.loads(await ws.recv())  # hello
        await ws.send(json.dumps({"type": "config", "mode": "wake", "wake_word": "Ming", "follow_up_s": FOLLOW_UP_S,
                                  "voice": "US_Stella"}))
        await ws.send(json.dumps({"type": "reset"}))
        events, queue_ = [], []
        rng = np.random.default_rng(0)

        async def mic():  # real-time frames: queued speech, else near-silence
            t0, n, cur, idx = time.perf_counter(), 0, None, 0
            while True:
                if cur is None and queue_:
                    cur, idx = queue_.pop(0), 0
                if cur is not None:
                    frame = cur[idx:idx + FRAME]
                    idx += FRAME
                    frame = np.pad(frame, (0, FRAME - len(frame)))
                    if idx >= len(cur):
                        cur = None
                        events.append(("spoke", time.perf_counter()))
                else:
                    frame = (rng.standard_normal(FRAME) * 1e-4).astype(np.float32)
                await ws.send((np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes())
                n += 1
                await asyncio.sleep(max(0, t0 + n * FRAME / 16000 - time.perf_counter()))

        async def recv():
            while True:
                m = await ws.recv()
                if isinstance(m, bytes):
                    events.append(("audio", time.perf_counter()))
                else:
                    ev = json.loads(m)
                    events.append((ev["type"], ev))

        tasks = [asyncio.create_task(mic()), asyncio.create_task(recv())]
        await asyncio.sleep(1.5)

        async def say(text, wait=8.0):
            """Speak; return (answered?, reply text, ms from the end of speech to the first audio, events)."""
            del events[:]
            queue_.append(speech(text))
            t_end = time.perf_counter() + 60
            while time.perf_counter() < t_end:
                await asyncio.sleep(0.1)
                spoke = next((t for k, t in events if k == "spoke"), None)
                done = any(k == "done" for k, _ in events)
                if done or (spoke and time.perf_counter() - spoke > wait and not any(k == "user_end" for k, _ in events)):
                    break
            await asyncio.sleep(0.3)
            spoke = next((t for k, t in events if k == "spoke"), None)
            first = next((t for k, t in events if k == "audio"), None)
            text_ = "".join(e.get("delta", "") for k, e in events if k == "text")
            lat = round((first - spoke) * 1000) if first and spoke else None
            return any(k == "user_end" for k, _ in events), text_, lat, [k if k != "wake" else f"wake:{e['state']}"
                                                                          for k, e in events if k not in ("audio", "text")]

        async def wait_playback(ev_text):  # until its audio has played (the reply's ~audio_s) + a little
            await asyncio.sleep(max(1.0, len(ev_text) / 15))

        ans, txt, lat, ev = await say("What is the capital of France?", wait=4)
        print(f"asleep, no wake word: answered={ans} {ev}")
        check(not ans, "no wake word while asleep -> ignored")

        ans, txt, lat, ev = await say("Hey Ming, what is the capital of France?")
        print(f"wake word + question: {txt!r} after {lat} ms {ev}")
        check(ans and "paris" in txt.lower(), "'Hey Ming, ...' is answered")
        await wait_playback(txt)

        ans, txt, lat, ev = await say("And what about Germany?")
        print(f"follow-up in the window: {txt!r} after {lat} ms {ev}")
        check(ans and "berlin" in txt.lower(), "follow-up within the window needs no wake word")
        await wait_playback(txt)

        await asyncio.sleep(FOLLOW_UP_S + 1.0)
        check(any(k == "wake" and e["state"] == "asleep" for k, e in events), "back to sleep after the window")
        ans, txt, lat, ev = await say("Tell me a short joke.", wait=4)
        print(f"after the window, no wake word: answered={ans} {ev}")
        check(not ans, "after the window the wake word is needed again")

        ans, txt, lat, ev = await say("Hey Ming.", wait=2.5)
        print(f"wake word alone: answered={ans} {ev}")
        check(not ans and "wake:listening" in ev, "'Hey Ming' alone -> chime, no reply")
        ans, txt, lat, ev = await say("What is two plus two?")
        print(f"then a question: {txt!r} after {lat} ms {ev}")
        check(ans and ("4" in txt or "four" in txt.lower()), "the question after the chime is answered")
        for t in tasks:
            t.cancel()
        await ws.send(json.dumps({"type": "config", "mode": "vad"}))
    print("\nALL PASS" if not fails else f"\n{len(fails)} FAILED: {fails}")


asyncio.run(main())
