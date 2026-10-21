"""End-to-end test: acts like a browser microphone against the running server.

Streams spoken questions over the WebSocket in real time (32 ms frames, silence in between),
records the spoken replies, measures latency from the end of the user's speech to the first
audio byte, transcribes each reply via /api/asr, and checks barge-in and speculative rollback.
"""
import asyncio, io, json, os, sys, time, urllib.request
import numpy as np, soundfile as sf, websockets

URL = os.environ.get("MING_URL", "localhost:8000")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QDIR = os.path.join(HERE, "e2e", "questions")
FRAME = 512


def load(name):
    w, sr = sf.read(os.path.join(QDIR, name + ".wav"), dtype="float32")
    assert sr == 16000
    return w


def asr(pcm44: np.ndarray) -> str:
    buf = io.BytesIO()
    sf.write(buf, pcm44, 44100, format="WAV")
    req = urllib.request.Request(f"http://{URL}/api/asr", data=buf.getvalue(), method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=120).read())["text"]


class Mic:
    """Sends queued audio in real time, then near-silence, like a live microphone."""

    def __init__(self, ws):
        self.ws, self.q, self.t_last_speech = ws, [], None
        self.rng = np.random.default_rng(0)

    def say(self, audio):
        self.q.append(audio)

    @property
    def busy(self):
        return bool(self.q)

    async def run(self):
        t0, n = time.perf_counter(), 0
        cur, idx = None, 0
        while True:
            if cur is None and self.q:
                cur, idx = self.q.pop(0), 0
            if cur is not None:
                frame = cur[idx:idx + FRAME]
                idx += FRAME
                if len(frame) < FRAME:
                    frame = np.pad(frame, (0, FRAME - len(frame)))
                if idx >= len(cur):
                    cur = None
                    self.t_last_speech = time.perf_counter()
            else:
                frame = (self.rng.standard_normal(FRAME) * 1e-4).astype(np.float32)
            await self.ws.send((np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes())
            n += 1
            await asyncio.sleep(max(0, t0 + n * FRAME / 16000 - time.perf_counter()))


async def main():
    results = []
    retracts = [0]
    async with websockets.connect(f"ws://{URL}/ws", max_size=2**24) as ws:
        hello = json.loads(await ws.recv())
        await ws.send(json.dumps({"type": "config", "voice": "US_Stella", "end_ms": 500, "spec_ms": int(os.environ.get("E2E_SPEC_MS", "32")),
                                  "smart_turn": os.environ.get("E2E_SMART", "1") == "1",
                                  **({"fast_end_ms": int(os.environ["E2E_FAST_END"])} if "E2E_FAST_END" in os.environ else {}),
                                  "barge_in": True}))
        if os.environ.get("E2E_RESET", "1") == "1":
            await ws.send(json.dumps({"type": "reset"}))
        mic = Mic(ws)
        events: asyncio.Queue = asyncio.Queue()

        async def recv():
            async for m in ws:
                events.put_nowait((time.perf_counter(), m))
        tasks = [asyncio.create_task(mic.run()), asyncio.create_task(recv())]

        async def turn(name, audio, interrupt_with=None, interrupt_after=1.5, clear=True, settle=0.0):
            while clear and not events.empty():
                events.get_nowait()
            audio_in = audio
            mic.say(audio)
            while mic.busy:
                await asyncio.sleep(0.01)
            # wait until the whole clip has been sent
            t_start_wait = time.perf_counter()
            while mic.t_last_speech is None or mic.t_last_speech < t_start_wait - 0.05:
                await asyncio.sleep(0.005)
            t_end = mic.t_last_speech
            pcm, text, first_audio, done, interrupted, saw_interrupt = [], "", None, None, False, False
            sent_interrupt = False
            multi = interrupt_with is not None  # several utterances: no single end of speech to time from
            pause_ms = None
            deadline = time.perf_counter() + 90
            while time.perf_counter() < deadline:
                # talk over the reply once it has *played* for interrupt_after seconds (audio arrives faster)
                if (interrupt_with is not None and not sent_interrupt and first_audio is not None
                        and time.perf_counter() - first_audio >= interrupt_after):
                    sent_interrupt = True
                    mic.say(interrupt_with)
                try:
                    t, m = await asyncio.wait_for(events.get(), timeout=0.02)
                except asyncio.TimeoutError:
                    continue
                if isinstance(m, bytes):
                    if first_audio is None:
                        first_audio = t
                    pcm.append(np.frombuffer(m, "<i2").astype(np.float32) / 32768)
                    continue
                ev = json.loads(m)
                if ev["type"] == "text":
                    text += ev["delta"]
                elif ev["type"] == "interrupt":
                    saw_interrupt = True
                elif ev["type"] == "retract":  # it answered during a pause and the user carried on: drop it
                    retracts[0] += 1
                    pcm, text, first_audio, multi = [], "", None, True
                elif ev["type"] == "user_end" and pause_ms is None:
                    pause_ms = ev.get("pause_ms")
                elif ev["type"] == "done":
                    done = ev
                    if settle > 0:  # keep collecting: the turn may have been split into several replies
                        deadline = min(deadline, time.perf_counter() + settle)
                        settle = 0.0
                        continue
                    if interrupt_with is None or ev.get("interrupted"):
                        break
            audio = np.concatenate(pcm) if pcm else np.zeros(1, np.float32)
            os.makedirs(os.path.join(HERE, "e2e", "out"), exist_ok=True)
            sf.write(os.path.join(HERE, "e2e", "out", f"{name}.wav"), audio, 44100)
            lat = None if first_audio is None else round((first_audio - t_end) * 1000)
            # from the end of the actual speech (clips carry ~0.4 s of trailing silence) to the first audio
            loud = np.flatnonzero(np.abs(audio_in) > 0.01)
            trail = (len(audio_in) - loud[-1]) / 16000 if len(loud) else 0.0
            speech_lat = None if first_audio is None or multi else round((first_audio - t_end + trail) * 1000)
            heard = (await asyncio.get_running_loop().run_in_executor(None, asr, audio)) if len(audio) > 4410 else ""
            fa = ((done or {}).get("metrics") or {}).get("first_audio_sent")
            # what a listener perceives: silence until the server ended the turn (VAD-measured) + time to audio
            perceived = None if pause_ms is None or fa is None else pause_ms + max(0, fa)
            r = dict(name=name, latency_ms=lat, speech_end_ms=speech_lat, perceived_ms=perceived, pause_ms=pause_ms, reply_text=text.strip(), heard=heard,
                     audio_s=round(len(audio) / 44100, 2), metrics=(done or {}).get("metrics"),
                     interrupted=bool(done and done.get("interrupted")), saw_interrupt=saw_interrupt)
            results.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            if not saw_interrupt:  # let playback finish before the next question
                await asyncio.sleep(max(0.5, len(audio) / 44100 - (time.perf_counter() - (first_audio or t_end)) + 0.3))
            return r

        await asyncio.sleep(1.0)
        await turn("capital", load("capital"))
        await turn("followup", load("followup"))
        await turn("pause", load("pause"), settle=6.0)
        await turn("sleep", load("sleep"))
        r = await turn("story_bargein", load("story"), interrupt_with=load("interrupt"))
        # the barge-in question's own reply follows the interrupted one
        await turn("after_bargein", np.zeros(1600, np.float32), clear=False)
        for t in tasks:
            t.cancel()

    print("\n=== summary ===")
    lats = [r["latency_ms"] for r in results if r["latency_ms"] is not None and r["name"] != "after_bargein"]
    print("clip end -> first audio byte (ms):", lats, " median", int(np.median(lats)) if lats else None)
    per = [r["perceived_ms"] for r in results if r["perceived_ms"] is not None and r["name"] != "after_bargein"]
    print("perceived: silence after speech -> first audio (ms):", per, " median", int(np.median(per)) if per else None)
    sl = [r["speech_end_ms"] for r in results if r["speech_end_ms"] is not None and r["name"] != "after_bargein"]
    print("end of speech -> first audio byte (ms):", sl, " median", int(np.median(sl)) if sl else None)
    print("withdrawn replies (resume after a pause):", retracts[0])
    checks = {
        "every turn produced audio": all(r["audio_s"] > 0.3 for r in results[:5]),
        "capital answer mentions Paris": "paris" in (results[0]["reply_text"] + results[0]["heard"]).lower(),
        "follow-up used context (mentions million)": "million" in (results[1]["reply_text"] + results[1]["heard"]).lower(),
        # the clip's hesitation sits right at the end-of-turn threshold: it may split into two turns
        "pause question answered (Everest)": "everest" in (results[2]["reply_text"] + results[2]["heard"] +
                                                           results[3]["reply_text"] + results[3]["heard"]).lower(),
        "barge-in interrupted the story": results[4]["saw_interrupt"],
        # the reply to the interrupting question may be collected into the story turn or the next one
        "barge-in question answered (four)": any(w in (results[4]["reply_text"] + results[4]["heard"] +
                                                       results[5]["reply_text"] + results[5]["heard"]).lower()
                                                 for w in ("four", " 4")),
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    sys.exit(0 if all(checks.values()) else 1)


asyncio.run(main())
