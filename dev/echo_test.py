"""Speakerphone echo: does the assistant interrupt itself when its own voice leaks back into the microphone?

Asks a question that gets a long answer, then streams a "microphone" made of the reply audio itself (resampled to
16 kHz, delayed, attenuated, smeared by a short synthetic room response) plus a little noise. A real browser's
echo canceller removes most of this; gains here model what is left. Then checks that a real interruption mixed
over the echo still barges in.

    ECHO_GAINS=0.05,0.15,0.4 python dev/echo_test.py
"""
import asyncio, json, os, time
import numpy as np, soundfile as sf, torch, torchaudio, websockets

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = os.environ.get("URL", "ws://localhost:8000/ws")
FRAME = 512
rng = np.random.default_rng(0)


def load(name):
    w, sr = sf.read(os.path.join(HERE, "e2e", "questions", name + ".wav"), dtype="float32")
    return w


_IR = rng.standard_normal(int(0.08 * 16000)) * np.exp(-np.arange(int(0.08 * 16000)) / (0.02 * 16000))
_IR[0] = 3.0
_IR /= np.abs(_IR).sum() / 3
DELAY = int(float(os.environ.get("ECHO_DELAY", "0.12")) * 16000)  # acoustic + capture path, beyond playback


def echo_at(timeline, t, n):
    """samples t..t+n of the microphone echo (fixed room response, delayed) of the playback timeline"""
    a = max(0, t - DELAY - len(_IR))
    x = timeline[a:max(a, t + n - DELAY)]
    if len(x) == 0:
        return np.zeros(n, np.float32)
    y = np.convolve(x, _IR)[: len(x)]
    out = y[t - DELAY - a:] if t - DELAY >= a else np.concatenate([np.zeros(a - (t - DELAY)), y])
    return np.pad(out, (0, max(0, n - len(out))))[:n].astype(np.float32)


async def run(gain, interrupt_at=None):
    async with websockets.connect(URL, max_size=2**24) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "config", "barge_in": True, "spec_ms": 32, "smart_turn": True,
                                  "echo_guard": os.environ.get("ECHO_GUARD", "1") == "1"}))
        await ws.send(json.dumps({"type": "reset"}))
        # playback timeline like the web client: audio is queued after what is playing and dropped on interrupt
        timeline = np.zeros(0, np.float32)
        pos = 0  # samples played so far (= samples of microphone sent)
        received = 0
        events = {"interrupt": 0, "user_end": 0, "done": 0}
        t_audio = None

        async def recv():
            nonlocal timeline, received, t_audio
            async for m in ws:
                if isinstance(m, bytes):
                    x = np.frombuffer(m, "<i2").astype(np.float32) / 32768
                    y = torchaudio.functional.resample(torch.from_numpy(x), 44100, 16000).numpy()
                    at = max(len(timeline), pos)
                    timeline = np.concatenate([timeline, np.zeros(at - len(timeline), np.float32), y])
                    received += len(y)
                    if t_audio is None:
                        t_audio = time.perf_counter()
                else:
                    d = json.loads(m)
                    if d["type"] in events:
                        events[d["type"]] += 1
                    if d["type"] == "interrupt":
                        timeline = timeline[:pos]
        rt = asyncio.create_task(recv())
        question = load("story")
        intr = load("interrupt") if interrupt_at is not None else None
        intr_pos = None
        t_start = time.perf_counter()
        deadline = t_start + len(question) / 16000 + 15
        n = 0
        while time.perf_counter() < deadline:
            frame = question[pos:pos + FRAME]
            frame = np.pad(frame, (0, FRAME - len(frame)))
            echo = echo_at(timeline, pos, FRAME)
            frame = frame + gain * echo + rng.standard_normal(FRAME).astype(np.float32) * 3e-4
            if intr is not None and t_audio is not None and time.perf_counter() - t_audio > interrupt_at:
                intr_pos = 0 if intr_pos is None else intr_pos
                seg = intr[intr_pos:intr_pos + FRAME]
                frame[:len(seg)] += seg
                intr_pos += FRAME
            await ws.send((np.clip(frame, -1, 1) * 32767).astype("<i2").tobytes())
            pos += FRAME
            n += 1
            await asyncio.sleep(max(0, t_start + n * FRAME / 16000 - time.perf_counter()))
        rt.cancel()
        return events, received / 16000


async def main():
    gains = [float(g) for g in os.environ.get("ECHO_GAINS", "0.05,0.15,0.4").split(",")]
    for g in gains:
        ev, secs = await run(g)
        print(f"echo gain {g:.2f}: interrupts {ev['interrupt'] - 1} (should be 0), extra turns {ev['user_end'] - 1}, "
              f"reply audio {secs:.1f} s", flush=True)
    ev, secs = await run(gains[len(gains) // 2], interrupt_at=3.0)
    print(f"echo gain {gains[len(gains) // 2]:.2f} + real interruption at 3 s: interrupts {ev['interrupt'] - 1} (should be >= 1)")

asyncio.run(main())
