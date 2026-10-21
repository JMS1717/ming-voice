"""Edge cases against the running server: very long utterance, Chinese speech, noise only, rapid barge-ins."""
import asyncio, json, os, time
import numpy as np, soundfile as sf, torch, torchaudio, websockets

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(HERE, "e2e", "questions")


def load(path):
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(1)
    if sr != 16000:
        w = torchaudio.functional.resample(torch.from_numpy(w), sr, 16000).numpy()
    return w


async def scenario(name, chunks, wait=25.0, config=None):
    """chunks: list of (audio, gap_seconds_after). Streams in real time; collects events."""
    async with websockets.connect("ws://localhost:8000/ws", max_size=2**24) as ws:
        await ws.recv()
        if config:
            await ws.send(json.dumps({"type": "config", **config}))
        await ws.send(json.dumps({"type": "reset"}))
        ev = {"user_end": 0, "interrupt": 0, "done": [], "error": [], "audio_s": 0.0}

        async def recv():
            async for m in ws:
                if isinstance(m, bytes):
                    ev["audio_s"] += len(m) / 2 / 44100
                    continue
                d = json.loads(m)
                if d["type"] == "user_end":
                    ev["user_end"] += 1
                elif d["type"] == "interrupt":
                    ev["interrupt"] += 1
                elif d["type"] == "done":
                    ev["done"].append(d["text"][:80])
                elif d["type"] == "error":
                    ev["error"].append(d["message"])
        rt = asyncio.create_task(recv())
        rng = np.random.default_rng(0)
        t0, n = time.perf_counter(), 0
        stream = []
        for audio, gap in chunks:
            stream.append(audio)
            stream.append((rng.standard_normal(int(gap * 16000)) * 1e-4).astype(np.float32))
        stream.append((rng.standard_normal(int(wait * 16000)) * 1e-4).astype(np.float32))
        x = np.concatenate(stream)
        for i in range(0, len(x) - 511, 512):
            await ws.send((np.clip(x[i:i + 512], -1, 1) * 32767).astype("<i2").tobytes())
            n += 1
            await asyncio.sleep(max(0, t0 + n * 512 / 16000 - time.perf_counter()))
        rt.cancel()
    print(f"{name:18s} turns={ev['user_end']} interrupts={ev['interrupt']} audio={ev['audio_s']:.1f}s errors={ev['error']}\n"
          f"{'':18s} replies={ev['done']}", flush=True)


async def main():
    story, capital, sleep_q, interrupt = (load(os.path.join(Q, f + ".wav")) for f in ("story", "capital", "sleep", "interrupt"))
    zh = load(os.path.join(HERE, "samples", "introduce_ant_group.wav"))
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(16000 * 3) * 0.02).astype(np.float32)
    def trim(w, keep=0.12):  # drop the clips' leading/trailing silence so they join into continuous speech
        loud = np.flatnonzero(np.abs(w) > 0.02)
        return w[max(0, loud[0] - int(keep * 16000)): loud[-1] + int(keep * 16000)]
    long_q = np.concatenate([np.concatenate([trim(capital), trim(sleep_q), trim(story)]) for _ in range(3)] + [trim(capital)])
    only = os.environ.get("ONLY")  # long | chinese | noise | barge
    if only in (None, "long"):
        # one 55 s turn: the clips are complete questions, which Smart Turn would (rightly) answer one by one
        await scenario("long utterance", [(long_q, 0)], wait=20, config={"smart_turn": False})
    if only in (None, "chinese"):
        await scenario("chinese", [(zh, 0)], wait=15)
    if only in (None, "noise"):
        await scenario("noise only", [(noise, 0)], wait=6)
    if only in (None, "barge"):
        await scenario("double barge-in", [(story, 3.0), (interrupt, 1.5), (capital, 0)], wait=15)

asyncio.run(main())
