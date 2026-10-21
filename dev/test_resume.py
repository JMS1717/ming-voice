"""Resume after a pause: the user stops long enough for the reply to start, then keeps talking.

The reply must be withdrawn ("retract"), the two parts must be answered as one turn, and the conversation must
not keep the withdrawn exchange. Also checks that a late interruption (after resume_ms) is still a plain
interruption.

    python dev/test_resume.py            # against ws://localhost:8000/ws
"""
import asyncio, json, os, time
import numpy as np, soundfile as sf, websockets

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = os.environ.get("URL", "ws://localhost:8000/ws")
FRAME = 512


def load(name):
    w, sr = sf.read(os.path.join(HERE, "e2e", "questions", name + ".wav"), dtype="float32")
    assert sr == 16000
    return w


def trim(w, th=0.01):
    idx = np.flatnonzero(np.abs(w) > th)
    return w[max(0, idx[0] - 1600): idx[-1] + 1600]


async def converse(parts, cfg, listen_s=12):
    """parts: list of (audio, gap_after_s). Streams them in real time; returns the events."""
    async with websockets.connect(URL, max_size=2**24) as ws:
        await ws.recv()
        await ws.send(json.dumps({"type": "config", "barge_in": True, "spec_ms": 64, "smart_turn": True, **cfg}))
        await ws.send(json.dumps({"type": "reset"}))
        await asyncio.sleep(0.3)
        ev = []
        t0 = time.perf_counter()

        async def recv():
            async for m in ws:
                if isinstance(m, str):
                    d = json.loads(m)
                    if d["type"] in ("user_end", "retract", "interrupt", "done", "transcript"):
                        ev.append((round(time.perf_counter() - t0, 2), d))
        rt = asyncio.create_task(recv())
        stream = np.concatenate([np.concatenate([a, np.zeros(int(g * 16000), np.float32)]) for a, g in parts])
        stream = np.concatenate([stream, np.zeros(int(listen_s * 16000), np.float32)])
        for n, i in enumerate(range(0, len(stream) - FRAME + 1, FRAME)):
            await ws.send((stream[i:i + FRAME] * 32767).astype("<i2").tobytes())
            await asyncio.sleep(max(0, t0 + (n + 1) * FRAME / 16000 - time.perf_counter()))
        rt.cancel()
        return ev


def summary(ev):
    # ignore the "interrupt" that answers our own reset, and replies finished before they were withdrawn
    first = next((i for i, (_, d) in enumerate(ev) if d["type"] == "user_end"), len(ev))
    ev = ev[first:]
    kinds = [d["type"] for _, d in ev]
    last_retract = max([i for i, (_, d) in enumerate(ev) if d["type"] == "retract"], default=-1)
    done = [d for i, (_, d) in enumerate(ev) if d["type"] == "done" and not d.get("interrupted") and i > last_retract]
    heard = [d["text"] for _, d in ev if d["type"] == "transcript"]
    return kinds, done, heard


async def main():
    ok = True
    a, b = trim(load("capital")), trim(load("followup"))
    # 1) pause 0.6 s mid-turn: the reply starts during the pause, then the user carries on
    ev = await converse([(a, 0.6), (b, 0)], {})
    kinds, done, heard = summary(ev)
    print("resume:", [(t, d["type"], str(d.get("text", d.get("id", "")))[:80]) for t, d in ev])
    final = done[-1]["text"] if done else ""
    c1 = kinds.count("retract") == 1 and kinds.count("interrupt") == 0
    c2 = len(done) == 1 and any(w in final.lower() for w in ("million", "people", "population"))
    c3 = bool(heard) and "france" in heard[-1].lower() and ("people" in heard[-1].lower() or "live" in heard[-1].lower())
    c4 = done and done[-1]["metrics"]["ctx_tokens"] < 400  # one exchange in context, not two
    for name, c in (("reply withdrawn (retract, no interrupt)", c1), ("one answer to the joined question", c2),
                    ("transcript has both parts", c3), (f"context holds one exchange ({done[-1]['metrics']['ctx_tokens'] if done else '-'} tokens)", c4)):
        print(("PASS " if c else "FAIL ") + name)
        ok &= bool(c)
    # 2) resume disabled: the same audio is an interruption (two turns)
    ev = await converse([(a, 0.6), (b, 0)], {"resume_ms": 0})
    kinds, done, heard = summary(ev)
    c = kinds.count("retract") == 0 and kinds.count("user_end") == 2
    print(("PASS " if c else "FAIL ") + f"resume_ms 0: two turns ({kinds})")
    ok &= c
    # 3) a late interruption (the reply has been playing for ~3 s) is not a resume
    story = trim(load("story"))
    ev = await converse([(story, 3.5), (trim(load("interrupt")), 0)], {})
    kinds, done, heard = summary(ev)
    c = kinds.count("retract") == 0 and kinds.count("interrupt") >= 1
    print(("PASS " if c else "FAIL ") + f"late barge-in stays an interruption ({kinds})")
    ok &= c
    print("ALL PASS" if ok else "SOME FAILED")

asyncio.run(main())
