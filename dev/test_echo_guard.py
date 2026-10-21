"""Offline check of echo_guard.EchoGuard (no server or GPU needed).

1. Speakerphone: the mic hears the assistant's reply through a simulated room (several levels and delays), alone
   (frames should be called echo) and with the user talking over it (the user's frames should not).
2. Headphones: no echo at all, the user talking around the start/end of playback: nothing may count as echo and
   no echo level may be learned (a wrong one would later swallow real speech).
3. Talking over a reply with no echo: the reply in dev/data/echo_reply_short.wav once matched the user's words
   by chance (rho ~0.6), taught the guard a wrong echo level and swallowed a barge-in.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf, torch, torchaudio
from echo_guard import EchoGuard

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)


def guard():
    return EchoGuard()


def load(path, sr_out=16000):
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = torch.from_numpy(w.mean(1))
    return torchaudio.functional.resample(w, sr, sr_out).numpy() if sr != sr_out else w.numpy()


def room(x, delay_s, gain):
    n = int(0.08 * 16000)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / (0.02 * 16000))
    ir[0] = 3.0
    ir /= np.abs(ir).sum() / 3
    y = np.convolve(x, ir)[: len(x)] * gain
    y = np.tanh(y * 3) / 3  # a little loudspeaker nonlinearity
    return np.concatenate([np.zeros(int(delay_s * 16000), np.float32), y])[: len(x)].astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


reply = load(os.path.join(HERE, "dev", "data", "echo_reply.wav"))
intr = load(os.path.join(HERE, "e2e", "questions", "interrupt.wav"))
user = np.pad(intr, (int(3.0 * 16000), len(reply)))[: len(reply)]
worst = dict(echo=1.0, user=0.0, user_loud=0.0, learned=0)

print("1. speakerphone")
for gain in (0.03, 0.1, 0.3, 0.6):
    for delay in (0.15, 0.45, 0.8):
        res = {}
        played = np.concatenate([np.zeros(int(delay * 16000), np.float32), reply])[: len(reply)]
        for name, mic in (("echo", room(reply, delay, gain)), ("user", room(reply, delay, gain) + user)):
            g = guard()
            g.add_playback(reply, 100.0)  # playback starts at t=100 s
            flagged = n = 0
            for i in range(0, len(mic) - 512, 512):
                now = 100.0 + (i + 512) / 16000 + 0.03  # arrival a bit after capture
                g.push_mic(mic[i:i + 512] + rng.standard_normal(512).astype(np.float32) * 3e-4, now)
                ref_loud = rms(played[i:i + 512]) > 0.02  # loud reply audio is echoing in this frame
                if (name == "echo" and ref_loud) or (name == "user" and rms(user[i:i + 512]) > 0.02):
                    n += 1
                    flagged += g.is_echo(now)
            res[name] = flagged / max(1, n)
        if gain >= 0.1:
            worst["echo"] = min(worst["echo"], res["echo"])
        # at gain 0.6 the echo is about as loud as the user: some of their frames can't be told apart by level
        key = "user_loud" if gain >= 0.6 else "user"
        worst[key] = max(worst[key], res["user"])
        print(f"   gain {gain:.2f} delay {delay:.2f}s: echo frames flagged {res['echo']:4.0%} (want ~100%), "
              f"user frames flagged {res['user']:3.0%} (want ~0%)")

print("2. headphones, user talking around the edges of playback")
for label, offset in (("0.3 s before playback ends", -0.3), ("0.2 s after playback ends", 0.2),
                      ("as playback starts", None)):
    g = guard()
    g.add_playback(reply[: 16000 * 3], 100.0)
    t_user = 100.0 + (3.0 + offset if offset is not None else 0.0)
    flagged = n = 0
    for i in range(0, len(intr) - 512, 512):
        now = t_user + (i + 512) / 16000 + 0.03
        g.push_mic(intr[i:i + 512] + rng.standard_normal(512).astype(np.float32) * 3e-4, now)
        if rms(intr[i:i + 512]) > 0.02:
            n += 1
            flagged += g.is_echo(now)
    worst["user"] = max(worst["user"], flagged / max(1, n))
    worst["learned"] += g.erl is not None
    print(f"   user starts {label}: user frames flagged {flagged / max(1, n):.0%} (want 0%), echo level {g.erl} (want None)")

print("3. talking over a reply, no echo (a reply that once matched the user's words by chance)")
short = load(os.path.join(HERE, "dev", "data", "echo_reply_short.wav"))
loud = np.flatnonzero(np.sqrt(np.convolve(intr ** 2, np.ones(320) / 320, "same")) > 0.01)
gap = max((b - a, b) for a, b in zip(loud[:-1], loud[1:]))  # the pause in "actually never mind ... what is 2+2"
second = intr[gap[1] - 800:]
for start in (0.1, 0.2, 0.3, 0.45):
    mic = np.random.default_rng(1).standard_normal(int((start + 2.5) * 16000)).astype(np.float32) * 1e-4
    s0 = int(start * 16000)
    mic[s0:s0 + len(second)] += second[: len(mic) - s0]
    g = guard()
    g.add_playback(short, 100.0)
    flagged = n = 0
    for i in range(0, len(mic) - 512, 512):
        now = 100.0 + (i + 512) / 16000 + 0.03
        g.push_mic(mic[i:i + 512], now)
        if rms(mic[i:i + 512]) > 0.004:
            n += 1
            flagged += g.is_echo(now)
    worst["user"] = max(worst["user"], flagged / max(1, n))
    worst["learned"] += g.erl is not None
    print(f"   user starts {start:.2f} s into the reply: frames flagged {flagged}/{n} (want 0), echo level {g.erl} (want None)")

ok = worst["echo"] >= 0.85 and worst["user"] <= 0.1 and worst["user_loud"] <= 0.25 and worst["learned"] == 0
print(f"{'PASS' if ok else 'FAIL'}: worst echo detection (gain >= 0.1) {worst['echo']:.0%}, worst user frames flagged "
      f"{worst['user']:.0%} (echo as loud as the user: {worst['user_loud']:.0%}), wrong echo levels learned {worst['learned']}")
