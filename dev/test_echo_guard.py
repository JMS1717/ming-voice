"""Offline check of echo_guard.EchoGuard: fraction of frames called echo when the mic hears only the assistant
(should be ~1 during speech-like frames) vs. when a user talks over it (should be ~0)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf, torch, torchaudio
from echo_guard import EchoGuard

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0)


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


reply = load(os.path.join(HERE, "e2e", "out", "story_bargein.wav"))  # whole reply queued at once, as the server streams ahead
user = load(os.path.join(HERE, "e2e", "questions", "interrupt.wav"))
user = np.pad(user, (int(3.0 * 16000), len(reply)))[: len(reply)]
for gain in (0.03, 0.1, 0.3, 0.6):
    for delay in (0.15, 0.45, 0.8):
        res = {}
        played = np.concatenate([np.zeros(int(delay * 16000), np.float32), reply])[: len(reply)]
        for name, mic in (("echo only", room(reply, delay, gain)), ("user + echo", room(reply, delay, gain) + user)):
            g = EchoGuard()
            g.add_playback(reply, 100.0)  # playback starts at t=100 s
            flagged = speech = 0
            for i in range(0, len(mic) - 512, 512):
                now = 100.0 + (i + 512) / 16000 + 0.03  # arrival a bit after capture
                frame = mic[i:i + 512] + rng.standard_normal(512).astype(np.float32) * 3e-4
                g.push_mic(frame, now)
                active = np.sqrt(np.mean(played[i:i + 512] ** 2)) > 0.02  # the echo of loud reply audio is in this frame
                u_active = np.sqrt(np.mean(user[i:i + 512] ** 2)) > 0.02
                if name == "echo only" and active:
                    speech += 1
                    flagged += g.is_echo(now)
                if name == "user + echo" and u_active:
                    speech += 1
                    flagged += g.is_echo(now)
            res[name] = flagged / max(1, speech)
        print(f"gain {gain:.2f} delay {delay:.2f}s: echo frames flagged {res['echo only']:.0%} (want ~100%), "
              f"user frames flagged {res['user + echo']:.0%} (want ~0%)")

# No echo at all (headphones), user talking right around the edges of playback: nothing may count as echo, and
# no echo level may be learned (a wrong one would later swallow real speech).
u = load(os.path.join(HERE, "e2e", "questions", "interrupt.wav"))
for label, offset in (("starts 0.3 s before playback ends", -0.3), ("starts 0.2 s after playback ends", 0.2),
                      ("starts as playback starts", None)):
    g = EchoGuard()
    short = reply[: 16000 * 3]
    g.add_playback(short, 100.0)
    t_user = 100.0 + (3.0 + offset if offset is not None else 0.0)
    flagged = speech = 0
    for i in range(0, len(u) - 512, 512):
        now = t_user + (i + 512) / 16000 + 0.03
        g.push_mic(u[i:i + 512] + rng.standard_normal(512).astype(np.float32) * 3e-4, now)
        if np.sqrt(np.mean(u[i:i + 512] ** 2)) > 0.02:
            speech += 1
            flagged += g.is_echo(now)
    print(f"headphones, user {label}: user frames flagged {flagged / max(1, speech):.0%} (want 0%), "
          f"learned echo level {g.erl} (want None)")
