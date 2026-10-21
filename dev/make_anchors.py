"""Make the voice anchors (anchors/<voice>.pt) and check them. Run with the server stopped (needs the GPU):

    HSA_ENABLE_DXG_DETECTION=1 .venv/bin/python dev/make_anchors.py [voice ...] [--eval-only]

For each English built-in voice: speak the anchor line K times with the plain voice, keep the takes that
transcribe back exactly and stay the same speaker throughout, and save the fastest. Then compare test sentences
with and without the anchor: speaker drift (CAM++ speaker embeddings of 1.5 s windows against the voice's average),
speaking rate, and transcription.
"""
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["MING_VOICE_ANCHORS"] = "0"  # (made here: start without them)
from engine import Talker  # noqa: E402

LINE = "Sure, I can help with that, and this is what I found for you today."
FEMALE = {"US_Stella": True, "US_Emma": True, "UK_Charlotte": True, "US_James": False, "UK_George": False}
TESTS = [
    "Today and Friday are overcast with highs around 20 to 22°C and only a tiny chance of rain.",
    "The weather is mild and sunny, the crowds thin out after the summer rush, and prices are lower than in July or August.",
    "October is a great pick.",
    "If you want warmer water for swimming, June or September are the sweet spots.",
]
K, N = 10, 6
SR = Talker.SR

_asr = None


def asr(x):
    global _asr
    if _asr is None:
        import whisper
        _asr = whisper.load_model("base.en", device="cpu")
    import torchaudio
    y = torchaudio.functional.resample(torch.from_numpy(x), SR, 16000).numpy()
    return _asr.transcribe(y, fp16=False, language="en")["text"].strip()


def words(s):
    s = s.lower().replace("°c", " degrees celsius").replace("twenty two", "22").replace("twenty", "20")
    return re.findall(r"[a-z0-9']+", s)


def spk(tk, x):
    """Unit-length CAM++ speaker embedding (the talker's own voice-cloning encoder)."""
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    w = torchaudio.functional.resample(torch.from_numpy(x).reshape(1, -1), SR, 16000)
    f = kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
    f -= f.mean(0, keepdim=True)
    enc = tk.t.speaker_encoder
    e = enc.run(None, {enc.get_inputs()[0].name: f.unsqueeze(0).numpy()})[0].reshape(-1)
    return e / np.linalg.norm(e)


def min_sim(tk, x, ref):
    """Lowest similarity to the voice over 1.5 s windows: a dip means part of it is another speaker."""
    n, hop, sims = int(1.5 * SR), int(0.75 * SR), []
    for i in range(0, max(1, len(x) - n + hop), hop):
        seg = x[i:i + n]
        if len(seg) > SR // 2 and np.sqrt(np.mean(seg ** 2)) > 0.01:
            sims.append(float(spk(tk, seg) @ ref))
    return min(sims) if sims else 1.0


def trim(latents, x):
    """Drop the trailing silence (keep one latent of it): continuing from "finished speaking" makes the talker
    sometimes stop right away."""
    n = latents.shape[1]
    hop = len(x) / n
    rms = np.array([np.sqrt(np.mean(x[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12) for i in range(n)])
    last = int(np.nonzero(rms > 0.05 * rms.max())[0][-1])
    return latents[:, :min(n, last + 2)], x[:int(min(n, last + 2) * hop)]


def run(tk, text, voice, anchor):
    lat = []
    x = np.concatenate(list(tk.synth_stream(text, voice, anchor=anchor, collect=lat)))
    return x, lat


def main():
    voices = [a for a in sys.argv[1:] if not a.startswith("--")] or list(FEMALE)
    tk = Talker()
    out = ROOT / "anchors"
    out.mkdir(exist_ok=True)
    for v in voices:
        other = "UK_George" if FEMALE[v] else "US_Stella"
        takes = []
        for k in range(K):
            x, lat = run(tk, LINE, v, False)
            ok = words(asr(x)) == words(LINE)
            takes.append((x, lat, ok))
        ref = np.mean([spk(tk, x) for x, _, ok in takes if ok], 0)
        ref /= np.linalg.norm(ref)
        own = [min_sim(tk, x, ref) for x, _, _ in takes]
        control = [min_sim(tk, run(tk, TESTS[1], other, False)[0], ref) for _ in range(3)]
        thr = (np.median(own) + max(control)) / 2
        print(f"{v}: own takes min-sim {np.round(own, 2)}, {other} {np.round(control, 2)} -> threshold {thr:.2f}",
              flush=True)
        if "--eval-only" not in sys.argv:
            best = None
            for (x, lat, ok), s in zip(takes, own):
                rate = len(LINE.split()) / (len(x) / SR)
                print(f"  take {len(x) / SR:.2f}s {rate:.2f} w/s sim {s:.2f} {'OK' if ok else 'misheard'} "
                      f"{len(lat)} fragment(s)")
                if ok and len(lat) == 1 and s >= np.median(own) and (best is None or rate > best[0]):
                    best = (rate, lat[0], x)
            if best is None:
                print(f"{v}: no clean take, skipped")
                continue
            frag, latents = best[1]
            latents, x = trim(latents, best[2])
            best = (best[0], best[1], x)
            torch.save({"text": frag.strip() + " ", "latents": latents.float().cpu()}, out / f"{v}.pt")
            import soundfile as sf
            sf.write(out / f"{v}.wav", best[2], SR)  # (to listen to; not used by the server)
            print(f"{v}: saved anchor, {best[0]:.2f} w/s, {latents.shape[1]} latents "
                  f"({latents.shape[1] / (len(best[2]) / SR):.1f}/s): {frag!r}", flush=True)
        a = torch.load(out / f"{v}.pt")
        anchor = tk.make_anchor(a["text"], a["latents"])
        for mode, an in (("plain", False), ("anchor", anchor)):
            stats = []
            t0 = time.time()
            for text in TESTS:
                for _ in range(N if text == TESTS[1] else 2):
                    x, _ = run(tk, text, v, an)
                    heard = asr(x)
                    stats.append((min_sim(tk, x, ref), len(text.split()) / (len(x) / SR), words(heard) == words(text)))
                    if words(heard) != words(text):
                        print(f"   {mode} misheard: {heard}")
            s, r, ok = np.array(stats).T
            print(f"{v} {mode:6s}: other-speaker takes {np.mean(s < thr):.2f} (min-sim mean {s.mean():.2f}, "
                  f"lowest {s.min():.2f}), rate {r.mean():.2f} w/s, exact {ok.mean():.2f} ({time.time() - t0:.0f}s)",
                  flush=True)


main()
