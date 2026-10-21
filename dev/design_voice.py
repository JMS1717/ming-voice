"""Try voice-design candidates for a persona voice. Run with the server stopped (needs the GPU):

    HSA_ENABLE_DXG_DETECTION=1 .venv/bin/python dev/design_voice.py OUT_DIR

Each candidate is a built-in voice (ip) and/or a style description (the talker's voice-design control). For each,
a few takes of some lines: transcription check, pitch, and how alike the takes sound (a designed voice is a fresh
draw every time unless it's anchored). Writes OUT_DIR/<candidate>.wav (all its takes) to listen to.
"""
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["MING_VOICE_ANCHORS"] = "0"
from engine import Talker  # noqa: E402

LINES = ["Good evening, sir. The living room lights are off, and I've taken the liberty of locking the car.",
         "Shall I run the numbers again, sir, or would you prefer to live dangerously?",
         "Certainly. Your reminder is set for nine o'clock tomorrow morning."]
EN = ("A refined, middle-aged English butler with a received pronunciation British accent: calm, crisp and "
      "articulate, warm baritone, dry understated wit, measured pace.")
ZH = "一位成熟稳重的英国男管家，标准英式口音，声音低沉温和，吐字清晰优雅，语气从容，略带冷幽默，语速适中。"
CANDIDATES = {  # name: (ip, style)
    "A_george": ("UK_George", None),
    "B_george_styled": ("UK_George", EN),
    "C_designed_en": (None, EN),
    "D_designed_zh": (None, ZH),
}
SR = Talker.SR


def pitch(x):
    """Median f0 of the voiced 40 ms frames (autocorrelation)."""
    n, f0 = int(0.04 * SR), []
    lo, hi = SR // 400, SR // 60
    for i in range(0, len(x) - n, n):
        f = x[i:i + n] - x[i:i + n].mean()
        if np.sqrt(np.mean(f ** 2)) < 0.02:
            continue
        ac = np.correlate(f, f, "full")[n - 1:]
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > 0.4 * ac[0]:
            f0.append(SR / k)
    return float(np.median(f0)) if f0 else 0.0


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
    return re.findall(r"[a-z0-9']+", s.lower().replace("9", "nine"))


def spk(tk, x):
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    w = torchaudio.functional.resample(torch.from_numpy(x).reshape(1, -1), SR, 16000)
    f = kaldi.fbank(w, num_mel_bins=80, dither=0, sample_frequency=16000)
    f -= f.mean(0, keepdim=True)
    enc = tk.t.speaker_encoder
    e = enc.run(None, {enc.get_inputs()[0].name: f.unsqueeze(0).numpy()})[0].reshape(-1)
    return e / np.linalg.norm(e)


ANCHOR_LINE = "Certainly sir, I have taken care of that and everything is exactly as you like it."


def takes(n: int, style: str, out: Path):
    """--takes N: N single-fragment takes of ANCHOR_LINE with a style (each a different speaker): wav + latents per
    take, to pick one and anchor it (--save)."""
    tk = Talker()
    instr = tk.tp.build_structured_instruction(style=style)
    prepare = tk.prepare
    tk.prepare = lambda text, voice: [(f, p, instr) for f, p, _ in prepare(text, "UK_George")]
    k = 0
    for _ in range(3 * n):
        if k == n:
            break
        lat = []
        x = np.concatenate(list(tk.synth_stream(ANCHOR_LINE, "UK_George", anchor=False, collect=lat)))
        heard = asr(x)
        if len(lat) != 1 or words(heard) != words(ANCHOR_LINE):
            print(f"  skipped: {len(lat)} fragment(s), heard {heard!r}", flush=True)
            continue
        k += 1
        sf.write(out / f"take{k}.wav", x, SR)
        torch.save({"text": lat[0][0].strip() + " ", "latents": lat[0][1].float().cpu(), "x": torch.from_numpy(x),
                    "instruction": instr}, out / f"take{k}.pt")
        print(f"take{k}: {len(x) / SR:.1f}s, pitch {pitch(x):.0f} Hz", flush=True)


def save(take: Path, name: str):
    """--save TAKE.pt NAME: anchors/NAME.pt from a take (trailing silence trimmed), then check it on test lines."""
    t = torch.load(take)
    latents, x = t["latents"], t["x"].numpy()
    n = latents.shape[1]  # (trailing silence trimmed, as in make_anchors.trim)
    hop = len(x) / n
    rms = np.array([np.sqrt(np.mean(x[int(i * hop):int((i + 1) * hop)] ** 2) + 1e-12) for i in range(n)])
    keep = min(n, int(np.nonzero(rms > 0.05 * rms.max())[0][-1]) + 2)
    latents, x = latents[:, :keep], x[:int(keep * hop)]
    dst = ROOT / "anchors" / f"{name}.pt"
    torch.save({"text": t["text"], "latents": latents, "instruction": t["instruction"]}, dst)
    tk = Talker()
    tk.load_anchors()
    ref = spk(tk, x)
    for line in LINES:
        y = np.concatenate(list(tk.synth_stream(line, name)))
        print(f"{name}: sim {float(spk(tk, y) @ ref):.2f}, pitch {pitch(y):.0f} Hz, "
              f"{'OK' if words(asr(y)) == words(line) else 'misheard: ' + asr(y)}", flush=True)
        sf.write(take.with_name(f"{name}_check{LINES.index(line)}.wav"), y, SR)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    if "--takes" in sys.argv:
        return takes(int(sys.argv[sys.argv.index("--takes") + 1]), EN, out)
    if "--save" in sys.argv:
        i = sys.argv.index("--save")
        return save(Path(sys.argv[i + 1]), sys.argv[i + 2])
    tk = Talker()
    tp = tk.tp
    prepare = tk.prepare
    for name, (ip, style) in CANDIDATES.items():
        instr = tp.build_structured_instruction(ip=ip, style=style)

        def patched(text, voice, _i=instr):
            return [(f, p, _i) for f, p, _ in prepare(text, "UK_George")]
        tk.prepare = patched
        clips, embs, ok, f0s = [], [], 0, []
        for line in LINES:
            for _ in range(2):
                x = np.concatenate(list(tk.synth_stream(line, "UK_George", anchor=False)))
                clips += [x, np.zeros(int(0.6 * SR), np.float32)]
                embs.append(spk(tk, x))
                f0s.append(pitch(x))
                good = words(asr(x)) == words(line)
                ok += good
                if not good:
                    print(f"   {name} misheard: {asr(x)}")
        e = np.array(embs)
        sim = e @ e.T
        alike = float(sim[np.triu_indices(len(e), 1)].mean())
        sf.write(out / f"{name}.wav", np.concatenate(clips), SR)
        print(f"{name}: exact {ok}/{len(embs)}, pitch {np.median(f0s):.0f} Hz, takes alike {alike:.2f}", flush=True)
    tk.prepare = prepare


main()
