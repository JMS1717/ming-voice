"""Pick the Talker CFM step count: speed vs intelligibility (ASR round-trip WER through the Thinker)."""
import os, re, sys, time
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Talker, Thinker, DEVICE, DTYPE
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
from talker.talker_module.cfm import get_epss_timesteps

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASR_PROMPT = ("Please recognize the language of this speech and transcribe it, placing a `\t` "
              "between the identified language and the transcribe.")
TEXTS = [
    "Sure, I can help with that. Let me walk you through the steps one at a time.",
    "I'd recommend starting with a simple recipe, like pasta with garlic, olive oil, and fresh basil.",
    "That's a great question. Photosynthesis is how plants turn sunlight, water, and carbon dioxide into sugar.",
    "Your meeting with the design team has been moved to Thursday afternoon.",
    "Hmm, I'm not completely sure, but I think the movie came out a few years ago.",
    "The quickest way downtown is to take the train and get off at the central station.",
    "Honestly, it sounds like you had a really long day. Maybe take a short walk and get some rest.",
    "Black holes form when massive stars collapse under their own gravity at the end of their lives.",
    "Okay, I've added milk, eggs, and bread to your shopping list.",
    "If you want to learn the guitar, start by practicing a few basic chords every single day.",
    "The museum is open every day except Monday, and admission is free for children.",
    "Good morning! It's going to be sunny and warm today, so don't forget your sunglasses.",
]

th = Thinker(use_graphs=False)
tk = Talker(use_graphs=True)


def asr(wav44):
    w = torchaudio.functional.resample(torch.from_numpy(wav44), 44100, 16000)
    msgs = [{"role": "user", "content": [{"type": "audio", "audio": (w, 16000)}, {"type": "text", "text": ASR_PROMPT}]}]
    inp = th.processor(msgs).to(DEVICE)
    inp["audio_feats"] = inp["audio_feats"].to(DTYPE)
    with torch.inference_mode():
        out = th.model.generate(**inp, max_new_tokens=120, do_sample=False)
    t = th.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return t.split("\t", 1)[-1]


def norm(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return s.split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def set_steps(n):
    tk.steps = n
    ts = get_epss_timesteps(n, DEVICE, DTYPE)
    coef = tk.t.cfm.sway_sampling_coef
    if coef is not None:
        ts = ts + coef * (torch.cos(torch.pi / 2 * ts) - 1 + ts)
    tk.ts = ts
    tk.dts = [ts[s + 1] - ts[s] for s in range(n)]
    tk._capture()


os.makedirs(os.path.join(HERE, "eval"), exist_ok=True)
for n in tuple(int(v) for v in os.environ.get("STEPS", "10,7,6").split(",")):
    set_steps(n)
    torch.manual_seed(0)
    tot_audio = tot_time = 0.0
    wers = []
    for i, text in enumerate(TEXTS * 2):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        wav = tk.synthesize(text, "US_Stella")
        dt = time.perf_counter() - t0
        tot_audio += len(wav) / 44100; tot_time += dt
        hyp = asr(wav)
        wers.append(wer(text, hyp))
        sf.write(os.path.join(HERE, f"eval/steps{n}_{i}.wav"), wav, 44100)
        if wers[-1] > 0.15:
            print(f"   [{n}] {text!r} -> {hyp!r}")
    print(f"steps={n:2d}: RTF {tot_time / tot_audio:.3f}  mean WER {np.mean(wers):.3f}  bad(>0.15) {sum(w > 0.15 for w in wers)}/{len(wers)}", flush=True)
