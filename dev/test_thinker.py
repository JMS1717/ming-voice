import os, sys, time
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Thinker, DEVICE, DTYPE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = "You are a helpful voice assistant. Answer briefly."
wav, sr = sf.read(os.path.join(HERE, "samples/introduce_ant_group.wav"), dtype="float32", always_2d=True)
wav = torchaudio.functional.resample(torch.from_numpy(wav.mean(1)), sr, 16000).numpy()

th = Thinker(use_graphs=os.environ.get("TG", "1") == "1")

# HF reference (dynamic cache path) with identical prompt
msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [{"type": "audio", "audio": (torch.from_numpy(wav), 16000)}]}]
inp = th.processor(msgs).to(DEVICE)
inp["audio_feats"] = inp["audio_feats"].to(DTYPE)
with torch.inference_mode():
    out = th.model.generate(**inp, max_new_tokens=40, do_sample=False)
ref = th.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
print("HF  :", repr(ref))

for i in range(3):
    th.reset(SYS)
    torch.cuda.synchronize(); t0 = time.perf_counter(); ts = []; txt = ""
    for d in th.stream(wav, max_new_tokens=40):
        ts.append(time.perf_counter()); txt += d
    print(f"FAST: {txt.strip()!r}  first {1000*(ts[0]-t0):.0f} ms, decode {1000*(ts[-1]-ts[0])/max(1,len(ts)-1):.1f} ms/token")
print("match:", txt.strip() == ref)
t0 = time.perf_counter(); f = "".join(th.stream(None, user_text="Now say it in English, one sentence.", max_new_tokens=60))
print(f"follow-up ({1000*(time.perf_counter()-t0):.0f} ms, ctx {th.ctx_len()}): {f!r}")
# longer generation speed
th.reset()
t0 = time.perf_counter(); n = 0
for d in th.stream(None, user_text="Tell me a short story about a robot learning to paint.", max_new_tokens=200):
    n += 1
dt = time.perf_counter() - t0
print(f"long gen: {n} pieces in {dt:.2f}s -> {n/dt:.1f} tok/s")
print("peak mem GiB", torch.cuda.max_memory_allocated() / 2**30)
