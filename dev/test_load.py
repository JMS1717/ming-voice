import os, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
peak = [0]
def watch():
    while True:
        peak[0] = max(peak[0], int(open("/proc/self/statm").read().split()[1]) * 4096); time.sleep(0.2)
threading.Thread(target=watch, daemon=True).start()
import numpy as np, soundfile as sf, torch, torchaudio
from engine import Thinker
t = time.time()
th = Thinker(use_graphs=False)
print(f"load {time.time()-t:.0f}s, peak RSS {peak[0]/2**30:.2f} GiB, gpu {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)
print({str(d): n for d, n in __import__("collections").Counter(p.dtype for p in th.model.parameters()).items()})
wav, sr = sf.read("samples/introduce_ant_group.wav", dtype="float32", always_2d=True)
wav = torchaudio.functional.resample(torch.from_numpy(wav.mean(1)), sr, 16000).numpy()
th.reset("You are a helpful voice assistant. Answer briefly.")
print("reply:", "".join(th.stream(wav, max_new_tokens=40)))
print("asr:", th.transcribe(wav))
