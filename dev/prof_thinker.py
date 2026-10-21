import os, sys, time
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Thinker, DEVICE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wav, sr = sf.read(os.path.join(HERE, "samples/introduce_ant_group.wav"), dtype="float32", always_2d=True)
wav = torchaudio.functional.resample(torch.from_numpy(wav.mean(1)), sr, 16000).numpy()
print("audio seconds", len(wav) / 16000)
th = Thinker()
for i in range(3):
    th.reset()
    torch.cuda.synchronize(); t0 = time.perf_counter(); times = []
    for d in th.stream(wav, max_new_tokens=60):
        times.append(time.perf_counter())
    first = (times[0] - t0) * 1000
    per = (times[-1] - times[0]) / max(1, len(times) - 1) * 1000
    print(f"run {i}: prefill+first token {first:.0f} ms, decode {per:.1f} ms/token over {len(times)} pieces")

# profile one decode step
from torch.profiler import profile, ProfilerActivity
th.reset()
g = th.stream(wav, max_new_tokens=200)
next(g); next(g)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
    for _ in range(5):
        next(g)
    torch.cuda.synchronize()
ka = p.key_averages()
n_kernels = sum(e.count for e in ka if e.device_type.name == "CUDA") if hasattr(ka[0], "device_type") else -1
print(ka.table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=60))
print("cuda kernel launches per token ~", n_kernels / 5)
