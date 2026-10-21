import os, sys, time
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Thinker, DEVICE, DTYPE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = "You are a helpful voice assistant. Answer briefly."
wav, sr = sf.read(os.path.join(HERE, "samples/introduce_ant_group.wav"), dtype="float32", always_2d=True)
wav = torchaudio.functional.resample(torch.from_numpy(wav.mean(1)), sr, 16000).numpy()
th = Thinker(use_graphs=False)

if False:
  pass
# 2) GPU kernel time per decode token
from torch.profiler import profile, ProfilerActivity
with torch.inference_mode():
    th.cache.attn_len = 256
    for _ in range(3):
        th._decode_step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            th._decode_step()
        torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=80))
