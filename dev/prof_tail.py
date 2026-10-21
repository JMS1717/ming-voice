import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Talker, DEVICE, DTYPE
from fast_talker import FastTail
tk = Talker(use_graphs=False)
ft = FastTail(tk, tk.ts, tk.dts, tk.cfg_strength)
from torch.profiler import profile, ProfilerActivity
h = torch.randn(1, 1, 896, device=DEVICE, dtype=DTYPE); hist = torch.zeros(1, 1, 64, device=DEVICE, dtype=DTYPE); n = torch.randn(1, 1, 64, device=DEVICE, dtype=DTYPE)
with torch.inference_mode():
    for _ in range(3): ft(h, hist, n)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        ft(h, hist, n); torch.cuda.synchronize()
ka = p.key_averages()
tot = sum(e.self_device_time_total for e in ka)
print("total GPU us", round(tot), "kernels", sum(e.count for e in ka))
for e in sorted(ka, key=lambda e: -e.self_device_time_total)[:14]:
    print(f"{e.self_device_time_total:9.0f} us {e.count:5d}x {e.self_device_time_total/max(1,e.count):7.1f} avg  {e.key[:90]}")
