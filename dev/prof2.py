import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Talker, DEVICE, DTYPE
tk = Talker(use_graphs=False)
from torch.profiler import profile, ProfilerActivity
with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
    for _ in range(3): tk._tail()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        tk._tail(); torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))
