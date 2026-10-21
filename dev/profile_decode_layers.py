"""Kernel launches and GPU time per decode step, broken down by module (attention type, MoE, norms, ...)."""
import os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from engine import Thinker

th = Thinker(use_graphs=False)
m = th.model.model
cfg = m.config
print("layers", th.n_layers, "types", [l.attention_layer_type for l in m.layers[:th.n_layers]])
print("mlp types", [type(l.mlp).__name__ for l in m.layers[:th.n_layers]])

# tag each submodule's forward with a record_function range
def tag(mod, name):
    orig = mod.forward
    def fwd(*a, **k):
        with record_function("@" + name):
            return orig(*a, **k)
    mod.forward = fwd

for i, layer in enumerate(m.layers[:th.n_layers]):
    kind = layer.attention_layer_type
    for cname, child in layer.named_children():
        tag(child, f"{kind}.{cname}" if cname == "attention" else f"{cname}:{type(child).__name__}")
for name, mod in (("final_norm", m.norm), ("lm_head", th.model.lm_head)):
    tag(mod, name)

with torch.inference_mode():
    th.reset()
    list(th.stream(None, user_text="Tell me a long story about a lighthouse keeper.", max_new_tokens=20))
    L = next(b for b in th._buckets() if b > th.cur_len)
    th.cache.attn_len = L
    for _ in range(3):
        th._decode_step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        th._decode_step()
        torch.cuda.synchronize()

# count kernel launches (runtime launch calls) under each "@" range on the CPU side
events = p.events()
stats = defaultdict(lambda: [0, 0.0])
launch_names = defaultdict(int)


def launches(e):
    n = 0
    for c in e.cpu_children:
        if "LaunchKernel" in c.name or "launch" in c.name.lower() and c.name.startswith("hip"):
            n += 1
            launch_names[c.name] += 1
        elif not c.name.startswith("@"):
            n += launches(c)
    return n


for e in events:
    if e.name.startswith("@"):
        stats[e.name][0] += launches(e)
        stats[e.name][1] += e.device_time_total
print("launch calls seen:", dict(launch_names))
tot_n = sum(v[0] for v in stats.values())
tot_t = sum(v[1] for v in stats.values())
print(f"total {tot_n} kernels, {tot_t / 1000:.2f} ms")
for k, (n, t) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
    print(f"{n:5d} kernels {t / 1000:7.3f} ms  {k}")
