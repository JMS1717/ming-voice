"""Thinker decode step: graph replay time per token, and a kernel breakdown of one (eager) step."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from engine import Thinker

th = Thinker()
with torch.inference_mode():
    th.reset()
    list(th.stream(None, user_text="Tell me a long story about a lighthouse keeper.", max_new_tokens=40))
    L = next(b for b in th._buckets() if b > th.cur_len)
    g = th.graphs[L]
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(100):
        g.replay()
    torch.cuda.synchronize()
    print(f"graph replay: {(time.perf_counter() - t) * 10:.2f} ms/token (bucket {L})")
    # one token through the graph incl. the host round trip that _step does (tok in, argmax out)
    t = time.perf_counter()
    for _ in range(50):
        th.tok_buf.fill_(100)
        g.replay()
        int(th.tok_buf.item())
    print(f"with host sync per token: {(time.perf_counter() - t) * 20:.2f} ms/token")
    from torch.profiler import profile, ProfilerActivity
    th.cache.attn_len = L
    for _ in range(3):
        th._decode_step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(10):
            th._decode_step()
        torch.cuda.synchronize()
    ka = p.key_averages()
    tot = sum(k.self_device_time_total for k in ka) / 10
    print(f"eager kernels: {tot / 1000:.2f} ms/token of GPU time, {sum(k.count for k in ka) / 10:.0f} launches/token")
    rows = sorted(ka, key=lambda k: -k.self_device_time_total)[:25]
    for k in rows:
        print(f"{k.self_device_time_total / 10 / 1000:7.3f} ms  {k.count / 10:5.0f}x  {k.key[:110]}")
