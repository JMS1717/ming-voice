import os, sys, time, threading
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def graph_us(fn, reps=50, inner=50):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        fn(); s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(inner): fn()
        g.replay(); s.synchronize(); t = time.perf_counter()
        for _ in range(reps): g.replay()
        s.synchronize()
    return (time.perf_counter() - t) / reps / inner * 1e6
a = torch.zeros(1024, device="cuda"); x = torch.ones(1024, device="cuda")
W = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); v = torch.randn(6, 1024, device="cuda", dtype=torch.bfloat16)
print("tiny add (graph):", round(graph_us(lambda: a.add_(x)), 1), "us | blas 6x1024x1024:", round(graph_us(lambda: v @ W.T), 1), "us")
big = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(20): big @ big
torch.cuda.synchronize(); print("8k gemm TFLOPS", round(20 * 2 * 8192**3 / (time.perf_counter() - t) / 1e12, 1))
print("tiny add right after big work:", round(graph_us(lambda: a.add_(x)), 1), "us")
