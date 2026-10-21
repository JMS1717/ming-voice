import os, sys, time
os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastlinear import small_linear, _graph_time
F = torch.nn.functional
for M, K, N in [(6, 1024, 1024), (6, 1024, 3072), (6, 1024, 4096), (6, 4096, 1024), (1, 896, 1152), (1, 896, 9728), (1, 4864, 896), (1, 1536, 157184)]:
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * .02; b = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    Wt = W.t().contiguous()
    r = {}
    r["x@W.T"] = _graph_time(lambda: x @ W.T)
    r["F.linear+b"] = _graph_time(lambda: F.linear(x, W, b))
    r["x@Wt(contig)"] = _graph_time(lambda: x @ Wt)
    r["mm+add b"] = _graph_time(lambda: (x @ W.T).add_(b))
    r["triton+b"] = _graph_time(lambda: small_linear(x, W, b))
    print(f"M={M} K={K} N={N}: " + "  ".join(f"{k} {v*1e6:.1f}" for k, v in r.items()), flush=True)
