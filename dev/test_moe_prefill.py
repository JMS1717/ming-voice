"""Grouped prefill MoE kernel vs. the per-expert reference loop: accuracy, speed, and a tile-size sweep."""
import itertools, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import fused_moe
from fused_moe import fused_moe_prefill, quantize_rows_int8

torch.manual_seed(0)
E, H, I, K = 128, 1536, 512, 8
dev = "cuda"
w_gu = torch.randn(E + 1, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
w_d = torch.randn(E + 1, H, I, device=dev, dtype=torch.bfloat16) * 0.02


def make(N):
    x = torch.randn(N, H, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(N, E, device=dev)
    ids = torch.cat([logits.topk(K, -1).indices, torch.full((N, 1), E, device=dev)], 1)
    wts = torch.cat([torch.rand(N, K, device=dev), torch.ones(N, 1, device=dev)], 1)
    return x, ids, wts


def reference(x, ids, wts):  # what the HF path does: one small GEMM pair per active expert
    y = torch.zeros(x.shape[0], H, device=dev)
    flat = ids.reshape(-1)
    order = flat.argsort()
    counts = torch.bincount(flat, minlength=E + 1).cpu().tolist()
    tok = order // (K + 1)
    s = 0
    for e, c in enumerate(counts):
        if c:
            t = tok[s:s + c]
            gu = x[t] @ w_gu[e].T
            a = F.silu(gu[:, :I]) * gu[:, I:]
            y.index_add_(0, t, (a @ w_d[e].T).float() * wts.reshape(-1)[order[s:s + c]][:, None])
        s += c
    return y


def bench(fn, reps=20):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps * 1e3


q_gu, s_gu = quantize_rows_int8(w_gu)
q_d, s_d = quantize_rows_int8(w_d)
if os.environ.get("TUNE") == "1":
    x, ids, wts = make(96)
    for part in ("A", "B"):
        best = None
        for bm, bn, bk, nw in itertools.product((16, 32), (32, 64, 128), (32, 64, 128), (4, 8)):
            fused_moe.GCFG["A" if part == "A" else "B"] = dict(BM=bm, BN=bn, BK=bk, num_warps=nw)
            other = "B" if part == "A" else "A"
            fused_moe.GCFG[other]["BM"] = bm
            try:
                ms = bench(lambda: fused_moe_prefill(x, w_gu, w_d, ids, wts), reps=10)
            except Exception:
                continue
            if best is None or ms < best[0]:
                best = (ms, dict(fused_moe.GCFG[part]))
        fused_moe.GCFG[part] = best[1]
        for p in ("A", "B"):
            fused_moe.GCFG[p]["BM"] = best[1]["BM"]
        print("best", part, best)
    print("GCFG =", fused_moe.GCFG)

for N in (8, 40, 120, 400):
    x, ids, wts = make(N)
    ref = reference(x, ids, wts)
    got = fused_moe_prefill(x, w_gu, w_d, ids, wts).float()
    got8 = fused_moe_prefill(x, q_gu, q_d, ids, wts, s_gu=s_gu, s_d=s_d).float()
    err = ((got - ref).norm() / ref.norm()).item()
    err8 = ((got8 - ref).norm() / ref.norm()).item()
    t_ref = bench(lambda: reference(x, ids, wts))
    t_new = bench(lambda: fused_moe_prefill(x, w_gu, w_d, ids, wts))
    t_q = bench(lambda: fused_moe_prefill(x, q_gu, q_d, ids, wts, s_gu=s_gu, s_d=s_d))
    print(f"N={N:4d}: rel err bf16 {err:.4f} int8 {err8:.4f} | per-expert loop {t_ref:.2f} ms -> grouped {t_new:.2f} ms "
          f"(int8 {t_q:.2f} ms) per layer")
