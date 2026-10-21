"""Fused single-token MoE for the Ming Thinker (RDNA3 / Triton).

Experts (plus the shared expert, stored as expert index E) live in stacked tensors
w_gu [E+1, 2I, H] and w_d [E+1, H, I]. Two kernels per MoE layer, no weight gather:
  A: a[k] = silu(x @ Wg[e_k]^T) * (x @ Wu[e_k]^T)         grid (K, I/BN)
  B: y    = sum_k w_k * (a[k] @ Wd[e_k]^T)                grid (H/BN)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _moe_gate_up(x_ptr, ids_ptr, wgu_ptr, a_ptr, H, I,
                 s_we, s_wn,
                 BN: tl.constexpr, BK: tl.constexpr):
    k = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(ids_ptr + k).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    base = wgu_ptr + e * s_we
    acc_g = tl.zeros((BN,), dtype=tl.float32)
    acc_u = tl.zeros((BN,), dtype=tl.float32)
    for k0 in range(0, H, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        wg = tl.load(base + offs_n[:, None] * s_wn + offs_k[None, :]).to(tl.float32)
        wu = tl.load(base + (offs_n[:, None] + I) * s_wn + offs_k[None, :]).to(tl.float32)
        acc_g += tl.sum(wg * x[None, :], axis=1)
        acc_u += tl.sum(wu * x[None, :], axis=1)
    # match the reference rounding: bf16 projections, bf16 silu, bf16 product
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    g = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(a_ptr + k * I + offs_n, (g * u).to(tl.bfloat16))


@triton.jit
def _moe_down(a_ptr, ids_ptr, w_ptr, wd_ptr, y_ptr, H, I, NK,
              s_we, s_wn,
              BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BN,), dtype=tl.float32)
    for k in range(0, NK):
        e = tl.load(ids_ptr + k).to(tl.int64)
        wk = tl.load(w_ptr + k)
        base = wd_ptr + e * s_we
        part = tl.zeros((BN,), dtype=tl.float32)
        for i0 in range(0, I, BK):
            offs_i = i0 + tl.arange(0, BK)
            a = tl.load(a_ptr + k * I + offs_i).to(tl.float32)
            w = tl.load(base + offs_n[:, None] * s_wn + offs_i[None, :]).to(tl.float32)
            part += tl.sum(w * a[None, :], axis=1)
        acc += wk * part.to(tl.bfloat16).to(tl.float32)
    tl.store(y_ptr + offs_n, acc.to(tl.bfloat16))


CFG = dict(A=dict(BN=16, BK=256, num_warps=4), B=dict(BN=16, BK=256, num_warps=4))


def fused_moe_decode(x, w_gu, w_d, ids, weights, a_buf=None, out=None):
    """x: [1, H] bf16; ids: [NK] int; weights: [NK] fp32 (shared expert weight 1.0 included)."""
    H = x.shape[-1]
    I = w_d.shape[-1]
    NK = ids.shape[0]
    a = torch.empty(NK, I, device=x.device, dtype=torch.bfloat16) if a_buf is None else a_buf
    y = torch.empty(1, H, device=x.device, dtype=torch.bfloat16) if out is None else out
    ca, cb = CFG["A"], CFG["B"]
    _moe_gate_up[(NK, I // ca["BN"])](x, ids, w_gu, a, H, I, w_gu.stride(0), w_gu.stride(1),
                                     BN=ca["BN"], BK=ca["BK"], num_warps=ca["num_warps"])
    _moe_down[(H // cb["BN"],)](a, ids, weights, w_d, y, H, I, NK, w_d.stride(0), w_d.stride(1),
                               BN=cb["BN"], BK=cb["BK"], num_warps=cb["num_warps"])
    return y


def _graph_us(fn, reps=50, inner=20):
    import time
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(inner):
                fn()
        g.replay(); s.synchronize(); t = time.perf_counter()
        for _ in range(reps):
            g.replay()
        s.synchronize()
    return (time.perf_counter() - t) / reps / inner * 1e6


if __name__ == "__main__":
    import itertools
    torch.manual_seed(0)
    E, H, I, K = 128, 1536, 512, 8
    dev = "cuda"
    w_gu = torch.randn(E + 1, 2 * I, H, device=dev, dtype=torch.bfloat16) * 0.02
    w_d = torch.randn(E + 1, H, I, device=dev, dtype=torch.bfloat16) * 0.02
    x = torch.randn(1, H, device=dev, dtype=torch.bfloat16)
    ids = torch.cat([torch.randperm(E, device=dev)[:K], torch.tensor([E], device=dev)])
    wts = torch.cat([torch.rand(K, device=dev), torch.ones(1, device=dev)])
    # reference
    ref = torch.zeros(1, H, device=dev, dtype=torch.float32)
    for k in range(K + 1):
        e = ids[k]
        g = torch.nn.functional.silu(x @ w_gu[e, :I].T)
        u = x @ w_gu[e, I:].T
        ref += wts[k] * ((g * u) @ w_d[e].T).float()
    got = fused_moe_decode(x, w_gu, w_d, ids, wts)
    print("rel err", ((got.float() - ref).norm() / ref.norm()).item())
    bytes_ = (K + 1) * 3 * I * H * 2
    best = None
    for bna, bka, nwa in itertools.product((8, 16, 32), (128, 256, 512), (2, 4, 8)):
        CFG["A"] = dict(BN=bna, BK=bka, num_warps=nwa)
        try:
            us = _graph_us(lambda: fused_moe_decode(x, w_gu, w_d, ids, wts))
        except Exception as ex:
            continue
        if best is None or us < best[0]:
            best = (us, dict(CFG["A"]))
    CFG["A"] = best[1]
    print("best A", best)
    best = None
    for bnb, bkb, nwb in itertools.product((8, 16, 32), (64, 128, 256, 512), (2, 4, 8)):
        CFG["B"] = dict(BN=bnb, BK=bkb, num_warps=nwb)
        try:
            us = _graph_us(lambda: fused_moe_decode(x, w_gu, w_d, ids, wts))
        except Exception:
            continue
        if best is None or us < best[0]:
            best = (us, dict(CFG["B"]))
    CFG["B"] = best[1]
    us = best[0]
    print("best B", best)
    print(f"fused MoE decode: {us:.1f} us ({bytes_ / us / 1e3:.0f} GB/s effective)")
    got = fused_moe_decode(x, w_gu, w_d, ids, wts)
    print("rel err (tuned)", ((got.float() - ref).norm() / ref.norm()).item())
