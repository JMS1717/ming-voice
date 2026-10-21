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
def _moe_gate_up(x_ptr, ids_ptr, wgu_ptr, sgu_ptr, a_ptr, H, I,
                 s_we, s_wn, s_se,
                 QUANT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
    if QUANT:  # int8 weights with one fp32 scale per output row
        acc_g *= tl.load(sgu_ptr + e * s_se + offs_n)
        acc_u *= tl.load(sgu_ptr + e * s_se + I + offs_n)
    # match the reference rounding: bf16 projections, bf16 silu, bf16 product
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    g = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(a_ptr + k * I + offs_n, (g * u).to(tl.bfloat16))


@triton.jit
def _moe_down(a_ptr, ids_ptr, w_ptr, wd_ptr, sd_ptr, y_ptr, H, I, NK,
              s_we, s_wn, s_se,
              QUANT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
        if QUANT:
            part *= tl.load(sd_ptr + e * s_se + offs_n)
        acc += wk * part.to(tl.bfloat16).to(tl.float32)
    tl.store(y_ptr + offs_n, acc.to(tl.bfloat16))


CFG = dict(A=dict(BN=16, BK=256, num_warps=4), B=dict(BN=16, BK=256, num_warps=4))


def fused_moe_decode(x, w_gu, w_d, ids, weights, a_buf=None, out=None, s_gu=None, s_d=None):
    """x: [1, H] bf16; ids: [NK] int; weights: [NK] fp32 (shared expert weight 1.0 included).
    w_gu/w_d are bf16, or int8 with per-output-row fp32 scales s_gu [E+1, 2I] / s_d [E+1, H]."""
    H = x.shape[-1]
    I = w_d.shape[-1]
    NK = ids.shape[0]
    quant = s_gu is not None
    a = torch.empty(NK, I, device=x.device, dtype=torch.bfloat16) if a_buf is None else a_buf
    y = torch.empty(1, H, device=x.device, dtype=torch.bfloat16) if out is None else out
    ca, cb = CFG["A"], CFG["B"]
    _moe_gate_up[(NK, I // ca["BN"])](x, ids, w_gu, s_gu if quant else w_gu, a, H, I,
                                     w_gu.stride(0), w_gu.stride(1), s_gu.stride(0) if quant else 0,
                                     QUANT=quant, BN=ca["BN"], BK=ca["BK"], num_warps=ca["num_warps"])
    _moe_down[(H // cb["BN"],)](a, ids, weights, w_d, s_d if quant else w_d, y, H, I, NK,
                               w_d.stride(0), w_d.stride(1), s_d.stride(0) if quant else 0,
                               QUANT=quant, BN=cb["BN"], BK=cb["BK"], num_warps=cb["num_warps"])
    return y


# ---------------------------------------------------------------- multi-token (prefill) grouped MoE
# (token, expert) pairs are sorted by expert and padded to BM-row blocks, all on the GPU (no host sync);
# each program computes one BM x BN tile of one expert's projection with tl.dot. Blocks past the last
# expert's padded range exit at once.


@triton.jit
def _moe_gu_grouped(x_ptr, sorted_ptr, bexp_ptr, wgu_ptr, sgu_ptr, a_ptr, P, H, I, KP1,
                    s_we, s_wn, s_se,
                    QUANT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(bexp_ptr + pid_m).to(tl.int64)
    if e < 0:
        return
    rows = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM))
    valid = rows < P
    tok = (rows // KP1).to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    base = wgu_ptr + e * s_we
    acc_g = tl.zeros((BM, BN), dtype=tl.float32)
    acc_u = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, H, BK):
        offs_k = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + tok[:, None] * H + offs_k[None, :], mask=valid[:, None], other=0.0)
        wg = tl.load(base + offs_n[None, :] * s_wn + offs_k[:, None]).to(tl.bfloat16)
        wu = tl.load(base + (offs_n[None, :] + I) * s_wn + offs_k[:, None]).to(tl.bfloat16)
        acc_g = tl.dot(a, wg, acc_g)
        acc_u = tl.dot(a, wu, acc_u)
    if QUANT:
        acc_g *= tl.load(sgu_ptr + e * s_se + offs_n)[None, :]
        acc_u *= tl.load(sgu_ptr + e * s_se + I + offs_n)[None, :]
    g = acc_g.to(tl.bfloat16).to(tl.float32)
    u = acc_u.to(tl.bfloat16).to(tl.float32)
    g = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(a_ptr + rows[:, None].to(tl.int64) * I + offs_n[None, :], (g * u).to(tl.bfloat16), mask=valid[:, None])


@triton.jit
def _moe_down_grouped(a_ptr, sorted_ptr, bexp_ptr, wd_ptr, sd_ptr, w_ptr, o_ptr, P, H, I,
                      s_we, s_wn, s_se,
                      QUANT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(bexp_ptr + pid_m).to(tl.int64)
    if e < 0:
        return
    rows = tl.load(sorted_ptr + pid_m * BM + tl.arange(0, BM))
    valid = rows < P
    r64 = rows.to(tl.int64)
    offs_n = pid_n * BN + tl.arange(0, BN)
    base = wd_ptr + e * s_we
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for i0 in range(0, I, BK):
        offs_i = i0 + tl.arange(0, BK)
        a = tl.load(a_ptr + r64[:, None] * I + offs_i[None, :], mask=valid[:, None], other=0.0)
        w = tl.load(base + offs_n[None, :] * s_wn + offs_i[:, None]).to(tl.bfloat16)
        acc = tl.dot(a, w, acc)
    if QUANT:
        acc *= tl.load(sd_ptr + e * s_se + offs_n)[None, :]
    wk = tl.load(w_ptr + rows, mask=valid, other=0.0)
    out = acc.to(tl.bfloat16).to(tl.float32) * wk[:, None]
    tl.store(o_ptr + r64[:, None] * H + offs_n[None, :], out, mask=valid[:, None])


GCFG = dict(A=dict(BM=32, BN=32, BK=128, num_warps=4), B=dict(BM=32, BN=64, BK=64, num_warps=4))  # dev/test_moe_prefill.py TUNE=1


def _align_blocks(flat, E1, BM):
    """Sort pair indices by expert, padding each expert's run to a multiple of BM (GPU only, no sync).
    Returns sorted pair ids (P = padding) and the expert of each BM block (-1 = unused)."""
    P = flat.numel()
    dev = flat.device
    # (not torch.bincount: on the GPU it reads min/max back to the host to size its output, a sync per call)
    counts = torch.zeros(E1, dtype=torch.long, device=dev).scatter_add_(0, flat.long(), torch.ones_like(flat, dtype=torch.long))
    padded = (counts + BM - 1) // BM * BM
    pend = padded.cumsum(0)
    pstart = pend - padded
    order = torch.argsort(flat, stable=True)
    se = flat[order]
    cstart = counts.cumsum(0) - counts
    dest = pstart[se] + torch.arange(P, device=dev) - cstart[se]
    nb = (P + E1 * (BM - 1) + BM - 1) // BM
    sorted_ids = torch.full((nb * BM,), P, dtype=torch.int32, device=dev)
    sorted_ids[dest] = order.to(torch.int32)
    bexp = torch.searchsorted(pend, torch.arange(nb, device=dev) * BM, right=True)
    bexp = torch.where(bexp < E1, bexp, -1).to(torch.int32)
    return sorted_ids, bexp, nb


def fused_moe_prefill(x, w_gu, w_d, ids, weights, s_gu=None, s_d=None):
    """x: [N, H] bf16; ids: [N, K+1] expert ids per token (shared expert included); weights: [N, K+1] fp32.
    Returns [N, H] bf16 = sum_k weights[:, k] * expert_k(x)."""
    N, H = x.shape
    KP1 = ids.shape[1]
    E1, I = w_d.shape[0], w_d.shape[-1]
    P = N * KP1
    quant = s_gu is not None
    ca, cb = GCFG["A"], GCFG["B"]
    assert ca["BM"] == cb["BM"]
    sorted_ids, bexp, nb = _align_blocks(ids.reshape(-1), E1, ca["BM"])
    a = torch.empty(P, I, device=x.device, dtype=torch.bfloat16)
    _moe_gu_grouped[(nb, I // ca["BN"])](x.contiguous(), sorted_ids, bexp, w_gu, s_gu if quant else w_gu, a, P, H, I, KP1,
                                         w_gu.stride(0), w_gu.stride(1), s_gu.stride(0) if quant else 0,
                                         QUANT=quant, BM=ca["BM"], BN=ca["BN"], BK=ca["BK"], num_warps=ca["num_warps"])
    o = torch.empty(P, H, device=x.device, dtype=torch.float32)
    _moe_down_grouped[(nb, H // cb["BN"])](a, sorted_ids, bexp, w_d, s_d if quant else w_d,
                                           weights.reshape(-1).float().contiguous(), o, P, H, I,
                                           w_d.stride(0), w_d.stride(1), s_d.stride(0) if quant else 0,
                                           QUANT=quant, BM=cb["BM"], BN=cb["BN"], BK=cb["BK"], num_warps=cb["num_warps"])
    return o.view(N, KP1, H).sum(1).to(x.dtype)


def quantize_rows_int8(w):
    """Symmetric per-output-row int8: w [..., N, K] -> (int8 [..., N, K], fp32 scales [..., N])."""
    amax = w.abs().amax(dim=-1).float().clamp_min(1e-8)
    scale = amax / 127.0
    q = torch.round(w.float() / scale[..., None]).clamp_(-127, 127).to(torch.int8)
    return q, scale


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
