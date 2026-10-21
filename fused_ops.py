"""Small fused Triton kernels for the Thinker's per-token path (each replaces a chain of tiny PyTorch ops;
on ROCm every launch inside a HIP graph still costs ~2 us, and a decode step had ~1600 of them).

  rms_norm(x, w, eps)           BailingMoeV3RMSNorm in one kernel (was ~8)
  moe_route(x, w32, bias, ...)  the MoE router for one token: fp32 gate GEMV, sigmoid, expert bias,
                                group-limited top-k (top 4 of 8 groups by their top-2 sum, then top 8),
                                weight normalization and scaling, in two kernels (was ~20 + an 85 us
                                fp32 rocBLAS GEMM). Writes the ids/weights the fused MoE kernels read,
                                with the shared expert appended.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm(x_ptr, w_ptr, y_ptr, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=m, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    y = (x * tl.rsqrt(var + eps)).to(tl.bfloat16).to(tl.float32)  # reference: normalize in fp32, cast, then scale
    w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * N + offs, (w * y).to(y_ptr.dtype.element_ty), mask=m)


def rms_norm(x, w, eps):
    N = x.shape[-1]
    x2 = x.reshape(-1, N)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    # the reference returns weight.dtype * input.dtype -> promoted dtype
    out_dtype = torch.promote_types(w.dtype, x.dtype)
    y = torch.empty(x2.shape, dtype=out_dtype, device=x.device)
    _rms_norm[(x2.shape[0],)](x2, w, y, N, eps, BLOCK=triton.next_power_of_2(N), num_warps=4 if N <= 2048 else 8)
    return y.view(x.shape)


@triton.jit
def _gate_gemv(x_ptr, w_ptr, out_ptr, H, BE: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_e = pid * BE + tl.arange(0, BE)
    acc = tl.zeros((BE,), dtype=tl.float32)
    for k0 in range(0, H, BK):
        offs_k = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + offs_k).to(tl.float32)
        w = tl.load(w_ptr + offs_e[:, None] * H + offs_k[None, :])
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(out_ptr + offs_e, acc)


@triton.jit
def _route(logits_ptr, bias_ptr, ids_ptr, wts_ptr, scale,
           E: tl.constexpr, G: tl.constexpr, TOPG: tl.constexpr, K: tl.constexpr):
    GS: tl.constexpr = E // G
    offs = tl.arange(0, E)
    logits = tl.load(logits_ptr + offs)
    scores = 1.0 / (1.0 + tl.exp(-logits))
    r = scores + tl.load(bias_ptr + offs).to(tl.float32)
    NEG = float("-inf")
    r2 = tl.reshape(r, (G, GS))
    # group score = sum of the two largest routing scores in the group
    m1 = tl.max(r2, axis=1)
    a1 = tl.argmax(r2, axis=1)
    cols = tl.arange(0, GS)
    r2b = tl.where(cols[None, :] == a1[:, None], NEG, r2)
    gscore = m1 + tl.max(r2b, axis=1)
    # keep the TOPG best groups
    gidx = tl.arange(0, G)
    keep = gidx < 0
    for _ in tl.static_range(TOPG):
        best = tl.argmax(gscore, axis=0)
        keep = keep | (gidx == best)
        gscore = tl.where(gidx == best, NEG, gscore)
    rmask = tl.reshape(tl.where(keep[:, None], r2, NEG), (E,))
    # top K experts, in descending order (as torch.topk), and their sigmoid scores
    kk = tl.arange(0, 16)
    sel = tl.zeros((16,), dtype=tl.int32)
    sw = tl.zeros((16,), dtype=tl.float32)
    for k in tl.static_range(K):
        e = tl.argmax(rmask, axis=0)
        s = tl.sum(tl.where(offs == e, scores, 0.0), axis=0)
        sel = tl.where(kk == k, e, sel)
        sw = tl.where(kk == k, s, sw)
        rmask = tl.where(offs == e, NEG, rmask)
    total = tl.sum(sw, axis=0)
    w = sw / (total + 1e-20) * scale
    # shared expert (index E) with weight 1 after the routed ones
    sel = tl.where(kk == K, E, sel)
    w = tl.where(kk == K, 1.0, w)
    tl.store(ids_ptr + kk, sel.to(ids_ptr.dtype.element_ty), mask=kk <= K)
    tl.store(wts_ptr + kk, w, mask=kk <= K)


def moe_route(x, w32, bias, n_group, topk_group, top_k, scale, logits_buf, ids_out, wts_out):
    """x: [1, H] (one token). Writes ids_out [top_k+1] (shared expert last) and wts_out [top_k+1] fp32."""
    E, H = w32.shape
    BE = 8
    _gate_gemv[(E // BE,)](x, w32, logits_buf, H, BE=BE, BK=256, num_warps=4)
    _route[(1,)](logits_buf, bias, ids_out, wts_out, float(scale), E=E, G=n_group, TOPG=topk_group, K=top_k,
                 num_warps=4)
    return ids_out, wts_out, logits_buf
