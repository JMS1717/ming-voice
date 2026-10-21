"""Triton small-M linear for decode-sized activations on RDNA3 (y = act(x @ W^T + b) + residual).

rocBLAS/hipBLASLt have a ~35 us floor for M<=16 GEMMs on this setup; these kernels are
bandwidth-bound instead and fuse the bias / activation / residual epilogue. Block configs are
tuned per (M, N, K, epilogue) by timing inside a captured graph (how they run in production)
and cached on disk, so a restart does not retune.
"""
import json
import os
import time

import torch
import triton
import triton.language as tl

CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fastlinear_tuning.json")


@triton.jit
def _small_m_linear(x_ptr, w_ptr, b_ptr, r_ptr, g_ptr, y_ptr, M, N, K, eps,
                    stride_xm, stride_wn, stride_rm, stride_ym,
                    HAS_BIAS: tl.constexpr, HAS_RES: tl.constexpr, ACT: tl.constexpr, HAS_NORM: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    if HAS_NORM:  # fused RMSNorm prologue: rows are tiny (M<=16), so each program recomputes the row rms
        ss = tl.zeros((BM,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            xx = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                         mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
            ss += tl.sum(xx * xx, axis=1)
        rstd = 1.0 / tl.sqrt(ss / K + eps)
    for k0 in range(0, K, BK):
        offs_k = k0 + tl.arange(0, BK)
        kmask = offs_k < K
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=(offs_m[:, None] < M) & kmask[None, :], other=0.0)
        if HAS_NORM:
            g = tl.load(g_ptr + offs_k, mask=kmask, other=0.0).to(tl.float32)
            x = ((x.to(tl.float32) * rstd[:, None]).to(tl.bfloat16).to(tl.float32) * g[None, :]).to(tl.bfloat16)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=(offs_n[:, None] < N) & kmask[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
    if HAS_BIAS:
        acc += tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)[None, :]
    if ACT == 1:  # gelu, tanh approximation (round to bf16 first, like an unfused bf16 linear)
        acc = acc.to(tl.bfloat16).to(tl.float32)
        c = 0.7978845608028654 * (acc + 0.044715 * acc * acc * acc)
        t = 2.0 / (1.0 + tl.exp(-2.0 * c)) - 1.0
        acc = 0.5 * acc * (1.0 + t)
    elif ACT == 2:  # silu
        acc = acc.to(tl.bfloat16).to(tl.float32)
        acc = acc / (1.0 + tl.exp(-acc))
    ymask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if HAS_RES:
        acc = acc.to(tl.bfloat16).to(tl.float32)
        acc += tl.load(r_ptr + offs_m[:, None] * stride_rm + offs_n[None, :], mask=ymask, other=0.0).to(tl.float32)
    tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(y_ptr.dtype.element_ty), mask=ymask)


_CONFIGS = [dict(BN=bn, BK=bk, num_warps=nw) for bn in (16, 32, 64) for bk in (64, 128, 256, 512) for nw in (2, 4, 8)]
_best: dict = {}
_dirty = False


def _load_cache():
    global _best
    try:
        with open(CACHE_FILE) as f:
            _best = {tuple(json.loads(k)): v for k, v in json.load(f).items()}
    except Exception:
        _best = {}


def save_cache():
    global _dirty
    if not _dirty:
        return
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({json.dumps(list(k)): v for k, v in _best.items()}, f, indent=0)
        _dirty = False
    except Exception:
        pass


_load_cache()


def _launch(x, w, b, res, act, cfg, y, norm=None):
    M, K = x.shape
    N = w.shape[0]
    g, eps = norm if norm is not None else (None, 0.0)
    _small_m_linear[(triton.cdiv(N, cfg["BN"]),)](
        x, w, b if b is not None else w, res if res is not None else x, g if g is not None else w, y, M, N, K,
        float(eps or 0.0), x.stride(0), w.stride(0), res.stride(0) if res is not None else 0, y.stride(0),
        HAS_BIAS=b is not None, HAS_RES=res is not None, ACT=act, HAS_NORM=norm is not None,
        BM=16, BN=cfg["BN"], BK=cfg["BK"], num_warps=cfg["num_warps"])


def _graph_time(fn, reps=20, inner=40):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        s.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            for _ in range(inner):
                fn()
        g.replay()
        s.synchronize()
        t = time.perf_counter()
        for _ in range(reps):
            g.replay()
        s.synchronize()
    torch.cuda.current_stream().wait_stream(s)
    return (time.perf_counter() - t) / reps / inner


def _tune(x, w, b, res, act, norm=None):
    global _dirty
    y = torch.empty(x.shape[0], w.shape[0], device=x.device, dtype=x.dtype)
    best, best_t = None, float("inf")
    for cfg in _CONFIGS:
        if cfg["BK"] > max(64, triton.next_power_of_2(w.shape[1])):
            continue
        try:
            dt = _graph_time(lambda: _launch(x, w, b, res, act, cfg, y, norm))
        except Exception:
            continue
        if dt < best_t:
            best, best_t = cfg, dt
    _dirty = True
    return best


def small_linear(x, w, b=None, act=0, residual=None, out=None, norm=None):
    """x: [..., K] with prod(leading) <= 16; w: [N, K] contiguous; act: 0 none, 1 gelu-tanh, 2 silu;
    residual: tensor broadcastable as [..., N] added after the activation;
    norm: (gamma, eps) applies RMSNorm to x first (fused)."""
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    M, N = x2.shape[0], w.shape[0]
    if M > 16:
        if norm is not None:
            x2 = torch.nn.functional.rms_norm(x2, (x2.shape[-1],), norm[0], norm[1])
        y = torch.nn.functional.linear(x2, w, b)
        if act == 1:
            y = torch.nn.functional.gelu(y, approximate="tanh")
        elif act == 2:
            y = torch.nn.functional.silu(y)
        y = y.reshape(*shape[:-1], N)
        return y + residual if residual is not None else y
    r2 = None
    if residual is not None:
        r2 = residual.reshape(-1, N)
        if not r2.is_contiguous():
            r2 = r2.contiguous()
    key = (M, N, x2.shape[1], b is not None, r2 is not None, act, str(x.dtype), norm is not None)
    cfg = _best.get(key)
    if cfg is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"small_linear shape {key} not tuned before graph capture")
        cfg = _best[key] = _tune(x2, w, b, r2, act, norm)
    y = torch.empty(M, N, device=x.device, dtype=x.dtype) if out is None else out.reshape(M, N)
    _launch(x2, w, b, r2, act, cfg, y, norm)
    return y.reshape(*shape[:-1], N)


if __name__ == "__main__":
    dev, dt = "cuda", torch.bfloat16
    for M in (1, 2, 6):
        for K, N in [(1024, 1024), (1024, 3072), (1024, 4096), (4096, 1024), (896, 1152), (896, 9728), (4864, 896)]:
            W = torch.randn(N, K, device=dev, dtype=dt) * 0.02
            b = torch.randn(N, device=dev, dtype=dt)
            x = torch.randn(M, K, device=dev, dtype=dt)
            r = torch.randn(M, N, device=dev, dtype=dt)
            ref = torch.nn.functional.gelu(torch.nn.functional.linear(x, W, b), approximate="tanh") + r
            got = small_linear(x, W, b, act=1, residual=r)
            err = ((ref.float() - got.float()).norm() / ref.float().norm()).item()
            us = _graph_time(lambda: small_linear(x, W, b, act=1, residual=r)) * 1e6
            us_ref = _graph_time(lambda: torch.nn.functional.linear(x, W, b)) * 1e6
            print(f"M={M} K={K} N={N}: fused {us:.1f} us ({N*K*2/us/1e3:.0f} GB/s) vs blas linear {us_ref:.1f} us, err {err:.1e}",
                  flush=True)
    save_cache()
