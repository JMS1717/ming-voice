"""Fused fast path for the Ming Talker's per-latent work (CFM/DiT sampling + aggregator + stop head).

Same math as talker_module (DiT / Aggregator / CFM.sample with temperature 0), restructured for
decode-sized tensors on RDNA3:
  * q/k/v fused into one projection; bias, GELU and residual adds fused into Triton GEMM epilogues
    (rocBLAS' biased linear takes a ~45 us slow path at M<=16, the fused kernel ~20 us);
  * rotary embedding + 3-token attention in one tiny Triton kernel per block;
  * everything constant across the 10 flow-matching steps is hoisted: timestep embeddings,
    condition embeddings (cond + CFG-null rows), history embedding, rotary tables, cls token.
"""
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fastlinear import small_linear


def _fuse(mods, attr="weight"):
    """Concatenate the given Linear params into one tensor and make the originals views of it
    (no duplicate VRAM; the reference modules keep working)."""
    parts = [getattr(m, attr) for m in mods]
    fused = torch.cat([p.detach() for p in parts]).contiguous()
    off = 0
    for m, p in zip(mods, parts):
        n = p.shape[0]
        p.data = fused[off:off + n]
        off += n
    return fused


@triton.jit
def _tiny_rope_attn(qkv_ptr, cos_ptr, sin_ptr, out_ptr, S, H, scale,
                    D: tl.constexpr, HD: tl.constexpr, SP: tl.constexpr):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    P: tl.constexpr = HD // 2
    offs_s = tl.arange(0, SP)
    offs_p = tl.arange(0, P)
    smask = offs_s[:, None] < S
    row = qkv_ptr + b * S * 3 * D + offs_s[:, None] * 3 * D + h * HD + 2 * offs_p[None, :]
    qe = tl.load(row, mask=smask, other=0.0).to(tl.float32)
    qo = tl.load(row + 1, mask=smask, other=0.0).to(tl.float32)
    ke = tl.load(row + D, mask=smask, other=0.0).to(tl.float32)
    ko = tl.load(row + D + 1, mask=smask, other=0.0).to(tl.float32)
    ve = tl.load(row + 2 * D, mask=smask, other=0.0).to(tl.float32)
    vo = tl.load(row + 2 * D + 1, mask=smask, other=0.0).to(tl.float32)
    cs = tl.load(cos_ptr + offs_s[:, None] * P + offs_p[None, :], mask=smask, other=1.0)
    sn = tl.load(sin_ptr + offs_s[:, None] * P + offs_p[None, :], mask=smask, other=0.0)
    # interleaved-pair rotary (x_transformers), computed in fp32 then rounded to bf16 like the reference
    qe2 = (qe * cs - qo * sn).to(tl.bfloat16).to(tl.float32)
    qo2 = (qo * cs + qe * sn).to(tl.bfloat16).to(tl.float32)
    ke2 = (ke * cs - ko * sn).to(tl.bfloat16).to(tl.float32)
    ko2 = (ko * cs + ke * sn).to(tl.bfloat16).to(tl.float32)
    sc = tl.sum(qe2[:, None, :] * ke2[None, :, :], axis=2) + tl.sum(qo2[:, None, :] * ko2[None, :, :], axis=2)
    sc = sc * scale
    sc = tl.where(offs_s[None, :] < S, sc, float("-inf"))
    mx = tl.max(sc, axis=1)
    p = tl.exp(sc - mx[:, None])
    p = p / tl.sum(p, axis=1)[:, None]
    p = p.to(tl.bfloat16).to(tl.float32)
    oe = tl.sum(p[:, :, None] * ve[None, :, :], axis=1)
    oo = tl.sum(p[:, :, None] * vo[None, :, :], axis=1)
    orow = out_ptr + b * S * D + offs_s[:, None] * D + h * HD + 2 * offs_p[None, :]
    tl.store(orow, oe.to(tl.bfloat16), mask=smask)
    tl.store(orow + 1, oo.to(tl.bfloat16), mask=smask)


def rope_attn(qkv, cos, sin, heads):
    B, S, D3 = qkv.shape
    D = D3 // 3
    HD = D // heads
    out = torch.empty(B, S, D, device=qkv.device, dtype=qkv.dtype)
    _tiny_rope_attn[(B * heads,)](qkv, cos, sin, out, S, heads, 1.0 / math.sqrt(HD),
                                  D=D, HD=HD, SP=triton.next_power_of_2(max(S, 2)), num_warps=1)
    return out


class _Block:
    def __init__(self, blk):
        a = blk.attn
        assert a.q_norm is None and a.pe_attn_head is None
        self.heads = a.heads
        self.wqkv = _fuse([a.to_q, a.to_k, a.to_v])
        self.bqkv = _fuse([a.to_q, a.to_k, a.to_v], "bias")
        self.wo, self.bo = a.to_out[0].weight, a.to_out[0].bias
        self.n1, self.e1 = blk.norm1.weight, blk.norm1.eps
        self.n2, self.e2 = blk.norm2.weight, blk.norm2.eps
        ff = blk.mlp.ff
        self.w1, self.b1 = ff[0][0].weight, ff[0][0].bias
        self.w2, self.b2 = ff[2].weight, ff[2].bias
        assert ff[0][1].approximate == "tanh"

    def __call__(self, h, cos, sin):
        qkv = small_linear(h, self.wqkv, self.bqkv, norm=(self.n1, self.e1))
        a = rope_attn(qkv, cos, sin, self.heads)
        h = small_linear(a, self.wo, self.bo, residual=h)
        f = small_linear(h, self.w1, self.b1, act=1, norm=(self.n2, self.e2))
        return small_linear(f, self.w2, self.b2, residual=h)


def _rope_tables(rotary, n):
    freqs, _ = rotary.forward_from_seq_len(n)  # [1, n, hd] interleaved duplicates
    f = freqs[0, :, ::2].float()
    return f.cos().contiguous(), f.sin().contiguous()


class FastTail:
    """y_latent, next_input_embedding, stop_prob = tail(hidden, history, noise)."""

    def __init__(self, talker, ts, dts, cfg_strength):
        t = talker.t
        dit, agg = t.cfm.model, t.aggregator
        assert dit.spk_embedder is None
        dev = t.stop_head.weight.device
        self.cfg = cfg_strength
        self.dts = [d.reshape(1) for d in dts]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # t_embedder output per step (identical for both CFG rows)
            self.t_emb = [dit.t_embedder(ts[s].reshape(1)).to(torch.bfloat16) for s in range(len(dts))]
            self.cls = agg.word_embedder(torch.zeros(1, 1, dtype=torch.long, device=dev)).to(torch.bfloat16)
        self.c_w, self.c_b = dit.c_embedder.cond_embedder.weight, dit.c_embedder.cond_embedder.bias
        self.x_w, self.x_b = dit.x_embedder.weight, dit.x_embedder.bias
        self.f_n, self.f_e = dit.final_layer.norm_final.weight, dit.final_layer.norm_final.eps
        self.f_w, self.f_b = dit.final_layer.linear.weight, dit.final_layer.linear.bias
        self.blocks = [_Block(b) for b in dit.blocks]
        self.cos3, self.sin3 = _rope_tables(dit.rotary_embed, 3)
        self.a_x_w, self.a_x_b = agg.x_embedder.weight, agg.x_embedder.bias
        self.a_blocks = [_Block(b) for b in agg.blocks]
        self.cos2, self.sin2 = _rope_tables(agg.rotary_embed, 2)
        self.a_n, self.a_e = agg.final_layer.norm_final.weight, agg.final_layer.norm_final.eps
        self.a_w, self.a_b = agg.final_layer.linear.weight, agg.final_layer.linear.bias
        self.s_w, self.s_b = t.stop_head.weight, t.stop_head.bias
        self.null_c = self.c_b.reshape(1, -1)  # c_embedder(zeros) == bias

    def __call__(self, hidden, hist, noise):
        D = self.x_w.shape[0]
        c = small_linear(hidden.reshape(1, -1), self.c_w, self.c_b)           # [1, D]
        c2 = torch.cat([c, self.null_c], 0)                                     # [2, D] cond / CFG-null
        he = small_linear(hist.reshape(1, -1), self.x_w, self.x_b)             # [1, D]
        y = noise.reshape(1, -1)
        for s, dt in enumerate(self.dts):
            ye = small_linear(y, self.x_w, self.x_b)                            # [1, D]
            h = torch.stack([c2 + self.t_emb[s], he.expand(2, D), ye.expand(2, D)], 1)  # [2, 3, D]
            for blk in self.blocks:
                h = blk(h, self.cos3, self.sin3)
            out = small_linear(h[:, -1], self.f_w, self.f_b, norm=(self.f_n, self.f_e))  # [2, 64]
            pred, null = out[0:1], out[1:2]
            y = y + (pred + (pred - null) * self.cfg) * dt
        lat = y.reshape(1, 1, -1)
        # aggregator: [cls, x_embed(latent)] -> blocks -> final layer on the cls token
        a = torch.stack([self.cls.reshape(1, D), small_linear(y, self.a_x_w, self.a_x_b)], 1)  # [1, 2, D]
        for blk in self.a_blocks:
            a = blk(a, self.cos2, self.sin2)
        x_next = small_linear(a[:, 0], self.a_w, self.a_b, norm=(self.a_n, self.a_e)).reshape(1, 1, -1)
        stop = small_linear(hidden.reshape(1, -1), self.s_w, self.s_b).float().softmax(-1)[0, 1]
        return lat, x_next, stop


# --------------------------------------------------------------------------- Talker LLM (Qwen2) decode step

@triton.jit
def _qwen_decode_attn(qkv_ptr, kc_ptr, vc_ptr, pos_ptr, inv_freq_ptr, out_ptr, max_len, scale,
                      NH: tl.constexpr, NKV: tl.constexpr, HD: tl.constexpr, BLOCK: tl.constexpr):
    """One program per query head: rotary on q/k, write k/v into the static cache at `pos`
    (first head of each GQA group), then online-softmax attention over positions [0, pos]."""
    h = tl.program_id(0)
    G: tl.constexpr = NH // NKV
    HALF: tl.constexpr = HD // 2
    kvh = h // G
    pos = tl.load(pos_ptr)
    d = tl.arange(0, HD)
    partner = (d + HALF) % HD
    sign = tl.where(d < HALF, -1.0, 1.0)
    f = pos.to(tl.float32) * tl.load(inv_freq_ptr + (d % HALF))
    c = tl.cos(f).to(tl.bfloat16).to(tl.float32)
    s = tl.sin(f).to(tl.bfloat16).to(tl.float32)
    qb = qkv_ptr + h * HD
    kb = qkv_ptr + NH * HD + kvh * HD
    vb = qkv_ptr + (NH + NKV) * HD + kvh * HD
    q = tl.load(qb + d).to(tl.float32)
    q = (q * c + sign * tl.load(qb + partner).to(tl.float32) * s).to(tl.bfloat16).to(tl.float32)
    k = tl.load(kb + d).to(tl.float32)
    k = (k * c + sign * tl.load(kb + partner).to(tl.float32) * s).to(tl.bfloat16)
    v = tl.load(vb + d)
    kcache = kc_ptr + kvh * max_len * HD
    vcache = vc_ptr + kvh * max_len * HD
    if h % G == 0:
        tl.store(kcache + pos * HD + d, k)
        tl.store(vcache + pos * HD + d, v)
    # current position first (from registers), then the cached prefix
    m = tl.sum(q * k.to(tl.float32), axis=0) * scale
    l = 1.0
    acc = v.to(tl.float32)
    offs_b = tl.arange(0, BLOCK)
    for start in range(0, pos, BLOCK):
        offs = start + offs_b
        msk = offs < pos
        K = tl.load(kcache + offs[:, None] * HD + d[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        sc = tl.sum(K * q[None, :], axis=1) * scale
        sc = tl.where(msk, sc, float("-inf"))
        m_new = tl.maximum(m, tl.max(sc, axis=0))
        alpha = tl.exp(m - m_new)
        p = tl.exp(sc - m_new)
        V = tl.load(vcache + offs[:, None] * HD + d[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * V, axis=0)
        l = l * alpha + tl.sum(p, axis=0)
        m = m_new
    tl.store(out_ptr + h * HD + d, (acc / l).to(tl.bfloat16))


@triton.jit
def _silu_mul(gu_ptr, out_ptr, I, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    msk = offs < I
    g = tl.load(gu_ptr + offs, mask=msk, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + I + offs, mask=msk, other=0.0).to(tl.float32)
    g = (g / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    tl.store(out_ptr + offs, (g * u).to(tl.bfloat16), mask=msk)


class FastQwen2Step:
    """Single-token decode for the Talker's Qwen2 over StaticQwen2's KV cache (same layout)."""

    def __init__(self, static_llm):
        m = static_llm.m
        self.sl = static_llm
        self.nh, self.nkv, self.hd = static_llm.nh, static_llm.nkv, static_llm.hd
        self.layers = []
        for layer in m.layers:
            a, mlp = layer.self_attn, layer.mlp
            self.layers.append(dict(
                wqkv=_fuse([a.q_proj, a.k_proj, a.v_proj]),
                bqkv=_fuse([a.q_proj, a.k_proj, a.v_proj], "bias"),
                wo=a.o_proj.weight,
                wgu=_fuse([mlp.gate_proj, mlp.up_proj]),
                wd=mlp.down_proj.weight,
                n1=(layer.input_layernorm.weight, layer.input_layernorm.variance_epsilon),
                n2=(layer.post_attention_layernorm.weight, layer.post_attention_layernorm.variance_epsilon),
                inter=mlp.gate_proj.weight.shape[0]))
        self.norm = (m.norm.weight, m.norm.variance_epsilon)
        self.scale = 1.0 / math.sqrt(self.hd)

    def __call__(self, x, pos):
        """x: [1, 1, H] input embedding; pos: [1] long (position of x). Returns final-normed hidden [1, 1, H]."""
        H = x.shape[-1]
        h = x.reshape(1, H)
        sl = self.sl
        for i, L in enumerate(self.layers):
            qkv = small_linear(h, L["wqkv"], L["bqkv"], norm=L["n1"])
            a = torch.empty(1, self.nh * self.hd, device=h.device, dtype=h.dtype)
            _qwen_decode_attn[(self.nh,)](qkv, sl.k[i], sl.v[i], pos, sl.inv_freq, a, sl.max_len, self.scale,
                                          NH=self.nh, NKV=self.nkv, HD=self.hd, BLOCK=64, num_warps=2)
            h = small_linear(a, L["wo"], None, residual=h)
            gu = small_linear(h, L["wgu"], None, norm=L["n2"])
            act = torch.empty(1, L["inter"], device=h.device, dtype=h.dtype)
            _silu_mul[(triton.cdiv(L["inter"], 1024),)](gu, act, L["inter"], BLOCK=1024, num_warps=4)
            h = small_linear(act, L["wd"], None, residual=h)
        return F.rms_norm(h, (H,), self.norm[0], self.norm[1]).reshape(1, 1, H)
