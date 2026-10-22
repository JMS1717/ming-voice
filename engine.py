"""Low-latency streaming engine for Ming-3.0-audio-tiny (Thinker + Talker) on one GPU.

Thinker: HF remote code, with a sync-free batched MoE decode path and a persistent
multi-turn cache (each turn only prefills the new utterance).
Talker: per-step work (Qwen2 LLM step with a static KV cache + 10-step CFM/DiT +
aggregator + stop head) captured into HIP/CUDA graphs; AudioVAE decoded in small
streaming chunks so audio starts after the first couple of latents (80 ms each).
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

if os.path.exists("/dev/dxg"):  # WSL: ROCm reaches the GPU through the Windows driver
    os.environ.setdefault("HSA_ENABLE_DXG_DETECTION", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Encode >30 s inputs one 30 s Whisper chunk at a time (lower peak VRAM; same result).
os.environ.setdefault("HF_SFT_WHISPER_MICRO_BATCH", "1")
# Flash attention for SDPA on RDNA3 (AOTriton). Without it PyTorch falls back to the math path, which
# materializes fp32 attention matrices: the Whisper encoder over a 30 s chunk is ~2x slower.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = Path(os.environ.get("MING_MODEL_DIR", Path(__file__).resolve().parent / "model")).expanduser()
DEVICE = torch.device("cuda:0")
DTYPE = torch.bfloat16
EOS_IDS = {156892, 156895}
ROLE_END = "<|role_end|>"
TOOL_CALL_ID = 156896  # <tool_call>: the model wants to call a function instead of answering (tools.py)
MAX_TOOL_ROUNDS = 3  # call -> results -> (another call -> results ...) -> answer
TOOL_RESERVE = 1500  # context kept free for tool results when tools are on
ROLLOVER_MAX = 4096  # at most this much earlier conversation (as text) carries over a context rollover
COMPACT_HEADROOM = 1200  # compact between turns once less than this (+ the tool reserve) is left for the next turn
TOOL_ANSWER_WORDS = 35  # an answer from tool results ends at the first sentence end after this many words
BRIEF_TOOLS = {"get_current_time", "calculate", "timer"}  # (one-line results: nothing to read out at length)
ANSWER_WORDS = 80  # any other answer (~30 s spoken) too, unless the user asked for something long:
LONG_ASK = re.compile(r"\b(stor(y|ies)|poem|song|lyrics|essay|in detail|detailed|more detail|step by step|steps|"
                      r"instructions|recipe|walk me through|everything|all (of )?the|full|long|longer|keep going|"
                      r"go on|continue|read)\b", re.I)
SENTENCE_END = re.compile(r"(?<!\d)[.!?]$|\n$")
REFUSAL_START = re.compile(r"(sorry|unfortunately|i('m| am)? (do not|don't|can't|cannot|am not|'m not|unable|have no"
                           r"|currently)|i'm (afraid|unable|not able))", re.I)
CANT_LOOK_UP = re.compile(r"\b((live|real[- ]time|current|up[- ]to[- ]date|recent) (data|info|information|results|"
                          r"prices|rankings|news|weather|forecasts?|events)|the internet|internet access|"
                          r"a search engine|search(ing)? the (web|internet)|brows(e|ing)( the (web|internet))?|"
                          r"look (things|that|it) up|weather (data|forecasts?|information|updates)|"
                          r"a (weather )?forecast)\b", re.I)
DIRECT_MAX_S = 8  # utterances up to this long are checked for a plain device command (Toolbox.direct)
# ... and that dodge a music request instead of calling home_assistant ("What song would you like me to skip?")
MUSIC_DODGE_START = re.compile(r"(what|which) (song|track|music|speaker)\b", re.I)
MUSIC_DODGE = re.compile(r"\b(what|which) (song|track|music|speaker)\b|\bnothing (is )?(currently )?playing\b|"
                         r"\b(no|any) (songs?|music|tracks?) (is |are )?(currently )?playing\b|\bmusic app\b", re.I)
# replies that say a list or memory changed: (pattern, tool, action) to run when no tool was called
CLAIMS = [
    (re.compile(r"\b(off (of )?(your|the)( [\w-]+)? list|removed|crossed|no longer on|list now (only )?has)\b",
                re.I), "lists", "remove"),
    (re.compile(r"\b((i've|i have) forgotten|forgotten (that|it|where|what|about)|won't (remember|remind)|"
                r"(erased|deleted) (that|it))\b", re.I), "memory", "forget"),
    (re.compile(r"\b((added|put)\b.*\b(to|on) (your|the)( [\w-]+)? list|on (your|the)( [\w-]+)? list now)\b", re.I),
     "lists", "add"),
    (re.compile(r"\b(i'll|i will) remember\b", re.I), "memory", "remember"),
]
BYE = re.compile(r"\b(bye|goodbye|good ?night|talk to you (later|soon)|see you|have a (great|good|nice|lovely|wonderful) "
                 r"(day|night|evening|weekend|one)|until next time|take care)\b", re.I)  # (a reply saying goodbye)

DEFAULT_SYSTEM_PROMPT = (
    "You are Ming, a friendly, helpful voice assistant in a live spoken conversation. "
    "Listen to the user's speech and reply naturally in the same language they speak. "
    "Keep replies short and conversational, usually one to three sentences, unless the user asks for more detail. "
    "Use plain spoken language only: no markdown, bullet points, emojis, or code."
)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def pad_to_bucket(wav16k: np.ndarray, bucket_s: float = 0.5) -> np.ndarray:
    """Pad with trailing silence to a multiple of `bucket_s`. Every new input length costs a one-time
    kernel compile (MIOpen convs, Triton autotune keys) of ~0.3 s; buckets keep the shape set small."""
    n = len(wav16k)
    b = int(bucket_s * 16000)
    target = max(b, -(-n // b) * b)
    if target == n:
        return wav16k
    return np.concatenate([np.asarray(wav16k, dtype=np.float32), np.zeros(target - n, np.float32)])


class GpuKeepAwake(threading.Thread):
    """Submit a trivial kernel every `interval` s so the GPU never idles into a low-power state.

    Under WSL, with no display attached to the dGPU, an idle GPU stops serving WSL work promptly:
    the next host<->GPU sync stalls for 80-300 s. A few tiny kernels per second prevent that."""

    def __init__(self, interval: float = 0.25):
        super().__init__(daemon=True, name="gpu-keepawake")
        self.interval = interval

    def run(self):
        s = torch.cuda.Stream(device=DEVICE)
        x = torch.zeros(1, device=DEVICE)
        ev = torch.cuda.Event()
        with torch.cuda.stream(s):
            while True:
                if ev.query():  # never queue behind a stalled kernel
                    x.add_(1)
                    ev.record(s)
                time.sleep(self.interval)


# --------------------------------------------------------------------------- Thinker


_PRESTACK_INT8 = [False]


def _int8_expert_loaders(model):
    """Loader hooks: checkpoint key -> fn(cpu_tensor) quantizing that expert matrix into the int8 buffers."""
    from fused_moe import quantize_rows_int8

    def into(q, s, e, off):
        def fn(src):
            qq, ss = quantize_rows_int8(src.to(DEVICE))
            q[e, off:off + qq.shape[0]].copy_(qq)
            s[e, off:off + qq.shape[0]].copy_(ss)
        return fn

    hooks = {}
    for name, m in model.named_modules():
        if type(m).__name__ != "BailingMoeV3SparseMoeBlock" or getattr(m, "q_gu", None) is None:
            continue
        I = m.config.moe_intermediate_size
        for e in range(m.config.num_experts):
            p = f"{name}.experts.{e}."
            hooks[p + "gate_proj.weight"] = into(m.q_gu, m.s_gu, e, 0)
            hooks[p + "up_proj.weight"] = into(m.q_gu, m.s_gu, e, I)
            hooks[p + "down_proj.weight"] = into(m.q_d, m.s_d, e, 0)
    return hooks


def _prestack_experts(model_dir):
    """Make the MoE blocks allocate their routed experts as views into stacked [E+1, ...] tensors at
    construction time, so checkpoint tensors load straight into the layout the fused kernels use
    (restacking 14 GB afterwards took ~50 s near the VRAM ceiling). Slot E is for the shared expert."""
    import sys as _sys

    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    get_class_from_dynamic_module(cfg.auto_map["AutoModelForCausalLM"], str(model_dir))  # imports the module
    mod = next(m for name, m in list(_sys.modules.items())
               if name.endswith("modeling_bailing_moe_v3") and hasattr(m, "BailingMoeV3SparseMoeBlock"))
    block = mod.BailingMoeV3SparseMoeBlock
    if getattr(block, "_prestacked", False):
        return
    orig = block._setup_experts

    def _setup_experts(self):
        with torch.device("meta"):
            orig(self)
        E, H = self.config.num_experts, self.config.hidden_size
        I = self.config.moe_intermediate_size
        if _PRESTACK_INT8[0]:  # int8 buffers only; the loader quantizes expert tensors as it reads them
            self.q_gu = torch.empty(E + 1, 2 * I, H, dtype=torch.int8)
            self.q_d = torch.empty(E + 1, H, I, dtype=torch.int8)
            self.s_gu = torch.empty(E + 1, 2 * I, dtype=torch.float32)
            self.s_d = torch.empty(E + 1, H, dtype=torch.float32)
            self.w_gu = self.w_d = None
            return  # expert params stay on the meta device
        self.w_gu = torch.empty(E + 1, 2 * I, H)  # device/dtype from the surrounding build context
        self.w_d = torch.empty(E + 1, H, I)
        for e, ex in enumerate(self.experts):
            ex.gate_proj.weight = torch.nn.Parameter(self.w_gu[e, :I], requires_grad=False)
            ex.up_proj.weight = torch.nn.Parameter(self.w_gu[e, I:], requires_grad=False)
            ex.down_proj.weight = torch.nn.Parameter(self.w_d[e], requires_grad=False)

    block._setup_experts = _setup_experts
    block._prestacked = True


def _get_audio_features_trimmed(self, audio_feats, audio_feats_lengths):
    """The model's _get_audio_features, but each 30 s Whisper chunk is encoded at its own length: the
    reference pads every chunk of a batch to the longest, so a 35 s turn paid for 60 s of encoder."""
    chunk_size = getattr(self.config.audio_config, "whisper_chunk_size", 3000)
    lengths = [int(n) for n in audio_feats_lengths[0].tolist() if int(n) > 0]
    chunks = []
    for seg in audio_feats[0, :sum(lengths)].split(lengths):
        chunks.extend(seg.split(chunk_size))
    out = []
    for c in chunks:
        n = torch.tensor([c.shape[0]], dtype=torch.long, device=audio_feats.device)
        encoded, enc_len = self.model.audio(c[None], n)[:2]
        projected = self.linear_proj_audio(encoded.transpose(-1, -2))
        for module in self.linear_proj_audio:
            if isinstance(module, torch.nn.Conv1d):
                enc_len = (enc_len + 2 * module.padding[0] - module.kernel_size[0]) // module.stride[0] + 1
        out.append(projected[0, :int(enc_len[0])])
    audio_embeds = torch.cat(out, dim=0).unsqueeze(0)
    if self.config.norm_query_embeds:
        audio_embeds = F.normalize(audio_embeds, dim=2)
    return audio_embeds


_FUSED_PREFILL = os.environ.get("MING_FUSED_PREFILL", "1") == "1"
_FUSED_OPS = os.environ.get("MING_FUSED_OPS", "1") == "1"  # Triton RMSNorm + one-token MoE router
_PREFILL_GRAPHS = os.environ.get("MING_PREFILL_GRAPHS", "1") == "1"  # graph-captured prefill, padded to buckets
PREFILL_BUCKETS = (32, 64, 128)  # prompt lengths with a prefill graph (longer prompts run eagerly)
PREFILL_CHUNK = 512  # longer eager prompts run in pieces this long (bounds the attention scratch memory)


def _patch_norms(model):
    """Every BailingMoeV3RMSNorm -> one Triton kernel (the PyTorch version is ~8 kernels per call)."""
    from types import MethodType

    from fused_ops import rms_norm
    n = 0
    for module in model.modules():
        if type(module).__name__ == "BailingMoeV3RMSNorm":
            module.forward = MethodType(lambda self, h: rms_norm(h, self.weight, self.variance_epsilon), module)
            n += 1
    return n


def _kda_padded(a, h, st, pad):
    """Kimi-delta-attention prefill over a prompt padded at the end to a fixed length (for graph capture).

    Same computation as the model's KDA forward, except that padding tokens leave the layer state untouched:
    their write strength (beta) is 0 and their decay gate is pushed to exp(0) = 1, so the recurrent state after
    the padded prompt equals the state after the real tokens; the short-convolution caches are rebuilt from the
    real tokens only (each holds the last W raw inputs). pad: valid [1, Tmax] (1 = real token), conv_idx [W]
    (= n_real + arange(W), positions in [old cache | new inputs])."""
    from fla.ops.kda import fused_recurrent_kda
    T = h.shape[1]
    valid = pad["valid"][:, :T, None]
    state = st.keys
    cq, ck, cv = st.values
    xq, xk, xv = a.q_proj(h), a.k_proj(h), a.v_proj(h)
    q, _ = a.q_conv1d(x=xq, cache=cq, output_final_state=False)
    k, _ = a.k_conv1d(x=xk, cache=ck, output_final_state=False)
    v, _ = a.v_conv1d(x=xv, cache=cv, output_final_state=False)
    g = a.f_proj(h) if a.no_kda_lora else a.f_b_proj(a.f_a_proj(h))
    g = torch.where(valid > 0, g, torch.full_like(g, -1e30))
    beta = a.b_proj(h).float().sigmoid() * valid
    q = q.view(1, T, -1, a.head_k_dim)
    k = k.view(1, T, -1, a.head_k_dim)
    v = v.view(1, T, -1, a.head_dim)
    g = g.view(1, T, -1, a.head_dim)
    o, new_state = fused_recurrent_kda(q=q, k=k, v=v, g=g, beta=beta, A_log=a.A_log, dt_bias=a.dt_bias,
                                       initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True,
                                       use_gate_in_kernel=True, lower_bound=a.lower_bound)
    for c, x in ((cq, xq), (ck, xk), (cv, xv)):
        c.copy_(torch.cat([c, x.transpose(1, 2).to(c.dtype)], 2).index_select(2, pad["conv_idx"]))
    state.copy_(new_state)
    gg = a.g_proj(h) if a.no_kda_lora else a.g_b_proj(a.g_a_proj(h))
    o = a.o_norm(o, gg.view(1, T, -1, a.head_dim))
    return a.o_proj(o.reshape(1, T, -1))


def _patch_kda(model, th):
    """KDA layers take the padded path (_kda_padded) while th._pad is set (prefill-graph capture)."""
    for layer in model.model.layers:
        a = layer.attention
        if not hasattr(a, "mode"):
            continue

        def forward(hidden_states, attention_mask=None, past_key_value=None, _a=a, _orig=a.forward, **kw):
            if th._pad is None:
                return _orig(hidden_states=hidden_states, attention_mask=attention_mask,
                             past_key_value=past_key_value, **kw)
            return _kda_padded(_a, hidden_states, past_key_value.layers[_a.layer_idx], th._pad), None, past_key_value
        a.forward = forward


def _route_one(block, hidden_states):
    """Router for a single token -> (ids incl. the shared expert, fp32 weights, aux) via fused_ops.moe_route."""
    from fused_ops import moe_route
    g = block.gate
    if not hasattr(block, "r_ids"):
        E = g.w32.shape[0]
        block.r_logits = torch.empty(E, dtype=torch.float32, device=DEVICE)
        block.r_ids = torch.empty(g.top_k + 1, dtype=torch.long, device=DEVICE)
        block.r_wts = torch.empty(g.top_k + 1, dtype=torch.float32, device=DEVICE)
        block.r_bias = g.expert_bias.float()
    moe_route(hidden_states.reshape(1, -1), g.w32, block.r_bias, g.n_group, g.topk_group, g.top_k,
              g.routed_scaling_factor, block.r_logits, block.r_ids, block.r_wts)
    aux = (block.r_logits.view(1, 1, -1), block.r_ids[:g.top_k].view(1, 1, -1))
    return block.r_ids, block.r_wts, aux


def _moe_prefill(block, x, topk_idx, topk_weight):
    """Multi-token MoE (prompt prefill) through the grouped Triton kernels: one launch pair per layer
    instead of one small GEMM pair per active expert plus a host sync."""
    from fused_moe import fused_moe_prefill
    N = x.shape[0]
    E = block.shared_id
    ids = torch.cat([topk_idx, E.expand(N, 1).to(topk_idx.dtype)], 1)
    wts = torch.cat([topk_weight.float(), block.one.expand(N, 1)], 1)
    if getattr(block, "w_gu", None) is not None:
        return fused_moe_prefill(x, block.w_gu, block.w_d, ids, wts)
    return fused_moe_prefill(x, block.q_gu, block.q_d, ids, wts, s_gu=block.s_gu, s_d=block.s_d)


def _patch_moe(model):
    """Restack expert weights (+ shared expert as index E) as views, cache fp32 router weights,
    and route single-token MoE through the fused Triton decode kernels."""
    from types import MethodType

    from fused_moe import fused_moe_decode

    n = 0
    for module in model.modules():
        if type(module).__name__ == "BailingMoeV3Gate":
            gate = module
            gate.w32 = gate.weight.detach().float()

            def gate_forward(self, hidden_states):
                hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
                logits = F.linear(hidden_states.float(), self.w32)
                scores = torch.sigmoid(logits)
                _, topk_idx = self.group_limited_topk(scores + self.expert_bias)
                scores = torch.gather(scores, dim=1, index=topk_idx)
                topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
                return topk_idx, topk_weight * self.routed_scaling_factor, logits

            gate.forward = MethodType(gate_forward, gate)
            continue
        if type(module).__name__ != "BailingMoeV3SparseMoeBlock":
            continue
        experts = module.experts
        E = len(experts)
        I, H = experts[0].gate_proj.weight.shape
        if getattr(module, "q_gu", None) is not None and getattr(module, "w_gu", None) is None:
            module.inter = I  # int8 at load time: forward is installed by _quantize_moe_int8
            module.shared_id = torch.tensor([E], device=DEVICE)
            module.one = torch.ones(1, device=DEVICE, dtype=torch.float32)
            n += 1
            continue
        shared = module.shared_experts
        assert shared is not None and shared.gate_proj.weight.shape == (I, H)
        if getattr(module, "w_gu", None) is not None:  # pre-stacked at construction (see _prestack_experts)
            w_gu, w_d = module.w_gu, module.w_d
            todo = [(E, shared)]
        else:
            w_gu = torch.empty(E + 1, 2 * I, H, dtype=DTYPE, device=DEVICE)
            w_d = torch.empty(E + 1, H, I, dtype=DTYPE, device=DEVICE)
            todo = list(enumerate(experts)) + [(E, shared)]
        with torch.no_grad():
            for e, ex in todo:
                assert ex.swiglu_limit is None
                w_gu[e, :I].copy_(ex.gate_proj.weight)
                w_gu[e, I:].copy_(ex.up_proj.weight)
                w_d[e].copy_(ex.down_proj.weight)
                ex.gate_proj.weight = torch.nn.Parameter(w_gu[e, :I], requires_grad=False)
                ex.up_proj.weight = torch.nn.Parameter(w_gu[e, I:], requires_grad=False)
                ex.down_proj.weight = torch.nn.Parameter(w_d[e], requires_grad=False)
        module.w_gu, module.w_d, module.inter = w_gu, w_d, I
        module.shared_id = torch.tensor([E], device=DEVICE)
        module.one = torch.ones(1, device=DEVICE, dtype=torch.float32)
        orig_forward = module.forward

        def forward(self, hidden_states, _orig=orig_forward):
            bsz, seq_len, h = hidden_states.shape
            if bsz * seq_len != 1 and not _FUSED_PREFILL:
                return _orig(hidden_states)
            if bsz * seq_len == 1 and _FUSED_OPS:
                ids, wts, aux = _route_one(self, hidden_states)
                y = fused_moe_decode(hidden_states.view(1, h), self.w_gu, self.w_d, ids, wts)
                return y.view(bsz, seq_len, h), aux
            topk_idx, topk_weight, router_logits = self.gate(hidden_states)
            aux = (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))
            if bsz * seq_len != 1:
                return _moe_prefill(self, hidden_states.view(-1, h), topk_idx, topk_weight).view(bsz, seq_len, h), aux
            ids = torch.cat([topk_idx.view(-1), self.shared_id])
            wts = torch.cat([topk_weight.view(-1).float(), self.one])
            y = fused_moe_decode(hidden_states.view(1, h), self.w_gu, self.w_d, ids, wts)
            return y.view(bsz, seq_len, h), aux

        module.forward = MethodType(forward, module)
        n += 1
    torch.cuda.empty_cache()
    return n


def _quantize_moe_int8(model):
    """Weight-only int8 (per-output-row scales) for all routed + shared experts, in place, layer by layer.
    Halves expert VRAM (~14 GB -> ~7 GB) and speeds up the bandwidth-bound decode kernel. Single-token
    steps use the int8 fused kernels; multi-token (prefill) passes dequantize the layer into a shared
    bf16 scratch buffer and run the reference per-expert loop."""
    from types import MethodType

    from fused_moe import fused_moe_decode, quantize_rows_int8

    n = 0
    for module in model.modules():
        if type(module).__name__ != "BailingMoeV3SparseMoeBlock" or not hasattr(module, "shared_id"):
            continue
        if getattr(module, "w_gu", None) is None:  # quantized while loading: add the shared expert (slot E)
            q_gu, q_d, s_gu, s_d = module.q_gu, module.q_d, module.s_gu, module.s_d
            E1, I = q_gu.shape[0], module.inter
            sh = module.shared_experts
            with torch.no_grad():
                for w, q, s, off in ((sh.gate_proj.weight, q_gu, s_gu, 0), (sh.up_proj.weight, q_gu, s_gu, I),
                                     (sh.down_proj.weight, q_d, s_d, 0)):
                    qq, ss = quantize_rows_int8(w)
                    q[E1 - 1, off:off + qq.shape[0]].copy_(qq)
                    s[E1 - 1, off:off + qq.shape[0]].copy_(ss)
        else:
            w_gu, w_d = module.w_gu, module.w_d
            E1 = w_gu.shape[0]
            q_gu = torch.empty(w_gu.shape, dtype=torch.int8, device=DEVICE)
            q_d = torch.empty(w_d.shape, dtype=torch.int8, device=DEVICE)
            s_gu = torch.empty(w_gu.shape[:2], dtype=torch.float32, device=DEVICE)
            s_d = torch.empty(w_d.shape[:2], dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                for i in range(0, E1, 16):  # chunked to bound the fp32 temporaries
                    q_gu[i:i + 16], s_gu[i:i + 16] = quantize_rows_int8(w_gu[i:i + 16])
                    q_d[i:i + 16], s_d[i:i + 16] = quantize_rows_int8(w_d[i:i + 16])
            del w_gu, w_d
        # drop every reference to bf16 expert weights (the routed ones were views into the stacks)
        module.experts = torch.nn.ModuleList()
        module.shared_experts = None
        module.w_gu = module.w_d = None
        module.q_gu, module.q_d, module.s_gu, module.s_d = q_gu, q_d, s_gu, s_d
        module.n_routed = E1 - 1

        def forward(self, hidden_states):
            bsz, seq_len, h = hidden_states.shape
            if bsz * seq_len == 1 and _FUSED_OPS:
                ids, wts, aux = _route_one(self, hidden_states)
                y = fused_moe_decode(hidden_states.view(1, h), self.q_gu, self.q_d, ids, wts,
                                     s_gu=self.s_gu, s_d=self.s_d)
                return y.view(bsz, seq_len, h), aux
            topk_idx, topk_weight, router_logits = self.gate(hidden_states)
            aux = (router_logits.view(bsz, seq_len, -1), topk_idx.view(bsz, seq_len, -1))
            if bsz * seq_len == 1:
                ids = torch.cat([topk_idx.view(-1), self.shared_id])
                wts = torch.cat([topk_weight.view(-1).float(), self.one])
                y = fused_moe_decode(hidden_states.view(1, h), self.q_gu, self.q_d, ids, wts,
                                     s_gu=self.s_gu, s_d=self.s_d)
                return y.view(bsz, seq_len, h), aux
            if _FUSED_PREFILL:
                return _moe_prefill(self, hidden_states.view(-1, h), topk_idx, topk_weight).view(bsz, seq_len, h), aux
            return self._prefill(hidden_states.view(-1, h), topk_idx, topk_weight).view(bsz, seq_len, h), aux

        def _prefill(self, x, topk_idx, topk_weight):
            N, K = topk_idx.shape
            E1, I = self.q_gu.shape[0], self.inter
            ids = torch.cat([topk_idx, torch.full((N, 1), E1 - 1, device=x.device, dtype=topk_idx.dtype)], 1)
            wts = torch.cat([topk_weight.float(), torch.ones(N, 1, device=x.device)], 1)
            flat = ids.reshape(-1)
            order = flat.argsort()
            counts = torch.bincount(flat, minlength=E1).cpu().tolist()
            tok = order // (K + 1)
            xs = x[tok]
            out = torch.empty(xs.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
            starts = [0]
            for c in counts:
                starts.append(starts[-1] + c)
            CH = 16  # dequantize a few experts at a time (bounded scratch, skips unused ones)
            for c0 in range(0, E1, CH):
                c1 = min(E1, c0 + CH)
                if starts[c1] == starts[c0]:
                    continue
                d_gu = (self.q_gu[c0:c1].half() * self.s_gu[c0:c1, :, None].half()).to(DTYPE)
                d_d = (self.q_d[c0:c1].half() * self.s_d[c0:c1, :, None].half()).to(DTYPE)
                for e in range(c0, c1):
                    if counts[e] == 0:
                        continue
                    seg = xs[starts[e]:starts[e + 1]]
                    gu = F.linear(seg, d_gu[e - c0])
                    a = F.silu(gu[:, :I]) * gu[:, I:]
                    out[starts[e]:starts[e + 1]] = F.linear(a, d_d[e - c0])
                del d_gu, d_d
            y = torch.zeros(N, x.shape[1], device=x.device, dtype=torch.float32)
            y.index_add_(0, tok, out.float() * wts.reshape(-1)[order][:, None])
            return y.to(x.dtype)

        module.forward = MethodType(forward, module)
        module._prefill = MethodType(_prefill, module)
        torch.cuda.empty_cache()
        n += 1
    return n


class _KDAState:
    """Cache layer for a Kimi-delta-attention layer, backed by fixed buffers (setters copy in place)."""

    def __init__(self, state, convs):
        self._state, self._convs = state, convs

    @property
    def keys(self):
        return self._state

    @keys.setter
    def keys(self, v):
        if v is not None and v is not self._state:
            self._state.copy_(v)

    @property
    def values(self):
        return self._convs

    @values.setter
    def values(self, v):
        if v is None:
            return
        for dst, src in zip(self._convs, v):
            if src is not None and src is not dst:
                dst.copy_(src)


class _MLAState:
    def __init__(self, k, v):
        self.k, self.v = k, v
        self.keys = self.values = None


def _make_static_cache_class():
    from transformers.cache_utils import Cache

    class StaticHybridCache(Cache):
        """Fixed-size cache for the hybrid KDA/MLA Thinker so decode steps can be graph-captured."""

        def __init__(self, layers):  # deliberately skips Cache.__init__
            self.layers = layers
            self.layer_class_to_replicate = None
            self.offloading = False
            self.attn_len = None

        def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
            st = self.layers[layer_idx]
            pos = cache_kwargs["cache_position"]
            st.k.index_copy_(2, pos, key_states)
            st.v.index_copy_(2, pos, value_states)
            L = self.attn_len
            return st.k[:, :, :L], st.v[:, :, :L]

        def get_seq_length(self, layer_idx=0):
            return 0

    return StaticHybridCache


def _want_int8_experts() -> bool:
    """MING_INT8_EXPERTS=1/0/auto (default auto: int8 when the GPU has < 23 GiB, i.e. 16-20 GB cards)."""
    v = os.environ.get("MING_INT8_EXPERTS", "auto").lower()
    if v in ("1", "true", "yes"):
        return True
    if v in ("0", "false", "no"):
        return False
    return torch.cuda.get_device_properties(DEVICE).total_memory < 23 * 2**30


class Thinker:
    int8_experts = None

    def __init__(self, model_dir: Path = MODEL_DIR, system_prompt: str = DEFAULT_SYSTEM_PROMPT,
                 use_graphs: bool = True):
        from transformers import AutoModelForCausalLM, AutoProcessor
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
        t = time.time()
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
        self.tok = self.processor.tokenizer
        if os.environ.get("MING_HF_LOAD") == "1":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_dir, trust_remote_code=True, dtype=DTYPE, device_map=str(DEVICE)
            ).eval()
        else:  # stream weights straight to the GPU (HF's loader peaks at ~14.6 GiB host RAM -> swap)
            from direct_load import load_remote_model_direct
            _PRESTACK_INT8[0] = self.int8_experts = _want_int8_experts()
            _prestack_experts(model_dir)
            self.model, info = load_remote_model_direct(
                model_dir, device=str(DEVICE), dtype=DTYPE,
                special=_int8_expert_loaders if self.int8_experts else None)
            log(f"thinker weights: build {info['build_s']}s, read+upload {info['gb']} GB in {info['load_s']}s")
            if info["unexpected"]:
                log(f"thinker load: {info['unexpected']} unused checkpoint tensors, e.g. {info['unexpected_examples']}")
        t_w = time.time()
        from types import MethodType
        self.model._get_audio_features = MethodType(_get_audio_features_trimmed, self.model)
        n = _patch_moe(self.model)
        if _FUSED_OPS:
            _patch_norms(self.model)
        if getattr(self, "int8_experts", None) is None:
            self.int8_experts = _want_int8_experts()
        if self.int8_experts:
            _quantize_moe_int8(self.model)
        t_p = time.time()
        import whisper

        self._mel = whisper.log_mel_spectrogram
        self.n_mels = getattr(self.model.config.audio_config, "n_mels", 128)
        self.system_prompt = system_prompt
        self.tools = None  # tools.Toolbox offered to the model (part of the system prompt: change -> reset)
        self._tools_sig = None
        self._sys_cache = {}  # system block text -> saved cache state (_prefill_system)
        # The MLA cache is ~60 MB per 1k tokens, allocated up front. A bigger context doesn't slow short ones down
        # (decode/prefill graphs exist per attention-length bucket and only read the filled part), so it's sized
        # by VRAM: 8k on a 24 GB card (16k fits, but leaves the BLAS/kernel libraries too little working memory
        # next to the Talker: they allocate outside PyTorch and fail hard), 16k from 32 GB, 4k in int8 mode.
        big = torch.cuda.get_device_properties(DEVICE).total_memory >= 30 * 2**30
        default = "4096" if self.int8_experts else "16384" if big else "8192"
        self.max_ctx = int(os.environ.get("MING_MAX_CTX", default))
        self._build_cache()
        self.graph = None
        self._pad = None
        self.prefill_graphs = {}
        _patch_kda(self.model, self)
        self.reset()
        t_c = time.time()
        if use_graphs:
            self._capture()
            if _PREFILL_GRAPHS:
                self._capture_prefill()
        log(f"Thinker loaded in {time.time() - t:.1f}s (processor+weights {t_w - t:.1f}s, MoE restack {t_p - t_w:.1f}s, "
            f"setup {t_c - t_p:.1f}s, graphs {time.time() - t_c:.1f}s), {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    # ---- static state ----
    def _build_cache(self):
        m = self.model.model
        self.n_layers = m.config.num_hidden_layers
        self._kda_attn = [l.attention for l in m.layers[: self.n_layers] if hasattr(getattr(l, "attention", None), "mode")]
        layers = []
        for layer in m.layers[: self.n_layers]:
            a = layer.attention
            if layer.attention_layer_type == "attention":
                k = torch.zeros(1, a.num_heads, self.max_ctx, a.qk_head_dim, dtype=DTYPE, device=DEVICE)
                v = torch.zeros(1, a.num_heads, self.max_ctx, a.v_head_dim, dtype=DTYPE, device=DEVICE)
                layers.append(_MLAState(k, v))
            else:
                st = torch.zeros(1, a.num_heads, a.head_k_dim, a.head_dim, dtype=torch.float32, device=DEVICE)
                convs = tuple(torch.zeros(1, c.hidden_size, c.kernel_size[0], dtype=DTYPE, device=DEVICE)
                              for c in (a.q_conv1d, a.k_conv1d, a.v_conv1d))
                layers.append(_KDAState(st, convs))
        self._layers = layers
        self.cache = _make_static_cache_class()(layers)
        self.ar = torch.arange(self.max_ctx, device=DEVICE)
        self.tok_buf = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)
        self.pos_buf = torch.zeros(1, dtype=torch.long, device=DEVICE)

    def _clear_state(self):
        for st in self._layers:
            if isinstance(st, _MLAState):
                st.k.zero_()
                st.v.zero_()
            else:
                st.keys.zero_()
                for c in st.values:
                    c.zero_()

    def _forward(self, h, pos, last=None):
        """h: [1, T, H] input embeddings, pos: [T] positions. Returns logits of the last position [1, V]
        (or of position `last`, a 1-element index tensor, when the prompt is padded)."""
        m = self.model.model
        position_ids = pos[None]
        cos_sin = m.rotary_emb(h, position_ids)
        L = self.cache.attn_len
        mask = torch.zeros(pos.shape[0], L, dtype=h.dtype, device=DEVICE)
        mask.masked_fill_(self.ar[None, :L] > pos[:, None], torch.finfo(h.dtype).min)
        mask = mask[None, None]
        for layer in m.layers[: self.n_layers]:
            h = layer(h, attention_mask=mask, position_ids=position_ids, past_key_value=self.cache,
                      cache_position=pos, use_cache=True, position_embeddings=cos_sin)[0]
        h = m.norm(h[:, -1:] if last is None else h.index_select(1, last))
        return self.model.lm_head(h)[:, -1]

    def _decode_step(self):
        h = self.model.model.word_embeddings(self.tok_buf)
        logits = self._forward(h, self.pos_buf)
        self.tok_buf.copy_(logits.argmax(-1, keepdim=True))
        self.pos_buf.add_(1)

    def _buckets(self):
        b, out = 256, []
        while b < self.max_ctx:
            out.append(b)
            b *= 2
        return out + [self.max_ctx]

    def _capture(self):
        """One decode graph per attention-length bucket (attention only reads the filled prefix)."""
        s = torch.cuda.Stream(device=DEVICE)
        s.wait_stream(torch.cuda.current_stream())
        self.graphs = {}
        pool = torch.cuda.graph_pool_handle()
        with torch.inference_mode(), torch.cuda.stream(s):
            for L in self._buckets():
                self.cache.attn_len = L
                self.pos_buf.fill_(1)
                for _ in range(2):
                    self._decode_step()
                s.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool, stream=s):
                    self._decode_step()
                s.synchronize()
                self.graphs[L] = g
        torch.cuda.current_stream().wait_stream(s)
        self.graph = True
        self._clear_state()

    def _capture_prefill(self):
        """One prefill graph per (padded prompt length, attention-length bucket). Eager prefill of a short prompt
        is CPU-bound (~700 kernel launches: ~45 ms for 32 tokens while the GPU needs a fraction of that)."""
        m = self.model.model
        Tmax, W = max(PREFILL_BUCKETS), self._kda_attn[0].conv_size
        self.p_h = torch.zeros(1, Tmax, m.config.hidden_size, dtype=DTYPE, device=DEVICE)
        self.p_pos = torch.zeros(Tmax, dtype=torch.long, device=DEVICE)
        self.p_valid = torch.zeros(1, Tmax, dtype=torch.float32, device=DEVICE)
        self.p_conv_idx = torch.arange(W, dtype=torch.long, device=DEVICE)
        self.p_last = torch.zeros(1, dtype=torch.long, device=DEVICE)
        self.p_logits = None
        self.p_W = W
        s = torch.cuda.Stream(device=DEVICE)
        s.wait_stream(torch.cuda.current_stream())
        pool = torch.cuda.graph_pool_handle()
        self._pad = dict(valid=self.p_valid, conv_idx=self.p_conv_idx)
        try:
            with torch.inference_mode(), torch.cuda.stream(s):
                for Tb in PREFILL_BUCKETS:
                    for L in self._buckets():
                        self.cache.attn_len = L
                        self.p_pos[:Tb].copy_(torch.arange(Tb, device=DEVICE))
                        self.p_valid.fill_(1.0)
                        self.p_last.fill_(Tb - 1)

                        def run():
                            out = self._forward(self.p_h[:, :Tb], self.p_pos[:Tb], self.p_last)
                            if self.p_logits is None:
                                self.p_logits = torch.empty_like(out)
                            self.p_logits.copy_(out)
                        for _ in range(2):
                            run()
                        s.synchronize()
                        g = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g, pool=pool, stream=s):
                            run()
                        s.synchronize()
                        self.prefill_graphs[(Tb, L)] = g
        finally:
            self._pad = None
        torch.cuda.current_stream().wait_stream(s)
        self._clear_state()

    def reset(self, system_prompt: str | None = None):
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self._clear_state()
        self.cur_len = 0
        self.pending = None  # text to prepend before the next human turn
        self.history: list[dict] = []
        self.epoch = getattr(self, "epoch", 0) + 1  # the cache was cleared: older snapshots can't be restored as-is
        self.rebuild = False  # next turn re-enters the history as text (after restoring across a reset)

    def snapshot(self):
        """Cheap copy of the conversation state (recurrent KDA states + positions) for rollback."""
        kda = [(st.keys.clone(), tuple(c.clone() for c in st.values)) if isinstance(st, _KDAState) else None
               for st in self._layers]
        return dict(cur_len=self.cur_len, pending=self.pending, n_hist=len(self.history), kda=kda, epoch=self.epoch)

    @contextmanager
    def isolated(self):
        """Run a stateless request (API chat) on the fast cached path without disturbing the live
        conversation: saves the whole state (recurrent states + the filled MLA KV prefix), restores it after.
        The caller must hold the GPU lock."""
        saved = self._save_state()
        meta = (self.pending, list(self.history), self.system_prompt, self.epoch, self.rebuild, self.tools, self._tools_sig)
        try:
            yield self
        finally:
            self._load_state(saved)
            self.pending, self.history, self.system_prompt, self.epoch, self.rebuild, self.tools, self._tools_sig = meta

    def _save_state(self):
        """Copy of the whole cache state: recurrent KDA states + the filled part of the MLA KV cache."""
        n = self.cur_len
        return n, [(st.keys.clone(), tuple(c.clone() for c in st.values)) if isinstance(st, _KDAState)
                   else (st.k[:, :, :n].clone(), st.v[:, :, :n].clone()) for st in self._layers]

    def _load_state(self, saved):
        n, layers = saved
        for st, (a, b) in zip(self._layers, layers):
            if isinstance(st, _KDAState):
                st.keys.copy_(a)
                for dst, src in zip(st.values, b):
                    dst.copy_(src)
            else:
                st.k[:, :, :n].copy_(a)
                st.v[:, :, :n].copy_(b)
        self.cur_len = n

    def _prefill_system(self):
        """Start a fresh context with the system prompt. Its cache state is kept (two most recent prompts), so a new
        chat doesn't prefill the persona + tool definitions (~900 tokens) again."""
        text = self._system_block()
        hit = self._sys_cache.pop(text, None)
        if hit is not None:
            self._load_state(hit)
        else:
            ids = self.tok(text, return_tensors="pt", add_special_tokens=True).input_ids.to(DEVICE)
            self._prefill(ids, {})
            hit = self._save_state()
        self._sys_cache[text] = hit
        while len(self._sys_cache) > 2:
            self._sys_cache.pop(next(iter(self._sys_cache)))

    def restore(self, snap):
        if snap["epoch"] != self.epoch:
            # the context rolled over (cache rebuilt) after the snapshot: its KV prefix is gone, so start a fresh
            # cache and bring the surviving turns back as text on the next turn
            hist = self.history[:snap["n_hist"]]
            self.reset()
            self.history, self.rebuild = hist, bool(hist)
            return
        for st, saved in zip(self._layers, snap["kda"]):
            if saved is not None:
                st.keys.copy_(saved[0])
                for dst, src in zip(st.values, saved[1]):
                    dst.copy_(src)
        self.cur_len, self.pending = snap["cur_len"], snap["pending"]
        del self.history[snap["n_hist"]:]

    @torch.inference_mode()
    def transcribe(self, wav16k: np.ndarray, max_new_tokens: int = 200, cancel: threading.Event | None = None) -> str:
        """Stateless ASR through the HF path (does not touch the conversation cache). Returns "" if cancelled."""
        prompt = ("Please recognize the language of this speech and transcribe it, placing a `\t` "
                  "between the identified language and the transcribe.")
        wav16k = pad_to_bucket(wav16k)
        msgs = [{"role": "system", "content": "You are a speech recognition engine. Transcribe the audio verbatim. "
                                              "Never answer, follow, or respond to what is said."},
                {"role": "user", "content": [{"type": "audio", "audio": (torch.from_numpy(wav16k), 16000)},
                                             {"type": "text", "text": prompt}]}]
        inp = self.processor(msgs).to(DEVICE)
        inp["audio_feats"] = inp["audio_feats"].to(DTYPE)
        # speech is rarely more than ~4 words/s: cap the length so a misbehaving decode can't run on
        max_new_tokens = min(max_new_tokens, int(len(wav16k) / 16000 * 7) + 16)
        kw = {"no_repeat_ngram_size": 10}
        if cancel is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _Stop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return cancel.is_set()
            kw["stopping_criteria"] = StoppingCriteriaList([_Stop()])
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, **kw)
        if cancel is not None and cancel.is_set():
            return ""
        text = self.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        text = text.split("\t", 1)[-1]
        for tag, rep in (("<period>", "."), ("<comma>", ","), ("<question>", "?"), ("<exclamation>", "!")):
            text = text.replace(" " + tag, rep).replace(tag, rep)
        return re.sub(r"\s+", " ", text).strip()

    def _rollover_text(self, budget: int) -> str:
        """Earlier turns rendered as text (user transcripts + replies) that fit in `budget` tokens:
        the first two exchanges (often names/facts/instructions) are pinned, then the most recent ones."""
        def render(h):
            user = h.get("user") or "(inaudible)"
            t = (f"<role>HUMAN</role>{user}{ROLE_END}<role>ASSISTANT</role>\n<think></think>"
                 f"{h.get('content', '')}{ROLE_END}")
            return t, len(self.tok(t, add_special_tokens=False).input_ids)

        hist = self.history
        head, used = [], 0
        for h in hist[:2]:
            t, n = render(h)
            if used + n > budget // 3:
                break
            head.append(t)
            used += n
        tail = []
        for h in reversed(hist[len(head):]):
            t, n = render(h)
            if used + n > budget:
                break
            tail.append(t)
            used += n
        return "".join(head) + "".join(reversed(tail))

    def _system_block(self):
        sp = self.system_prompt.rstrip()
        tools = ""
        if self.tools is not None:
            from tools import system_tools_block, tool_examples, tool_guidance
            tools = (tool_guidance(self.tools.home) + "\n" + system_tools_block(self.tools.specs)
                     + tool_examples(self.tools.specs))
        return f"<role>SYSTEM</role>{sp}\n{tools}detailed thinking off{ROLE_END}"

    def set_tools(self, tools):
        """Offer a different set of tools (None = none). They live in the cached system prompt, so when they (or the
        user's location) change, the context is rebuilt from the conversation's text on the next turn."""
        sig = None if tools is None else tools.signature()
        changed = sig != self._tools_sig
        self.tools, self._tools_sig = tools, sig
        if changed and self.cur_len:
            hist = self.history
            self.reset()
            self.history, self.rebuild = hist, bool(hist)

    def ctx_len(self):
        return self.cur_len

    def needs_compact(self) -> bool:
        reserve = TOOL_RESERVE if self.tools is not None else 0
        return self.cur_len > 0 and self.cur_len + reserve + COMPACT_HEADROOM > self.max_ctx

    @torch.inference_mode()
    def compact(self):
        """Roll the context over now, between turns, instead of at the start of the next one (where rebuilding
        it from text would delay the reply): fresh cache = system prompt + the conversation so far as text.
        The caller must hold the GPU lock."""
        t = time.time()
        n0 = self.cur_len
        text = self._rollover_text(budget=min(self.max_ctx // 2, ROLLOVER_MAX))
        hist = self.history
        self.reset()
        self.history = hist
        self._prefill_system()
        if text:
            self._prefill(self.tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE), {})
        torch.cuda.synchronize()
        log(f"context compacted between turns: {n0} -> {self.cur_len} tokens in {time.time() - t:.2f}s")

    def _prefill(self, ids, feats):
        m = self.model
        h = m.model.word_embeddings(ids)
        if feats:
            audio = m._get_audio_features(feats["audio_feats"], feats["audio_feats_lengths"]).to(h.dtype)
            amask = (ids == m.config.audio_token_id).unsqueeze(-1).expand_as(h)
            h = h.masked_scatter(amask, audio)
        T = ids.shape[1]
        # KDA: the chunked kernels are a dozen small launches per layer; up to a few hundred tokens the single
        # recurrent kernel is faster (first turn with the system prompt: 67 -> 56 ms to the first token)
        for a in self._kda_attn:
            a.mode = "fused_recurrent" if T <= 192 else "chunk"
        Tb = next((b for b in PREFILL_BUCKETS if b >= T), None)
        if self.prefill_graphs and Tb is not None and self.cur_len + Tb <= self.max_ctx:
            L = next(b for b in self._buckets() if b >= self.cur_len + Tb)
            self.p_h[:, :T].copy_(h)
            self.p_h[:, T:Tb].zero_()
            self.p_pos[:Tb].copy_(torch.arange(self.cur_len, self.cur_len + Tb, device=DEVICE))
            self.p_valid.zero_()
            self.p_valid[:, :T] = 1.0
            self.p_conv_idx.copy_(torch.arange(T, T + self.p_W, device=DEVICE))
            self.p_last.fill_(T - 1)
            self.prefill_graphs[(Tb, L)].replay()
            logits = self.p_logits
        else:
            # long prompts (a context rollover) in chunks: attention scores for all of it at once take ~1 GB
            for s in range(0, T, PREFILL_CHUNK):
                n = min(PREFILL_CHUNK, T - s)
                for a in self._kda_attn:
                    a.mode = "fused_recurrent" if n <= 192 else "chunk"
                pos = torch.arange(self.cur_len + s, self.cur_len + s + n, device=DEVICE)
                self.cache.attn_len = self.cur_len + s + n
                logits = self._forward(h[:, s:s + n], pos)
        self.cur_len += T
        return int(logits[0].argmax())

    def _step(self, tok):
        """Feed `tok` at position cur_len and return the next greedy token."""
        self.tok_buf.fill_(tok)
        self.pos_buf.fill_(self.cur_len)
        L = next(b for b in self._buckets() if b > self.cur_len)
        if self.graph is not None:
            self.graphs[L].replay()
        else:
            self.cache.attn_len = L
            self._decode_step()
        self.cur_len += 1
        return int(self.tok_buf.item())

    def render_history(self, turns: list[tuple[str, str]]) -> str:
        """Earlier (user, assistant) text exchanges in the chat format, e.g. from an API request."""
        return "".join(f"<role>HUMAN</role>{u}{ROLE_END}<role>ASSISTANT</role>\n<think></think>{a}{ROLE_END}"
                       for u, a in turns)

    @torch.inference_mode()
    def stream(self, wav16k: np.ndarray, cancel: threading.Event | None = None, max_new_tokens: int = 400,
               user_text: str | None = None, history_text: str = "", events: bool = False, gate=None):
        """Yield text deltas of the assistant reply to one user utterance (16 kHz mono float32) and/or text.
        history_text: earlier turns (render_history) inserted after the system prompt on a fresh context.
        With self.tools set, the model may call tools first; they run here (GPU idle meanwhile) and the reply
        continues from their results. events=True also yields {"type": "tool_call" | "tool_result", ...} dicts.
        gate(): called before a tool that changes something (tools.SIDE_EFFECTS); blocks until the reply is
        confirmed and returns False if it was dropped (speculative replies), which ends the reply there."""
        cancel = cancel or threading.Event()
        feats = {}
        human = ""
        heard = {}

        def said() -> str:  # what the user said (only transcribed when a tool needs it: ~0.2 s)
            if "text" not in heard:
                spoken = self.transcribe(wav0, max_new_tokens=60) if wav0 is not None else ""
                heard["text"] = (spoken + " " + (user_text or "")).strip()
            return heard["text"]
        wav0 = wav16k
        if wav16k is not None:
            wav16k = pad_to_bucket(wav16k)
            mel = self._mel(torch.from_numpy(np.ascontiguousarray(wav16k, dtype=np.float32)).to(DEVICE),
                            n_mels=self.n_mels)
            mel = mel.transpose(0, 1).unsqueeze(0).contiguous()
            n_audio = self.processor._num_audio_tokens(mel.shape[1])
            human = "<|audio_start|>" + "<|audio_pad|>" * n_audio + "<|audio_end|>"
            feats = dict(audio_feats=mel.to(DTYPE),
                         audio_feats_lengths=torch.tensor([[mel.shape[1]]], dtype=torch.long))
        human += user_text or ""
        # With tools, the reply starts from a private note of the local time (asked "what time is it?", the model
        # would rather offer to check than call get_current_time; the system prompt can't hold a changing time).
        note = time.strftime("(Today is %A, %B %d, %Y. It's %I:%M %p now.)").replace(" 0", " ") \
            if self.tools is not None else ""
        tail = f"<role>HUMAN</role>{human}{ROLE_END}<role>ASSISTANT</role>\n<think>{note}</think>"
        n_tail = len(self.tok(tail, add_special_tokens=False).input_ids)
        rollover = ""
        reserve = TOOL_RESERVE if self.tools is not None else 0
        if self.tools is not None:
            self.tools.new_turn(said)  # (a risky action asked about in an earlier turn may now be confirmed)
        budget = min(self.max_ctx // 2, ROLLOVER_MAX) - n_tail
        if self.cur_len and self.cur_len + n_tail + max_new_tokens + reserve + 8 > self.max_ctx:
            rollover = self._rollover_text(budget=budget)
            log(f"context limit reached: restarting from the last turns as text ({len(rollover)} chars)")
            hist = self.history
            self.reset()
            self.history = hist
        elif self.cur_len == 0 and self.rebuild and self.history:
            rollover = self._rollover_text(budget=budget)
        self.rebuild = False
        if self.cur_len == 0:
            self._prefill_system()
            prefix = rollover or history_text
        else:
            prefix = self.pending or ""
        ids = self.tok(prefix + tail, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
        max_new_tokens = min(max_new_tokens, self.max_ctx - self.cur_len - ids.shape[1] - 1)
        # Primed by earlier "I don't have access to your email" replies, the model sometimes forgets its tools and
        # says it can't look anything up. So a reply that opens like a refusal is held back to its first sentence
        # end; if that claims no live data / search / weather, the reply starts over as that tool call instead.
        tool_names = {s.get("function", s).get("name") for s in self.tools.specs} if self.tools is not None else set()
        snap = self.snapshot() if tool_names & {"web_search", "home_assistant"} else None
        tok = self._prefill(ids, feats)
        held, retry, forced, quiet = snap is not None, None, None, False
        gen: list[int] = []  # the spoken reply (text of every round, tool calls excluded)
        emitted = ""
        seen, used_tools, ended = set(), False, False
        answer_from = None  # where the answer from tool results starts in `emitted`
        capped = True  # ANSWER_WORDS applies (until the transcript shows the user asked for something long)
        try:
            for rnd in range(MAX_TOOL_ROUNDS + 1):
                call: list[int] = forced or []
                forced = None
                n = 0
                room = self.max_ctx - self.cur_len - 2
                limit = min(max_new_tokens, room)
                # past the limit, a reply may run on a little to finish its sentence (not stop at "But the hardest")
                hard = min(max_new_tokens + (60 if max_new_tokens >= 100 else 0), room)
                while tok not in EOS_IDS and n < hard and not cancel.is_set():
                    n += 1
                    if tok == TOOL_CALL_ID and rnd == MAX_TOOL_ROUNDS:
                        break  # yet another call after the last allowed round: end the reply (never speak it)
                    if call or (tok == TOOL_CALL_ID and self.tools is not None):
                        call.append(tok)  # a function call: collected, not spoken
                    else:
                        gen.append(tok)
                        full = self.tok.decode(gen, skip_special_tokens=True)
                        if held and not full.endswith("�"):
                            head = full.strip()
                            if len(head.split()) >= 4 or SENTENCE_END.search(full):
                                music = "home_assistant" in tool_names
                                # A plain device command ("turn off the bedroom lights") answered without the tool:
                                # what the user said decides (a short utterance: transcribing it takes ~0.1 s).
                                if music and (wav0 is None or len(wav0) <= DIRECT_MAX_S * 16000) \
                                        and hasattr(self.tools, "direct") and self.tools.direct(said()):
                                    held, retry = False, "home_assistant"
                                    break
                                if not (REFUSAL_START.match(head) or music and MUSIC_DODGE_START.match(head)) \
                                        or len(head.split()) > 30:
                                    held = False
                                elif SENTENCE_END.search(full):
                                    held = False
                                    if music and MUSIC_DODGE.search(head):
                                        retry = "home_assistant"
                                        break
                                    m = CANT_LOOK_UP.search(head)
                                    if m and "web_search" in tool_names:
                                        weather = re.search(r"weather|forecast", m.group(), re.I)
                                        retry = "get_weather" if weather and "get_weather" in tool_names \
                                            else "web_search"
                                        break
                        if not held and not full.endswith("�") and len(full) > len(emitted):
                            yield full[len(emitted):]
                            emitted = full
                            # Search results tempt it into reading out whole lists: after a tool, stop at the
                            # first sentence end past TOOL_ANSWER_WORDS words. A spoken monologue is no better
                            # without tools: stop past ANSWER_WORDS, unless asked for a story, steps, details...
                            # (the transcript is only made then, while the audio already sent is still playing).
                            if SENTENCE_END.search(full):
                                if n >= limit:
                                    break
                                if answer_from is not None:
                                    if len(full[answer_from:].split()) >= TOOL_ANSWER_WORDS:
                                        break
                                elif capped and len(full.split()) >= ANSWER_WORDS:
                                    capped = False
                                    if not LONG_ASK.search(said()):
                                        log(f"long answer: ending it at {len(full.split())} words")
                                        break
                    tok = self._step(tok)
                if retry and not cancel.is_set():
                    log(f"reply dodged its tool ({self.tok.decode(gen, skip_special_tokens=True)!r}): "
                        f"calling {retry} instead")
                    self.restore(snap)
                    arg = {"get_weather": "location", "home_assistant": "command"}.get(retry, "query")
                    forced = self.tok(f"<tool_call>{retry}\n<arg_key>{arg}</arg_key>\n<arg_value>",
                                      add_special_tokens=False).input_ids
                    tok = self._prefill(torch.cat([ids, torch.tensor([forced], device=DEVICE)], 1), feats)
                    gen, retry = [], None
                    continue  # (the next round starts inside that call)
                if held:  # ended (or a tool call began) while the opening was still held back: say it now
                    held = False
                    full = self.tok.decode(gen, skip_special_tokens=True)
                    if len(full) > len(emitted):
                        yield full[len(emitted):]
                        emitted = full
                if not call and not used_tools and tok in EOS_IDS and not cancel.is_set():
                    # "The eggs are off your list" without calling lists: make it true with that call (quietly,
                    # it was already said), continuing the reply as if the model had made it
                    full = self.tok.decode(gen, skip_special_tokens=True)
                    claim = next(((t, a) for rx, t, a in CLAIMS if t in tool_names and rx.search(full)), None)
                    if claim:
                        log(f"reply claimed a {claim[0]} {claim[1]} it didn't do ({full!r}): calling it now")
                        forced = self.tok(f"\n<tool_call>{claim[0]}\n<arg_key>action</arg_key>\n<arg_value>"
                                          f"{claim[1]}</arg_value>\n<arg_key>", add_special_tokens=False).input_ids
                        tok = self._prefill(torch.tensor([forced], device=DEVICE), {})
                        quiet = True
                        continue
                if not call or cancel.is_set():
                    break
                from tools import parse_tool_calls
                calls = parse_tool_calls(self.tok.decode(call, skip_special_tokens=False))[:3]
                calls = [c for c in calls if repr(c) not in seen]  # the same call again won't go differently
                if not calls:
                    break
                seen.update(repr(c) for c in calls)
                used_tools = True
                from tools import SIDE_EFFECTS
                if gate is not None and any(n in SIDE_EFFECTS for n, _ in calls) and not gate():
                    break  # a speculative reply that was dropped: never switch lights or start tasks for it
                results, says = [], []
                for name, args in calls:
                    if events:
                        yield {"type": "tool_call", "name": name, "args": args}
                    text, info = self.tools.run(name, args, cancel)
                    log(f"tool {name}({args}) -> {len(text)} chars in {info.get('elapsed_ms')} ms")
                    if events:
                        yield {"type": "tool_result", "name": name, "args": args, **info}
                    results.append(text)
                    says.append(info.get("say"))
                    ended = ended or bool(info.get("end"))
                if cancel.is_set():
                    break
                if all(says):
                    # fire-and-forget tools (a task handed to Hermes) come with their own short acknowledgement:
                    # say it and end the turn (the model tends to call them again instead of answering)
                    say = " ".join(says)
                    obs = (ROLE_END + "<role>OBSERVATION</role>"
                           + "".join(f"\n<tool_response>\n{r}\n</tool_response>" for r in results)
                           + ROLE_END + "<role>ASSISTANT</role>\n<think></think>")
                    o_ids = self.tok(obs, add_special_tokens=False).input_ids
                    if (quiet and "n't" not in say) or (emitted.strip() and all(n == "end_conversation"
                                                                                for n, _ in calls)):
                        # (done as the reply already said / it already said goodbye: nothing to add)
                        if self.cur_len + len(o_ids) + 8 < self.max_ctx:
                            self._prefill(torch.tensor([o_ids], device=DEVICE), {})
                        break
                    s_ids = self.tok(say, add_special_tokens=False).input_ids
                    if self.cur_len + len(o_ids) + len(s_ids) + 8 < self.max_ctx:
                        self._prefill(torch.tensor([o_ids + s_ids], device=DEVICE), {})  # (the reply, as said)
                        gen += s_ids
                    yield (" " if emitted else "") + say
                    emitted += say
                    break
                # the role_end after </tool_call> was sampled but not fed; the results come back as an observation
                room = self.max_ctx - self.cur_len - min(max_new_tokens, 200) - 16
                while True:
                    obs = (ROLE_END + "<role>OBSERVATION</role>"
                           + "".join(f"\n<tool_response>\n{r}\n</tool_response>" for r in results)
                           + ROLE_END + "<role>ASSISTANT</role>\n<think></think>")
                    ids = self.tok(obs, return_tensors="pt", add_special_tokens=False).input_ids.to(DEVICE)
                    if ids.shape[1] <= room or max(map(len, results)) < 200:
                        break
                    results = [r[: len(r) * 2 // 3] for r in results]  # too long for what's left of the context
                if ids.shape[1] > room:
                    break
                tok = self._prefill(ids, {})
                if any(n not in BRIEF_TOOLS for n, _ in calls):
                    answer_from = len(emitted)
            if (events and not ended and "end_conversation" in tool_names and not cancel.is_set()
                    and BYE.search(emitted)):
                # it said goodbye without ending the conversation: end it if the user did say goodbye
                from tools import END_RE
                if END_RE.search(said()):
                    log(f"reply said goodbye ({emitted!r}) without end_conversation: ending it")
                    yield {"type": "tool_result", "name": "end_conversation", "args": {}, "end": True}
            if used_tools and not emitted.strip() and not cancel.is_set():  # tools led nowhere: don't go silent
                sorry = ("My apologies, sir. I couldn't manage that." if getattr(self.tools, "persona", "") == "jarvis"
                         else "Sorry, I couldn't get that done.")
                gen += self.tok(sorry, add_special_tokens=False).input_ids
                yield sorry
        finally:
            # The last sampled token (EOS or the interrupted one) was never fed; close the turn.
            self.pending = ROLE_END
            self.history.append({"role": "assistant", "content": self.tok.decode(gen, skip_special_tokens=True),
                                 "interrupted": cancel.is_set(), "user": user_text})


# --------------------------------------------------------------------------- Talker


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class StaticQwen2:
    """Graph-capturable Qwen2 forward over a preallocated KV cache, reusing HF module weights."""

    def __init__(self, m, max_len: int):
        cfg = m.config
        self.m = m
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.nh
        self.rep = self.nh // self.nkv
        L = cfg.num_hidden_layers
        self.k = torch.zeros(L, 1, self.nkv, max_len, self.hd, dtype=DTYPE, device=DEVICE)
        self.v = torch.zeros_like(self.k)
        theta = getattr(cfg, "rope_theta", None)
        if theta is None:
            theta = (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", 10000.0)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.hd, 2, device=DEVICE, dtype=torch.float32) / self.hd))
        self.ar = torch.arange(max_len, device=DEVICE)
        self.max_len = max_len

    def forward(self, x, pos):
        """x: [1, T, H] embeddings, pos: [T] long positions (on GPU). Returns final-normed hidden [1, T, H]."""
        T = x.shape[1]
        f = pos.float()[:, None] * self.inv_freq[None]
        emb = torch.cat([f, f], dim=-1)
        cos = emb.cos().to(DTYPE)[None, None]
        sin = emb.sin().to(DTYPE)[None, None]
        mask = (self.ar[None, :] <= pos[:, None])[None, None]  # [1,1,T,max_len]
        for i, layer in enumerate(self.m.layers):
            a = layer.self_attn
            h = layer.input_layernorm(x)
            q = a.q_proj(h).view(1, T, self.nh, self.hd).transpose(1, 2)
            k = a.k_proj(h).view(1, T, self.nkv, self.hd).transpose(1, 2)
            v = a.v_proj(h).view(1, T, self.nkv, self.hd).transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin
            self.k[i].index_copy_(2, pos, k)
            self.v[i].index_copy_(2, pos, v)
            kk = self.k[i][:, :, None].expand(1, self.nkv, self.rep, self.max_len, self.hd).reshape(
                1, self.nh, self.max_len, self.hd)
            vv = self.v[i][:, :, None].expand(1, self.nkv, self.rep, self.max_len, self.hd).reshape(
                1, self.nh, self.max_len, self.hd)
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
            x = x + a.o_proj(o.transpose(1, 2).reshape(1, T, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.m.norm(x)


class Talker:
    SR = 44100

    def __init__(self, model_dir: Path = MODEL_DIR, max_len: int = 1024, use_graphs: bool = True,
                 steps: int | None = None):
        sys.path.insert(0, str(model_dir))
        from talker import MingTalker
        from talker.front.number_en import normalize_numbers
        from talker.front.text_segment_cut import is_chinese
        from talker.front.toolkit import normalize_text_kit
        from talker.talker_module.cfm import get_epss_timesteps
        from talker import tts_prompt as tp

        self._normalize_numbers, self._is_chinese, self._normalize_text_kit = normalize_numbers, is_chinese, normalize_text_kit
        self.tp = tp
        t = time.time()
        # Build the modules directly on the GPU in bf16 (the reference builds in fp32 on the CPU, then
        # converts: minutes under WSL's capped RAM, or ~4.5 GB of transient VRAM if built on the GPU).
        # The fp32 checkpoint tensors (~3 GB) are staged in host RAM, then copied into the bf16 params.
        from direct_load import prefetch
        prefetch([model_dir / "talker" / "model.safetensors", model_dir / "talker" / "vae" / "model.safetensors"])
        t_read = time.time()
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(DTYPE)
        try:
            with torch.device(DEVICE):
                self.t = MingTalker.from_pretrained(model_dir / "talker", dtype=DTYPE, device=str(DEVICE))
        finally:
            torch.set_default_dtype(prev_dtype)
        t_build = time.time()
        # The VAE encoder is only used for reference-audio voice cloning; free its VRAM.
        self.t.audio_vae.encoder = None
        torch.cuda.empty_cache()
        cfg = self.t.config
        assert self.t.patch_size == 1 and self.t.history_patch_size == 1
        self.steps = int(steps or os.environ.get("MING_CFM_STEPS", cfg.steps))
        self.cfg_strength = 2.0
        self.max_len = max_len
        self.stream = torch.cuda.Stream(device=DEVICE)
        self.lock = threading.Lock()
        H = self.t.model.config.hidden_size
        with torch.cuda.stream(self.stream):
            self.llm = StaticQwen2(self.t.model, max_len)
            ts = get_epss_timesteps(self.steps, DEVICE, DTYPE)
            coef = self.t.cfm.sway_sampling_coef
            if coef is not None:
                ts = ts + coef * (torch.cos(torch.pi / 2 * ts) - 1 + ts)
            self.ts = ts
            self.dts = [ts[s + 1] - ts[s] for s in range(self.steps)]
            self.x_buf = torch.zeros(1, 1, H, dtype=DTYPE, device=DEVICE)
            self.hidden_buf = torch.zeros(1, 1, H, dtype=DTYPE, device=DEVICE)
            self.hist_buf = torch.zeros(1, 1, self.t.latent_dim, dtype=DTYPE, device=DEVICE)
            self.latent_buf = torch.zeros(1, 1, self.t.latent_dim, dtype=DTYPE, device=DEVICE)
            self.stop_buf = torch.zeros((), dtype=torch.float32, device=DEVICE)
            self.pos_buf = torch.zeros(1, dtype=torch.long, device=DEVICE)
        # ts/dts and the buffers above were produced on self.stream; everything below (the fused path
        # precomputes timestep embeddings from ts) runs on the default stream, so wait for them first.
        # Without this the precompute could read ts before it was written whenever the GPU was busy.
        self.stream.synchronize()
        self.fast_tail = self.fast_llm = None
        if os.environ.get("MING_FAST_TALKER", "1") == "1":
            from fast_talker import FastQwen2Step, FastTail
            self.fast_tail = FastTail(self, self.ts, self.dts, self.cfg_strength)
            self.fast_llm = FastQwen2Step(self.llm)
            torch.cuda.synchronize()  # precomputed constants are consumed later from other streams/threads
            torch.cuda.empty_cache()
        t_fast = time.time()
        self.use_graphs = use_graphs
        self.g_tail = self.g_step = None
        if use_graphs:
            self._capture()
        self.anchors = {}
        self.load_anchors(designed_only=os.environ.get("MING_VOICE_ANCHORS", "1") != "1")
        log(f"Talker loaded in {time.time() - t:.1f}s (read {t_read - t:.1f}s, build {t_build - t_read:.1f}s, "
            f"fast path {t_fast - t_build:.1f}s, graphs {time.time() - t_fast:.1f}s), "
            f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    # ---- per-step compute (captured) ----
    def _tail(self):
        if self.fast_tail is not None:
            noise = torch.randn(self.latent_buf.shape, device=DEVICE, dtype=DTYPE)
            lat, x_next, stop = self.fast_tail(self.hidden_buf, self.hist_buf, noise)
            self.latent_buf.copy_(lat)
            self.hist_buf.copy_(lat)
            self.x_buf.copy_(x_next)
            self.stop_buf.copy_(stop)
            return
        dit = self.t.cfm.model
        y = torch.randn(self.latent_buf.shape, device=DEVICE, dtype=DTYPE)
        for s in range(self.steps):
            pred_cfg = dit.forward_with_cfg(y, self.ts[s], self.hidden_buf, self.hist_buf, None)
            pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
            y = y + (pred + (pred - null_pred) * self.cfg_strength) * self.dts[s]
        self.latent_buf.copy_(y)
        self.hist_buf.copy_(y)
        self.x_buf.copy_(self.t.aggregator(y))
        self.stop_buf.copy_(self.t.stop_head(self.hidden_buf[:, -1]).float().softmax(dim=-1)[0, 1])

    def _llm_step(self):
        step = self.fast_llm if self.fast_llm is not None else self.llm.forward
        self.hidden_buf.copy_(step(self.x_buf, self.pos_buf))
        self.pos_buf.add_(1)

    def _full_step(self):
        self._llm_step()
        self._tail()

    def _capture(self):
        ctx = lambda: torch.autocast(device_type="cuda", dtype=DTYPE, cache_enabled=False)  # noqa: E731
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            self.pos_buf.fill_(8)
            for _ in range(3):  # warmup (also JITs kernels)
                with ctx():
                    self._full_step()
            self.stream.synchronize()
            try:
                import fastlinear
                fastlinear.save_cache()
            except Exception:
                pass
            pool = torch.cuda.graph_pool_handle()
            self.g_tail = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g_tail, pool=pool, stream=self.stream), ctx():
                self._tail()
            self.g_step = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g_step, pool=pool, stream=self.stream), ctx():
                self._full_step()
            self.stream.synchronize()

    def _run_tail(self):
        if self.g_tail is not None:
            self.g_tail.replay()
        else:
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                self._tail()

    def _run_step(self):
        if self.g_step is not None:
            self.g_step.replay()
        else:
            with torch.autocast(device_type="cuda", dtype=DTYPE):
                self._full_step()

    # ---- voice anchors ----
    # A built-in voice is only named in the instruction, so each fragment is a fresh draw of that voice: now and
    # then one comes out as a different speaker mid-reply. An anchor is a short clip of the voice (its text and
    # latents); every fragment continues from it, which holds the speaker and the clip's pace steady.
    ANCHOR_DIR = Path(__file__).resolve().parent / "anchors"

    # A designed voice (dev/design_voice.py) is a voice-design instruction ("a refined English butler...") and an
    # anchor made with it: on its own, every fragment would be a different speaker that fits the description.
    def load_anchors(self, designed_only=False):
        for f in sorted(self.ANCHOR_DIR.glob("*.pt")):
            a = torch.load(f, map_location="cpu")
            if a.get("instruction") or not designed_only:
                self.anchors[f.stem] = dict(self.make_anchor(a["text"], a["latents"]),
                                            instruction=a.get("instruction"))
        if self.anchors:
            log(f"voice anchors: {', '.join(self.anchors)}")

    @property
    def designed_voices(self) -> list[str]:
        return [v for v, a in self.anchors.items() if a.get("instruction")]

    @torch.inference_mode()
    def make_anchor(self, text: str, latents: torch.Tensor) -> dict:
        D, H = self.t.latent_dim, self.t.model.config.hidden_size
        lat = latents.to(DEVICE, DTYPE).reshape(1, -1, D)
        with torch.cuda.stream(self.stream), torch.autocast(device_type="cuda", dtype=DTYPE):
            emb = self.t.aggregator(lat.reshape(-1, 1, D)).reshape(1, -1, H)
        self.stream.synchronize()
        return {"text": text, "latents": lat, "emb": emb}

    # ---- text front-end ----
    @torch.inference_mode()
    def speaker_from_audio(self, samples: np.ndarray, sample_rate: int):
        """Speaker embedding (CAM++ on CPU -> spk_head) for voice cloning; returns a [1, H] tensor."""
        with torch.cuda.stream(self.stream):
            speaker, _, _ = self.t.reference_features({"samples": samples, "sample_rate": sample_rate}, False)
            self.stream.synchronize()
        return speaker[0].detach().clone()

    def prepare(self, text: str, voice):
        """Return list of (normalized_fragment, prompt, instruction) for one chunk of reply text."""
        tp = self.tp
        text = re.sub(r"^[A-Za-z]{1,16}\t", "", text)
        cloned = isinstance(voice, torch.Tensor)  # speaker embedding from speaker_from_audio()
        language, text = tp.resolve_request_language(text, language=None, taskname="TTS" if cloned else "IP")
        prompt = tp.append_optional_target_language(tp.DEFAULT_TTS_PROMPT, language)
        designed = (self.anchors.get(voice) or {}).get("instruction") if not cloned else None
        instruction = None if cloned else designed or tp.build_task_instruction("IP", ip=voice)
        skip_tn = tp.preserve_multilingual_text(tp.target_language_from_prompt(prompt))
        out = []
        for fragment in self.t.text_fragments(text):
            protected = tp.contains_paralinguistic_tag(fragment)
            fragment = tp.protect_paralinguistic_tags(fragment) if protected else fragment
            if not skip_tn:
                if not self._is_chinese(fragment):
                    fragment = self._normalize_numbers(fragment)
                fragment = self._normalize_text_kit(self.t.normalizer.normalize(fragment)).lstrip("，")
            if protected:
                fragment = tp.restore_paralinguistic_tags(fragment)
            if fragment.strip():
                out.append((fragment, prompt, instruction))
        return out

    # ---- streaming synthesis ----
    @torch.inference_mode()
    def synth_stream(self, text: str, voice: str = "US_Stella", cancel: threading.Event | None = None,
                     chunk_latents: int = 3, max_steps: int = 750, anchor: dict | None = None,
                     collect: list | None = None):
        """Yield float32 numpy PCM chunks (44.1 kHz) as soon as they are decoded.
        anchor: override the voice's anchor (False: none); collect: gets each fragment's latents."""
        cancel = cancel or threading.Event()
        with self.lock:
            speaker = [voice] if isinstance(voice, torch.Tensor) else None
            if anchor is None:
                anchor = self.anchors.get(voice) if isinstance(voice, str) else None
            for fragment, prompt, instruction in self.prepare(text, voice):
                if cancel.is_set():
                    return
                a = anchor if anchor and not self._is_chinese(fragment) else None  # (anchors are English clips)
                lat = [] if collect is not None else None
                yield from self._synth_fragment(fragment, prompt, instruction, cancel, chunk_latents, max_steps,
                                                speaker, a, lat)
                if lat is not None:
                    collect.append((fragment, torch.cat(lat, 1)))

    def _synth_fragment(self, fragment, prompt, instruction, cancel, chunk_latents, max_steps, speaker=None,
                        anchor=None, collect=None):
        # No `with` block may span a `yield`: autocast/stream state is thread-local and would leak.
        vae = self.t.audio_vae
        s = self.stream
        with torch.cuda.stream(s), torch.autocast(device_type="cuda", dtype=DTYPE):
            emb = self.t.input_embeddings(fragment, prompt, instruction, speaker,
                                          anchor["text"] if anchor else None, anchor["emb"] if anchor else None)
            T = emb.shape[1]
            max_steps = min(max_steps, self.max_len - T - 1)
            hid = self.llm.forward(emb, torch.arange(T, device=DEVICE))
            self.hidden_buf.copy_(hid[:, -1:])
            if anchor:
                self.hist_buf.copy_(anchor["latents"][:, -1:])
            else:
                self.hist_buf.zero_()
            self.pos_buf.fill_(T)
            self._run_tail()
            pending = [self.latent_buf.clone()]
            if collect is not None:
                collect.append(pending[0])
        state, pkv, calls, step = (None, None, None), None, 0, 0
        # (an anchor ending in silence made the talker stop after a few latents: don't let it end implausibly early)
        min_steps = max(3, int(0.25 * len(fragment))) if anchor else 3

        def decode(last):
            nonlocal state, pkv, calls, pending
            wav, state, pkv = vae.decode(torch.cat(pending, 1), past_key_values=pkv, use_cache=True,
                                         stream_state=state, last_chunk=last)
            calls += 1
            pending = []
            return wav.float().reshape(-1).cpu().numpy() if wav.numel() else None

        while True:
            out = None
            with torch.cuda.stream(s):
                stop = float(self.stop_buf.item())  # sync: latent `step` and its stop prob are ready
                done = (step > min_steps and stop > 0.5) or step + 1 >= max_steps or cancel.is_set()
                if done:
                    if not cancel.is_set():
                        out = decode(True)
                else:
                    if len(pending) >= (1 if calls < 2 else chunk_latents):
                        out = decode(False)
                    self._run_step()
                    pending.append(self.latent_buf.clone())
                    if collect is not None:
                        collect.append(pending[-1])
                    step += 1
            if out is not None and out.size:
                yield out
            if done:
                return

    def synthesize(self, text, voice="US_Stella", **kw):
        chunks = list(self.synth_stream(text, voice, **kw))
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


# --------------------------------------------------------------------------- text chunking

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def _units(s: str) -> int:
    return len(_CJK.sub(" ", s).split()) + len(_CJK.findall(s)) // 2


_CURRENCY = {"$": ("dollar", "dollars", "cent", "cents"), "€": ("euro", "euros", "cent", "cents"),
             "£": ("pound", "pounds", "penny", "pence")}


def _money(m: re.Match) -> str:
    one, many, sub1, subn = _CURRENCY[m[1]]
    whole, cents, scale = m[2], m[3], m[4]
    if scale:  # $5 million -> 5 million dollars ($1.5 billion too)
        return f"{whole}{'.' + cents if cents else ''} {scale} {many}"
    s = f"{whole} {one if whole == '1' else many}"
    if cents and cents.strip("0"):
        c = str(int((cents + "0")[:2]))  # ($1.5: 50 cents)
        s += f" and {c} {sub1 if c == '1' else subn}"
    return s


# What the talker reads wrong (checked by speaking each and transcribing it back): it drops "×", stops reading at
# "km/h" and at spaced dashes, says "minus" for scores and ranges, loses the sign of "-5", reads "€40" as
# "euro forty point" and "12:00" as "twelve zero".
_SPEECH_SUBS = [(re.compile(p, re.I), r) for p, r in (
    (r"\s*×\s*", " times "),
    (r"(?<=\d)x\b", " times"),  # 3x faster
    (r"(?<=\d)\s*[–-]\s*(?=\d)", " to "),  # 24-17, 5-10 minutes, 2025–2026
    (r"\s+[–—-]\s+|—", ", "),  # spaced dashes (and em dashes) are pauses
    (r"(?<![\w.])[-−](?=\d)", "minus "),  # -5°C
    (r"\bkm/h\b|\bkph\b", "kilometers per hour"),
    (r"\bmph\b", "miles per hour"),
    (r"\bm/s\b", "meters per second"),
    (r"\b(\d{1,2}):00\b", r"\1"),  # 12:00 PM -> 12 PM
    (r"\b1/2\b", "a half"), (r"\b1/3\b", "a third"), (r"\b1/4\b", "a quarter"), (r"\b3/4\b", "three quarters"),
    (r"\s&\s", " and "),
    (r"(?<=[a-z])-(?=[a-z])", " "),  # Wi-Fi (sometimes said as "why")
)]
_MONEY = re.compile(r"([$€£])(\d+(?:,\d{3})*)(?:\.(\d+))?(?:\s*(thousand|million|billion|trillion)\b)?", re.I)
# It also says "20 to 22" as "twenty to twenty, two" about half the time; spelled out it reads them right.
_COMPOUND = re.compile(r"(?<![\d.,:/])\b([2-9])([1-9])\b(?![.,:/]\d)")
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()
_ONES = "one two three four five six seven eight nine".split()


def clean_for_speech(s: str) -> str:
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"^\s*(?:[-*+•]|\d+[.)])\s+", "", s, flags=re.M)
    s = re.sub(r"[*_#`>|~]+", "", s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = _MONEY.sub(_money, s)
    for p, r in _SPEECH_SUBS:
        s = p.sub(r, s)
    s = _COMPOUND.sub(lambda m: f"{_TENS[int(m[1]) - 2]}-{_ONES[int(m[2]) - 1]}", s)
    return re.sub(r"\s+", " ", s).strip()


class Fragmenter:
    """Cut streamed text into speakable pieces: tiny first piece for latency, bigger ones after."""

    STRONG = re.compile(r"(?:[.!?…]+[\"')\]]*\s)|[。！？；\n]")
    WEAK = re.compile(r"(?:[,;:]\s)|[，、：]")

    def __init__(self, first_min=3, first_weak=5, min_units=6, weak_units=14, max_units=28):
        self.buf = ""
        self.n = 0
        self.first_min, self.first_weak = first_min, first_weak
        self.min_units, self.weak_units, self.max_units = min_units, weak_units, max_units

    def _cut(self):
        first = self.n == 0
        smin = self.first_min if first else self.min_units
        wmin = self.first_weak if first else self.weak_units
        for m in self.STRONG.finditer(self.buf):
            if _units(self.buf[: m.end()]) >= smin:
                return m.end()
        for m in self.WEAK.finditer(self.buf):
            if _units(self.buf[: m.end()]) >= wmin:
                return m.end()
        if _units(self.buf) >= self.max_units:
            i = self.buf.rfind(" ")
            return i + 1 if i > 0 else len(self.buf)
        return None

    def push(self, delta: str):
        self.buf += delta
        out = []
        while (c := self._cut()) is not None:
            piece, self.buf = self.buf[:c], self.buf[c:]
            piece = clean_for_speech(piece)
            if piece:
                out.append(piece)
                self.n += 1
        return out

    def flush(self):
        piece = clean_for_speech(self.buf)
        self.buf = ""
        if piece and re.search(r"\w", piece):
            self.n += 1
            return [piece]
        return []
