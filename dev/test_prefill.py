"""Fused prefill MoE vs. the reference path on the real Thinker: logits on a forced continuation, greedy
replies, ASR, and prefill latency (time to first token) for text and audio turns."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soundfile as sf, torch
import engine
from engine import Thinker, DEVICE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
clips = {q: sf.read(os.path.join(HERE, "e2e", "questions", q + ".wav"), dtype="float32")[0]
         for q in ("capital", "sleep", "story")}
cont = "The capital of France is Paris, a city known for art, food, and the Eiffel Tower."


def ttft(th, **kw):
    th.reset()
    torch.cuda.synchronize()
    t = time.perf_counter()
    next(iter(th.stream(**kw)))
    return (time.perf_counter() - t) * 1000


def probe(th):
    out = {}
    with torch.inference_mode():
        th.reset()
        ids = th.tok(th._system_block() + "<role>HUMAN</role>Tell me about Paris.<|role_end|><role>ASSISTANT</role>\n<think></think>"
                     + cont, return_tensors="pt").input_ids.to(DEVICE)
        th.cache.attn_len = ids.shape[1]
        m = th.model.model
        h = m.word_embeddings(ids)
        pos = torch.arange(ids.shape[1], device=DEVICE)
        mask = torch.zeros(pos.shape[0], ids.shape[1], dtype=h.dtype, device=DEVICE)
        mask.masked_fill_(th.ar[None, :ids.shape[1]] > pos[:, None], torch.finfo(h.dtype).min)
        cs = m.rotary_emb(h, pos[None])
        for layer in m.layers[:th.n_layers]:
            h = layer(h, attention_mask=mask[None, None], position_ids=pos[None], past_key_value=th.cache,
                      cache_position=pos, use_cache=True, position_embeddings=cs)[0]
        out["logits"] = th.model.lm_head(m.norm(h)).float()[0]
    out["replies"] = {}
    for q, w in clips.items():
        th.reset()
        out["replies"][q] = "".join(th.stream(w, max_new_tokens=50)).strip()
    out["asr"] = th.transcribe(clips["story"])
    for _ in range(2):
        ttft(th, wav16k=clips["capital"]); ttft(th, wav16k=None, user_text="Give me three tips for learning to cook.")
    out["ttft_audio"] = sorted(ttft(th, wav16k=clips["capital"]) for _ in range(5))[2]
    out["ttft_text"] = sorted(ttft(th, wav16k=None, user_text="Give me three tips for learning to cook.") for _ in range(5))[2]
    t = time.perf_counter()
    th.transcribe(clips["story"])
    out["asr_ms"] = (time.perf_counter() - t) * 1000
    return out


th = Thinker(use_graphs=True)
engine._FUSED_PREFILL = False
ref = probe(th)
engine._FUSED_PREFILL = True
new = probe(th)
cos = torch.nn.functional.cosine_similarity(ref["logits"], new["logits"], dim=-1)
top1 = (ref["logits"].argmax(-1) == new["logits"].argmax(-1)).float().mean().item()
print(f"prefill logits: cos min {cos.min():.4f} mean {cos.mean():.4f} | top-1 agree {top1:.3f}")
for q in clips:
    same = "same" if ref["replies"][q] == new["replies"][q] else "DIFF"
    print(f"[{same}] {q}:\n   ref: {ref['replies'][q][:110]!r}\n   new: {new['replies'][q][:110]!r}")
print(f"asr ref {ref['asr']!r}\n    new {new['asr']!r}")
for k in ("ttft_audio", "ttft_text", "asr_ms"):
    print(f"{k}: {ref[k]:.0f} ms -> {new[k]:.0f} ms")
