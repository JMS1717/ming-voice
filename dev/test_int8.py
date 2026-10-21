"""int8 experts vs bf16: logits agreement on a forced continuation, greedy replies, ASR, speed, VRAM."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf, torch, torchaudio
import engine
from engine import Thinker, DEVICE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QS = ["capital", "followup", "sleep", "story"]
clips = {}
for q in QS:
    w, sr = sf.read(os.path.join(HERE, "e2e", "questions", q + ".wav"), dtype="float32")
    clips[q] = w
zh, sr = sf.read(os.path.join(HERE, "samples", "introduce_ant_group.wav"), dtype="float32", always_2d=True)
clips["zh"] = torchaudio.functional.resample(torch.from_numpy(zh.mean(1)), sr, 16000).numpy()
cont = "The capital of France is Paris, a city known for art, food, and the Eiffel Tower."


def probe(th):
    out = {}
    with torch.inference_mode():
        th.reset()
        ids = th.tok(th._system_block() + "<role>HUMAN</role>Tell me about Paris.<|role_end|><role>ASSISTANT</role>\n<think></think>",
                     return_tensors="pt").input_ids.to(DEVICE)
        cids = th.tok(cont, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
        full = torch.cat([ids, cids], 1)
        th.cache.attn_len = full.shape[1]
        h = th.model.model.word_embeddings(full)
        m = th.model.model
        pos = torch.arange(full.shape[1], device=DEVICE)
        mask = torch.zeros(pos.shape[0], full.shape[1], dtype=h.dtype, device=DEVICE)
        mask.masked_fill_(th.ar[None, :full.shape[1]] > pos[:, None], torch.finfo(h.dtype).min)
        cs = m.rotary_emb(h, pos[None])
        for layer in m.layers[:th.n_layers]:
            h = layer(h, attention_mask=mask[None, None], position_ids=pos[None], past_key_value=th.cache,
                      cache_position=pos, use_cache=True, position_embeddings=cs)[0]
        out["logits"] = th.model.lm_head(m.norm(h[:, ids.shape[1] - 1:-1])).float()[0]
        out["targets"] = cids[0]
    out["replies"] = {}
    for q, w in clips.items():
        th.reset()
        out["replies"][q] = "".join(th.stream(w, max_new_tokens=60)).strip()
    out["asr"] = {q: th.transcribe(clips[q]) for q in ("capital", "story")}
    th.reset()
    t0 = time.perf_counter(); n = 0
    for _ in th.stream(None, user_text="Tell me a short story about a robot learning to paint.", max_new_tokens=150):
        n += 1
    out["tok_s"] = n / (time.perf_counter() - t0)
    out["vram"] = torch.cuda.memory_allocated() / 2**30
    return out


th = Thinker(use_graphs=True)
ref = probe(th)
t = time.time()
engine._quantize_moe_int8(th.model)
th._capture()  # re-capture decode graphs for the int8 kernels
print(f"quantized in {time.time() - t:.1f}s")
q8 = probe(th)

lr, lq = ref["logits"], q8["logits"]
cos = torch.nn.functional.cosine_similarity(lr, lq, dim=-1)
top1 = (lr.argmax(-1) == lq.argmax(-1)).float().mean().item()
pr, pq = lr.log_softmax(-1), lq.log_softmax(-1)
kl = (pr.exp() * (pr - pq)).sum(-1)
nll = lambda lp: -lp.gather(1, ref["targets"][:, None]).mean().item()
print(f"forced continuation ({len(ref['targets'])} tok): cos min {cos.min():.4f} mean {cos.mean():.4f} | top-1 agree {top1:.3f} "
      f"| KL mean {kl.mean():.4f} | NLL bf16 {nll(pr):.3f} int8 {nll(pq):.3f}")
for q in clips:
    same = "same" if ref["replies"][q] == q8["replies"][q] else "DIFF"
    print(f"[{same}] {q}:\n   bf16: {ref['replies'][q][:100]!r}\n   int8: {q8['replies'][q][:100]!r}")
for q in ref["asr"]:
    print(f"asr {q}: bf16 {ref['asr'][q]!r} | int8 {q8['asr'][q]!r}")
print(f"decode: bf16 {ref['tok_s']:.1f} tok/s -> int8 {q8['tok_s']:.1f} tok/s | VRAM {ref['vram']:.2f} -> {q8['vram']:.2f} GiB")
