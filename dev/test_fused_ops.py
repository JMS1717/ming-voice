"""fused_ops (Triton RMSNorm + one-token MoE router) vs. the reference modules, then whole-model replies and
decode speed with the fused ops off and on."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import soundfile as sf, torch
import engine
from engine import Thinker, DEVICE
from fused_ops import moe_route, rms_norm

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
th = Thinker()
m = th.model.model
norms = [x for x in th.model.modules() if type(x).__name__ == "BailingMoeV3RMSNorm"]
blocks = [x for x in th.model.modules() if type(x).__name__ == "BailingMoeV3SparseMoeBlock"]

with torch.inference_mode():
    # RMSNorm on real-scale inputs
    worst = 0.0
    for nm in norms[:12]:
        H = nm.weight.shape[0]
        for rows in (1, 7, 64):
            x = (torch.randn(rows, H, device=DEVICE) * 3).to(torch.bfloat16)
            ref = type(nm).forward(nm, x).float()
            got = rms_norm(x, nm.weight, nm.variance_epsilon).float()
            worst = max(worst, ((got - ref).abs().max() / ref.abs().max()).item())
    print(f"rms_norm: max rel diff {worst:.2e} over {min(12, len(norms))} norms x 3 shapes")

    # router on real hidden states (MoE inputs captured during a prefill)
    captured = {}
    hooks = [b.register_forward_pre_hook(lambda mod, a, i=i: captured.setdefault(i, a[0].detach().clone()))
             for i, b in enumerate(blocks)]
    th.reset()
    list(th.stream(None, user_text="Tell me about the history of the printing press in Europe.", max_new_tokens=1))
    for h in hooks:
        h.remove()
    same = total = 0
    wdiff = 0.0
    for i, b in enumerate(blocks):
        if i not in captured:  # blocks past n_layers (e.g. an MTP head) are never run
            continue
        x = captured[i][0]  # [T, H]
        g = b.gate
        logits = torch.empty(g.w32.shape[0], device=DEVICE)
        ids = torch.empty(g.top_k + 1, dtype=torch.long, device=DEVICE)
        wts = torch.empty(g.top_k + 1, device=DEVICE)
        bias = g.expert_bias.float()
        for t in range(x.shape[0]):
            ref_idx, ref_w, _ = g(x[t:t + 1])
            moe_route(x[t:t + 1], g.w32, bias, g.n_group, g.topk_group, g.top_k, g.routed_scaling_factor, logits, ids, wts)
            total += 1
            if set(ref_idx[0].tolist()) == set(ids[:g.top_k].tolist()):
                same += 1
                order = {e: j for j, e in enumerate(ref_idx[0].tolist())}
                rw = torch.tensor([ref_w[0, order[e]].item() for e in ids[:g.top_k].tolist()])
                wdiff = max(wdiff, (rw - wts[:g.top_k].cpu()).abs().max().item())
            assert ids[g.top_k].item() == g.w32.shape[0] and wts[g.top_k].item() == 1.0
    print(f"moe_route: same expert set for {same}/{total} (token, layer) pairs; max weight diff {wdiff:.2e}")

clips = {q: sf.read(os.path.join(HERE, "e2e", "questions", q + ".wav"), dtype="float32")[0] for q in ("capital", "sleep", "story")}


def probe():
    out = {}
    for q, w in clips.items():
        th.reset()
        out[q] = "".join(th.stream(w, max_new_tokens=60)).strip()
    th.reset()
    gen = th.stream(None, user_text="Tell me a long story about a lighthouse keeper.", max_new_tokens=200)
    next(gen)
    t, n = time.perf_counter(), 0
    for _ in gen:
        n += 1
    out["tok_s"] = n / (time.perf_counter() - t)
    return out


def set_fused(on):
    engine._FUSED_OPS = on
    for nm in norms:
        if on:
            engine._patch_norms(th.model)
            break
        nm.__dict__.pop("forward", None)
    th._capture()


set_fused(False)
ref = probe()
set_fused(True)
new = probe()
for q in clips:
    print(f"[{'same' if ref[q] == new[q] else 'DIFF'}] {q}:\n   ref: {ref[q][:110]!r}\n   new: {new[q][:110]!r}")
print(f"decode: {ref['tok_s']:.1f} -> {new['tok_s']:.1f} tok/s ({1000 / ref['tok_s']:.2f} -> {1000 / new['tok_s']:.2f} ms/token)")
