"""Per-layer check of the fused prefill MoE on real hidden states: error of the HF path and of the fused
kernel, each against an fp32 computation of the same MoE."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import engine
from engine import Thinker, DEVICE

th = Thinker()
captured = {}
blocks = [m for m in th.model.modules() if type(m).__name__ == "BailingMoeV3SparseMoeBlock"]
hooks = [b.register_forward_pre_hook(lambda mod, args, i=i: captured.setdefault(i, args[0].detach().clone()))
         for i, b in enumerate(blocks)]
engine._FUSED_PREFILL = False
with torch.inference_mode():
    th.reset()
    list(th.stream(None, user_text="Tell me about Paris and why so many people visit it every year.", max_new_tokens=1))
for h in hooks:
    h.remove()


def fp32_moe(b, x):
    idx, w, _ = b.gate(x)
    N, h = x.shape[1], x.shape[2]
    xf = x.view(-1, h).float()
    y = torch.zeros(N, h, device=DEVICE)
    I = b.inter
    ids = torch.cat([idx, b.shared_id.expand(N, 1)], 1)
    wts = torch.cat([w.float(), torch.ones(N, 1, device=DEVICE)], 1)
    for n in range(N):
        for k in range(ids.shape[1]):
            e = ids[n, k]
            gu = xf[n] @ b.w_gu[e].float().T
            y[n] += wts[n, k] * ((F.silu(gu[:I]) * gu[I:]) @ b.w_d[e].float().T)
    return y


with torch.inference_mode():
    for i in (0, 5, 11, 22):
        x = captured[i]
        b = blocks[i]
        ref32 = fp32_moe(b, x)
        engine._FUSED_PREFILL = False
        hf = b(x)[0].view(-1, x.shape[-1]).float()
        engine._FUSED_PREFILL = True
        fu = b(x)[0].view(-1, x.shape[-1]).float()
        e = lambda a: ((a - ref32).norm() / ref32.norm()).item()
        print(f"MoE layer {i}: N={x.shape[1]} | HF path err {e(hf):.4f} | fused err {e(fu):.4f} | "
              f"max |fused-HF| {(fu - hf).abs().max().item():.3g} (ref max {ref32.abs().max().item():.3g})")
