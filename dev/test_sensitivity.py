import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Talker, DEVICE, DTYPE
from fast_talker import FastTail
tk = Talker(use_graphs=False)
ft = tk.fast_tail
dit, agg = tk.t.cfm.model, tk.t.aggregator
def ref_tail(hidden, hist, noise):
    y = noise
    for s in range(tk.steps):
        pc = dit.forward_with_cfg(y, tk.ts[s], hidden, hist, None)
        p, n = torch.chunk(pc, 2, dim=0)
        y = y + (p + (p - n) * tk.cfg_strength) * tk.dts[s]
    return y
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
torch.manual_seed(0)
with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
    frag, prompt, instr = tk.prepare("Hello there, this is a quick test of the fast talker path.", "US_Stella")[0]
    emb = tk.t.input_embeddings(frag, prompt, instr, None, None, None)
    hid = tk.llm.forward(emb, torch.arange(emb.shape[1], device=DEVICE))
    fe, pe = [], []
    hist = torch.zeros(1, 1, 64, device=DEVICE, dtype=DTYPE)
    for i in range(20):
        hidden = hid[:, -1 - (i % 6):hid.shape[1] - (i % 6)].contiguous()
        noise = torch.randn(1, 1, 64, device=DEVICE, dtype=DTYPE)
        r = ref_tail(hidden, hist, noise)
        f = ft(hidden, hist, noise)[0]
        pert = hidden * (1 + 0.004 * torch.randn_like(hidden))   # ~bf16 rounding-level perturbation
        rp = ref_tail(pert, hist, noise)
        fe.append(rel(f, r)); pe.append(rel(rp, r))
        hist = r
    import statistics as st
    print("fast vs ref     : median %.4f  max %.4f  >0.1: %d/20" % (st.median(fe), max(fe), sum(e > 0.1 for e in fe)))
    print("ref perturbed   : median %.4f  max %.4f  >0.1: %d/20" % (st.median(pe), max(pe), sum(e > 0.1 for e in pe)))
