import os, sys, threading, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Talker, DEVICE, DTYPE
rel = lambda a, b: ((a.float() - b.float()).norm() / b.float().norm()).item()
def run(tag):
    tk = Talker(use_graphs=False)
    dit = tk.t.cfm.model
    torch.manual_seed(0)
    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        frag, prompt, instr = tk.prepare("Hello there, this is a quick test of the fast talker path.", "US_Stella")[0]
        emb = tk.t.input_embeddings(frag, prompt, instr, None, None, None)
        T = emb.shape[1]
        hid = tk.llm.forward(emb, torch.arange(T, device=DEVICE))
        hidden = hid[:, -1:].contiguous(); hist = torch.zeros(1, 1, 64, device=DEVICE, dtype=DTYPE)
        noise = torch.randn(1, 1, 64, device=DEVICE, dtype=DTYPE)
        y = noise
        for s in range(tk.steps):
            pc = dit.forward_with_cfg(y, tk.ts[s], hidden, hist, None); p, n = torch.chunk(pc, 2, 0)
            y = y + (p + (p - n) * tk.cfg_strength) * tk.dts[s]
        f = tk.fast_tail(hidden, hist, noise)
        x = torch.randn(1, 1, hid.shape[-1], device=DEVICE, dtype=DTYPE) * emb.float().std().to(DTYPE)
        kk, vv = tk.llm.k.clone(), tk.llm.v.clone()
        r = tk.llm.forward(x, torch.tensor([T], device=DEVICE))
        tk.llm.k.copy_(kk); tk.llm.v.copy_(vv)
        g = tk.fast_llm(x, torch.tensor([T], device=DEVICE))
        st_ref = tk.t.stop_head(hidden[:, -1]).float().softmax(-1)[0, 1].item()
    print(f"[{tag}] tail latent err {rel(f[0], y):.4f} | llm step err {rel(g, r):.4f} | stop ref {st_ref:.4f} fast {f[2].item():.4f}", flush=True)
run("main thread")
t = threading.Thread(target=run, args=("worker thread",)); t.start(); t.join()
