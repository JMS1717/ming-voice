import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Talker, DEVICE, DTYPE
from fast_talker import FastTail
import fastlinear

tk = Talker(use_graphs=False)
ft = FastTail(tk, tk.ts, tk.dts, tk.cfg_strength)
dit, agg = tk.t.cfm.model, tk.t.aggregator


def ref_tail(hidden, hist, noise):
    y = noise
    for s in range(tk.steps):
        pred_cfg = dit.forward_with_cfg(y, tk.ts[s], hidden, hist, None)
        pred, null = torch.chunk(pred_cfg, 2, dim=0)
        y = y + (pred + (pred - null) * tk.cfg_strength) * tk.dts[s]
    return y, agg(y), tk.t.stop_head(hidden[:, -1]).float().softmax(-1)[0, 1]


with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
    frag, prompt, instr = tk.prepare("Hello there, this is a quick test of the fast talker path.", "US_Stella")[0]
    emb = tk.t.input_embeddings(frag, prompt, instr, None, None, None)
    hid = tk.llm.forward(emb, torch.arange(emb.shape[1], device=DEVICE))
    hidden = hid[:, -1:].contiguous()
    hist = torch.zeros(1, 1, 64, device=DEVICE, dtype=DTYPE)
    for trial in range(3):
        noise = torch.randn(1, 1, 64, device=DEVICE, dtype=DTYPE)
        r = ref_tail(hidden, hist, noise)
        f = ft(hidden, hist, noise)
        errs = [((a.float() - b.float()).norm() / (a.float().norm() + 1e-9)).item() for a, b in zip(r, f)]
        print(f"trial {trial}: rel err latent {errs[0]:.4f}  x_next {errs[1]:.4f}  stop {r[2].item():.4f} vs {f[2].item():.4f}")
        hist = r[0]
        hidden = hid[:, -2 - trial:-1 - trial].contiguous()
    fastlinear.save_cache()

    s = tk.stream
    def gtime(fn, reps=30):
        with torch.cuda.stream(s):
            fn(); s.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                fn()
            g.replay(); s.synchronize(); t = time.perf_counter()
            for _ in range(reps):
                g.replay()
            s.synchronize()
        return (time.perf_counter() - t) / reps * 1000
    noise = torch.randn(1, 1, 64, device=DEVICE, dtype=DTYPE)
    print(f"reference tail (graph): {gtime(lambda: ref_tail(hidden, hist, noise)):.2f} ms")
    print(f"fast tail      (graph): {gtime(lambda: ft(hidden, hist, noise)):.2f} ms")
