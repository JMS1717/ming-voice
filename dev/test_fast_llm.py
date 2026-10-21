import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Talker, DEVICE, DTYPE
from fast_talker import FastQwen2Step
import fastlinear
tk = Talker(use_graphs=False)
fs = FastQwen2Step(tk.llm)
H = tk.t.model.config.hidden_size
with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
    frag, prompt, instr = tk.prepare("Hello there, this is a quick test of the fast talker path.", "US_Stella")[0]
    emb = tk.t.input_embeddings(frag, prompt, instr, None, None, None)
    T = emb.shape[1]
    tk.llm.forward(emb, torch.arange(T, device=DEVICE))
    k_ref, v_ref = tk.llm.k.clone(), tk.llm.v.clone()
    xs = torch.randn(4, 1, 1, H, device=DEVICE, dtype=DTYPE) * emb.float().std().to(DTYPE)
    # reference path
    refs = [tk.llm.forward(xs[j], torch.tensor([T + j], device=DEVICE)) for j in range(4)]
    k_after_ref = tk.llm.k.clone()
    tk.llm.k.copy_(k_ref); tk.llm.v.copy_(v_ref)
    outs = [fs(xs[j], torch.tensor([T + j], device=DEVICE)) for j in range(4)]
    for j in range(4):
        print(f"pos {T+j}: rel err {((refs[j].float()-outs[j].float()).norm()/refs[j].float().norm()).item():.4f}")
    kd = (k_after_ref[:, :, :, :T + 4].float() - tk.llm.k[:, :, :, :T + 4].float()).norm() / k_after_ref[:, :, :, :T+4].float().norm()
    print(f"kv-cache rel diff {kd.item():.4f}")
    fastlinear.save_cache()
    s = tk.stream
    pos = torch.tensor([T + 10], device=DEVICE)
    def gtime(fn, reps=30):
        with torch.cuda.stream(s):
            fn(); s.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                fn()
            g.replay(); s.synchronize(); t = time.perf_counter()
            for _ in range(reps): g.replay()
            s.synchronize()
        return (time.perf_counter() - t) / reps * 1000
    print(f"reference LLM step (graph): {gtime(lambda: tk.llm.forward(xs[0], pos)):.2f} ms")
    print(f"fast LLM step      (graph): {gtime(lambda: fs(xs[0], pos)):.2f} ms")
