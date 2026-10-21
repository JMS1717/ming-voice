import os, sys, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Talker, DEVICE, DTYPE
tk = Talker(use_graphs=True)
s = tk.stream
def tm(fn, n=20):
    with torch.cuda.stream(s):
        fn(); s.synchronize(); t = time.perf_counter()
        for _ in range(n): fn()
        s.synchronize()
    return (time.perf_counter() - t) / n * 1000
with torch.inference_mode():
    tk.pos_buf.fill_(100)
    print("g_step  ms", tm(tk.g_step.replay))
    print("g_tail  ms", tm(tk.g_tail.replay))
    def llm():
        tk.pos_buf.fill_(100); tk._llm_step()
    print("llm eager ms", tm(llm))
    def tail():
        with torch.autocast("cuda", dtype=DTYPE): tk._tail()
    print("tail eager ms", tm(tail))
    lat = torch.randn(1, 3, 64, device=DEVICE, dtype=DTYPE)
    def dec():
        st=(None,None,None); pkv=None
        w, st, pkv = tk.t.audio_vae.decode(lat, use_cache=True, stream_state=st)
        w, st, pkv = tk.t.audio_vae.decode(lat, past_key_values=pkv, use_cache=True, stream_state=st)
    print("vae 2x3 latents ms", tm(dec, 10))
    frag = tk.prepare("Hello there, this is a quick test of the talker.", "US_Stella")
    def pre():
        with torch.autocast("cuda", dtype=DTYPE):
            emb = tk.t.input_embeddings(*frag[0], None, None, None)
            tk.llm.forward(emb, torch.arange(emb.shape[1], device=DEVICE))
    print("prefill ms", tm(pre, 5))
    t=time.perf_counter(); tk.prepare("Hello there, this is a quick test of the talker, number 42 on May 3rd.", "US_Stella"); print("prepare(TN) ms", (time.perf_counter()-t)*1000)
