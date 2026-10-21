import os, sys, threading, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Talker, Thinker
out = {}
def load():
    out["th"] = Thinker(use_graphs=False)
    out["tk"] = Talker(use_graphs=False)
t = threading.Thread(target=load); t.start(); t.join()
tk = out["tk"]
ft, fl = tk.fast_tail, tk.fast_llm
texts = ["The capital of France is Paris.", "Hi! This is a quick test.", "Keep your bedroom cool, dark, and quiet."]
for name, a, b in [("both fast", ft, fl), ("ref tail + fast llm", None, fl), ("fast tail + ref llm", ft, None), ("both ref", None, None)]:
    tk.fast_tail, tk.fast_llm = a, b
    torch.manual_seed(0)
    d = [round(len(tk.synthesize(x, "US_Stella")) / 44100, 1) for x in texts]
    print(f"{name:22s}: {d}", flush=True)
