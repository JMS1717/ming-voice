"""Graph-captured (padded) prefill vs. eager prefill: logits, KDA recurrent + conv states, replies, and speed.

    python dev/test_prefill_graphs.py      (standalone, needs the GPU: stop the server first)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf, torch
import torch.nn.functional as F
import engine
from engine import Thinker, _KDAState

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load = lambda n: sf.read(os.path.join(HERE, "e2e", "questions", n + ".wav"), dtype="float32")[0]
th = Thinker()
assert th.prefill_graphs, "no prefill graphs captured"
graphs = th.prefill_graphs


def states():
    out = []
    for st in th._layers:
        if isinstance(st, _KDAState):
            out.append(st.keys.float().clone())
            out += [c.float().clone() for c in st.values]
    return out


def turn(wav, use_graphs, text=None):
    """Prefill one turn (after the earlier turns already in the cache); returns logits, states, first token, time."""
    th.prefill_graphs = graphs if use_graphs else {}
    cap = {}
    orig = th._prefill

    def pf(ids, feats):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        tok = orig(ids, feats)
        torch.cuda.synchronize(); cap["ms"] = (time.perf_counter() - t0) * 1000
        cap["T"] = ids.shape[1]
        cap["logits"] = (th.p_logits if use_graphs and cap["T"] <= max(engine.PREFILL_BUCKETS) else None)
        cap["states"] = states()
        cap["tok"] = tok
        return tok
    th._prefill = pf
    try:
        reply = "".join(th.stream(wav, max_new_tokens=40, user_text=text))
    finally:
        th._prefill = orig
        th.prefill_graphs = graphs
    return cap, reply


ok = True
convo = [("capital", None), ("followup", None), ("sleep", None), (None, "And what is two plus two?")]
for rep in range(2):
    th.reset(); ea = []
    for name, text in convo:
        ea.append(turn(load(name) if name else None, False, text))
    th.reset(); gr = []
    for name, text in convo:
        gr.append(turn(load(name) if name else None, True, text))
    if rep == 0:
        continue  # first pass warms up both paths
    for (name, text), (ce, re_), (cg, rg) in zip(convo, ea, gr):
        # relative L2 per state tensor; bf16 matmuls over padded shapes round differently, and the drift grows
        # through the layers: merely switching eager KDA between its chunked and recurrent kernels moves these
        # states by up to ~0.09 in one turn, and later turns carry the earlier drift: 0.15 (layer 0, before any
        # matmul drift, is bit-identical)
        dif = max(float((a - b).norm() / (a.norm() + 1e-9)) for a, b in zip(ce["states"], cg["states"]))
        same = re_ == rg
        print(f"{name or 'text':9s} T={ce['T']:3d}: eager {ce['ms']:5.1f} ms -> graph {cg['ms']:5.1f} ms | "
              f"first token {'same' if ce['tok'] == cg['tok'] else 'DIFFERENT'} | worst state rel. L2 diff {dif:.4f} | "
              f"reply {'same' if same else 'differs'}")
        if not same:
            print("   eager:", repr(re_[:100]), "\n   graph:", repr(rg[:100]))
        ok &= ce["tok"] == cg["tok"] and dif < 0.15
print("PASS" if ok else "FAIL")
