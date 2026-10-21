"""Context sizing + between-turn compaction: VRAM at the default context, short-turn speed, how long compacting a
full conversation takes, and that the conversation survives it. Run with the server stopped (needs the GPU)."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from engine import Thinker, ROLLOVER_MAX

th = Thinker(use_graphs=True)
free, total = torch.cuda.mem_get_info()
print(f"max_ctx {th.max_ctx}, allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB, free {free / 2**30:.2f} GiB")


def turn(q, n=60):
    t = time.perf_counter()
    first, out = None, ""
    for d in th.stream(None, user_text=q, max_new_tokens=n):
        if first is None:
            first = time.perf_counter() - t
        out += d
    print(f"[ctx {th.ctx_len():5d}] first {first * 1000:4.0f} ms  Q: {q}\n               A: {out}", flush=True)
    return out


turn("My name is Sam and my favorite color is green. Remember that.")
turn("What is the capital of Japan?")
for i in range(150):  # a long day of chatting, as text
    th.history.append({"role": "assistant", "content": f"Sure, fact number {i}: octopuses have three hearts.",
                       "interrupted": False, "user": f"Tell me fun fact number {i} about octopuses."})
th.cur_len = th.max_ctx - 1000  # pretend the cache filled up
assert th.needs_compact()
t = time.time()
th.compact()
print(f"compact: {time.time() - t:.2f}s -> ctx {th.ctx_len()} (rollover budget {ROLLOVER_MAX})")
assert not th.needs_compact()
a = turn("What is my name and what is my favorite color?")
ok = "sam" in a.lower() and "green" in a.lower()
print("PASS" if ok else "FAIL")
