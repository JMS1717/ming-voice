import os, sys
os.environ["MING_MAX_CTX"] = "300"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Thinker
th = Thinker(use_graphs=True)
qs = ["My name is Jason and my favorite color is green. Remember that.",
      "Name three fruits.", "What is the capital of Japan?", "Tell me a fun fact about octopuses.",
      "What is twelve times twelve?", "Give me a tip for learning guitar.",
      "What is my name and what is my favorite color?"]
for q in qs:
    a = "".join(th.stream(None, user_text=q, max_new_tokens=60))
    print(f"[ctx {th.ctx_len():4d}] Q: {q}\n           A: {a}", flush=True)
