import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
import engine
th = engine.Thinker.__new__(engine.Thinker)
th.tok = AutoTokenizer.from_pretrained(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model"))
th.history = [{"user": "My name is Jason and my favorite color is green.", "content": "Nice to meet you, Jason!"}]
th.history += [{"user": f"Question number {i} about something else entirely?", "content": f"Here is a fairly ordinary answer number {i}."} for i in range(200)]
txt = th._rollover_text(budget=1500)
n = len(th.tok(txt, add_special_tokens=False).input_ids)
print("tokens", n, "| pinned first turn:", "Jason" in txt, "| has latest turn:", "number 199" in txt, "| oldest kept recent:", txt.count("<role>HUMAN</role>"), "turns")
