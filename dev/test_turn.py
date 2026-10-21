"""Smart Turn endpoint scores on the e2e clips: complete questions vs. prefixes cut mid-sentence."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf
from turn_detect import TurnDetector

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
td = TurnDetector()
qdir = os.path.join(HERE, "e2e", "questions")
sil = np.zeros(int(0.15 * 16000), np.float32)  # what the server keeps after the speech
for name in sorted(os.listdir(qdir)):
    w, sr = sf.read(os.path.join(qdir, name), dtype="float32")
    full = td.prob(np.concatenate([w, sil]))
    cuts = [td.prob(np.concatenate([w[:int(len(w) * f)], sil])) for f in (0.35, 0.55, 0.75)]
    print(f"{name:14s} complete {full:.2f} | cut 35/55/75%: " + " ".join(f"{c:.2f}" for c in cuts))
w, _ = sf.read(os.path.join(qdir, "pause.wav"), dtype="float32")
a = w[: int(1.05 * 16000)]
print("'Can you tell me' alone:", round(td.prob(np.concatenate([a, sil])), 3))
t = time.perf_counter()
for _ in range(20):
    td.prob(w)
print(f"{(time.perf_counter() - t) / 20 * 1000:.1f} ms per call")
