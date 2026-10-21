"""Synthesize spoken test questions (16 kHz) for the end-to-end test, using the Talker with another voice."""
import os, sys
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Talker

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "e2e", "questions")
os.makedirs(OUT, exist_ok=True)
Q = {
    "capital": "What is the capital of France?",
    "followup": "And roughly how many people live there?",
    "sleep": "Can you give me one quick tip for sleeping better?",
    "story": "Tell me a long story about a dragon who learns how to cook.",
    "interrupt": "Actually, never mind. What is two plus two?",
    "pause_a": "Can you tell me",
    "pause_b": "what the tallest mountain in the world is?",
}
tk = Talker()
torch.manual_seed(1)


def to16k(w):
    return torchaudio.functional.resample(torch.from_numpy(w), 44100, 16000).numpy()


clips = {}
for name, text in Q.items():
    clips[name] = to16k(tk.synthesize(text, "US_James"))
for name in ("capital", "followup", "sleep", "story", "interrupt"):
    sf.write(os.path.join(OUT, f"{name}.wav"), clips[name], 16000)
# one question with a 380 ms hesitation in the middle: should trigger and then roll back a speculative reply
gap = np.zeros(int(0.38 * 16000), np.float32)
sf.write(os.path.join(OUT, "pause.wav"), np.concatenate([clips["pause_a"], gap, clips["pause_b"]]), 16000)
print("wrote", sorted(os.listdir(OUT)))
