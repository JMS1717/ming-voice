"""Audio encoder: trimmed per-chunk encoding vs. the reference (padded batch), and flash vs. math SDPA.
Run twice, with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=0 and =1; each run saves its features and compares
with the other run's file if present."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf, torch
import engine
from engine import Thinker, DEVICE, DTYPE, pad_to_bucket

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
q = [sf.read(os.path.join(HERE, "e2e", "questions", f + ".wav"), dtype="float32")[0] for f in ("capital", "sleep", "story")]
speech = np.concatenate(q * 12)
flash = os.environ.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1"
th = Thinker()
ref_fn = type(th.model)._get_audio_features.__get__(th.model)
feats = {}
with torch.inference_mode():
    for sec in (3, 20, 35, 55):
        wav = pad_to_bucket(speech[: sec * 16000])
        mel = th._mel(torch.from_numpy(wav).to(DEVICE), n_mels=th.n_mels).transpose(0, 1).unsqueeze(0).contiguous().to(DTYPE)
        ln = torch.tensor([[mel.shape[1]]], dtype=torch.long)
        a = ref_fn(mel, ln).float()
        b = th.model._get_audio_features(mel, ln).float()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(3):
            th.model._get_audio_features(mel, ln)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t) / 3 * 1e3
        cos = torch.nn.functional.cosine_similarity(a[0], b[0], dim=-1)
        print(f"{sec:2d} s: shapes {tuple(a.shape)} vs {tuple(b.shape)} | trimmed vs padded cos min {cos.min():.5f} | "
              f"{ms:.1f} ms (flash={flash})")
        feats[sec] = b.cpu()
    reply = "".join(th.stream(speech[: 3 * 16000], max_new_tokens=40))
    asr = th.transcribe(speech[: 40 * 16000], max_new_tokens=400)
    trimmed_fn = th.model._get_audio_features
    th.model._get_audio_features = ref_fn
    asr_ref = th.transcribe(speech[: 40 * 16000], max_new_tokens=400)
    th.model._get_audio_features = trimmed_fn
print("reply:", reply)
print("asr 40 s, trimmed:", asr)
print("asr 40 s, padded: ", asr_ref)
out = os.path.join(HERE, "dev", "scratch", f"enc_flash{int(flash)}.pt")
os.makedirs(os.path.dirname(out), exist_ok=True)
torch.save(feats, out)
other = os.path.join(HERE, "dev", "scratch", f"enc_flash{int(not flash)}.pt")
if os.path.exists(other):
    o = torch.load(other)
    for sec in feats:
        cos = torch.nn.functional.cosine_similarity(feats[sec][0], o[sec][0], dim=-1)
        print(f"flash vs math, {sec} s: cos min {cos.min():.5f} mean {cos.mean():.5f}")
