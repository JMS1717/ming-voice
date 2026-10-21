"""Where does prefill time go for spoken turns of different lengths? mel / audio encoder / LLM layers.
PROFILE=<seconds> additionally prints a torch.profiler kernel table for that length."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf, torch
from engine import Thinker, DEVICE, DTYPE, pad_to_bucket

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
q = [sf.read(os.path.join(HERE, "e2e", "questions", f + ".wav"), dtype="float32")[0] for f in ("capital", "sleep", "story")]
speech = np.concatenate(q * 12)
th = Thinker()


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def run(sec, profile=False):
    wav = pad_to_bucket(speech[: int(sec * 16000)])
    th.reset()
    t0 = sync()
    mel = th._mel(torch.from_numpy(wav).to(DEVICE), n_mels=th.n_mels).transpose(0, 1).unsqueeze(0).contiguous()
    n_audio = th.processor._num_audio_tokens(mel.shape[1])
    feats = dict(audio_feats=mel.to(DTYPE), audio_feats_lengths=torch.tensor([[mel.shape[1]]], dtype=torch.long))
    t1 = sync()
    audio = th.model._get_audio_features(feats["audio_feats"], feats["audio_feats_lengths"])
    t2 = sync()
    tail = f"<role>HUMAN</role><|audio_start|>{'<|audio_pad|>' * n_audio}<|audio_end|><|role_end|><role>ASSISTANT</role>\n<think></think>"
    ids = th.tok(th._system_block() + tail, return_tensors="pt").input_ids.to(DEVICE)
    h = th.model.model.word_embeddings(ids)
    h = h.masked_scatter((ids == th.model.config.audio_token_id).unsqueeze(-1).expand_as(h), audio.to(h.dtype))
    pos = torch.arange(ids.shape[1], device=DEVICE)
    th.cache.attn_len = ids.shape[1]
    t3 = sync()
    th._forward(h, pos)
    t4 = sync()
    return dict(sec=sec, tokens=ids.shape[1], mel=(t1 - t0) * 1e3, encoder=(t2 - t1) * 1e3, llm=(t4 - t3) * 1e3)


with torch.inference_mode():
    for sec in (2, 5, 10, 20, 35, 55):
        run(sec)
        r = min((run(sec) for _ in range(3)), key=lambda r: r["llm"] + r["encoder"])
        print(f"{r['sec']:3d} s audio, {r['tokens']:4d} tokens: mel {r['mel']:5.1f} ms | encoder {r['encoder']:6.1f} ms | "
              f"LLM {r['llm']:6.1f} ms", flush=True)
    if os.environ.get("PROFILE"):
        sec = int(os.environ["PROFILE"])
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            run(sec)
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=30))
