import os, sys, time
import numpy as np, soundfile as sf, torch, torchaudio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import Thinker, DEVICE, DTYPE

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = "You are a helpful voice assistant. Answer briefly."
wav, sr = sf.read(os.path.join(HERE, "samples/introduce_ant_group.wav"), dtype="float32", always_2d=True)
wav = torchaudio.functional.resample(torch.from_numpy(wav.mean(1)), sr, 16000).numpy()
th = Thinker(use_graphs=False)

# 1) logits agreement: HF dynamic path vs static path on the same prompt + forced continuation
msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": [{"type": "audio", "audio": (torch.from_numpy(wav), 16000)}]}]
inp = th.processor(msgs).to(DEVICE)
inp["audio_feats"] = inp["audio_feats"].to(DTYPE)
cont = th.tok("蚂蚁集团是一家以支付宝为核心的互联网开放平台", add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
with torch.inference_mode():
    full_ids = torch.cat([inp["input_ids"], cont], 1)
    ref = th.model(input_ids=full_ids, audio_feats=inp["audio_feats"], audio_feats_lengths=inp["audio_feats_lengths"]).logits[0, -1].float()
    th.reset(SYS)
    # static: prefill prompt, then feed continuation tokens one at a time (decode path)
    th._prefill(inp["input_ids"], dict(audio_feats=inp["audio_feats"], audio_feats_lengths=inp["audio_feats_lengths"]))
    for t in cont[0].tolist():
        th.tok_buf.fill_(t); th.pos_buf.fill_(th.cur_len)
        h = th.model.model.word_embeddings(th.tok_buf)
        lg = th._forward(h, th.pos_buf)[0].float()
        th.cur_len += 1
    for name, l in (("HF", ref), ("static", lg)):
        top = l.topk(3)
        print(name, [(th.tok.decode([i]), round(v, 2)) for v, i in zip(top.values.tolist(), top.indices.tolist())])
    print("logit max abs diff", (ref - lg).abs().max().item(), "cos", torch.nn.functional.cosine_similarity(ref, lg, 0).item())

# 2) GPU kernel time per decode token
from torch.profiler import profile, ProfilerActivity
with torch.inference_mode():
    for _ in range(3):
        th._decode_step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(5):
            th._decode_step()
        torch.cuda.synchronize()
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=80))
