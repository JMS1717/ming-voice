import gc, os, torch
from silero_vad import load_silero_vad
def rss(): return int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**20
print(f"start {rss():.0f} MB")
for i in range(6):
    m = load_silero_vad()
    for _ in range(50):
        m(torch.zeros(512), 16000)
    del m; gc.collect()
    print(f"after load #{i+1}: {rss():.0f} MB")
