import io, os, json, urllib.request, urllib.parse, soundfile as sf, numpy as np
B = "http://localhost:8000"
ref = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model", "voice_presets", "voice_audio", "US_James.wav"), "rb").read()
r = json.loads(urllib.request.urlopen(urllib.request.Request(B + "/api/voices?name=James%20clone", data=ref, method="POST"), timeout=120).read())
print("create:", r)
text = "Hi! This is a test of a cloned voice. The weather is lovely today."
wav = urllib.request.urlopen(B + "/api/tts?" + urllib.parse.urlencode({"text": text, "voice": "James clone"}), timeout=120).read()
a, sr = sf.read(io.BytesIO(wav)); print("tts:", round(len(a) / sr, 2), "s at", sr)
open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "e2e", "clone_test.wav"), "wb").write(wav)
heard = json.loads(urllib.request.urlopen(urllib.request.Request(B + "/api/asr", data=wav, method="POST"), timeout=120).read())["text"]
print("asr:", heard)
