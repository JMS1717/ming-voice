import io, sys, json, time, urllib.request, urllib.parse, soundfile as sf
for text in ["The capital of France is Paris.", "Hi! This is a quick test.", "Keep your bedroom cool, dark, and quiet for better sleep."]:
    t = time.time()
    wav = urllib.request.urlopen("http://localhost:8000/api/tts?" + urllib.parse.urlencode({"text": text, "voice": "US_Stella"}), timeout=200).read()
    a, sr = sf.read(io.BytesIO(wav))
    heard = json.loads(urllib.request.urlopen(urllib.request.Request("http://localhost:8000/api/asr", data=wav, method="POST"), timeout=200).read())["text"]
    print(f"{len(a)/sr:5.2f}s audio in {time.time()-t:4.1f}s | {text!r} -> {heard[:90]!r}", flush=True)
