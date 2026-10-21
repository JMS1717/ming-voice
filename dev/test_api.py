"""Exercise the OpenAI-compatible API against a running server (stdlib only, so it doubles as a usage example).

    python dev/test_api.py                                  # http://localhost:8000
    BASE=https://<pc-ip>:8443 KEY=<key> python dev/test_api.py   # through the phone/LAN entry point
"""
import base64, io, json, os, ssl, sys, time, uuid, urllib.error, urllib.request, wave

BASE = os.environ.get("BASE", "http://localhost:8000").rstrip("/")
KEY = os.environ.get("KEY")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CTX = ssl._create_unverified_context()  # the LAN entry point uses a self-signed certificate
results = []


def req(path, data=None, headers=None, method=None):
    h = dict(headers or {})
    if KEY:
        h["Authorization"] = f"Bearer {KEY}"
    if isinstance(data, (dict, list)):
        data = json.dumps(data).encode()
        h.setdefault("Content-Type", "application/json")
    r = urllib.request.Request(BASE + path, data=data, headers=h, method=method)
    return urllib.request.urlopen(r, context=CTX, timeout=120)


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)


def wav_seconds(data):
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate(), w.getframerate()


def multipart(fields, files):
    b = uuid.uuid4().hex
    out = io.BytesIO()
    for k, v in fields.items():
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, (fname, data, mime) in files.items():
        out.write(f"--{b}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{fname}\"\r\n"
                  f"Content-Type: {mime}\r\n\r\n".encode() + data + b"\r\n")
    out.write(f"--{b}--\r\n".encode())
    return out.getvalue(), {"Content-Type": f"multipart/form-data; boundary={b}"}


def sse(path, body):
    events = []
    with req(path, body) as r:
        for line in r:
            line = line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                events.append((time.perf_counter(), json.loads(line[6:])))
    return events


def voice_turn(ws_conn, text):
    """One typed turn in the live (WebSocket) conversation; returns the reply text."""
    import asyncio
    async def go():
        await ws_conn.send(json.dumps({"type": "text", "text": text}))
        while True:
            m = await ws_conn.recv()
            if isinstance(m, str) and json.loads(m)["type"] == "done":
                return json.loads(m)["text"]
    return asyncio.get_event_loop().run_until_complete(go())


try:  # a live conversation running alongside the API calls (needs the `websockets` package)
    import asyncio, websockets
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    hdrs = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    url, sslctx = BASE.replace("http", "ws", 1) + "/ws", CTX if BASE.startswith("https") else None
    try:
        live = loop.run_until_complete(websockets.connect(url, additional_headers=hdrs, ssl=sslctx, max_size=2**24))
    except TypeError:  # websockets < 14
        live = loop.run_until_complete(websockets.connect(url, extra_headers=hdrs, ssl=sslctx, max_size=2**24))
    loop.run_until_complete(live.recv())
    loop.run_until_complete(live.send(json.dumps({"type": "reset"})))
    voice_turn(live, "Remember this: my favorite color is green. Just say okay.")
except ImportError:
    live = None
ctx0 = json.load(req("/api/health")).get("ctx_tokens")

# models
m = json.load(req("/v1/models"))
check("GET /v1/models", m["data"][0]["id"] == "ming-3.0-audio-tiny")

# speech: buffered wav, streamed wav (time to first byte), mp3, pcm
text = "The quick brown fox jumps over the lazy dog."
t = time.perf_counter()
wav = req("/v1/audio/speech", {"input": text, "voice": "alloy", "stream": False}).read()
dur, sr = wav_seconds(wav)
check("speech wav", sr == 44100 and 1.5 < dur < 8, f"{dur:.2f} s in {time.perf_counter() - t:.2f} s")
t = time.perf_counter()
with req("/v1/audio/speech", {"input": text, "response_format": "pcm"}) as r:
    first = r.read(4096)
    t_first = time.perf_counter() - t
    pcm = first + r.read()
check("speech pcm streamed (24 kHz)", abs(len(pcm) / 2 / 24000 - dur) < 0.25,
      f"{len(pcm) / 48000:.2f} s, first bytes after {t_first * 1000:.0f} ms")
mp3 = req("/v1/audio/speech", {"input": "Hello there.", "response_format": "mp3"}).read()
check("speech mp3", mp3[:3] == b"ID3" or mp3[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"), f"{len(mp3)} bytes")
try:
    req("/v1/audio/speech", {"input": "x", "response_format": "aac"})
    check("speech rejects bad format", False)
except urllib.error.HTTPError as e:
    check("speech rejects bad format", e.code == 400)

# transcription of our own speech
body, h = multipart({"model": "ming-3.0-audio-tiny"}, {"file": ("fox.wav", wav, "audio/wav")})
tr = json.load(req("/v1/audio/transcriptions", body, h))
check("transcription", "fox" in tr["text"].lower() and "lazy" in tr["text"].lower(), repr(tr["text"]))
body, h = multipart({"response_format": "text"}, {"file": ("fox.wav", wav, "audio/wav")})
check("transcription text format", "fox" in req("/v1/audio/transcriptions", body, h).read().decode().lower())

# chat: text, multi-turn, audio in, audio out, streaming
c = json.load(req("/v1/chat/completions", {"model": "ming-3.0-audio-tiny", "messages": [
    {"role": "user", "content": "What is the capital of Japan? One short sentence."}]}))
check("chat text", "tokyo" in c["choices"][0]["message"]["content"].lower(), repr(c["choices"][0]["message"]["content"]))
c = json.load(req("/v1/chat/completions", {"messages": [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "My name is Priya."}, {"role": "assistant", "content": "Nice to meet you, Priya."},
    {"role": "user", "content": "What is my name?"}]}))
check("chat history", "priya" in c["choices"][0]["message"]["content"].lower(), repr(c["choices"][0]["message"]["content"]))
q = open(os.path.join(HERE, "e2e", "questions", "capital.wav"), "rb").read()
c = json.load(req("/v1/chat/completions", {"messages": [{"role": "user", "content": [
    {"type": "input_audio", "input_audio": {"data": base64.b64encode(q).decode(), "format": "wav"}}]}],
    "modalities": ["text", "audio"], "audio": {"voice": "US_Emma", "format": "wav"}}))
msg = c["choices"][0]["message"]
adur, _ = wav_seconds(base64.b64decode(msg["audio"]["data"]))
check("chat audio in -> audio out", "paris" in msg["audio"]["transcript"].lower() and adur > 1,
      f"{msg['audio']['transcript']!r}, {adur:.1f} s of audio, usage {c['usage']}")
t = time.perf_counter()
ev = sse("/v1/chat/completions", {"stream": True, "messages": [{"role": "user", "content": "Count from one to five."}],
                                   "stream_options": {"include_usage": True}})
txt = "".join(e["choices"][0]["delta"].get("content") or "" for _, e in ev if e["choices"])
first_tok = next((ts for ts, e in ev if e["choices"] and e["choices"][0]["delta"].get("content")), None)
check("chat streaming text", "five" in txt.lower() and bool(ev[-1][1].get("usage")),
      f"{txt!r}, first token after {(first_tok - t) * 1000:.0f} ms")
t = time.perf_counter()
ev = sse("/v1/chat/completions", {"stream": True, "modalities": ["text", "audio"], "audio": {"voice": "UK_George", "format": "pcm16"},
                                   "messages": [{"role": "user", "content": "Say hello in one short sentence."}]})
audio = b"".join(base64.b64decode(e["choices"][0]["delta"]["audio"].get("data", "")) for _, e in ev
                 if e["choices"] and e["choices"][0]["delta"].get("audio"))
first_audio = next((ts for ts, e in ev if e["choices"] and (e["choices"][0]["delta"].get("audio") or {}).get("data")), None)
check("chat streaming audio (pcm16 24 kHz)", len(audio) / 48000 > 0.5,
      f"{len(audio) / 48000:.2f} s, first audio after {(first_audio - t) * 1000:.0f} ms")

ctx1 = json.load(req("/api/health")).get("ctx_tokens")
check("API calls left the voice conversation untouched", ctx0 == ctx1, f"context {ctx0} -> {ctx1}")
if live is not None:
    a = voice_turn(live, "What is my favorite color?")
    check("live conversation still remembers its context", "green" in a.lower(), repr(a))
    loop.run_until_complete(live.close())
print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
