"""Wyoming protocol server: Ming as speech-to-text, text-to-speech and conversation agent for Home Assistant.

Home Assistant's Assist pipelines (Voice PE, ESPHome and Wyoming satellites, the Assist button in the app) can
then use Ming for any of their steps. In Home Assistant: Settings -> Devices & services -> Add integration ->
Wyoming Protocol, host = this PC's address, port 10300; then pick Ming in Settings -> Voice assistants.

    python wyoming_bridge.py [--port 10300] [--server http://127.0.0.1:8000] [--allow 10.0.0.0/24,...]

Standard library only, so it runs with any Python 3.9+: with a WSL install, Windows' own Python (WSL2 is only
reachable from this PC, like windows/lan_proxy.py); on Linux, the app's .venv. It forwards to the server's HTTP
API: /v1/audio/transcriptions, /v1/audio/speech, and /v1/chat/completions with the assistant's own tools (web
search, weather, timers, Home Assistant...). The Wyoming protocol has no authentication, so only addresses on
private networks may connect (--allow narrows that down further).
"""
from __future__ import annotations

import argparse
import asyncio
import io
import ipaddress
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave

VERSION = "1.7.0"
PROTOCOL = "1.8.0"  # (sent in event headers like the wyoming package does; clients don't check it)
MODEL = "ming-3.0-audio-tiny"
ATTRIBUTION = {"name": "inclusionAI", "url": "https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny"}
LANGUAGES = ["en", "zh"]
FALLBACK_VOICES = ["US_Stella", "US_Emma", "US_James", "UK_George", "UK_Charlotte"]
PCM = {"rate": 24000, "width": 2, "channels": 1}  # the API's "pcm" speech format
# A sentence end followed by more text: streamed text (HA's LLM replies) is spoken a sentence at a time
SENTENCE_END = re.compile(r"[.!?。！？…]+[\"'”)\]]*\s+")
HISTORY_TURNS = 8  # exchanges of a conversation kept as context
HISTORY_IDLE_S = 1800  # a conversation unused this long is forgotten


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------ Ming's HTTP API

class Ming:
    def __init__(self, base: str, key: str = ""):
        self.base, self.key = base.rstrip("/"), key

    def open(self, path: str, body: bytes | None = None, ctype: str = "application/json", timeout: float = 120):
        req = urllib.request.Request(self.base + path, data=body, method="POST" if body is not None else "GET")
        if body is not None:
            req.add_header("Content-Type", ctype)
        if self.key:
            req.add_header("Authorization", "Bearer " + self.key)
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:  # the API's {"error": {"message": ...}}
            try:
                msg = json.loads(e.read()).get("error", {}).get("message")
            except Exception:
                msg = None
            raise RuntimeError(msg or f"HTTP {e.code}") from None
        except OSError as e:
            raise RuntimeError(f"Ming Voice is not reachable at {self.base} ({e})") from None

    def get_json(self, path: str, obj=None, timeout: float = 120):
        body = json.dumps(obj).encode() if obj is not None else None
        with self.open(path, body, timeout=timeout) as r:
            return json.loads(r.read())

    def transcribe(self, wav_bytes: bytes) -> str:
        b = "ming" + uuid.uuid4().hex
        body = (f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
                f"Content-Type: audio/wav\r\n\r\n").encode() + wav_bytes + f"\r\n--{b}--\r\n".encode()
        with self.open("/v1/audio/transcriptions", body, f"multipart/form-data; boundary={b}") as r:
            return json.loads(r.read()).get("text", "").strip()

    async def speech(self, text: str, voice: str | None):
        """Yields s16le mono 24 kHz audio as it is generated."""
        body = json.dumps({"input": text, "voice": voice or "", "response_format": "pcm", "stream": True}).encode()
        r = await asyncio.to_thread(self.open, "/v1/audio/speech", body)
        try:
            while chunk := await asyncio.to_thread(r.read1, 32768):
                yield chunk
        finally:
            r.close()


# ------------------------------------------------------------------------------------------------ Wyoming events

async def read_event(reader: asyncio.StreamReader):
    """-> (type, data, payload), or None at the end of the connection. Header line (JSON), then optional extra data
    (JSON, data_length bytes) and payload (payload_length bytes)."""
    line = await reader.readline()
    if not line:
        return None
    head = json.loads(line)
    data = head.get("data") or {}
    if head.get("data_length"):
        data.update(json.loads(await reader.readexactly(head["data_length"])))
    payload = await reader.readexactly(head["payload_length"]) if head.get("payload_length") else None
    return head["type"], data, payload


def encode_event(etype: str, data: dict | None = None, payload: bytes | None = None) -> bytes:
    head = {"type": etype, "version": PROTOCOL}
    extra = json.dumps(data, ensure_ascii=False).encode() if data else b""
    if extra:
        head["data_length"] = len(extra)
    if payload:
        head["payload_length"] = len(payload)
    return json.dumps(head).encode() + b"\n" + extra + (payload or b"")


# ------------------------------------------------------------------------------------------------ the service

class Bridge:
    def __init__(self, ming: Ming, services: set[str], voice: str, allow: list):
        self.ming, self.services, self.voice, self.allow = ming, services, voice, allow
        self.voices = list(FALLBACK_VOICES)
        self.home_control = False
        self.convos: dict[str, tuple[float, list]] = {}  # conversation_id -> (last used, messages)

    def allowed(self, host: str) -> bool:
        try:
            ip = ipaddress.ip_address(host.split("%")[0])
        except ValueError:
            return False
        if getattr(ip, "ipv4_mapped", None):
            ip = ip.ipv4_mapped
        if self.allow:
            return ip.is_loopback or any(ip in net for net in self.allow)
        return ip.is_private or ip.is_loopback or ip.is_link_local

    async def refresh(self):
        """The server's voices and whether it controls Home Assistant (kept from last time if it's down)."""
        try:
            v = await asyncio.to_thread(self.ming.get_json, "/api/voices", None, 5)
            self.voices = list(v.get("builtin") or []) + list(v.get("custom") or []) or self.voices
            h = await asyncio.to_thread(self.ming.get_json, "/api/health", None, 5)
            self.home_control = bool(h.get("home_assistant"))
        except Exception as e:
            log(f"couldn't ask the server for its voices ({e})")

    def info(self) -> dict:
        def artifact(name, description, **kw):
            return {"name": name, "description": description, "attribution": ATTRIBUTION, "installed": True,
                    "version": VERSION, **kw}
        model = artifact(MODEL, "Ming-3.0-audio-tiny", languages=LANGUAGES)
        data = {"asr": [], "tts": [], "handle": [], "intent": [], "wake": [], "mic": [], "snd": []}
        if "stt" in self.services:
            data["asr"].append(artifact("ming-stt", "Ming speech-to-text", models=[model],
                                        supports_transcript_streaming=False))
        if "tts" in self.services:
            voices = [artifact(v, v.replace("_", " "), languages=LANGUAGES, speakers=None) for v in self.voices]
            data["tts"].append(artifact("ming-tts", "Ming text-to-speech", voices=voices,
                                        supports_synthesize_streaming=True))
        if "conversation" in self.services:
            data["handle"].append(artifact("ming", "Ming conversation agent", models=[model],
                                           supports_handled_streaming=False,
                                           supports_home_control=self.home_control))
        return data

    def history(self, cid: str) -> list:
        now = time.time()
        for k in [k for k, (t, _) in self.convos.items() if now - t > HISTORY_IDLE_S]:
            del self.convos[k]
        _, msgs = self.convos.get(cid, (now, []))
        self.convos[cid] = (now, msgs)
        return msgs

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = (writer.get_extra_info("peername") or ("?",))[0]
        if not self.allowed(peer):
            log(f"refused a connection from {peer} (not an allowed address)")
            writer.close()
            return
        await Connection(self, reader, writer).run()


class Connection:
    """One client connection. Home Assistant opens one per request (a transcription, a reply to speak...)."""

    def __init__(self, bridge: Bridge, reader, writer):
        self.b, self.reader, self.writer = bridge, reader, writer
        self.audio, self.fmt, self.language = bytearray(), None, None
        self.streaming, self.text, self.tts_voice, self.audio_started = False, "", None, False

    async def send(self, etype, data=None, payload=None):
        self.writer.write(encode_event(etype, data, payload))
        await self.writer.drain()

    async def error(self, text):
        log("error:", text)
        await self.send("error", {"text": text, "code": "ming"})

    async def run(self):
        try:
            while (ev := await read_event(self.reader)) is not None:
                await self.handle(*ev)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except Exception as e:
            log(f"connection failed: {type(e).__name__}: {e}")
        finally:
            try:
                self.writer.close()
            except Exception:
                pass

    async def handle(self, etype, data, payload):
        if etype == "describe":
            await self.b.refresh()
            await self.send("info", self.b.info())
        elif etype == "ping":
            await self.send("pong", {"text": data.get("text")} if data.get("text") else None)
        # speech to text: transcribe, audio-start, audio-chunk..., audio-stop -> transcript
        elif etype == "transcribe":
            self.language, self.audio, self.fmt = data.get("language"), bytearray(), None
        elif etype == "audio-start":
            self.audio, self.fmt = bytearray(), (data["rate"], data["width"], data["channels"])
        elif etype == "audio-chunk":
            self.fmt = self.fmt or (data["rate"], data["width"], data["channels"])
            self.audio += payload or b""
        elif etype == "audio-stop":
            await self.transcribe()
        # text to speech: synthesize, or synthesize-start, synthesize-chunk..., synthesize-stop (streamed text)
        elif etype == "synthesize":
            if not self.streaming:  # (in a stream, this is the whole text again for older servers)
                await self.speak(data.get("text", ""), self._voice(data), final=True)
        elif etype == "synthesize-start":
            self.streaming, self.text, self.tts_voice, self.audio_started = True, "", self._voice(data), False
        elif etype == "synthesize-chunk":
            self.text += data.get("text", "")
            cut = None
            for m in SENTENCE_END.finditer(self.text):
                cut = m.end()
            if cut and len(self.text[:cut].split()) >= 2:
                piece, self.text = self.text[:cut], self.text[cut:]
                await self.speak(piece, self.tts_voice)
        elif etype == "synthesize-stop":
            await self.speak(self.text, self.tts_voice, final=True)
            self.streaming, self.text = False, ""
            await self.send("synthesize-stopped")
        # conversation agent: transcript (with context) -> handled
        elif etype == "transcript":
            await self.converse(data)

    def _voice(self, data):
        v = (data.get("voice") or {}).get("name")
        return v if v in self.b.voices else self.b.voice

    async def transcribe(self):
        if not self.audio or not self.fmt:
            await self.send("transcript", {"text": ""})
            return
        rate, width, channels = self.fmt
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setframerate(rate)
            w.setsampwidth(width)
            w.setnchannels(channels)
            w.writeframes(bytes(self.audio))
        secs = len(self.audio) / (rate * width * channels)
        self.audio = bytearray()
        t0 = time.perf_counter()
        try:
            text = await asyncio.to_thread(self.b.ming.transcribe, buf.getvalue())
        except Exception as e:
            await self.error(f"transcription failed: {e}")
            return
        log(f"heard ({secs:.1f} s audio, {time.perf_counter() - t0:.2f} s): {text!r}")
        await self.send("transcript", {"text": text, **({"language": self.language} if self.language else {})})

    async def speak(self, text, voice, final=False):
        """Synthesize text into the current audio stream (audio-start first, audio-stop when final)."""
        text = text.strip()
        if (text or final) and not self.audio_started:
            await self.send("audio-start", {**PCM, "timestamp": 0})
            self.audio_started = True
        if text:
            t0, odd = time.perf_counter(), b""
            try:
                async for chunk in self.b.ming.speech(text, voice):
                    chunk, odd = (odd + chunk), b""
                    if len(chunk) % 2:
                        chunk, odd = chunk[:-1], chunk[-1:]
                    if chunk:
                        await self.send("audio-chunk", PCM, chunk)
            except (RuntimeError, OSError) as e:
                await self.error(f"speech failed: {e}")
                final = True
            log(f"said ({time.perf_counter() - t0:.2f} s, {voice}): {text[:80]!r}")
        if final:
            await self.send("audio-stop", {"timestamp": None})
            self.audio_started = False

    async def converse(self, data):
        text = (data.get("text") or "").strip()
        ctx = data.get("context") or {}
        cid = str(ctx.get("conversation_id") or uuid.uuid4().hex)
        msgs = self.b.history(cid)
        req = {"messages": msgs + [{"role": "user", "content": text}],
               "ming": {"tools": True, "device_id": ctx.get("device_id") or ""}}
        t0 = time.perf_counter()
        try:
            r = await asyncio.to_thread(self.b.ming.get_json, "/v1/chat/completions", req, 180)
            reply = (r["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            log(f"conversation failed: {e}")
            await self.send("not-handled", {"text": "Sorry, Ming isn't available right now.", "context": ctx})
            return
        log(f"replied ({time.perf_counter() - t0:.2f} s) to {text!r}: {reply!r}")
        msgs += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        del msgs[:-2 * HISTORY_TURNS]
        await self.send("handled", {"text": reply, "context": ctx})


async def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default=os.environ.get("MING_WYOMING_HOST", "0.0.0.0"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MING_WYOMING_PORT", "10300")))
    ap.add_argument("--server", default=os.environ.get("MING_SERVER", f"http://127.0.0.1:{os.environ.get('MING_PORT', '8000')}"))
    ap.add_argument("--key", default=os.environ.get("MING_API_KEY", ""), help="the server's MING_API_KEY, if it has one")
    ap.add_argument("--voice", default=os.environ.get("MING_WYOMING_VOICE", "US_Stella"),
                    help="voice when Home Assistant doesn't pick one")
    ap.add_argument("--services", default="stt,tts,conversation", help="which to offer (default: all three)")
    ap.add_argument("--allow", default=os.environ.get("MING_WYOMING_ALLOW", ""),
                    help="addresses/networks allowed to connect, comma-separated (default: any private address)")
    args = ap.parse_args()
    allow = [ipaddress.ip_network(a.strip(), strict=False) for a in args.allow.split(",") if a.strip()]
    services = {s.strip() for s in args.services.split(",") if s.strip()}
    if not services <= {"stt", "tts", "conversation"}:
        sys.exit("--services: choose from stt, tts, conversation")
    bridge = Bridge(Ming(args.server, args.key), services, args.voice, allow)
    try:
        server = await asyncio.start_server(bridge.serve, args.host, args.port)
    except OSError as e:
        if e.errno in (98, 10048):  # (address in use: Linux, Windows)
            log(f"port {args.port} is taken: the bridge is probably running already")
            return
        raise
    await bridge.refresh()
    log(f"Wyoming bridge for Home Assistant on port {args.port} ({', '.join(sorted(services))}) -> {args.server}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
