"""OpenAI-compatible HTTP API (the subset that maps onto a speech model), on the same server as the web app.

  GET  /v1/models
  POST /v1/audio/speech          text -> speech (wav, mp3, flac, pcm; streamed for wav/pcm)
  POST /v1/audio/transcriptions  speech -> text (multipart upload, like Whisper's endpoint)
  POST /v1/chat/completions      chat with text and/or audio input (`input_audio` parts), text and optionally
                                 spoken output (`modalities: ["text", "audio"]`), streamed with `stream: true`

Requests share the GPU with the live voice session (one at a time); chat requests run on an isolated copy
of the conversation state, so they never disturb an ongoing voice conversation. Generation is greedy
(temperature/top_p are accepted and ignored). `pcm` audio is 16-bit mono at 24 kHz like OpenAI's;
wav/mp3/flac keep the model's native 44.1 kHz.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
import uuid

import numpy as np
import soundfile as sf
import torch
import torchaudio
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from engine import DEFAULT_SYSTEM_PROMPT, Fragmenter, Talker, log

router = APIRouter(tags=["OpenAI-compatible API"])
MODEL_ID = "ming-3.0-audio-tiny"
SR = Talker.SR  # 44100
PCM_SR = 24000
CTX: dict = {}  # engine, gpu lock, voices: filled by setup()

# OpenAI voice names -> the closest built-in voice, so existing clients work unchanged
OPENAI_VOICES = {"alloy": "US_Stella", "nova": "US_Emma", "shimmer": "UK_Charlotte", "coral": "US_Emma",
                 "sage": "UK_Charlotte", "echo": "US_James", "onyx": "UK_George", "fable": "UK_George",
                 "ash": "US_James", "ballad": "UK_George", "verse": "US_James"}


def setup(engine: dict, gpu_lock: threading.Lock, voices: list[str]):
    CTX.update(engine=engine, lock=gpu_lock, voices=voices)


def _err(msg: str, status: int = 400, code: str | None = None):
    return JSONResponse({"error": {"message": msg, "type": "invalid_request_error", "param": None, "code": code}},
                        status_code=status)


def _voice(name: str | None):
    eng, voices = CTX["engine"], CTX["voices"]
    name = name or voices[0]
    if name in eng.get("custom_voices", {}):
        return eng["custom_voices"][name]
    if name in voices:
        return name
    return OPENAI_VOICES.get(name.lower(), voices[0])


def _ready():
    return bool(CTX["engine"].get("ready"))


class StreamResampler:
    """Chunk-wise 44.1 kHz -> 24 kHz with no seams: each call resamples a little history along with the new
    audio and only emits output that is far enough from the edges to be final."""

    def __init__(self, sr_in=SR, sr_out=PCM_SR):
        from math import gcd
        g = gcd(sr_in, sr_out)
        self.sr_in, self.sr_out, self.step = sr_in, sr_out, sr_in // g  # history is kept in whole steps
        self.buf = np.zeros(0, np.float32)
        self.out_pos = 0  # output samples (relative to buf start) already emitted

    def _resample(self, x):
        return torchaudio.functional.resample(torch.from_numpy(x), self.sr_in, self.sr_out).numpy()

    def push(self, x: np.ndarray, final=False) -> np.ndarray:
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        y = self._resample(self.buf)
        end = len(y) if final else max(self.out_pos, len(y) - 64)  # last ~2.7 ms may still change
        out = y[self.out_pos:end]
        self.out_pos = end
        keep = (min(len(self.buf), 4 * 1024) // self.step) * self.step  # ~90 ms of filter history
        drop = len(self.buf) - keep
        if drop > 0 and not final:
            drop = (drop // self.step) * self.step
            self.buf = self.buf[drop:]
            self.out_pos -= drop * self.sr_out // self.sr_in
        return out


def _pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


def _encode(audio: np.ndarray, fmt: str) -> tuple[bytes, str]:
    if fmt == "pcm":
        return _pcm16(StreamResampler().push(audio, final=True)), "audio/pcm"
    buf = io.BytesIO()
    if fmt == "mp3":
        sf.write(buf, audio, SR, format="MP3")
        return buf.getvalue(), "audio/mpeg"
    if fmt == "flac":
        sf.write(buf, audio, SR, format="FLAC")
        return buf.getvalue(), "audio/flac"
    sf.write(buf, audio, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue(), "audio/wav"


def _wav_stream_header(sr=SR):
    """WAV header with 'unknown' sizes, for streaming (players read until the connection closes)."""
    import struct
    return (b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", 0xFFFFFFFF))


async def _thread_stream(work, cancel: threading.Event):
    """Run work(emit) in a thread holding the GPU lock; yield what it emits. Stops the work if the client goes away."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()
    done = object()

    def run():
        try:
            with CTX["lock"]:
                if not cancel.is_set():
                    work(lambda item: loop.call_soon_threadsafe(q.put_nowait, item))
        except Exception as e:  # surfaced to the client as an error event
            log("api error:", repr(e))
            loop.call_soon_threadsafe(q.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, done)

    threading.Thread(target=run, daemon=True).start()
    try:
        while (item := await q.get()) is not done:
            yield item
    finally:
        cancel.set()


def _read_audio(data: bytes) -> np.ndarray:
    """Any soundfile-readable audio -> 16 kHz mono float32."""
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav.numpy()


# ------------------------------------------------------------------------------------------------ models

@router.get("/v1/models")
def list_models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "inclusionAI"}]}


# ------------------------------------------------------------------------------------------------ speech

class SpeechRequest(BaseModel):
    input: str = Field(..., description="Text to speak (up to 4096 characters).")
    model: str = MODEL_ID
    voice: str = Field("US_Stella", description="Built-in voice (US_Stella, US_Emma, US_James, UK_George, "
                       "UK_Charlotte, 灵小柔, 灵楷), a cloned voice's name, or an OpenAI voice name (mapped).")
    response_format: str = Field("wav", description="wav | mp3 | flac | pcm (s16le mono 24 kHz)")
    speed: float = Field(1.0, description="Accepted for compatibility; ignored.")
    stream: bool = Field(True, description="Stream audio as it is generated (wav and pcm only).")


@router.post("/v1/audio/speech", response_class=Response)
async def speech(req: SpeechRequest):
    """Text to speech. wav and pcm stream as they are generated (first audio after ~0.1-0.2 s)."""
    if not _ready():
        return _err("model is still loading", 503)
    text = req.input.strip()[:4096]
    fmt = req.response_format.lower()
    if fmt not in ("wav", "mp3", "flac", "pcm"):
        return _err(f"unsupported response_format {fmt!r} (wav, mp3, flac, pcm)")
    if not text:
        return _err("input is empty")
    voice = _voice(req.voice)
    cancel = threading.Event()

    def work(emit):
        for pcm in CTX["engine"]["talker"].synth_stream(text, voice, cancel=cancel):
            if cancel.is_set():
                break
            emit(pcm)

    if req.stream and fmt in ("wav", "pcm"):
        async def body():
            rs = StreamResampler() if fmt == "pcm" else None
            if fmt == "wav":
                yield _wav_stream_header()
            async for item in _thread_stream(work, cancel):
                if isinstance(item, Exception):
                    return
                yield _pcm16(rs.push(item) if rs else item)
            if rs:
                yield _pcm16(rs.push(np.zeros(0, np.float32), final=True))
        return StreamingResponse(body(), media_type="audio/pcm" if fmt == "pcm" else "audio/wav")
    chunks = []
    async for item in _thread_stream(work, cancel):
        if isinstance(item, Exception):
            return _err(f"synthesis failed: {item}", 500)
        chunks.append(item)
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    data, mime = _encode(audio, fmt)
    return Response(data, media_type=mime)


# ------------------------------------------------------------------------------------------------ transcription

@router.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), model: str = Form(MODEL_ID), language: str | None = Form(None),
                         prompt: str | None = Form(None), response_format: str = Form("json"),
                         temperature: float = Form(0.0)):
    """Speech to text (any length; English and Chinese). response_format: json | text | verbose_json."""
    if not _ready():
        return _err("model is still loading", 503)
    try:
        wav = _read_audio(await file.read())
    except Exception as e:
        return _err(f"could not read audio: {e}")
    dur = len(wav) / 16000

    def run():
        with CTX["lock"]:
            return CTX["engine"]["thinker"].transcribe(wav, max_new_tokens=int(dur * 7) + 16)
    text = await asyncio.get_running_loop().run_in_executor(None, run)
    if response_format == "text":
        return PlainTextResponse(text)
    if response_format == "verbose_json":
        return {"task": "transcribe", "language": language or "", "duration": round(dur, 2), "text": text}
    return {"text": text}


# ------------------------------------------------------------------------------------------------ chat

def _parts(content):
    """-> (text, list of base64 audio payloads with formats) for a message's content."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    texts, audio = [], []
    for p in content:
        t = p.get("type")
        if t == "text":
            texts.append(p.get("text", ""))
        elif t == "input_audio":
            audio.append(p.get("input_audio", {}).get("data", ""))
    return "\n".join(texts), audio


def _parse_messages(messages: list[dict]):
    """-> (system prompt or None, [(user, assistant) earlier turns], last user text, last user audio b64 or None)"""
    system = [m for m in messages if m.get("role") in ("system", "developer")]
    convo = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not convo or convo[-1].get("role") != "user":
        raise ValueError("the last message must be from the user")
    sys_prompt = "\n".join(_parts(m.get("content"))[0] for m in system).strip() or None
    turns, pending_user = [], None
    for m in convo[:-1]:
        text, audio = _parts(m.get("content"))
        if m["role"] == "user":
            text = text or ("(spoken message)" if audio else "")
            pending_user = text if pending_user is None else pending_user + "\n" + text
        else:
            if not text and isinstance(m.get("audio"), dict):
                text = m["audio"].get("transcript", "")
            turns.append((pending_user or "", text))
            pending_user = None
    last_text, last_audio = _parts(convo[-1].get("content"))
    if pending_user:
        last_text = pending_user + "\n" + last_text
    if len(last_audio) > 1:
        raise ValueError("at most one input_audio part is supported in the last message")
    return sys_prompt, turns, last_text.strip(), (last_audio[0] if last_audio else None)


class ChatRequest(BaseModel):
    model_config = {"extra": "allow", "json_schema_extra": {"examples": [
        {"model": MODEL_ID, "messages": [{"role": "user", "content": "Tell me a fun fact about otters."}]},
        {"model": MODEL_ID, "modalities": ["text", "audio"], "audio": {"voice": "US_Emma", "format": "wav"},
         "messages": [{"role": "user", "content": [{"type": "input_audio",
                                                     "input_audio": {"data": "<base64 wav>", "format": "wav"}}]}]}]}}
    messages: list[dict]
    model: str = MODEL_ID
    stream: bool = False
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    modalities: list[str] | None = Field(None, description='["text"] or ["text", "audio"]')
    audio: dict | None = Field(None, description='{"voice": "US_Stella", "format": "wav|mp3|flac|pcm16"} '
                                                 '(pcm16 when streaming)')
    stream_options: dict | None = None


@router.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    """Chat completions with optional audio in and out (OpenAI audio-chat format):
    user content parts `{"type": "input_audio", "input_audio": {"data": <base64>, "format": "wav"}}`;
    `"modalities": ["text", "audio"], "audio": {"voice": "US_Stella", "format": "wav"}` for a spoken reply.
    Generation is greedy; tools/function calling are not supported."""
    if not _ready():
        return _err("model is still loading", 503)
    body = req.model_dump()
    try:
        sys_prompt, turns, text, audio_b64 = _parse_messages(body.get("messages") or [])
        wav = _read_audio(base64.b64decode(audio_b64)) if audio_b64 else None
    except Exception as e:
        return _err(str(e))
    if not text and wav is None:
        return _err("the last user message is empty")
    max_tokens = int(body.get("max_completion_tokens") or body.get("max_tokens") or 512)
    modalities = body.get("modalities") or ["text"]
    want_audio = "audio" in modalities
    audio_cfg = body.get("audio") or {}
    voice = _voice(audio_cfg.get("voice"))
    afmt = (audio_cfg.get("format") or ("pcm16" if body.get("stream") else "wav")).lower()
    if want_audio and afmt not in ("wav", "mp3", "flac", "pcm16"):
        return _err(f"unsupported audio format {afmt!r} (wav, mp3, flac, pcm16)")
    if want_audio and body.get("stream") and afmt != "pcm16":
        return _err("streaming audio output requires audio.format = pcm16")
    stream = bool(body.get("stream"))
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    cancel = threading.Event()
    eng = CTX["engine"]
    th, tk = eng["thinker"], eng["talker"]
    usage = {}

    def work(emit):
        """Emits ("text", delta) and ("audio", float32 chunk) items."""
        with th.isolated():
            th.reset(sys_prompt or DEFAULT_SYSTEM_PROMPT)  # not the live session's persona
            hist = th.render_history(turns)
            # keep the history within half the context (oldest turns dropped first)
            while turns and len(th.tok(hist, add_special_tokens=False).input_ids) > th.max_ctx // 2:
                turns.pop(0)
                hist = th.render_history(turns)
            frag_q = None
            if want_audio:
                import queue
                frag_q = queue.Queue()

                def talker_loop():
                    while (piece := frag_q.get()) is not None and not cancel.is_set():
                        for pcm in tk.synth_stream(piece, voice, cancel=cancel):
                            emit(("audio", pcm))
                tt = threading.Thread(target=talker_loop, daemon=True)
                tt.start()
            fr = Fragmenter()
            n = 0
            try:
                for delta in th.stream(wav, cancel=cancel, max_new_tokens=max_tokens, user_text=text or None,
                                       history_text=hist):
                    n += 1
                    emit(("text", delta))
                    if frag_q is not None:
                        for piece in fr.push(delta):
                            frag_q.put(piece)
                if frag_q is not None:
                    for piece in fr.flush():
                        frag_q.put(piece)
            finally:
                if frag_q is not None:
                    frag_q.put(None)
                    tt.join()
            gen_tokens = len(th.tok(th.history[-1]["content"], add_special_tokens=False).input_ids) if th.history else 0
            usage.update(prompt_tokens=max(0, th.cur_len - gen_tokens), completion_tokens=gen_tokens,
                         total_tokens=th.cur_len)

    if not stream:
        reply, chunks = "", []
        async for item in _thread_stream(work, cancel):
            if isinstance(item, Exception):
                return _err(f"generation failed: {item}", 500)
            kind, val = item
            if kind == "text":
                reply += val
            else:
                chunks.append(val)
        msg = {"role": "assistant", "content": reply.strip(), "refusal": None}
        if want_audio:
            audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
            data, _ = _encode(audio, "pcm" if afmt == "pcm16" else afmt)
            msg["content"] = None
            msg["audio"] = {"id": "audio_" + uuid.uuid4().hex[:16], "expires_at": created + 3600,
                            "data": base64.b64encode(data).decode(), "transcript": reply.strip()}
        return {"id": cid, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}], "usage": usage}

    async def sse():
        def chunk(delta, finish=None):
            return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                          "model": MODEL_ID, "choices": [{"index": 0, "delta": delta,
                                                                          "finish_reason": finish}]},
                                         ensure_ascii=False) + "\n\n"
        yield chunk({"role": "assistant", "content": "" if not want_audio else None})
        rs = StreamResampler() if want_audio else None
        aid = "audio_" + uuid.uuid4().hex[:16]
        async for item in _thread_stream(work, cancel):
            if isinstance(item, Exception):
                yield "data: " + json.dumps({"error": {"message": str(item)}}) + "\n\n"
                break
            kind, val = item
            if kind == "text":
                yield chunk({"audio": {"id": aid, "transcript": val}} if want_audio else {"content": val})
            else:
                yield chunk({"audio": {"id": aid, "data": base64.b64encode(_pcm16(rs.push(val))).decode()}})
        if rs is not None:
            tail = rs.push(np.zeros(0, np.float32), final=True)
            if len(tail):
                yield chunk({"audio": {"id": aid, "data": base64.b64encode(_pcm16(tail)).decode()}})
        yield chunk({}, "stop")
        if (body.get("stream_options") or {}).get("include_usage"):
            yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                         "model": MODEL_ID, "choices": [], "usage": usage}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers={"cache-control": "no-cache"})
