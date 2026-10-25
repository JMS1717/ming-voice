"""Realtime speech-to-speech server for Ming-3.0-audio-tiny.

Browser streams 16 kHz mono PCM16 over a WebSocket; server-side Silero VAD tracks speech.
Speculative turn-taking: after a short pause (spec_ms) the reply starts generating with its output
held back; if the pause reaches the end-of-turn threshold (end_ms) the held output is released at
once, and if the user keeps talking the reply is discarded and the Thinker state rolled back.
The Thinker streams text, cut into small fragments that are streamed through the Talker;
44.1 kHz PCM16 chunks go back as binary frames. Barge-in cancels the reply.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import warnings

# harmless library chatter that would otherwise bury the server log
warnings.filterwarnings("ignore", message=".*Efficient attention on Current AMD GPU.*")
warnings.filterwarnings("ignore", message=".*attention mask API.*", category=FutureWarning)

import numpy as np
import soundfile as sf
import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

import api
import lan
from echo_guard import EchoGuard
from engine import DEFAULT_SYSTEM_PROMPT, Fragmenter, GpuKeepAwake, Talker, Thinker, log
from tools import FILLERS, Toolbox

TOOLBOX = Toolbox()  # web search, weather, calculator, clock, timers, Home Assistant, Jellyfin, Hermes
# Things to announce at the next quiet moment (finished Hermes tasks, timers going off):
# {"prompt": for the model, "events": JSON messages for the client}
NOTICES: list = []


FATAL_GPU = re.compile(r"HIP error|CUDA error|hipBLASLt|HIPBLAS_STATUS|CUBLAS_STATUS|illegal memory access")


def check_fatal(e: BaseException):
    """A GPU error that leaves the device unusable (unlike torch's own out-of-memory, which the next turn survives):
    under run.sh's supervisor (MING_SUPERVISE=1), exit so it starts a fresh server."""
    if FATAL_GPU.search(str(e)) and not isinstance(e, torch.cuda.OutOfMemoryError) \
            and os.environ.get("MING_SUPERVISE") == "1":
        log(f"fatal GPU error, exiting for a restart: {str(e)[:200]}")
        os._exit(75)


def _notify(prompt: str, events: list, ttl: float | None = None):
    """prompt: 'what happened) what to do', completing '(Notification for you, not said by the user: '.
    ttl: drop it if it couldn't be said within this many seconds (nobody connected, a long conversation)."""
    NOTICES.append({"prompt": f"(Notification for you, not said by the user: {prompt}", "events": events,
                    "expires": time.time() + ttl if ttl else None})
    sess = ACTIVE.get("sess")
    if sess is not None:
        sess.loop.call_soon_threadsafe(sess.deliver_notices)


def _delegate_done(task: str, ok, text: str):
    """A Hermes run finished (ok True/False) or needs approval (ok None): have the assistant tell the user."""
    log(f"hermes {'done' if ok else 'failed' if ok is False else 'waiting'}: {task!r} -> {text[:200]!r}")
    what = (f'Hermes finished the task "{task}". Its answer: {text}' if ok else
            f'Hermes could not finish the task "{task}": {text}' if ok is False else f'{text} (Task: "{task}")')
    _notify(f"{what}) Tell the user briefly, in your own words.", [
        {"type": "tool", "status": "done", "name": "hermes_result", "args": {"task": task},
         "error": None if ok or ok is None else "failed"}])


def _timer_fired(t: dict, late_s: float):
    from tools import spoken_duration
    if t.get("ha"):  # "turn off the lights in 20 minutes": do it (quietly; only a failure is said)
        if late_s > 900:
            log(f"scheduled {t['ha']!r} skipped: {spoken_duration(late_s)} late")
            return
        from tools import home_assistant
        try:
            text, info = home_assistant(t["ha"], TOOLBOX.room)
            log(f"scheduled {t['ha']!r}: {text}")
            ok = not info.get("failed") and info.get("response_type") != "error"
        except Exception as e:
            log(f"scheduled {t['ha']!r} failed: {e!r}")
            ok = False
        if not ok:
            announce(f"I couldn't {t['label']} as planned.", source="Scheduled action")
        return
    say = t.get("say") or (f"Your timer for {spoken_duration(t['duration'])} is done." if t["duration"] else "Time's up.")
    if late_s > 90:
        say += f" That was {spoken_duration(late_s)} ago."
    log(f"timer fired: {say!r}")
    _alarm(say, t["label"] or f"{spoken_duration(t['duration'] or 0)} timer", ALARM_REPEATS)


# A timer that went off is said again every ALARM_REPEAT_S, up to ALARM_REPEATS more times, until the user says
# anything (a kitchen timer is easy to miss from the next room).
ALARM_REPEAT_S = float(os.environ.get("MING_ALARM_REPEAT_S", "60"))
ALARM_REPEATS = int(os.environ.get("MING_ALARM_REPEATS", "2"))
RINGING: dict = {}  # the alarm last said and not yet answered: {"say", "label", "left", "said_at"}


def _alarm(say: str, label: str, left: int):
    # (a fixed sentence: asked to phrase it, the small model sometimes "confirms" the reminder instead of giving it)
    _notify(f'a timer or reminder the user set has just gone off.) Say exactly this to the user, nothing else: "{say}"',
            [{"type": "alarm", "say": say, "label": label, "left": left},
             {"type": "tool", "status": "done", "name": "timer_done", "args": {"message": label}}])


def announce(message: str, chime: bool = True, verbatim: bool = True, source: str = "Home Assistant",
             ttl: float = 300) -> bool:
    """Say something to the user at the next quiet moment (doorbell, washer done, garage left open...).
    verbatim: say exactly `message`; else it's information to pass on in the assistant's own words."""
    message = " ".join(str(message or "").split())[:500]
    if not message:
        return False
    log(f"announcement from {source}: {message!r}")
    if verbatim:
        prompt = f'{source} sent an announcement for the user.) Say exactly this to the user, nothing else: "{message}"'
    else:
        prompt = f"{source} sent this for the user: {message}) Tell the user briefly, in your own words."
    _notify(prompt, ([{"type": "chime"}] if chime else []) +
            [{"type": "tool", "status": "done", "name": "announcement", "args": {"message": message, "source": source}}],
            ttl)
    return True


def _ha_events():
    """Listen on Home Assistant's event bus for `ming_announce` events and say them. In an HA automation:
        action: event
        event: ming_announce
        event_data: {message: "Someone is at the front door."}   # optional: chime: false, verbatim: false
    Nothing to configure in HA beyond that (the event bus needs no integration or restart)."""
    from websockets.sync.client import connect
    from tools import _ha_url, ha_token
    url = re.sub(r"^http", "ws", _ha_url()) + "/api/websocket"
    delay = 5
    while True:
        try:
            with connect(url, open_timeout=10, max_size=2**22) as ws:
                json.loads(ws.recv())  # auth_required
                ws.send(json.dumps({"type": "auth", "access_token": ha_token()}))
                if json.loads(ws.recv()).get("type") != "auth_ok":
                    log("Home Assistant events: the token was rejected; not listening")
                    return
                ws.send(json.dumps({"id": 1, "type": "subscribe_events", "event_type": "ming_announce"}))
                r = json.loads(ws.recv())
                if not r.get("success"):
                    log(f"Home Assistant events: can't subscribe ({r.get('error')}); not listening")
                    return
                log("Home Assistant: listening for ming_announce events")
                delay = 5
                for raw in ws:
                    m = json.loads(raw)
                    if m.get("type") == "event":
                        d = m["event"].get("data") or {}
                        no = (False, "false", "no", 0)
                        announce(d.get("message", ""), chime=d.get("chime", True) not in no,
                                 verbatim=d.get("verbatim", True) not in no, ttl=float(d.get("ttl", 300)))
        except Exception as e:
            log(f"Home Assistant events: {type(e).__name__}: {e}; reconnecting in {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, 300)


def _ha_watch(every_s: float = 60.0):
    """Heads-ups from Home Assistant (ha.alerts): a lock or garage door left open at night, the vacuum stuck or done.
    MING_HA_ALERTS=0 turns them off; MING_ALERT_NIGHT="22-6" is when "at night" is."""
    import ha
    from tools import _ha_url, ha_token, in_character
    try:
        start, end = (int(x) for x in os.environ.get("MING_ALERT_NIGHT", "22-6").split("-"))
    except ValueError:
        start, end = 22, 6
    seen: dict = {}
    first = True
    while True:
        try:
            msgs = ha.alerts(ha.devices(_ha_url(), ha_token(), fresh=True), seen, night=(start, end))
            for m in ([] if first else msgs):  # (what was already so at startup isn't news)
                announce(in_character(m, TOOLBOX.persona), source="Home Assistant", ttl=1800)
            first = False
        except Exception as e:
            log(f"Home Assistant heads-ups: {type(e).__name__}: {e}")
        time.sleep(every_s)


TOOLBOX.on_delegate_done = _delegate_done
TOOLBOX.timers.start(_timer_fired)

ECHO_DEBUG = bool(os.environ.get("MING_ECHO_DEBUG"))

VOICES = ["US_Stella", "US_Emma", "US_James", "UK_George", "UK_Charlotte", "灵小柔", "灵楷"]
VOICE_DIR = HERE / "voices"  # cloned voices: tiny speaker-embedding files
FRAME = 512  # 32 ms @ 16 kHz (Silero VAD window)
ENGINE: dict = {}
GPU_TURN = threading.Lock()  # one reply generating at a time
TURN_POOL = ThreadPoolExecutor(1)  # Smart Turn runs off the event loop (~20 ms on the CPU)
WAKE_POOL = ThreadPoolExecutor(1)  # wake-word checks (a short transcription, ~0.2 s on the GPU)
WAKE_CHECK_S = 1.6  # how much of an utterance's start is checked for the wake word
WAKE_ALIKE = {"jarvis": {"jarvis's", "javis", "jervis", "jarvi", "jarves", "travis", "garvis"}}  # ASR mishearings


def find_wake_word(text: str, word: str) -> tuple[bool, str]:
    """Is the wake word among the first few words ("Hey Ming, ...", "Okay Ming", "Ming, ...")? -> (found, the
    words after it). ASR sometimes writes a lone "Hey Ming" in Chinese (嘿，明)."""
    toks = re.findall(r"[a-z0-9']+|[一-鿿]", text.lower())
    name = (word.lower().split() or ["ming"])[-1]
    names = {name, name + "'s"} | ({"明", "铭"} if name == "ming" else set()) | WAKE_ALIKE.get(name, set())
    for i, t in enumerate(toks[:4]):
        if t in names:
            return True, " ".join(toks[i + 1:])
    return False, ""


def _load():
    torch.set_grad_enabled(False)
    from silero_vad import load_silero_vad

    if os.environ.get("MING_VRAM_LIMIT_GB"):  # emulate a smaller GPU (testing)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["MING_VRAM_LIMIT_GB"]) * 2**30 / total))


    ENGINE["thinker"] = Thinker()  # load first: loading it after the Talker exceeds WSL's 15 GB RAM cap
    import gc
    gc.collect()
    ENGINE["talker"] = Talker(use_graphs=os.environ.get("MING_GRAPHS", "1") == "1")
    VOICES.extend(v for v in ENGINE["talker"].designed_voices if v not in VOICES)  # (e.g. Jarvis: dev/design_voice.py)
    ENGINE["vad_proto"] = load_silero_vad
    ENGINE["turn"] = None
    if os.environ.get("MING_SMART_TURN", "1") == "1":
        import turn_detect
        if turn_detect.available():
            ENGINE["turn"] = turn_detect.TurnDetector()
            ENGINE["turn"].prob(np.zeros(16000, np.float32))
        else:
            log(f"smart turn model not found ({turn_detect.MODEL}); using the fixed end-of-turn pause")
    ENGINE["custom_voices"] = {}
    VOICE_DIR.mkdir(exist_ok=True)
    for f in sorted(VOICE_DIR.glob("*.pt")):
        try:
            ENGINE["custom_voices"][f.stem] = torch.load(f, map_location="cuda:0")
        except Exception as e:
            log(f"could not load voice {f.name}: {e}")
    _warmup()
    load_conversation(ENGINE["thinker"])
    # Start only after all graph captures (it must not launch kernels while a capture is running).
    # Without it, the first request after ~20 s of GPU idle stalls for minutes under WSL (not needed on native Linux).
    if os.environ.get("MING_GPU_KEEPAWAKE", "1" if os.path.exists("/dev/dxg") else "0") == "1":
        GpuKeepAwake().start()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    log(f"GPU memory: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, "
        f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB reserved, {free / 2**30:.2f} GiB free of {total / 2**30:.2f}")
    ENGINE["ready"] = True
    log("READY — open http://localhost:%s" % os.environ.get("PORT", "8000"))


CONVERSATION_FILE = HERE / ".conversation.json"
RESUME_S = float(os.environ.get("MING_RESUME_S", "1800"))


def save_conversation(th: Thinker):
    """Keep the conversation on disk after each turn, so a restart (a crash, an update) can pick it up again."""
    try:
        data = {"saved": time.time(), "system_prompt": th.system_prompt, "history": th.history[-100:]}
        tmp = CONVERSATION_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, default=str))
        tmp.replace(CONVERSATION_FILE)
    except Exception as e:
        log(f"could not save the conversation: {e}")


def load_conversation(th: Thinker):
    """Continue the conversation from before a restart, if it was saved less than RESUME_S ago (the first turn then
    re-enters it as text, as after a context rollover). MING_RESUME_S=0: always start afresh."""
    try:
        data = json.loads(CONVERSATION_FILE.read_text())
    except (OSError, ValueError):
        return
    age = time.time() - data.get("saved", 0)
    if not data.get("history") or age > RESUME_S:
        return
    th.reset(data.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
    th.history, th.rebuild = data["history"], True
    log(f"continuing the conversation from {age / 60:.0f} min ago ({len(th.history)} turns)")


def _warmup():
    """Compile/autotune Triton + HIP kernels for the prompt lengths real turns produce, so no user turn
    pays a JIT stall (fla autotunes per block count, i.e. per prompt-length range)."""
    th, tk = ENGINE["thinker"], ENGINE["talker"]
    t = time.time()
    rng = np.random.default_rng(0)
    txt = ""
    # Compiled kernels and autotune results persist on disk (Triton/MIOpen caches), so the full sweep
    # over input shapes is only needed once per code version; later starts do a quick warmup.
    import hashlib
    src = b"".join((HERE / f).read_bytes() for f in ("engine.py", "fast_talker.py", "fastlinear.py",
                                                     "fused_moe.py", "direct_load.py"))
    stamp_val = hashlib.sha1(src + torch.__version__.encode()).hexdigest()
    stamp = HERE / ".warmup_stamp"
    full = not (stamp.exists() and stamp.read_text().strip() == stamp_val)
    lengths = [x / 2 for x in range(1, 25)] + [14, 17, 20, 25, 30, 40, 55] if full else [1.0, 3.0, 8.0]
    log(f"warmup: {'full shape sweep (first run for this code version)' if full else 'quick'}")
    for first in (True, False):
        for sec in lengths:
            n = int(sec * 16000)
            tt = np.arange(n) / 16000
            wav = (0.05 * np.sin(2 * np.pi * 220 * tt) + 0.01 * rng.standard_normal(n)).astype(np.float32)
            if first:
                th.reset()
            snap = th.snapshot()
            txt = "".join(th.stream(wav, max_new_tokens=3))
            th.restore(snap)
    t_think = time.time()
    th.reset()
    for _ in th.stream(None, user_text="Say hi.", max_new_tokens=8):
        pass
    for n in ((500, 1500, 4000) if full else ()):  # long text prompts (a context rollover): ~14 s of JIT when cold
        th.reset()
        for _ in th.stream(None, user_text="Hello there. " * (n // 3), max_new_tokens=2):
            pass
    for k in (range(1, 25) if full else (2, 6)):  # background ASR uses the stateless HF path: warm its shapes too
        n = k * 8000
        th.transcribe((0.05 * np.sin(np.arange(n) / 16000 * 2 * np.pi * 180)).astype(np.float32), max_new_tokens=4)
    t_asr = time.time()
    th.reset()
    for text in ("Hi.", "Hello, warming up the voice. This is a slightly longer sentence to cover more shapes."):
        for _ in tk.synth_stream(text, VOICES[0]):
            pass
    torch.cuda.synchronize()
    if full:
        stamp.write_text(stamp_val)
    log(f"warmup done in {time.time() - t:.1f}s (thinker {t_think - t:.1f}s, asr {t_asr - t_think:.1f}s, "
        f"talker {time.time() - t_asr:.1f}s)")


@asynccontextmanager
async def lifespan(_app):
    await asyncio.get_running_loop().run_in_executor(None, _load)
    from tools import ha_token
    if ha_token() and os.environ.get("MING_HA_EVENTS", "1") != "0":
        threading.Thread(target=_ha_events, daemon=True, name="ha-events").start()
    if ha_token() and os.environ.get("MING_HA_ALERTS", "1") != "0":
        threading.Thread(target=_ha_watch, daemon=True, name="ha-watch").start()
    yield


app = FastAPI(lifespan=lifespan, title="Ming Voice", version="1.5",
              description="Local speech-to-speech assistant. The web app is at `/`; the OpenAI-compatible API is "
                          "under `/v1`. From other devices (LAN port) send `Authorization: Bearer <access key>`.")
api.setup(ENGINE, GPU_TURN, VOICES, TOOLBOX)
app.include_router(api.router)
app.add_middleware(lan.AccessControl)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(HERE / "static" / "index.html", headers={"Cache-Control": "no-cache"})  # pick up updates


@app.get("/manifest.webmanifest", include_in_schema=False)
def manifest():
    return JSONResponse({"name": "Ming Voice", "short_name": "Ming Voice", "start_url": "/", "display": "standalone",
                         "background_color": "#0f0f10", "theme_color": "#0f0f10",
                         "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml"}]},
                        media_type="application/manifest+json")


@app.get("/icon.svg", include_in_schema=False)
def icon():
    from fastapi.responses import Response
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><linearGradient id="g" x1="0" y1="0" '
           'x2="1" y2="1"><stop offset="0" stop-color="#7aa2ff"/><stop offset="1" stop-color="#ff8a5c"/>'
           '</linearGradient></defs><rect width="64" height="64" rx="14" fill="#0f0f10"/>'
           '<circle cx="32" cy="32" r="18" fill="url(#g)"/></svg>')
    return Response(svg, media_type="image/svg+xml")


@app.get("/api/lan")
def lan_status(request: Request):
    """Phone links + QR code (only shown to the PC itself: the key must not leak to LAN clients)."""
    if (request.scope.get("server") or (None, None))[1] == lan.LAN_PORT:
        return {"local": False}
    urls = lan.phone_urls()
    return {"local": True, "running": bool(urls), "urls": urls,
            "qr_svg": lan.qr_svg(urls[0]) if urls else None, "key": lan.access_key()}


@app.get("/api/health")
def health():
    th = ENGINE.get("thinker")
    return {"ready": bool(ENGINE.get("ready")), "ctx_tokens": th.ctx_len() if th else 0,
            "max_ctx": th.max_ctx if th else 0, "connected": ACTIVE.get("sess") is not None,
            "talker_steps": getattr(ENGINE.get("talker"), "steps", None),
            "home_assistant": "home_assistant" in TOOLBOX.names,
            "gpu_mem_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
            "gpu_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
            "gpu_free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2)}


@app.get("/api/ha/areas")
def ha_areas():
    """The Home Assistant areas that have devices (the client's "Room" setting)."""
    from tools import HA_DIRECT, _ha_url, ha_token
    if not (HA_DIRECT and ha_token()):
        return {"areas": []}
    try:
        import ha
        return {"areas": sorted({d["area"] for d in ha.devices(_ha_url(), ha_token()) if d["area"]})}
    except Exception as e:
        return JSONResponse({"areas": [], "error": f"{type(e).__name__}: {e}"}, status_code=502)


@app.post("/api/announce")
async def api_announce(req: Request):
    """Have the assistant say something at the next quiet moment, e.g. from a Home Assistant rest_command or a
    script: {"message": "The washer is done.", "chime": true, "verbatim": true, "ttl": 300}. verbatim=false passes
    it on in the assistant's own words. Dropped if it can't be said within ttl seconds (nobody connected)."""
    try:
        d = await req.json()
    except Exception:
        d = {}
    if not isinstance(d, dict) or not str(d.get("message", "")).strip():
        return JSONResponse({"error": "message is required"}, status_code=400)
    no = (False, "false", "no", 0)
    announce(d["message"], chime=d.get("chime", True) not in no, verbatim=d.get("verbatim", True) not in no,
             source=str(d.get("source") or "Home Assistant")[:40], ttl=float(d.get("ttl", 300)))
    return {"queued": True, "connected": ACTIVE.get("sess") is not None}


def _all_voices():
    return {"builtin": VOICES, "custom": sorted(ENGINE.get("custom_voices", {}))}


_MEM_SNAP: dict = {}


@app.get("/api/debug/mem")
def debug_mem(top: int = 15):
    """Host-memory growth since the previous call (Python allocations by line when MING_TRACEMALLOC=1)."""
    import gc
    import tracemalloc
    gc.collect()
    rss = int(open("/proc/self/statm").read().split()[1]) * 4096 / 2**20
    gpu = torch.cuda.memory_allocated() / 2**20
    out = {"rss_mb": round(rss), "rss_growth_mb": round(rss - _MEM_SNAP.get("rss", rss)),
           "gpu_mb": round(gpu), "gpu_growth_mb": round(gpu - _MEM_SNAP.get("gpu", gpu)), "threads": threading.active_count()}
    _MEM_SNAP["rss"], _MEM_SNAP["gpu"] = rss, gpu
    if tracemalloc.is_tracing():
        snap = tracemalloc.take_snapshot()
        prev = _MEM_SNAP.get("snap")
        if prev is not None:
            stats = snap.compare_to(prev, "lineno")[:top]
            out["python_growth"] = [f"{s.size_diff / 2**20:+.2f} MB  {s.count_diff:+d} blocks  {s.traceback}" for s in stats]
        out["python_traced_mb"] = round(tracemalloc.get_traced_memory()[0] / 2**20, 1)
        _MEM_SNAP["snap"] = snap
    return out


@app.get("/api/voices")
def list_voices():
    return _all_voices()


@app.post("/api/voices")
async def create_voice(request: Request, name: str):
    """Clone a voice from a 3-30 s sample (WAV/FLAC/OGG/MP3 body)."""
    import re as _re
    name = _re.sub(r"[^\w\- ]", "", name).strip()[:40]
    if not name or name in VOICES:
        return JSONResponse({"error": "pick a different name"}, status_code=400)
    data = await request.body()
    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as e:
        return JSONResponse({"error": f"could not read audio: {e}"}, status_code=400)
    wav = wav.mean(1)
    dur = len(wav) / sr
    if dur < 3:
        return JSONResponse({"error": "sample is too short; record at least 3 seconds"}, status_code=400)
    wav = wav[: int(30 * sr)]
    peak = float(np.abs(wav).max() or 1.0)
    wav = wav / peak * 0.9

    def run():
        with GPU_TURN:
            return ENGINE["talker"].speaker_from_audio(wav, sr)
    emb = await asyncio.get_running_loop().run_in_executor(None, run)
    ENGINE["custom_voices"][name] = emb
    torch.save(emb.cpu(), VOICE_DIR / f"{name}.pt")
    log(f"cloned voice {name!r} from {dur:.1f}s of audio")
    return {"ok": True, "name": name, "seconds": round(dur, 1), **_all_voices()}


@app.delete("/api/voices/{name}")
def delete_voice(name: str):
    if ENGINE["custom_voices"].pop(name, None) is None:
        return JSONResponse({"error": "no such voice"}, status_code=404)
    try:
        (VOICE_DIR / f"{name}.pt").unlink()
    except FileNotFoundError:
        pass
    return {"ok": True, **_all_voices()}


@app.get("/api/tts")
async def tts(text: str, voice: str = "US_Stella"):
    """Synthesize a short preview (WAV). Used by the client's voice preview button."""
    from fastapi.responses import Response
    v = ENGINE["custom_voices"].get(voice, voice if voice in VOICES else VOICES[0])

    def run():
        with GPU_TURN:
            return ENGINE["talker"].synthesize(text[:300], v)
    audio = await asyncio.get_running_loop().run_in_executor(None, run)
    buf = io.BytesIO()
    sf.write(buf, audio, Talker.SR, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


@app.post("/api/asr")
async def asr(request: Request):
    """Transcribe a WAV body (any rate). Used by the end-to-end test to check spoken replies."""
    data = await request.body()
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    def run():
        with GPU_TURN:
            return ENGINE["thinker"].transcribe(wav.numpy())
    text = await asyncio.get_running_loop().run_in_executor(None, run)
    return JSONResponse({"text": text})


class Reply:
    """One assistant turn. While speculative, everything it emits is held back until commit()."""

    _ids = 0

    def __init__(self, sess: "Session", audio, user_text, speculative: bool):
        Reply._ids += 1
        self.id = Reply._ids
        self.sess, self.audio, self.user_text = sess, audio, user_text
        self.cancel = threading.Event()
        self.decided = threading.Event()
        self.lock = threading.Lock()
        self.committed = not speculative
        self.speculative = speculative
        self.buf: list = []
        self.t_start = time.perf_counter()
        self.t_commit = None if speculative else self.t_start
        self.t_first_audio_sent = None
        self.snap = None  # conversation state before this turn (for undo)
        self.undone = False  # retracted: the user kept talking right after it started answering
        self.finished = False
        self.end = False  # it ends the conversation (end_conversation)
        if not speculative:
            self.decided.set()
        self.worker = threading.Thread(target=sess._run_reply, args=(self,), daemon=True)

    def emit(self, kind, payload, dur=0.0):
        with self.lock:
            if self.cancel.is_set():
                return
            if self.committed:
                self._send(kind, payload, dur)
            else:
                self.buf.append((kind, payload, dur))

    def _send(self, kind, payload, dur):
        if kind == "bytes" and self.t_first_audio_sent is None:
            self.t_first_audio_sent = time.perf_counter()
        self.sess._put(kind, payload)
        if dur:
            now = time.time()
            start = max(self.sess.play_until, now)
            self.sess.play_until = start + dur
            if kind == "bytes":
                self.sess.echo_reference(payload, start)

    def commit(self, user_end_msg):
        with self.lock:
            if self.cancel.is_set():
                return False
            self.t_commit = time.perf_counter()
            self.sess._put("json", user_end_msg)
            for kind, payload, dur in self.buf:
                self._send(kind, payload, dur)
            self.buf.clear()
            self.committed = True
        self.decided.set()
        return True

    def abort(self):
        with self.lock:
            self.cancel.set()
            self.buf.clear()
        self.decided.set()

    @property
    def active(self):
        return self.worker.is_alive()


class Session:
    def __init__(self, ws: WebSocket, loop: asyncio.AbstractEventLoop):
        self.ws, self.loop = ws, loop
        self.out: asyncio.Queue = asyncio.Queue()
        self.vad = ENGINE["vad_proto"]()
        self.voice = VOICES[0]
        self.barge_in = True
        self.tools = True  # let the model search the web, check the weather, calculate, tell the time
        self.home = ""  # the user's location, for "what's the weather?"
        self.persona = ""  # "jarvis": tool confirmations in the butler's words ("Certainly, sir. I've turned off...")
        self.room = ""  # the Home Assistant area this device is in ("turn off the lights": that room's)
        self.end_ms = 500
        self.spec_ms = 32  # start generating speculatively after this much of a pause (0 = off)
        self.short_extra_ms = 250  # extra end-of-turn wait when the utterance is under 1 s of speech
        # Smart Turn: once the pause starts, a classifier judges from the audio whether the user has finished.
        # Confident "finished" ends the turn after fast_end_ms; "unfinished" stretches the wait to slow_end_ms.
        self.smart_turn = ENGINE.get("turn") is not None
        self.fast_end_ms = 200
        self.slow_end_ms = 1500
        self.turn_fut = None
        self.turn_p = None
        # Echo guard: ignore the assistant's own voice coming back through the mic (speakerphones)
        self.echo = EchoGuard()
        self.echo_lock = threading.Lock()
        self.echo_guard = True
        self.echo_trace = []
        self.echo_suppressed = 0
        self.threshold = 0.5
        self.barge_threshold = 0.75
        self.barge_ms = 250
        # Resume: if you keep talking within this long after it starts answering, the answer is withdrawn and
        # what you say is joined to the same turn (you only paused). 0 = always treat it as an interruption.
        self.resume_ms = 1000
        self.undo_lock = threading.Lock()
        self.pending_restore = None  # snapshot to restore before the next turn (undo of a finished reply)
        self.mode = "vad"  # "vad" (hands-free), "wake" (hands-free after the wake word) or "ptt" (push-to-talk)
        self.ptt_down = False
        # Wake word: while asleep, speech only counts if it starts with the wake word, checked by transcribing its
        # first WAKE_CHECK_S while the user is still talking (so the decision is ready when they pause). After
        # each reply it stays awake for follow_up_s, so the conversation can go on without it.
        self.wake_word = "Ming"
        self.follow_up_s = 8.0
        self.awake_until = 0.0
        self.utt_id = 0
        self.gated = False  # this utterance started while asleep: it needs the wake word
        self.wake_fut = None
        self.wake = None  # (found, words after it, seconds checked) once checked
        self.wake_pending = None  # an utterance that ended before its check finished: (utt_id, audio, msg)
        self.reply: Reply | None = None
        self.spec: Reply | None = None
        self.asr_cancel = threading.Event()  # set when the user starts a new turn: drop background ASR
        self.play_until = 0.0
        self.speaking_frames = 0
        self.noise_rms = 0.0  # background level between utterances
        self.preroll: list[np.ndarray] = []
        self.reset_listen()

    def reset_listen(self):
        self.in_speech = False
        self.frames: list[np.ndarray] = []
        self.speech_frames = 0
        self.pause_frames = 0  # frames since the pause began (voice energy dropped, or Silero released)
        self.speech_rms, self.n_loud = 0.0, 0  # this utterance's average speech level
        self.turn_fut = None
        self.turn_p = None

    def _put(self, kind, payload):
        self.loop.call_soon_threadsafe(self.out.put_nowait, (kind, payload))

    def emit_json(self, obj):
        self._put("json", obj)

    def deliver_notices(self):
        """Announce finished background tasks at the next quiet moment (event loop)."""
        if not NOTICES or ACTIVE.get("sess") is not self:
            return
        if self._busy():
            self.loop.call_later(0.5, self.deliver_notices)
            return
        n = NOTICES.pop(0)
        if n.get("expires") and time.time() > n["expires"]:
            log(f"dropped a stale notification: {n['prompt'][:120]!r}")
            if NOTICES:
                self.loop.call_soon(self.deliver_notices)
            return
        for ev in n["events"]:
            if ev["type"] in ("alarm", "chime"):  # the chime plays before the announcement
                self.emit_json({"type": ev["type"]})
        r = self.start_reply(None, user_text=n["prompt"])
        alarm = next((ev for ev in n["events"] if ev["type"] == "alarm"), None)
        if alarm and alarm.get("left"):
            ringing = dict(alarm, said_at=time.time())
            RINGING.clear()
            RINGING.update(ringing)
            self.loop.call_later(ALARM_REPEAT_S, self._alarm_again, ringing["said_at"])
        for ev in n["events"]:
            if ev["type"] not in ("alarm", "chime"):
                self.emit_json(ev)
        if NOTICES:
            self.loop.call_later(1.0, self.deliver_notices)
        return r

    def _alarm_again(self, said_at: float):
        """No answer since the alarm was said: say it again (event loop)."""
        if RINGING.get("said_at") != said_at:  # answered, or a newer alarm took over
            return
        if ACTIVE.get("sess") is not self:  # whoever heard it has gone (a test, a closed tab): don't follow them
            RINGING.clear()
            return
        if self._busy() or self.in_speech:  # (talking, or the user may be answering right now)
            self.loop.call_later(2.0, self._alarm_again, said_at)
            return
        r = dict(RINGING)
        RINGING.clear()
        log(f"timer not answered, saying it again: {r['say']!r}")
        _alarm(r["say"], r["label"], r["left"] - 1)

    @property
    def responding(self):
        r = self.reply
        live = r is not None and r.committed and r.active and not r.cancel.is_set()
        return live or time.time() < self.play_until

    def echo_reference(self, pcm16_44k: bytes, start: float):
        """Record what the client will play (16 kHz) and when, for the echo guard. Called from reply threads."""
        x = torch.from_numpy(np.frombuffer(pcm16_44k, dtype="<i2").astype(np.float32) / 32768.0)
        y = torchaudio.functional.resample(x, Talker.SR, 16000).numpy()
        with self.echo_lock:
            self.echo.add_playback(y, start)

    def _is_echo(self, f: np.ndarray) -> bool:
        if not self.echo_guard:
            return False
        now = time.time()
        with self.echo_lock:
            if not self.echo.active(now):
                return False
            self.echo.push_mic(f, now)
            e = self.echo.is_echo(now)
            if ECHO_DEBUG:
                g = self.echo
                if g.learned:
                    log(f"echo level learned: erl={g.erl:.3f} from {g.learned}")
                    g.learned = []
                self.echo_trace.append((round(g.rho, 2), g.lag, round(g.mic_rms, 4), round(g.ref_rms, 4),
                                        None if g.erl is None else round(g.erl, 3), e))
                self.echo_trace = self.echo_trace[-24:]
            return e

    def _resume_turn(self):
        """The user started talking again right after the reply began: they had only paused. Withdraw the
        reply (the client drops it), roll the conversation back and return the earlier utterance so it can be
        joined with what follows. None if this is a real interruption."""
        r = self.reply
        if (not self.resume_ms or r is None or not r.committed or r.audio is None or r.user_text is not None
                or r.t_commit is None):
            return None
        onset_ms = (time.perf_counter() - r.t_commit) * 1000 - self.barge_ms - 100  # detection lags the onset
        if onset_ms > self.resume_ms:
            return None
        with self.undo_lock:
            r.undone = True
            r.cancel.set()
            if r.finished:  # its worker is done: roll back before the next turn instead
                self.pending_restore = r.snap
        self.asr_cancel.set()
        with self.echo_lock:
            self.echo.truncate(time.time() + 0.25)
        self.play_until = 0.0
        self.emit_json({"type": "retract", "id": r.id})
        log(f"resumed: the user kept talking {max(0, onset_ms):.0f} ms after the reply started; joined into one turn")
        return r.audio

    def interrupt(self, reason="barge-in"):
        if self.reply is not None:
            self.reply.cancel.set()
        with self.echo_lock:
            self.echo.truncate(time.time() + 0.25)  # the client stops playing; what is in flight still echoes
        self.play_until = 0.0
        self.emit_json({"type": "interrupt", "reason": reason})

    # -- audio in --
    def on_audio(self, pcm16: bytes):
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        for i in range(0, len(x) - FRAME + 1, FRAME):
            self.on_frame(x[i:i + FRAME])

    def _abort_spec(self):
        if self.spec is not None:
            self.spec.abort()
            self.spec = None

    def on_frame(self, f: np.ndarray):
        if self.mode == "ptt":
            if self.ptt_down:
                self.frames.append(f)
            return
        p = float(self.vad(torch.from_numpy(f), 16000).item())
        echo = self._is_echo(f)
        if self.responding:
            if not self.barge_in:
                return
            # require confident, sustained speech to barge in (helps against echo)
            self.preroll.append(f)
            self.preroll = self.preroll[-20:]  # 640 ms: confident-speech detection lags the onset
            if p < self.barge_threshold:
                if ECHO_DEBUG and self.speaking_frames >= 3:
                    log(f"barge-in attempt dropped after {self.speaking_frames} frames; trace: {self.echo_trace[-12:]}")
                self.speaking_frames = self.echo_suppressed = 0
            elif echo:  # the assistant's own voice: does not count (and slowly cancels earlier counts)
                self.speaking_frames = max(0, self.speaking_frames - 1)
                if ECHO_DEBUG:
                    self.echo_suppressed += 1
                    if self.echo_suppressed == 6:
                        log(f"speech ignored as echo; trace (rho, lag, mic, ref, erl, echo, p, n): {self.echo_trace}")
            else:
                self.speaking_frames += 1
            if ECHO_DEBUG:
                self.echo_trace[-1:] = [self.echo_trace[-1] + (round(p, 2), self.speaking_frames)] if self.echo_trace else []
            if self.speaking_frames * FRAME / 16 >= self.barge_ms:
                if ECHO_DEBUG:
                    log(f"barge-in; echo trace (rho, lag, mic, ref, erl, echo, p, n): {self.echo_trace}")
                prev = self._resume_turn()
                if prev is None:
                    self.interrupt()
                self.speaking_frames = 0
                self.reset_listen()
                self.utt_id += 1
                self.gated, self.wake_fut, self.wake = False, None, None  # (talking over a reply: no wake word needed)
                self.in_speech = True
                self.frames = ([prev] if prev is not None else []) + list(self.preroll)
                self.speech_frames = len(self.frames)
                self.emit_json({"type": "vad", "state": "speech"})
            return
        rms = float(np.sqrt(np.mean(f * f)))
        if not self.in_speech:
            if p < 0.3:
                self.noise_rms += 0.05 * (rms - self.noise_rms)
            self.preroll.append(f)
            self.preroll = self.preroll[-10:]  # 320 ms pre-roll
            if p >= self.threshold and not echo:  # (echo: the tail of the reply still coming back)
                self.asr_cancel.set()
                self.utt_id += 1
                self.gated, self.wake_fut, self.wake = not self.awake, None, None
                self.in_speech = True
                self.frames = list(self.preroll)
                self.speech_frames = 1
                self.pause_frames = 0
                self.emit_json({"type": "vad", "state": "speech"})
            return
        self.frames.append(f)
        if self.gated and self.wake_fut is None and len(self.frames) * FRAME >= WAKE_CHECK_S * 16000:
            self._check_wake()
        if p >= self.threshold:
            self.n_loud += 1
            self.speech_rms += (rms - self.speech_rms) / min(self.n_loud, 50)
        # Silero keeps saying "speech" for ~120-190 ms after the voice stops, so pauses are timed from the drop
        # in voice energy (20 dB under this utterance's level, or near the noise floor). That only starts the
        # speculative reply and the turn model earlier: ending the turn still needs Silero to agree.
        quiet = self.n_loud > 0 and rms < max(0.1 * self.speech_rms, 2.5 * self.noise_rms)
        released = p < self.threshold - 0.15
        soft_speech = not released and self.pause_frames >= 7  # still "speech" well past Silero's hangover
        if (not released and not quiet) or soft_speech:
            self.speech_frames += 1
            if self.pause_frames:
                self._abort_spec()  # user kept talking: discard the speculative reply
                self.turn_fut = self.turn_p = None
            self.pause_frames = 0
            return
        self.pause_frames += 1
        silence_ms = self.pause_frames * FRAME / 16
        speech_s = self.speech_frames * FRAME / 16000
        long_enough = speech_s >= 0.25
        end_ms = self._end_of_turn_ms(silence_ms, speech_s, long_enough)
        if self.gated and self.wake_fut is None and long_enough and silence_ms >= self.spec_ms:
            self._check_wake()  # (a short utterance: check what there is)
        if (self.spec_ms and self.spec is None and long_enough and silence_ms >= self.spec_ms
                and silence_ms < end_ms and (not self.gated or (self.wake and self.wake[0]))):
            self.spec = self.start_reply(self._utterance(), speculative=True)
        if silence_ms >= end_ms and released:
            audio = self._utterance()
            gated, wake = self.gated, self.wake
            self.reset_listen()
            self.preroll = []
            self.vad.reset_states()
            if not long_enough:
                self._abort_spec()
                self.emit_json({"type": "vad", "state": "ignored"})
                return
            msg = {"type": "user_end", "seconds": round(len(audio) / 16000, 2), "pause_ms": round(silence_ms),
                   "turn_p": None if self.turn_p is None else round(self.turn_p, 3)}
            if gated:
                if wake is None:  # the check is still running: decide when it's done (_wake_result)
                    self._abort_spec()
                    self.wake_pending = (self.utt_id, audio, msg)
                    return
                if not self._wake_go(wake, audio):
                    self._abort_spec()
                    return
            spec, self.spec = self.spec, None
            if spec is not None:
                msg["id"] = spec.id
            if spec is not None and spec.commit(msg):
                self.reply = spec
            else:
                self.start_reply(audio, user_end_msg=msg)

    def _end_of_turn_ms(self, silence_ms, speech_s, long_enough):
        # very short utterances ("Can you tell me...") are usually unfinished: wait a bit longer
        extra = self.short_extra_ms if speech_s < 1.0 else 0
        det = ENGINE.get("turn")
        if not (self.smart_turn and det is not None and long_enough):
            return self.end_ms + extra
        if self.turn_fut is None and silence_ms >= 64:
            self.turn_fut = TURN_POOL.submit(det.prob, np.concatenate(self.frames))
        if self.turn_p is None and self.turn_fut is not None and self.turn_fut.done():
            try:
                self.turn_p = self.turn_fut.result()
            except Exception as e:
                log("smart turn error:", repr(e))
                self.turn_p = 0.5
        p = self.turn_p
        if p is None:
            return self.end_ms + extra
        if p >= 0.8:
            return min(self.end_ms, self.fast_end_ms) + extra
        if p < 0.3:
            return max(self.end_ms, self.slow_end_ms)
        return self.end_ms + extra

    def _utterance(self):
        audio = np.concatenate(self.frames)
        keep = len(audio) - int(max(0, self.pause_frames * FRAME - 150 * 16))  # keep ~150 ms of silence
        return audio[:max(keep, FRAME)]

    # -- wake word --
    @property
    def awake(self):
        return (self.mode != "wake" or time.time() < self.awake_until or self.responding
                or (self.reply is not None and self.reply.active))

    def _check_wake(self):
        """Transcribe the start of the current utterance (in the background) and look for the wake word."""
        uid, clip, word = self.utt_id, np.concatenate(self.frames)[:int(WAKE_CHECK_S * 16000)], self.wake_word

        def run():
            with GPU_TURN:
                return ENGINE["thinker"].transcribe(clip, max_new_tokens=12)

        self.wake_fut = WAKE_POOL.submit(run)
        self.wake_fut.add_done_callback(
            lambda f: self.loop.call_soon_threadsafe(self._wake_result, uid, f, word, len(clip) / 16000))

    def _wake_result(self, uid, fut, word, clip_s):
        try:
            text = fut.result()
        except Exception as e:
            log("wake check error:", repr(e))
            text = ""
        found, rest = find_wake_word(text, word)
        if found:
            log(f"wake word: {text!r}")
            self.emit_json({"type": "wake", "state": "awake"})
        if uid == self.utt_id:
            self.wake = (found, rest, clip_s)
        pend, self.wake_pending = self.wake_pending, None
        if pend is not None and pend[0] == uid and self._wake_go((found, rest, clip_s), pend[1]):
            self.start_reply(pend[1], user_end_msg=pend[2])
        elif pend is not None and pend[0] != uid:
            self.wake_pending = pend

    def _wake_go(self, wake, audio) -> bool:
        """End of a gated utterance: True to answer it. Just the wake word ("Hey Ming") -> awake, chime, no reply."""
        found, rest, clip_s = wake
        if not found:
            self.emit_json({"type": "vad", "state": "ignored"})
            return False
        # (the check may have run at the first pause, "Hey Ming, | what's ...": words after the clip are a command)
        if len(rest.split()) < 2 and clip_s >= len(audio) / 16000 - 0.3:
            self._stay_awake(self.follow_up_s)
            self.emit_json({"type": "wake", "state": "listening"})  # (the client chimes)
            return False
        return True

    def _stay_awake(self, seconds):
        self.awake_until = max(self.awake_until, time.time() + seconds)
        self.loop.call_later(self.awake_until - time.time() + 0.05, self._maybe_sleep)

    def _replied(self):
        """A reply finished generating (event loop): stay awake until it has played, plus the follow-up window."""
        if self.mode == "wake":
            self._stay_awake(max(0.0, self.play_until - time.time()) + self.follow_up_s)
        if ENGINE["thinker"].needs_compact():
            self.loop.call_later(max(0.0, self.play_until - time.time()) + 1.5, self._maybe_compact)

    def _busy(self):
        return (self.in_speech or self.responding or self.spec is not None or self.ptt_down
                or (self.reply is not None and self.reply.active))

    def _maybe_compact(self):
        """The context is nearly full: rebuild it now, while nobody is talking, so the next turn doesn't pay for
        it. (If someone is, the next reply's end tries again; failing that, the turn itself rolls over.)"""
        th = ENGINE["thinker"]
        if ACTIVE.get("sess") is not self or self._busy() or not th.needs_compact():
            return

        def run():
            with GPU_TURN:
                with self.undo_lock:  # (an undo restores across it fine: the turns come back as text)
                    undo = self.pending_restore is not None
                if not undo and not self._busy() and th.needs_compact():
                    try:
                        th.compact()
                    except Exception as e:  # (the next turn rolls over instead)
                        log("compaction error:", repr(e)[:300])
                        check_fatal(e)
        threading.Thread(target=run, daemon=True).start()

    def _maybe_sleep(self):
        if self.mode == "wake" and not self.awake and not self.in_speech and self.wake_pending is None:
            self.emit_json({"type": "wake", "state": "asleep"})

    # -- reply --
    def start_reply(self, audio, user_text=None, speculative=False, user_end_msg=None):
        self.asr_cancel.set()
        self.asr_cancel = threading.Event()
        if not (user_text or "").startswith("(Notification"):
            RINGING.clear()  # the user said something: the alarm was heard
        r = Reply(self, audio, user_text, speculative)
        if not speculative:
            if self.reply is not None:
                self.reply.cancel.set()
            msg = dict(user_end_msg or {"type": "user_end", "seconds": 0})
            msg["id"] = r.id
            self._put("json", msg)
        if not speculative:
            self.reply = r
        r.worker.start()
        return r

    def _run_reply(self, r: Reply):
        th, tk = ENGINE["thinker"], ENGINE["talker"]
        with GPU_TURN:
            with self.undo_lock:
                pending, self.pending_restore = self.pending_restore, None
            if pending is not None:
                th.restore(pending)
            if r.cancel.is_set():
                return
            TOOLBOX.home, TOOLBOX.persona, TOOLBOX.room = self.home, self.persona, self.room
            th.set_tools(TOOLBOX if self.tools else None)  # (rebuilds the context when this changed)
            snap = r.snap = th.snapshot()
            try:
                self._generate(r, th, tk)
            finally:
                if r.speculative:
                    r.decided.wait(timeout=10)
                    if not r.committed:
                        th.restore(snap)
            if r.committed and not r.undone and r.audio is not None and len(r.audio) > 4000:
                self._transcribe_user(r, th)
            with self.undo_lock:  # (restore outside the lock: the event loop takes it)
                r.finished = True
                undo = r.undone and r.committed
            if undo:
                th.restore(snap)
            save_conversation(th)

    def _transcribe_user(self, r: Reply, th: Thinker):
        """Background ASR of the user's utterance (for the transcript and context rollover).
        Runs after the reply is generated; abandoned as soon as the user starts a new turn."""
        cancel = self.asr_cancel
        entry = th.history[-1] if th.history else None
        try:
            text = th.transcribe(r.audio, max_new_tokens=160, cancel=cancel)
        except Exception as e:
            log("transcribe error:", repr(e))
            return
        if text and not cancel.is_set():
            log(f"heard: {text!r}")
            if entry is not None and entry.get("user") is None:
                entry["user"] = text
            self._put("json", {"type": "transcript", "id": r.id, "text": text})

    def _generate(self, r: Reply, th: Thinker, tk: Talker):
        cancel = r.cancel
        m = {"first_text": None, "first_frag": None}
        frag_q: queue.Queue = queue.Queue()
        voice = ENGINE["custom_voices"].get(self.voice, self.voice)
        n_samples = [0]

        def talker_loop():
            while True:
                item = frag_q.get()
                if item is None or cancel.is_set():
                    break
                pieces = [item]  # merge whatever is already queued (talker behind -> bigger pieces)
                while True:
                    try:
                        nxt = frag_q.get_nowait()
                    except queue.Empty:
                        break
                    if nxt is None:
                        frag_q.put(None)
                        break
                    pieces.append(nxt)
                try:
                    for pcm in tk.synth_stream(" ".join(pieces), voice, cancel=cancel):
                        if cancel.is_set():
                            break
                        b = (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
                        r.emit("bytes", b, dur=len(pcm) / Talker.SR)
                        n_samples[0] += len(pcm)
                except Exception as e:
                    log("talker error:", repr(e))
                    check_fatal(e)
                    r.emit("json", {"type": "error", "message": f"talker: {e}"})

        tt = threading.Thread(target=talker_loop, daemon=True)
        tt.start()
        fr = Fragmenter()
        text = ""
        try:
            confirmed = lambda: r.decided.wait(30) and r.committed and not cancel.is_set()  # noqa: E731
            for delta in th.stream(r.audio, cancel=cancel, user_text=r.user_text, events=True, gate=confirmed):
                if isinstance(delta, dict):  # a tool call / its result
                    if delta.get("end"):  # the user said goodbye: end the conversation after this reply
                        r.end = True
                    if delta["type"] == "tool_call":
                        for piece in fr.flush():  # speak what it said before calling the tool
                            frag_q.put(piece)
                        filler = FILLERS.get(delta["name"])
                        if filler and not text.strip():  # it went straight to the tool: say something meanwhile
                            text += filler + " "
                            r.emit("json", {"type": "text", "delta": filler + " "})
                            frag_q.put(filler)
                    r.emit("json", {"type": "tool", "status": "running" if delta["type"] == "tool_call" else "done",
                                    **{k: v for k, v in delta.items() if k != "type"}})
                    continue
                if m["first_text"] is None:
                    m["first_text"] = time.perf_counter()
                text += delta
                r.emit("json", {"type": "text", "delta": delta})
                for piece in fr.push(delta):
                    m["first_frag"] = m["first_frag"] or time.perf_counter()
                    frag_q.put(piece)
            for piece in fr.flush():
                m["first_frag"] = m["first_frag"] or time.perf_counter()
                frag_q.put(piece)
        except Exception as e:  # keep the session alive
            import traceback
            where = " <- ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}"
                                for f in reversed(traceback.extract_tb(e.__traceback__)[-6:]))
            log("thinker error:", repr(e)[:300], "at", where)
            check_fatal(e)
            r.emit("json", {"type": "error", "message": f"thinker: {e}"})
        finally:
            frag_q.put(None)
        tt.join()
        t_done = time.perf_counter()
        if r.speculative:
            r.decided.wait(timeout=10)
        if not r.committed:
            return

        def rel(t):  # ms after the user's turn ended (negative = ready before the turn ended)
            return None if t is None else round((t - r.t_commit) * 1000)

        ms = {"first_text": rel(m["first_text"]), "first_frag": rel(m["first_frag"]),
              "first_audio_sent": rel(r.t_first_audio_sent), "done": rel(t_done),
              "speculative": r.speculative, "head_start_ms": round((r.t_commit - r.t_start) * 1000),
              "audio_s": round(n_samples[0] / Talker.SR, 2), "ctx_tokens": th.ctx_len()}
        log(f"turn: {text!r} | {ms}{' (interrupted)' if cancel.is_set() else ''}")
        r.emit("json", {"type": "done", "text": text, "metrics": ms, "interrupted": cancel.is_set()})
        self.loop.call_soon_threadsafe(self._replied)
        if r.end and not cancel.is_set():
            r.emit("json", {"type": "end"})  # (the browser stops listening once the goodbye has played)
            self.loop.call_soon_threadsafe(self._end_conversation)

    def _end_conversation(self):
        """Wake-word mode: back to waiting for the wake word as soon as the goodbye has played (no follow-up)."""
        if self.mode == "wake":
            self.awake_until = self.play_until
            self.loop.call_later(max(0.0, self.play_until - time.time()) + 0.3, self._maybe_sleep)

    # -- control --
    def on_control(self, msg: dict):
        t = msg.get("type")
        if t == "config":
            if msg.get("voice") in VOICES or msg.get("voice") in ENGINE["custom_voices"]:
                self.voice = msg["voice"]
            if "barge_in" in msg:
                self.barge_in = bool(msg["barge_in"])
            if "echo_guard" in msg:
                self.echo_guard = bool(msg["echo_guard"])
            if "smart_turn" in msg:
                self.smart_turn = bool(msg["smart_turn"]) and ENGINE.get("turn") is not None
            if "end_ms" in msg:
                self.end_ms = int(max(200, min(2000, msg["end_ms"])))
            if "short_extra_ms" in msg:
                self.short_extra_ms = int(max(0, min(1000, msg["short_extra_ms"])))
            if "spec_ms" in msg:
                self.spec_ms = int(max(0, min(1000, msg["spec_ms"])))
            if "resume_ms" in msg:
                self.resume_ms = int(max(0, min(3000, msg["resume_ms"])))
            if "tools" in msg:
                self.tools = bool(msg["tools"])
            if "home" in msg:
                self.home = str(msg["home"] or "").strip()[:100]
            if "persona" in msg:
                self.persona = str(msg["persona"] or "")[:40]
            if "room" in msg:
                self.room = str(msg["room"] or "").strip()[:60]
            if "fast_end_ms" in msg:
                self.fast_end_ms = int(max(100, min(1000, msg["fast_end_ms"])))
            if "threshold" in msg:
                self.threshold = float(max(0.2, min(0.95, msg["threshold"])))
            if "wake_word" in msg and str(msg["wake_word"] or "").strip():
                self.wake_word = str(msg["wake_word"]).strip()[:40]
            if "follow_up_s" in msg:
                self.follow_up_s = float(max(0, min(60, msg["follow_up_s"])))
            if msg.get("mode") in ("vad", "ptt", "wake") and msg["mode"] != self.mode:
                self.mode = msg["mode"]
                self._abort_spec()
                self.reset_listen()
                self.ptt_down = False
                self.vad.reset_states()
                self.awake_until, self.wake_pending = 0.0, None
            if self.mode == "wake":
                self.emit_json({"type": "wake", "state": "awake" if self.awake else "asleep"})
        elif t == "ptt":
            if msg.get("state") == "down":
                self.asr_cancel.set()
                self._abort_spec()
                if self.responding or (self.reply is not None and self.reply.active):
                    self.interrupt("push-to-talk")
                self.reset_listen()
                self.ptt_down = True
                self.emit_json({"type": "vad", "state": "speech"})
            elif self.ptt_down:
                self.ptt_down = False
                audio = np.concatenate(self.frames) if self.frames else np.zeros(0, np.float32)
                self.reset_listen()
                if len(audio) < 0.25 * 16000:
                    self.emit_json({"type": "vad", "state": "ignored"})
                else:
                    self.start_reply(audio, user_end_msg={"type": "user_end", "seconds": round(len(audio) / 16000, 2)})
        elif t == "reset":
            RINGING.clear()
            self._abort_spec()
            self.interrupt("reset")
            th = ENGINE["thinker"]

            def do_reset():
                with GPU_TURN:
                    with self.undo_lock:
                        self.pending_restore = None
                    th.reset(msg.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
                    save_conversation(th)  # (an empty one: a restart starts afresh too)
            threading.Thread(target=do_reset, daemon=True).start()
            self.emit_json({"type": "reset_ok"})
        elif t == "stop":
            RINGING.clear()
            self.interrupt("stop")
        elif t == "text":  # typed input, same pipeline
            self._abort_spec()
            if self.responding:
                self.interrupt()
            self.start_reply(None, user_text=str(msg.get("text", ""))[:2000])

    def close(self):
        self._abort_spec()
        if self.reply is not None:
            self.reply.cancel.set()


ACTIVE: dict = {}  # the one connected session (the conversation state is shared, so one tab at a time)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    sess = Session(ws, loop)
    old = ACTIVE.get("sess")
    if old is not None:
        old.close()
        try:
            await old.ws.send_text(json.dumps({"type": "taken_over"}))
            await old.ws.close(code=4000)
        except Exception:
            pass
    ACTIVE["sess"] = sess
    th = ENGINE["thinker"]
    # the conversation so far, so a reloaded page (or the phone after switching apps) can show it again
    history = [{"user": h.get("user"), "assistant": h.get("content", ""), "interrupted": h.get("interrupted", False)}
               for h in list(th.history)]
    await ws.send_text(json.dumps({"type": "hello", "voices": VOICES, "custom_voices": _all_voices()["custom"],
                                   "sample_rate": Talker.SR, "system_prompt": th.system_prompt,
                                   "history": history}, ensure_ascii=False))

    async def sender():
        while True:
            kind, payload = await sess.out.get()
            if kind == "json":
                await ws.send_text(json.dumps(payload, ensure_ascii=False))
            else:
                await ws.send_bytes(payload)

    send_task = asyncio.create_task(sender())
    loop.call_later(1.5, sess.deliver_notices)  # background results that finished while nobody was connected
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                sess.on_audio(msg["bytes"])
            elif msg.get("text"):
                sess.on_control(json.loads(msg["text"]))
    except WebSocketDisconnect:
        pass
    finally:
        sess.close()
        send_task.cancel()
        if ACTIVE.get("sess") is sess:
            ACTIVE.pop("sess", None)


if __name__ == "__main__":
    if os.environ.get("MING_TRACEMALLOC") == "1":
        import tracemalloc
        tracemalloc.start(8)
    # Local-only by default (the API accepts audio uploads and drives the GPU). Other devices reach the app
    # through windows/lan_proxy.py -> the key-protected LAN port (lan.py); HOST=0.0.0.0 exposes port 8000
    # without a key unless MING_API_KEY is set.
    import socket
    socks = []
    for host, port in ((os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))),
                       ("127.0.0.1", lan.LAN_PORT)):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError as e:  # (98: run.sh's supervisor doesn't retry: a server is most likely running already)
            log(f"cannot listen on {host}:{port} ({e.strerror}): is Ming Voice already running?")
            sys.exit(98)
        s.set_inheritable(True)
        socks.append(s)
    lan.access_key()  # create the key file on first start
    config = uvicorn.Config(app, log_level="warning", ws_max_size=2**24, ws_ping_interval=None, ws_ping_timeout=None)
    try:
        uvicorn.Server(config).run(sockets=socks)
    except KeyboardInterrupt:  # Ctrl-C: uvicorn.run() swallows this too
        pass
