"""Ming Voice without a browser: an always-on client that uses this PC's own microphone and speakers.

It connects to the server (ws://127.0.0.1:8000/ws), listens for the wake word ("Hey Ming, ...") and plays the
replies, timer alarms and announcements. No tab to keep open, no Start button to press.

Only one device can hold the conversation at a time. When you open the app on your phone (or in a browser), it takes
over and this client steps aside; once that's closed, this client takes the conversation back by itself.

    python windows\\ming_client.py                    # run (Ctrl-C to stop)
    python windows\\ming_client.py --list-devices     # microphones and speakers
    python windows\\ming_client.py --stop             # stop the one running in the background
    pythonw windows\\ming_client.py                   # no window (start_ming_voice.ps1 -Headless does this)

Settings: command-line flags, or a .client.json next to the project's server.py (not committed), e.g.
    {"wake_word": "Ming", "home": "Paris", "voice": "US_Emma", "input_device": "USB", "barge_in": false}
barge_in: with speakers (no headset) the microphone is muted while Ming talks, so it can't hear itself; set it to
true with a headset to be able to talk over it. Needs: pip install sounddevice websockets numpy
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import sys
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONFIG_FILE = HERE.parent / ".client.json"
PID_FILE = HERE / ".ming_client.pid"
LOCK_PORT = 47613  # single instance: whoever holds this localhost port is the running client
FRAME = 512  # 32 ms @ 16 kHz, what the server's VAD works in
DEFAULTS = {"url": "ws://127.0.0.1:8000/ws", "mode": "wake", "wake_word": "Ming", "follow_up_s": 8, "voice": None,
            "home": "", "tools": True, "barge_in": False, "input_device": None, "output_device": None,
            "volume": 1.0, "reclaim_s": 15}
_log_file = None


def log(*a):
    line = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    if _log_file is not None:
        _log_file.write(line + "\n")
        _log_file.flush()
    else:
        print(line, flush=True)


def _device(name):
    """A device index from a number or part of its name (None = the system default)."""
    if name in (None, ""):
        return None
    if isinstance(name, int) or str(name).isdigit():
        return int(name)
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if str(name).lower() in d["name"].lower():
            return i
    raise SystemExit(f"no audio device matching {name!r} (see --list-devices)")


class Player:
    """Mono PCM16 playback that can be cut off at once (an interruption, a withdrawn reply).
    With out_wav it plays into a file instead, in simulated real time (tests)."""

    def __init__(self, sr: int, device=None, volume=1.0, out_wav: str | None = None):
        self.sr, self.volume = sr, volume
        self.lock = threading.Lock()
        self.buf = np.zeros(0, np.int16)
        self.last_sound = 0.0
        self.saved, self.until = ([], 0.0) if out_wav else (None, 0.0)
        self.out_wav = out_wav
        self.stream = None
        if out_wav is None:
            import sounddevice as sd
            self.stream = sd.OutputStream(samplerate=sr, channels=1, dtype="int16", device=device,
                                          latency="low", callback=self._cb)
            self.stream.start()

    def _cb(self, out, frames, _t, _status):
        with self.lock:
            n = min(frames, len(self.buf))
            out[:n, 0] = self.buf[:n]
            out[n:, 0] = 0
            self.buf = self.buf[n:]
            if n:
                self.last_sound = time.monotonic()

    def play(self, pcm: np.ndarray):
        if self.volume != 1.0:
            pcm = np.clip(pcm.astype(np.float32) * self.volume, -32768, 32767).astype(np.int16)
        with self.lock:
            if self.saved is not None:
                self.saved.append(pcm)
                self.until = max(self.until, time.monotonic()) + len(pcm) / self.sr
            else:
                self.buf = np.concatenate([self.buf, pcm])

    def flush(self):
        with self.lock:
            self.buf = np.zeros(0, np.int16)
            self.until = min(self.until, time.monotonic())

    def busy(self, tail=0.35) -> bool:
        """Playing (or just finished: the room's echo) -> the microphone stays muted in half-duplex mode."""
        with self.lock:
            if self.saved is not None:
                return time.monotonic() < self.until + tail
            return len(self.buf) > 0 or time.monotonic() - self.last_sound < tail

    def chime(self, notes, gap=0.12, vol=0.2):
        t = np.arange(int(self.sr * gap * 1.7)) / self.sr
        env = np.minimum(1, t / 0.01) * np.exp(-t / (gap * 0.45))
        out = np.zeros(int(self.sr * gap * (len(notes) - 1)) + len(t), np.float32)
        for i, f in enumerate(notes):
            s = int(self.sr * gap * i)
            out[s:s + len(t)] += vol * env * np.sin(2 * np.pi * f * t)
        self.play(np.clip(out * 32767, -32768, 32767).astype(np.int16))

    def close(self):
        if self.stream is not None:
            self.stream.close()
        if self.out_wav and self.saved:
            with wave.open(self.out_wav, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(self.sr)
                w.writeframes(np.concatenate(self.saved).tobytes())


class Mic:
    """16 kHz PCM16 frames into an asyncio queue (oldest dropped when nobody is reading). With wav: that file,
    in real time, then silence (tests)."""

    def __init__(self, loop, device=None, wav: str | None = None):
        self.loop, self.q = loop, asyncio.Queue(maxsize=64)
        self.stream = None
        if wav:
            threading.Thread(target=self._from_wav, args=(wav,), daemon=True).start()
        else:
            import sounddevice as sd
            self.stream = sd.InputStream(samplerate=16000, channels=1, dtype="int16", blocksize=FRAME,
                                         device=device, callback=lambda d, *_: self._put(d[:, 0].tobytes()))
            self.stream.start()

    def _put(self, b: bytes):
        def put():
            if self.q.full():
                self.q.get_nowait()
            self.q.put_nowait(b)
        self.loop.call_soon_threadsafe(put)

    def _from_wav(self, path):
        with wave.open(path) as w:
            sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), np.int16)
            if w.getnchannels() > 1:
                x = x[::w.getnchannels()]
        if sr != 16000:
            x = np.interp(np.arange(0, len(x), sr / 16000), np.arange(len(x)), x).astype(np.int16)
        time.sleep(1.5)  # (let the session start)
        x = np.concatenate([x, np.zeros(16000 * 3600, np.int16)])
        t0 = time.monotonic()
        for k, i in enumerate(range(0, len(x) - FRAME, FRAME)):
            self._put(x[i:i + FRAME].tobytes())
            time.sleep(max(0.0, t0 + (k + 1) * FRAME / 16000 - time.monotonic()))


def _health(url: str) -> dict | None:
    http = url.replace("ws://", "http://").replace("wss://", "https://").rsplit("/ws", 1)[0]
    try:
        with urllib.request.urlopen(http + "/api/health", timeout=3) as r:
            return json.load(r)
    except Exception:
        return None


class Client:
    def __init__(self, cfg: dict, wav=None, out_wav=None):
        self.cfg, self.wav, self.out_wav = cfg, wav, out_wav
        self.player: Player | None = None
        self.mic: Mic | None = None
        self.reply = ""

    async def run(self, max_s: float | None = None):
        import websockets
        self.mic = Mic(asyncio.get_running_loop(), _device(self.cfg["input_device"]), self.wav)
        t_end = None if max_s is None else time.monotonic() + max_s
        warned = False
        while t_end is None or time.monotonic() < t_end:
            try:
                left = None if t_end is None else t_end - time.monotonic()
                result = await asyncio.wait_for(self.session(websockets), left)
                warned = False
            except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as e:
                if t_end is not None and time.monotonic() >= t_end:
                    break
                if not warned:
                    log(f"server not reachable ({type(e).__name__}); retrying until it is")
                    warned = True
                await asyncio.sleep(3)
                continue
            if result == "taken_over":
                log("another device took over the conversation; taking it back once that's closed")
                await self.wait_until_free()
            else:
                await asyncio.sleep(2)
        if self.player is not None:
            self.player.close()

    async def wait_until_free(self):
        free_since = None
        while True:
            await asyncio.sleep(5)
            h = await asyncio.to_thread(_health, self.cfg["url"])
            if h is not None and h.get("ready") and not h.get("connected"):
                free_since = free_since or time.monotonic()
                if time.monotonic() - free_since >= self.cfg["reclaim_s"]:
                    return
            else:
                free_since = None

    async def session(self, websockets) -> str:
        async with websockets.connect(self.cfg["url"], max_size=2**24, open_timeout=10) as ws:
            hello = json.loads(await ws.recv())
            if self.player is None or self.player.sr != hello["sample_rate"]:
                self.player = Player(hello["sample_rate"], _device(self.cfg["output_device"]), self.cfg["volume"],
                                     self.out_wav)
            c = self.cfg
            conf = {"type": "config", "mode": c["mode"], "wake_word": c["wake_word"], "follow_up_s": c["follow_up_s"],
                    "tools": c["tools"], "home": c["home"], "barge_in": c["barge_in"], "echo_guard": True}
            if c["voice"]:
                conf["voice"] = c["voice"]
            await ws.send(json.dumps(conf))
            log(f"connected ({c['mode']} mode, wake word {c['wake_word']!r})")
            while not self.mic.q.empty():  # (drop what was heard while disconnected)
                self.mic.q.get_nowait()
            pump = asyncio.create_task(self.pump(ws))
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        self.player.play(np.frombuffer(msg, "<i2"))
                    elif self.on_message(json.loads(msg)) == "taken_over":
                        return "taken_over"
            finally:
                pump.cancel()
                self.player.flush()
        log("disconnected")
        return "closed"

    async def pump(self, ws):
        silence = bytes(FRAME * 2)
        while True:
            frame = await self.mic.q.get()
            if not self.cfg["barge_in"] and self.player is not None and self.player.busy():
                frame = silence  # half-duplex: it mustn't hear (and answer) itself through the speakers
            await ws.send(frame)

    def on_message(self, m: dict):
        t, p = m.get("type"), self.player
        if t in ("user_end", "interrupt", "retract"):
            p.flush()
            self.reply = ""
        elif t == "text":
            self.reply += m.get("delta", "")
        elif t == "done":
            if self.reply.strip():
                log(f"Ming: {self.reply.strip()}")
            self.reply = ""
        elif t == "transcript":
            log(f"you: {m.get('text', '')}")
        elif t == "tool" and m.get("status") == "done":
            log(f"[{m.get('name')}] {json.dumps(m.get('args') or {}, ensure_ascii=False)[:200]}")
        elif t == "wake":
            if m.get("state") == "listening":
                p.chime([660, 880])
            elif m.get("state") in ("awake", "asleep"):
                log(f"({m['state']})")
        elif t == "alarm":
            p.chime([880, 660, 880, 660, 880, 660], 0.16, 0.25)
        elif t == "chime":
            p.chime([988, 784], 0.22, 0.2)
        elif t == "taken_over":
            p.flush()
            return "taken_over"
        elif t == "error":
            log("server error:", m.get("message"))
        return None


def _single_instance() -> socket.socket | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", LOCK_PORT))
    except OSError:
        return None
    return s


def main():
    global _log_file
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url")
    ap.add_argument("--mode", choices=["wake", "vad"], help="wake: answer 'Hey Ming, ...' (default); vad: everything")
    ap.add_argument("--wake-word")
    ap.add_argument("--voice")
    ap.add_argument("--home", help="your town, for the weather")
    ap.add_argument("--input-device", help="microphone: number or part of its name")
    ap.add_argument("--output-device", help="speakers: number or part of its name")
    ap.add_argument("--barge-in", action="store_true", default=None, help="keep the mic on while it talks (headset)")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--stop", action="store_true", help="stop the client running in the background")
    ap.add_argument("--log", help="log file (default: the console, or windows/ming_client.log without one)")
    ap.add_argument("--wav", help=argparse.SUPPRESS)  # tests: this file instead of the microphone
    ap.add_argument("--out-wav", help=argparse.SUPPRESS)  # tests: replies into this file instead of the speakers
    ap.add_argument("--seconds", type=float, help=argparse.SUPPRESS)  # tests: stop after this long
    a = ap.parse_args()

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if a.stop:
        probe = _single_instance()  # (a free lock port: nothing running, and the pid file may be stale)
        if probe is not None:
            probe.close()
            print("not running")
        else:
            try:
                os.kill(int(PID_FILE.read_text()), signal.SIGTERM)
                print("stopped")
            except (OSError, ValueError):
                print("couldn't stop it (no pid file)")
        PID_FILE.unlink(missing_ok=True)
        return

    cfg = dict(DEFAULTS)
    if CONFIG_FILE.exists():
        cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    for k in ("url", "mode", "wake_word", "voice", "home", "input_device", "output_device", "barge_in"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.log or sys.stdout is None:  # (pythonw: no console)
        _log_file = open(a.log or HERE / "ming_client.log", "a", encoding="utf-8")
    lock = None
    if not a.wav:
        lock = _single_instance()
        if lock is None:
            log("already running (windows\\ming_client.py --stop to stop it)")
            return
        PID_FILE.write_text(str(os.getpid()))
    try:
        asyncio.run(Client(cfg, a.wav, a.out_wav).run(a.seconds))
    except KeyboardInterrupt:
        pass
    finally:
        if lock is not None:
            PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
