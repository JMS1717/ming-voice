"""Echo-aware barge-in: tell the user's voice apart from the assistant's own voice leaking back into the mic.

On a speakerphone the browser's echo canceller leaves some echo behind, and Silero VAD happily calls that
speech, so the assistant would interrupt itself. The server knows exactly what it played, so it keeps that
reference (16 kHz, on the estimated playback timeline) and, for each microphone frame during playback:
  * cross-correlates the last 0.4 s of mic audio with the reference over 0-1 s of round-trip delay
    (client buffer + device + network latency, unknown in advance);
  * learns the echo level ("ERL", mic RMS / reference RMS) from frames that clearly are echo (high correlation);
  * calls a frame echo when it correlates with the reference, or is not clearly louder than the echo predicted
    from the reference at that delay.
Barge-in then counts only speech frames that are not echo. No assumptions about linearity: a strongly
processed residual still correlates with its source, and a real interruption is louder than the residual.
"""
from __future__ import annotations

import numpy as np

SR = 16000


class EchoGuard:
    def __init__(self, max_lag_s: float = 1.0, win_s: float = 0.4, keep_s: float = 3.0,
                 rho_echo: float = 0.45, margin: float = 2.5, min_eff_s: float = 0.02):
        self.max_lag = int(max_lag_s * SR)
        self.win = int(win_s * SR)
        self.keep = int(keep_s * SR)
        self.rho_echo, self.margin = rho_echo, margin
        self.min_eff = min_eff_s * SR
        self.ref = np.zeros(0, np.float32)
        self.ref_t0 = 0.0  # wall-clock playback time of ref[0]
        self.mic = np.zeros(self.win, np.float32)
        self.erl = None  # learned echo level (mic RMS / reference RMS)
        self.rho = 0.0
        self.lag = None
        self.mic_rms = 0.0
        self.ref_rms = 0.0
        self.active_until = 0.0

    # -- reference: what the client is playing --
    def add_playback(self, pcm16k: np.ndarray, start_time: float):
        end_time = self.ref_t0 + len(self.ref) / SR
        if len(self.ref) == 0 or start_time - end_time > 2.0:
            self.ref, self.ref_t0 = pcm16k.astype(np.float32), start_time
        else:
            gap = int(round((start_time - end_time) * SR))
            parts = [self.ref, np.zeros(gap, np.float32)] if gap > 0 else [self.ref]
            self.ref = np.concatenate(parts + [pcm16k.astype(np.float32)])
        self.active_until = max(self.active_until, self.ref_t0 + len(self.ref) / SR + self.max_lag / SR)

    def truncate(self, at_time: float):
        """Playback stopped (interrupt): nothing after `at_time` will be heard."""
        n = int(max(0.0, at_time - self.ref_t0) * SR)
        self.ref = self.ref[:n]
        self.active_until = min(self.active_until, at_time + self.max_lag / SR)

    def _trim(self, now: float):
        """Drop reference audio that can no longer reach the mic (audio not yet played is kept: the server
        streams well ahead of real time)."""
        drop = int((now - self.ref_t0) * SR) - (self.keep + self.max_lag + self.win)
        if drop > SR:  # in chunks, not every frame
            self.ref = self.ref[drop:]
            self.ref_t0 += drop / SR

    def active(self, now: float) -> bool:
        return len(self.ref) > 0 and now < self.active_until

    # -- microphone --
    def push_mic(self, frame: np.ndarray, now: float):
        """Feed one mic frame (arrival time `now`) and update the echo estimate for the latest window."""
        self.mic = np.concatenate([self.mic[len(frame):], frame])
        self.mic_rms = float(np.sqrt(np.mean(self.mic[-len(frame):] ** 2)) + 1e-9)
        self.rho, self.ref_rms = 0.0, 0.0
        if not self.active(now):
            return
        self._trim(now)
        # reference samples that could be in the mic window: played between now-win-max_lag and now
        i1 = int((now - self.ref_t0) * SR)
        i0 = i1 - self.win - self.max_lag
        seg = np.zeros(self.win + self.max_lag, np.float32)
        a, b = max(i0, 0), min(i1, len(self.ref))
        if b <= a:
            return
        seg[a - i0:b - i0] = self.ref[a:b]
        if not seg.any():
            return
        m = self.mic.astype(np.float64)
        s = seg.astype(np.float64)
        n = 1 << int(np.ceil(np.log2(len(s) + len(m))))
        # c[k] = sum_i m[i] * seg[k + i]: correlation of the mic window with each candidate delay
        c = np.fft.irfft(np.fft.rfft(s, n) * np.conj(np.fft.rfft(m, n)), n)[: len(s) - len(m) + 1]
        s2 = s * s
        e = np.concatenate([[0.0], np.cumsum(s2)])
        e4 = np.concatenate([[0.0], np.cumsum(s2 * s2)])
        seg_energy = np.maximum(e[len(m):] - e[: len(e) - len(m)], 0.0)
        seg_e4 = np.maximum(e4[len(m):] - e4[: len(e4) - len(m)], 1e-30)
        mic_energy = float(np.dot(m, m))
        if mic_energy < 1e-7:
            return
        # Delays where the reference window is (almost) silent can't explain the mic, and ones where only a
        # sliver of it has sound (the edges of playback) correlate with anything by chance: skip both.
        # (seg_energy^2 / seg_e4 = number of samples that effectively carry the window's energy)
        valid = (seg_energy > 1e-3 * seg_energy.max() + 1e-9) & (seg_energy * seg_energy > self.min_eff * seg_e4)
        rho = np.where(valid, np.abs(c) / np.sqrt(seg_energy * mic_energy + 1e-18), 0.0)
        np.clip(rho, 0.0, 1.0, out=rho)
        k = int(np.argmax(rho))
        self.rho = float(rho[k])
        self.lag = len(seg) - len(m) - k  # samples between playback and arrival
        # reference level at that delay, for the newest frame
        tail = seg[k + len(m) - len(frame): k + len(m)]
        self.ref_rms = float(np.sqrt(np.mean(tail ** 2)) + 1e-9)
        if self.rho > 0.6 and self.ref_rms > 3e-3:  # clearly echo: learn how loud the echo is
            r = self.mic_rms / self.ref_rms
            self.erl = r if self.erl is None else 0.9 * self.erl + 0.1 * r

    def is_echo(self, now: float) -> bool:
        """Is the newest mic frame (most likely) the assistant's own voice?"""
        if not self.active(now) or self.ref_rms <= 3e-3:
            return False
        if self.rho >= self.rho_echo:
            return True
        if self.erl is not None and self.mic_rms < self.margin * self.erl * self.ref_rms:
            return True
        return False
