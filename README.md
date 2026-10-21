# Ming Voice

Low-latency, fully local **speech-to-speech** voice assistant built on
[inclusionAI/Ming-3.0-audio-tiny](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny), running on an
**AMD Radeon GPU** (ROCm 7.2 inside WSL2 on Windows). Talk to it in the browser; it answers out loud,
typically starting to speak **~0.2 s after the end-of-turn pause** (≈0.7 s after you stop talking with the
default 500 ms pause).

* Hands-free conversation with voice activity detection, or push-to-talk
* **Speculative turn-taking**: the reply starts generating during your pause and is discarded if you keep talking
* Barge-in: talk over the assistant to interrupt it
* Streaming speech output (audio starts after the first ~160 ms of speech are generated)
* Multi-turn memory with KV-cache reuse, automatic rollover to a text history for long chats
* 7 built-in voices plus **voice cloning** from a 5–15 s sample
* Browser client: mic / speaker selection, live levels, transcript with replay & download, personas, settings
* English and Chinese (the model's languages)

Everything runs locally; no audio leaves the machine.

## Performance (RX 7900 XTX, 24 GB)

| | |
|---|---|
| Server time from end of your turn to first audio | ~0.2 s median (0–0.13 s after the end-of-turn pause, thanks to speculation) |
| Thinker (audio LLM, 9.3B MoE / 2.7B active) | ~67 tokens/s decode, ~170 ms prefill for a short utterance |
| Talker (speech synthesis) | ~0.3× real time end to end; 13 ms per 80 ms of audio for the core step |
| Startup | ~30 s (first start after install: ~1–2 min to compile kernels) |
| VRAM | ~20.5 GB |
| Speech quality | ASR round-trip WER 1.0% (same as the reference implementation) |

## Requirements

* Windows 11 with **WSL2** and **Ubuntu 24.04**
* AMD GPU supported by ROCm on WSL (tested: Radeon RX 7900 XTX) with ≥ 24 GB VRAM, and a recent AMD
  Adrenalin driver with WSL support
* ~30 GB free disk (model 20 GB + Python env)
* [uv](https://docs.astral.sh/uv/) in WSL

Native Linux with ROCm is untested but should need only the WSL-specific bits removed
(`HSA_ENABLE_DXG_DETECTION`, the `libhsa-runtime64` removal in `setup_env.sh`, `scripts/install_rocm_wsl.sh`).
NVIDIA GPUs are not targeted.

## Install

Inside WSL Ubuntu:

```bash
git clone https://github.com/JMS1717/ming-voice.git ~/ming-voice
cd ~/ming-voice
sudo bash scripts/install_rocm_wsl.sh      # ROCm 7.2.4 + librocdxg (WSL GPU bridge); log out/in afterwards
bash setup_env.sh                          # Python env (ROCm PyTorch wheels) + model download (~20 GB)
```

## Run

* **Windows:** double-click `windows\Ming Voice.cmd` (open the repo folder in Explorer via
  `\\wsl.localhost\Ubuntu\home\<you>\ming-voice`). It starts the server in its own window, waits for the
  models, and opens <http://localhost:8000>. Make a shortcut to it for your Desktop.
* **WSL shell:** `./run.sh`, then open <http://localhost:8000>.

Keep the server window open while you use it — WSL shuts its VM down when nothing is attached. Close it when
you're done: it holds ~20 GB of VRAM and keeps the GPU out of its deepest idle state (see Troubleshooting).

## Using it

* **Start conversation**, allow the microphone, talk, and pause when you're done.
* Interrupt by talking over it. With speakers instead of headphones, turn **barge-in** off if it interrupts
  itself.
* Shortcuts: <kbd>Space</kbd> hold to talk (push-to-talk mode), <kbd>M</kbd> mute, <kbd>Esc</kbd> stop reply,
  <kbd>N</kbd> new conversation.
* **Clone a voice** in the sidebar: record or upload 5–15 s of one person speaking. Voices are saved in
  `voices/`.

| Setting | Default | Meaning |
|---|---|---|
| End-of-turn pause | 500 ms | Silence that ends your turn. Lower = snappier, but more likely to cut you off. |
| Extra pause after short phrases | 250 ms | Added when you've said < 1 s ("Can you tell me…"), which is usually unfinished. |
| Start thinking after | 160 ms | When speculative generation starts. |
| Voice detection sensitivity | 0.50 | Silero VAD threshold. |
| Playback buffer | 100 ms | Client jitter buffer. |

### Configuration (environment variables for `run.sh`)

| Variable | Default | |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address. The server is local-only by default; `HOST=0.0.0.0` exposes it on your network (browsers only allow the microphone on `localhost` or HTTPS). |
| `MING_MODEL_DIR` | `./model` | Model folder. |
| `MING_MAX_CTX` | `8192` | Conversation context (tokens) before rolling over to a text history. |
| `MING_CFM_STEPS` | `10` | Talker flow-matching steps (fewer is faster but degrades speech). |
| `MING_FAST_TALKER` / `MING_GRAPHS` | `1` / `1` | Fused Triton Talker path / HIP graphs (set `0` to fall back to the reference code). |
| `MING_TRACEMALLOC` | `0` | Python allocation tracing for `/api/debug/mem`. |

## How it works

```
browser (static/index.html) ⇄ WebSocket: 16 kHz PCM in / 44.1 kHz PCM out ⇄ server.py
  server.py   Silero VAD · speculative turn-taking with rollback · barge-in · push-to-talk
              · voice cloning / TTS / ASR HTTP endpoints · background transcription
  engine.py   Thinker: Ming audio LLM (hybrid Kimi-delta-attention / MLA, 128-expert MoE)
                - static-state cache, one HIP graph per attention-length bucket (14 ms/token)
                - fused Triton MoE decode with the shared expert folded in (fused_moe.py)
                - multi-turn KV reuse, snapshot/restore for speculation, text-history rollover
              Talker: Qwen2 + 10-step flow-matching DiT + AudioVAE
                - fused Triton fast path (fast_talker.py, fastlinear.py): RMSNorm/bias/activation/
                  residual fused into small-M GEMMs, tiny rotary+attention kernel, decode attention
                  over a static KV cache, constants hoisted out of the flow-matching loop
                - HIP graph per step, streaming VAE decode
  direct_load.py  builds the model on the GPU and streams weights in with parallel reads
```

The main speedups over the reference implementation came from: replacing rocBLAS's slow small-M biased GEMMs
(~45 µs) with fused Triton kernels (~20 µs), cutting kernel launches (RMSNorm decomposed into ~6 kernels on
ROCm), graph-capturing every per-token step, and hiding model latency inside the end-of-turn pause.

## Troubleshooting

* **`No HIP GPUs are available`** — `HSA_ENABLE_DXG_DETECTION=1` must be set (`run.sh` does it).
* **First request after idle takes minutes** — under WSL, when the AMD GPU drives no display, an idle GPU stops
  serving WSL work promptly. The server runs a tiny keep-alive kernel a few times per second (`GpuKeepAwake`).
* **Slow loading / swap** — the loader streams weights straight to the GPU; Hugging Face's loader peaks at
  ~15 GB of host RAM, which exceeds WSL's default memory cap.
* **The assistant cuts you off** — raise the end-of-turn pause in Settings.
* Logs: `server.log`; health: <http://localhost:8000/api/health>.

## Development

`dev/` has the tests and benchmarks, run from the repo root against a running server unless noted:

* `dev/e2e_test.py` — acts like a microphone over the WebSocket: latency, multi-turn memory, hesitation
  handling, barge-in; transcribes the spoken replies (`dev/make_questions.py` regenerates its audio)
* `dev/robustness.py` — long input, Chinese, noise-only, double barge-in
* `dev/eval_steps.py` — speech quality (ASR round-trip WER) vs. diffusion steps (standalone, needs the GPU)
* `dev/test_fast_tail.py`, `dev/test_fast_llm.py`, `dev/test_sensitivity.py` — fused paths vs. reference
* `dev/tts_check.py`, `dev/test_clone.py`, `dev/test_takeover.py` — API checks

## Credits & licenses

* This project: MIT (see `LICENSE`).
* Model: [Ming-3.0-audio-tiny](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny) by inclusionAI (MIT);
  weights are downloaded from Hugging Face and not included here. The model's own code is loaded from the
  model folder with `trust_remote_code`.
* [Silero VAD](https://github.com/snakers4/silero-vad) (MIT), [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT),
  PyTorch / ROCm / Triton.
