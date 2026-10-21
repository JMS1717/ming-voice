# Ming Voice

Low-latency, fully local **speech-to-speech** voice assistant built on
[inclusionAI/Ming-3.0-audio-tiny](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny), running on an
**AMD Radeon GPU** (ROCm 7.2 inside WSL2 on Windows). Talk to it in the browser; it answers out loud,
typically starting to speak **~0.25 s after you stop talking**.

* Hands-free conversation with voice activity detection, push-to-talk, or **wake word** ("Hey Ming, ...") for an
  always-listening device, with a follow-up window so a conversation doesn't need it every time
* **Smart turn detection** ([Smart Turn v3](https://huggingface.co/pipecat-ai/smart-turn-v3)): judges from your
  words and intonation whether you've finished, answering sooner when you have and waiting when you're mid-sentence
* **Speculative turn-taking**: the reply starts generating during your pause and is discarded if you keep talking
* **Resume after a pause**: if it starts answering and you carry on within a second, the answer is withdrawn,
  the conversation rolls back, and everything you said is answered as one turn
* **Survives restarts**: the conversation is kept in `.conversation.json` after every turn; a server that
  restarts within 30 minutes (`MING_RESUME_S`) carries on where it left off. New chat clears it.
* **Speaks numbers and symbols properly**: "240 × 37", "−5°C", "24-17", "$1,299.99", "12 km/h" or "12:00 PM" are
  rewritten into words the voice reads correctly (each was checked by speaking it and transcribing it back)
* Barge-in: talk over the assistant to interrupt it, **even on a speakerphone**: the server recognises its own
  voice leaking back into the mic (echo-aware barge-in) and only stops for yours
* Streaming speech output (audio starts after the first ~160 ms of speech are generated)
* Multi-turn memory with KV-cache reuse, automatic rollover to a text history for long chats
* 7 built-in voices plus **voice cloning** from a 5–15 s sample
* Browser client: mic / speaker selection, live levels, transcript with replay & download, personas, settings
* **Use it from your phone** over HTTPS on your home network (QR code, access key, installable web app)
* **OpenAI-compatible API**: text-to-speech, transcription, chat with audio in and out
* **Tools**: web search, weather, calculator, clock, **timers and reminders**, **shopping and to-do lists**, and **remembering things you tell
  it** out of the box; **Home Assistant**
  (lights, plugs, media, thermostat, ...), **Jellyfin** and delegation to a **Hermes Agent** for bigger jobs when
  configured. The model itself decides when to call them (Ming-3.0's LLM was trained for function calling)
* English and Chinese (the model's languages)

Everything runs locally; no audio leaves the machine. With tools on, only the search terms / place names go to
DuckDuckGo (or Brave Search, if you add a key) and Open-Meteo (turn tools off in Settings to keep everything
offline).

## Performance (RX 7900 XTX, 24 GB)

| | |
|---|---|
| From the end of your speech to the first audio | ~0.23 s median, measured from the last audible sound of the question (the pause is timed from the drop in voice energy, 0.2 s when Smart Turn is confident; the reply is generated during it and is ready when the pause ends) |
| Thinker (audio LLM, 9.3B MoE / 2.7B active) | ~115 tokens/s decode; first token ~80 ms after a short spoken turn (~65 ms typed) |
| Talker (speech synthesis) | ~0.3× real time end to end; 13 ms per 80 ms of audio for the core step |
| Startup | ~30 s (first start after install: ~1–2 min to compile kernels) |
| VRAM | ~20.5 GB bf16 · ~12.5 GB with int8 experts (auto on < 23 GB cards) |
| Speech quality | ASR round-trip WER 1.0% (same as the reference implementation) |

## Requirements

* Windows 11 with **WSL2** and **Ubuntu 24.04**
* AMD GPU supported by ROCm on WSL with **16 GB+ VRAM** (tested: Radeon RX 7900 XTX), and a recent AMD
  Adrenalin driver with WSL support. Cards under 23 GB automatically use int8 expert weights (~12.5 GB total;
  verified under a 14 GB cap); 24 GB cards keep bf16 (~20.5 GB).
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

### Always on, without a browser

`windows\Ming Voice (always on).cmd` starts the server (minimized), phone access, and **`windows\ming_client.py`**,
a background client that uses the PC's own microphone and speakers. No tab to keep open and no Start button:
say "Hey Ming, …" any time. Timer alarms and announcements play through the speakers too.

* To start it whenever you sign in to Windows, run
  `powershell -ExecutionPolicy Bypass -File windows\install_autostart.ps1` once (`-Remove` undoes it;
  `-Browser` autostarts the browser app instead).
* Needs Python for Windows with `pip install sounddevice websockets numpy`.
* Settings go in `.client.json` in the project folder (not committed), e.g.
  `{"wake_word": "Ming", "home": "Paris", "voice": "US_Emma", "input_device": "USB Mic", "barge_in": false}`;
  `python windows\ming_client.py --test-mics` shows each microphone's level while you talk (to find the right
  one; `--list-devices` lists them all). The log is `windows\ming_client.log` (it says which microphone is in use,
  and warns when it has heard nothing for 10 minutes); `python windows\ming_client.py --stop` stops it.
* The server restarts by itself if it crashes or hits a GPU error it can't recover from (`run.sh` with
  `MING_SUPERVISE=1`, as the launcher starts it).
* On speakers the microphone is muted while Ming talks, so it never hears and answers itself. With a headset, set
  `"barge_in": true` to be able to talk over it.
* One device holds the conversation at a time: opening the app on your phone takes over, and the PC client takes
  it back by itself 15 s after the phone app is closed.

Keep the server window open while you use it — WSL shuts its VM down when nothing is attached. Close it when
you're done: it holds ~20 GB of VRAM and keeps the GPU out of its deepest idle state (see Troubleshooting).

## Using it

* **Start conversation**, allow the microphone, talk, and pause when you're done.
* Interrupt by talking over it. This works on speakers too: the server ignores its own voice coming back
  through the mic. If it still interrupts itself in a very echoey room, turn **barge-in** off in Settings.
* Shortcuts: <kbd>Space</kbd> hold to talk (push-to-talk mode), <kbd>M</kbd> mute, <kbd>Esc</kbd> stop reply,
  <kbd>N</kbd> new conversation.
* **Wake word** (Settings → Turn-taking): for a PC, tablet or old phone that stays on and listens all the time.
  It only answers speech that starts with its name: "Hey Ming, turn off the lamp", or "Hey Ming" on its own and
  wait for the chime. For a few seconds after each reply (8 s by default) you can follow up without the name.
  The name is checked by transcribing the first 1.6 s of each utterance while you're still talking, so answers
  are as quick as in hands-free mode; any name works (set it in Settings), "Ming" is recognised best. Timer
  alarms and Hermes results are announced whether it's asleep or not.
* **Clone a voice** in the sidebar: record or upload 5–15 s of one person speaking. Voices are saved in
  `voices/`.

### On your phone (or any device on your network)

Start with `windows\Ming Voice (phone).cmd` (or run `python windows\lan_proxy.py` on Windows next to a running
server), press **Phone** in the app on the PC, and scan the QR code. The phone opens
`https://<pc-ip>:8443/?key=…`; the key is swapped for a cookie, so the link only has to be opened once.

* Phones only allow the microphone on HTTPS, so the proxy uses a **self-signed certificate** (created on first
  run in `certs/`): the phone warns once that the connection is not private. Tap *Advanced → Proceed*
  (Safari: *Show Details → visit this website*).
* Everything arriving from the network needs the access key (`.lan_key`, created on first start). Delete the
  file and restart to change it.
* Needs Python for Windows (`lan_proxy.py` uses only the standard library). WSL2 is not reachable from the
  network by itself, which is why the proxy runs on the Windows side. If Windows Firewall asks, allow Python
  on private networks.
* Add the page to your home screen to use it like an app; the screen stays on during a conversation.
* One device talks to the assistant at a time: opening it on the phone takes over from the PC tab.

### Tools: web search, weather, timers, Home Assistant, Jellyfin, Hermes

With **Web search & tools** on (Settings, default), the assistant can call tools while it answers. It says
"Let me look that up" while a slow one runs, and the reply shows what it looked up and the sources.

| Tool | What it does | Needs |
|---|---|---|
| `web_search` | News, prices, scores, recent events (DuckDuckGo, no key; with a Brave key, Brave Search for web results, which are fresher, and DuckDuckGo as the fallback) | optional `.brave_key` or `MING_BRAVE_KEY` ([Brave Search API](https://brave.com/search/api/), free tier) |
| `get_weather` | Current weather and 3-day forecast (Open-Meteo, no key); "Your location" in Settings is the default | – |
| `calculate`, `get_current_time` | Exact arithmetic; the time in other time zones (the local time comes with every turn, so "what time is it?" needs no tool) | – |
| `timer` | "Set a timer for 10 minutes", "remind me to call mom at 5 pm", "how long is left?", "cancel the pasta timer". When one goes off the page chimes and says it (*"Time to call mom."*), again every minute (twice more) until you say something or press stop; kept in `.timers.json`, so they survive a restart, and one that went off while the server was down is announced late rather than lost | – |
| `lists` | "Add milk and eggs to my shopping list", "what's on my to-do list?", "I bought the eggs", "clear my shopping list": shopping, to-do or any list you name, kept in `.lists.json` on the server | – |
| `memory` | "Remember that I parked on level 3", "forget where I parked": facts kept in `.memory.json` (up to 40) and given to the model in every conversation, so "where did I park?" works days later | – |
| `home_assistant` | "Turn off the kitchen lights", "is the car locked?": passed to Home Assistant's own Assist intent handling, so device names, aliases and areas work as in HA, and **only entities exposed to voice assistants in HA** can be touched | `.ha_token` (a long-lived access token: HA → your profile → Security), `.ha_url` or `MING_HA_URL` (default `http://homeassistant.local:8123`) |
| `jellyfin` | "Play The Office on the TV", "pause", "what's playing?", "do I have Dune?": searches your library and remote-controls open Jellyfin players (a series continues from the next unwatched episode) | `.jellyfin.json`: `{"url": "http://<host>:8096", "key": "<API key>"}` (Jellyfin → Dashboard → API Keys) |
| `ask_hermes` | "Ask Hermes to ...": hands a job (research, your files and accounts, reminders, multi-step work) to a [Hermes Agent](https://github.com/NousResearch/hermes-agent) in the background; the answer is announced when it's ready, even if you've moved on. Only when you say "Hermes" (checked against what you said); `MING_HERMES_AUTO=1` lets Ming hand it anything its own tools can't do | `.hermes.json`: `{"url": "http://<host>:8642", "key": "<API_SERVER_KEY>"}` (Hermes: `API_SERVER_ENABLED=true`, `API_SERVER_KEY`, `API_SERVER_HOST` in `~/.hermes/.env`) |

Tools that change something (timers, lists, memory, Home Assistant, Jellyfin, Hermes) never run for a speculative reply: they wait until your
turn has really ended, so a pause mid-sentence can't switch a light. Files are read at server start.
Jellyfin only starts playback when you said to (play, watch, put on, listen, ...), and is only used when you talk
about watching or listening to something. An answer from search or weather results ends after a few sentences,
rather than reading out a whole list of search results; any other answer ends after about 30 seconds of speech,
unless you asked for something long (a story, a recipe, steps, "in detail", ...). Each reply privately starts
from the local date and time, so "what time is it?" needs no tool. And if a reply starts by claiming it can't
look things up ("I don't have access to real-time data..."), it's caught before it's spoken and becomes a web
search (or weather lookup) instead. Likewise, a reply that says it changed a list or forgot something without
calling the tool ("The eggs are off your list") gets that call made for it, so what it said is true.

**Risky actions are confirmed first.** Unlocking, opening a garage door or gate, and disarming or turning off an
alarm get a question back (*"Just to be sure: do you want me to unlock the front door?"*) and only happen when
you say yes in your next turn, within a minute. What runs is exactly what you were asked about; a yes in the same
reply that asked, or a different device, doesn't count. Questions ("is the garage open?") and locking/closing
don't ask. `MING_CONFIRM_RISKY=0` turns this off.

### Announcements from Home Assistant (or anything else)

Ming can say things that happen at home: the doorbell, the washer finishing, the garage left open. With a
Home Assistant token configured, the server listens on HA's event bus for **`ming_announce`** events, so all
it takes is an automation with an *event* action (no HA integration, `configuration.yaml` change or restart):

```yaml
alias: Doorbell → Ming
triggers:
  - trigger: state
    entity_id: binary_sensor.front_doorbell
    to: "on"
actions:
  - action: event
    event: ming_announce
    event_data:
      message: Someone is at the front door.
      # chime: false      no chime before it
      # verbatim: false   pass it on in Ming's own words instead of exactly
      # ttl: 300          drop it if it can't be said within this many seconds
```

Or from a script, another server, or an HA `rest_command`: `POST /api/announce` with the same fields as JSON
(on the LAN port, with `Authorization: Bearer <access key>`):

```bash
curl -X POST http://localhost:8000/api/announce -H "Content-Type: application/json" \
  -d '{"message": "The washer is done."}'
```

An announcement waits for a quiet moment (never mid-sentence), plays a short chime, and is said whether the
wake word has been heard or not. `MING_HA_EVENTS=0` turns the event listener off.

### API

An OpenAI-compatible API runs on the same server; interactive docs at <http://localhost:8000/docs>.

| Endpoint | |
|---|---|
| `POST /v1/audio/speech` | Text to speech: `{"input": "...", "voice": "US_Emma", "response_format": "wav"}` (wav, mp3, flac, pcm = s16le 24 kHz; wav/pcm stream as generated, first audio after ~0.12 s). OpenAI voice names (alloy, nova, …) map to built-in voices; cloned voices work by name. |
| `POST /v1/audio/transcriptions` | Speech to text (multipart `file`), any length, English/Chinese. |
| `POST /v1/chat/completions` | Chat with text and/or audio input (`input_audio` parts) and optional spoken output (`"modalities": ["text", "audio"]`); `stream: true` streams text (and pcm16 audio). Greedy decoding, no tools. |
| `GET /v1/models` | `ming-3.0-audio-tiny` |

```bash
curl http://localhost:8000/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input": "Hello from Ming Voice!", "voice": "US_Emma"}' -o hello.wav
curl http://localhost:8000/v1/audio/transcriptions -F file=@hello.wav
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused-on-localhost")
print(client.chat.completions.create(model="ming-3.0-audio-tiny",
      messages=[{"role": "user", "content": "Tell me a fun fact about otters."}]).choices[0].message.content)
```

From another device use `https://<pc-ip>:8443/v1` with `api_key=<contents of .lan_key>` (and accept the
self-signed certificate, e.g. `curl -k`, or pass `certs/lan.crt` as the CA). API requests share the GPU with
the voice session, one at a time; chat requests run on an isolated copy of the conversation state, so they
never disturb a live conversation. `dev/test_api.py` exercises every endpoint.

| Setting | Default | Meaning |
|---|---|---|
| Smart turn detection | on | Ends the turn after a 200 ms pause when you've clearly finished, waits up to 1.5 s when you sound mid-sentence, and uses the end-of-turn pause otherwise. |
| End-of-turn pause | 500 ms | Silence that ends your turn (when Smart Turn is off or unsure). Lower = snappier, but more likely to cut you off. |
| Extra pause after short phrases | 250 ms | Added when you've said < 1 s ("Can you tell me…"), which is usually unfinished. |
| Start thinking after | 32 ms | When speculative generation starts (it is discarded if you keep talking). |
| Keep my turn if I resume within | 1000 ms | If you carry on talking this soon after it starts answering, the answer is withdrawn and everything you said is answered as one turn (0 = carrying on interrupts it). |
| Ignore the assistant's own voice | on | Echo-aware barge-in for speakers: its own voice coming back through the mic doesn't count as you talking. |
| Voice detection sensitivity | 0.50 | Silero VAD threshold. |
| Playback buffer | 100 ms | Client jitter buffer. |

### Configuration (environment variables for `run.sh`)

| Variable | Default | |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address. The server is local-only by default; for other devices use the phone proxy (above). `HOST=0.0.0.0` exposes port 8000 directly, without HTTPS (no microphone in browsers) and without a key unless `MING_API_KEY` is set. |
| `MING_LAN_PORT` | `8001` | Key-protected port for the phone proxy (localhost only; `lan_proxy.py --target`). |
| `MING_API_KEY` | – | Require this key (`Authorization: Bearer …`) on every port, including localhost. |
| `MING_MODEL_DIR` | `./model` | Model folder. |
| `MING_MAX_CTX` | `8192` (`16384` from 32 GB of VRAM, `4096` in int8 mode) | Conversation context (tokens). It costs VRAM (~60 MB per 1k tokens allocated at start, plus bigger graph buffers: 16k takes ~1.5 GB more than 8k, which leaves a 24 GB card too little working memory) but not speed: attention only reads the filled part, so short conversations are as fast as with a small context. When it fills up, the context is rebuilt between turns (while nobody is talking, ~0.5 s) from the conversation so far as text: the first two exchanges and the most recent ones, up to 4k tokens. |
| `MING_INT8_EXPERTS` | `auto` | Weight-only int8 for the MoE experts (`1`/`0`/`auto` = int8 below 23 GB VRAM). Measured vs bf16: identical greedy replies and transcripts on the test set, logit cosine ≥ 0.997, +2% NLL. |
| `MING_VRAM_LIMIT_GB` | – | Cap PyTorch's VRAM use (to emulate a smaller card). |
| `MING_CFM_STEPS` | `10` | Talker flow-matching steps (fewer is faster but degrades speech). |
| `MING_SMART_TURN` | `1` | Smart Turn end-of-turn detection (needs `model_extra/smart-turn-v3.2-cpu.onnx`, downloaded by `setup_env.sh`). |
| `MING_FUSED_PREFILL` | `1` | Grouped Triton MoE kernels for prompt prefill (`0` = the reference per-expert loop). |
| `MING_PREFILL_GRAPHS` | `1` | Graph-captured prompt prefill for prompts up to 128 tokens, padded to 32/64/128 (`0` = eager). |
| `MING_FUSED_OPS` | `1` | Triton RMSNorm and one-token MoE router (`0` = the reference PyTorch ops). |
| `MING_ECHO_DEBUG` | unset | Log what the echo guard measured (correlation, delay, levels) whenever a barge-in fires. |
| `MING_FAST_TALKER` / `MING_GRAPHS` | `1` / `1` | Fused Triton Talker path / HIP graphs (set `0` to fall back to the reference code). |
| `MING_TRACEMALLOC` | `0` | Python allocation tracing for `/api/debug/mem`. |
| `MING_CONFIRM_RISKY` | `1` | Ask before unlocking, opening the garage/gate, or disarming an alarm (Home Assistant). |
| `MING_HA_EVENTS` | `1` | Say Home Assistant `ming_announce` events (needs `.ha_token`). |
| `MING_HERMES_AUTO` | `0` | `1`: hand Hermes anything Ming's own tools can't do, not only what you ask Hermes for by name. |
| `MING_SUPERVISE` | `0` | `1` (`run.sh`; the Windows launcher sets it): restart the server when it crashes. |
| `MING_ALARM_REPEATS` | `2` | How many more times a timer that went off is said when nobody answers (`0`: once only). |
| `MING_ALARM_REPEAT_S` | `60` | Seconds between those repeats. |
| `MING_RESUME_S` | `1800` | After a restart, carry on with a conversation saved less than this many seconds ago (`0`: always start afresh). |

## How it works

```
browser (static/index.html) ⇄ WebSocket: 16 kHz PCM in / 44.1 kHz PCM out ⇄ server.py
  server.py   Silero VAD · Smart Turn (turn_detect.py) · speculative turn-taking with rollback · barge-in
              · resume (a reply you talk over within resume_ms is undone and your turn continues)
              · echo guard (echo_guard.py: mic vs. played-audio correlation, so its own voice can't barge in)
              · push-to-talk · voice cloning · background transcription
  api.py      OpenAI-compatible /v1 API (speech, transcriptions, chat with audio in/out)
  lan.py      access key + cookie login for the LAN port; windows/lan_proxy.py terminates HTTPS on Windows
  engine.py   Thinker: Ming audio LLM (hybrid Kimi-delta-attention / MLA, 128-expert MoE)
                - static-state cache, one HIP graph per attention-length bucket (8.7 ms/token)
                - prefill graphs too: prompts padded to 32/64/128 tokens, padding made a no-op for the
                  recurrent KDA layers (zero write strength, unit decay, conv state rebuilt from real tokens)
                - fused Triton MoE with the shared expert folded in (fused_moe.py): a GEMV kernel pair
                  for decode, grouped tl.dot kernels for prefill (tokens sorted by expert on the GPU)
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
ROCm; the reference MoE prefill launched ~5 kernels per active expert per layer plus a host sync, ~120 ms per
turn, now ~15 ms), graph-capturing every per-token step, and hiding model latency inside the end-of-turn pause.

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
* `dev/e2e_compare.sh 32:1 64:1` — e2e under several `spec_ms:smart_turn` settings
* `dev/test_fast_tail.py`, `dev/test_fast_llm.py`, `dev/test_sensitivity.py` — fused paths vs. reference
* `dev/test_moe_prefill.py` (`TUNE=1` to re-tune), `dev/test_prefill.py`, `dev/test_prefill_layer.py` — fused
  prefill MoE: accuracy vs. fp32, replies, time to first token
* `dev/test_prefill_graphs.py` — graph-captured padded prefill vs. eager: same tokens, states within bf16 noise
* `dev/test_turn.py` — Smart Turn scores on complete vs. truncated questions
* `dev/test_resume.py` — pause mid-question until the reply starts, then continue: one answer to the whole
  question, nothing left of the withdrawn reply; late interruptions still interrupt
* `dev/echo_test.py` — speakerphone simulation: the reply leaks back into the "mic" at several levels and delays;
  must never interrupt itself, but must still stop for a real interruption (`dev/test_echo_guard.py`: offline)
* `dev/test_api.py` — every `/v1` endpoint, plus a live WebSocket conversation that must survive the API calls
  (`BASE=https://<pc-ip>:8443 KEY=… python dev/test_api.py` from another machine)
* `dev/test_encoder.sh`, `dev/profile_prefill.py` — audio encoder (trimmed chunks, flash attention) and prefill timing
* `dev/tts_check.py`, `dev/test_clone.py`, `dev/test_takeover.py` — API checks
* `dev/test_speech_text.py` — offline: symbols, money, ranges and units rewritten for the voice
* `dev/test_routing.py` — spoken everyday questions: which tools each one uses (none for general knowledge)

## Credits & licenses

* This project: MIT (see `LICENSE`).
* Model: [Ming-3.0-audio-tiny](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny) by inclusionAI (MIT);
  weights are downloaded from Hugging Face and not included here. The model's own code is loaded from the
  model folder with `trust_remote_code`.
* [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) by Pipecat (BSD-2-Clause), downloaded by `setup_env.sh`.
* [Silero VAD](https://github.com/snakers4/silero-vad) (MIT), [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT),
  PyTorch / ROCm / Triton.
