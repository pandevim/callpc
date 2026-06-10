# "Call My Computer" — voice-controlled computer-use agent

## Context

"Call" your computer from a phone — a WebRTC web page that looks like a phone call (answer animation, call timer, mute/hang-up), kept on speaker — talk naturally to a real-time voice agent, and have it operate the computer (open apps, send messages, browse) via a GUI agent model, narrating progress back over the call.

**The ASR engine is already vendored.** `server/asr/` contains a proven, working streaming Nemotron ONNX engine (details in "Vendored ASR engine" below). Everything else in this repo is built new.

**Decisions made:**

- Call leg: **plain WebRTC web page** styled like a phone call. It's a website, so the caller can be on any device/OS.
- Topology: **L40S = the remote server**, hosting everything heavy (ASR, TTS, orchestrator LLM, UI-TARS, voice gateway). The **only local component** is a tiny agent installed on whatever computer is being controlled — it connects outbound, executes input actions, returns screenshots.
- Controlled computer is **any OS**: Windows, macOS, and Linux. The agent is a single small static binary (Rust or Go) so installing it is download-and-run — see "Agent distribution" below.
- Brain: **self-hosted LLM on the L40S** via vLLM.
- Voice loop: **Pipecat** — for streaming TTS, Silero VAD, and real barge-in out of the box (a plain turn-based browser↔STT↔LLM↔TTS bridge works but lacks barge-in, which matters for interrupting a running task).
- STT: **Nemotron ONNX** (the vendored engine), chosen over Kyutai STT after comparison — see "STT choice" rationale under the vendored-engine section. Kyutai is used for **TTS only**.
- GPU target: a **single `gpu-l40s` (48 GB)** — the only GPU available; everything must fit in it.

## Architecture

```
Phone / any browser                    L40S server (remote)                   Controlled computer (Win/Mac/Linux)
┌──────────────────┐   WebRTC   ┌─────────────────────────────────┐   WSS    ┌──────────────────────┐
│ Call web page    │◀─────────▶│ Voice gateway (Pipecat 1.3)      │◀────────▶│ callpc-executor       │
│ (looks like a    │  (audio)   │  ├ STT: Nemotron ONNX (ported)  │ outbound │ (dumb I/O stub: ~5 MB │
│  phone call)     │            │  ├ Brain: Qwen3.5-9B (vLLM #2)  │  only    │  static binary,       │
└──────────────────┘            │  └ TTS: Kyutai 1.6B (moshi-srv) │          │  paired once)         │
                  HTTPS         │ Task runner ── UI-TARS-1.5-7B   │          │  ├ screenshots (xcap) │
                                │  (agent loop)    (vLLM #1)      │          │  └ input inject(enigo:│
                                └─────────────────────────────────┘          │    SendInput/CGEvent/ │
                                                                             │    uinput)            │
                                                                             └──────────────────────┘
```

All intelligence is server-side; the executor has zero smarts by design.

Voice loop: browser mic → 48 kHz Opus → Pipecat (resamples to 16 kHz for STT, Silero VAD, barge-in) → Nemotron streaming text → Qwen3.5 (persona + tools) → Kyutai TTS (24 kHz) → back over WebRTC.

Task loop: Qwen calls `computer_task(instruction)` → task runner loops {screenshot from the controlled computer → UI-TARS via vLLM → parse action → execute on the controlled computer} until `finished()` / max steps, emitting narration events spoken between actions. User can barge in to cancel/redirect.

## Repo layout

```
callpc/
  server/
    gateway/app.py          # FastAPI: /api/offer signaling, static call UI, Pipecat pipeline
    gateway/stt_nemotron.py # Pipecat STTService subclass wrapping our ASR (sample_rate=16000)
    gateway/tts_kyutai.py   # Pipecat InterruptibleTTSService — msgpack WS client to moshi-server
    gateway/brain.py        # persona prompt + tool defs (computer_task/task_status/cancel_task)
    asr/                    # VENDORED working code: onnx_engine.py, onnx_features.py, download_onnx.py, ws_engine.py, assets/*.npy
    tasks/runner.py         # UI-TARS agent loop: history (last 5 shots), smart_resize coord mapping
    tasks/uitars_client.py  # OpenAI-compatible calls to vLLM; `ui-tars` PyPI parser
    control/hub.py          # WS hub the laptop agent connects to; auth via pre-shared token
    deploy/                 # SLURM+Apptainer: build_images.sh + prefetch_weights.sh (login-node) + callpc.sbatch (5 services, one 24h job); moshi_with_git.def; dotenvx .env
  webapp/                   # call UI: dial → ringing → in-call screen; @pipecat-ai/client-js +
                            # @pipecat-ai/small-webrtc-transport (v0 can use pipecat-ai-small-webrtc-prebuilt)
  executor/                 # Rust — single static binary per OS; the ONLY thing users install.
                            # NOT an AI agent: a dumb I/O stub (receive action → inject input / return screenshot)
    src/main.rs             # outbound WSS client: pairing, action dispatch, reconnect, panic key
    src/screen.rs           # screenshots via xcap (Windows/macOS/X11/Wayland)
    src/input.rs            # input via enigo (SendInput / CGEvent / uinput+libei); Linux-Wayland fallback: dotool-style absolute-axis uinput device
    src/pair.rs             # callpc:// URL-scheme handler + pairing-code flow
  webapp/connect/           # "Connect this computer" page: OS detection, download link, pairing code
  docs/PROTOCOL.md          # control-channel protocol (below); ASR protocol already in docs/ASR_PROTOCOL.md
```

## Key implementation facts (researched, verified June 2026)

**Pipecat 1.3.0** (`pip install "pipecat-ai[webrtc,silero]"`):

- STT: subclass `STTService`; `run_stt(audio)` receives 16-bit PCM at our requested 16 kHz; push `InterimTranscriptionFrame`/`TranscriptionFrame` from a WS receive task. Our existing engine protocol maps 1:1.
- TTS: subclass `InterruptibleTTSService` (Kyutai streams can only be cancelled by closing the socket — this base handles reconnect-on-interrupt). Pipeline default output rate is 24 kHz = Kyutai's native rate, no resampling.
- `SmallWebRTCTransport` + `SileroVADAnalyzer` + `allow_interruptions=True` gives VAD, turn-taking, and barge-in for free. Signaling is one `POST /api/offer`.

**Kyutai TTS 1.6B** (`kyutai/tts-1.6b-en_fr`, weights CC-BY-4.0). Two ways to run it; prefer the proven one:
- **Preferred (lowest friction on the cluster):** the `moshi` Python library inside an Apptainer container — pytorch base image + `pip install git+…/moshi`, run with `TORCHDYNAMO_DISABLE=1`. Wrap it in a small **streaming** server (yield 24 kHz float32 PCM as frames generate) so Pipecat gets low time-to-first-audio. A plain HTTP `/synthesize` returning the whole clip is the simplest first cut; making it stream is the main TTS task.
- **Alt (official streaming):** Rust `moshi-server` exposing a WS msgpack API (`ws://host:8080/api/tts_streaming?voice=<voice>`, `kyutai-api-key` header; send `{"type":"Text","text":word}`/`Eos`, receive `{"type":"Audio","pcm":[f32]}`). Lower latency but adds a Rust build — only if the Python path underperforms.
~220–350 ms to first audio; native 24 kHz. Cap VRAM ~4–6 GB (small batch). Either way the Pipecat `InterruptibleTTSService` cancels speech by closing the connection on barge-in.

**Orchestrator LLM — Qwen3.5-9B** (newest small-dense generation, March 2026; Qwen leads open tool-calling benchmarks in this class; Qwen3.6 only ships 27B/35B-A3B — too big alongside UI-TARS). Serve FP8 (native on L40S Ada): `vllm serve Qwen/Qwen3.5-9B-Instruct-FP8 --max-model-len 16384 --gpu-memory-utilization 0.22 --enforce-eager`, thinking mode OFF for latency. Persona: terse, phone-style replies; confirms before destructive tasks. Upgrade path: Qwen3.6-35B-A3B-AWQ if we also move UI-TARS to FP8.

**UI-TARS-1.5-7B** (Apache-2.0, Qwen2.5-VL base):

- `vllm serve ByteDance-Seed/UI-TARS-1.5-7B --served-model-name ui-tars --max-model-len 32768 --limit-mm-per-prompt '{"image":5}' --gpu-memory-utilization 0.42 --enforce-eager`
- Use the official COMPUTER_USE prompt (Thought/Action; actions: `click/left_double/right_single/drag/hotkey/type/scroll/wait/finished` with `<point>x y</point>`).
- **Coordinates are absolute pixels in the smart_resize'd image** (factor 28) — map back: `screen_x = x / resized_w * orig_w`. Use `pip install ui-tars` → `parse_action_to_structure_output(..., model_type="qwen25vl")`. Getting this wrong = clicks land off-target (top risk; write a calibration test that clicks a known target).
- History: last **5** screenshots as user-image messages; older turns keep Thought/Action text only.

**callpc-executor (the only local install) — cross-platform from v1:**

The executor is deliberately dumb — receive an action JSON, execute it, return a screenshot — a few hundred lines total. It's a compiled binary only for distribution's sake (single ~5 MB file, no Python/runtime to install) and because input injection is a per-OS API (SendInput / CGEvent / uinput), which `enigo` wraps. Rust: `enigo` (input), `xcap` (screenshots), `tokio-tungstenite` (WS). Go + `robotgo` is the fallback choice if Rust friction appears.

| OS            | Input backend (enigo) | Screenshots (xcap) | One-time user friction                                                                                                                                                                         |
| ------------- | --------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Windows       | `SendInput`           | DXGI               | none (SmartScreen warning if unsigned)                                                                                                                                                         |
| macOS         | `CGEvent`             | CoreGraphics       | grant Screen Recording + Accessibility (agent guides through it, like Zoom)                                                                                                                    |
| Linux X11     | XTest/uinput          | X11                | none beyond input group                                                                                                                                                                        |
| Linux Wayland | uinput + libei        | portal/PipeWire    | one portal grant + a udev rule granting `/dev/uinput` to the `input` group. Fallback: dotool-style absolute-axis uinput device (NOT ydotool — `mousemove --absolute` broken, issues #250/#195) |

**Executor distribution (install must be smooth):**

1. Caller's web page → "Connect this computer" → detects OS, offers the right binary + a short pairing code.
2. User downloads and runs it — no installer, no runtime deps. The page also offers a `callpc://pair/<code>` link so the running executor picks up the token with zero typing (custom URL scheme, registered on first run); manual code entry as fallback.
3. Executor stores the token, connects **outbound WSS** to the server, shows "online" on the site. Auto-start at login is an opt-in toggle.
4. CI (GitHub Actions) cross-compiles win-x64, mac-universal, linux-x64 on every release. macOS notarization + Windows code-signing are post-v1 polish (until then: right-click-open on Mac, "run anyway" on Windows).

Note: a pure-browser executor (WASM) is impossible — browsers cannot synthesize OS-level input outside their own pages. The browser-extension route (tab screenshots + DOM-synthesized clicks, like UI-TARS's browser operator) is a possible zero-install v2 mode for web-only tasks.

**Control protocol** (server hub ⇄ callpc-executor, JSON over WSS, platform-agnostic — mirrors the ASR protocol's style): `hello{token, os, screen_w/h}` → `screenshot_request`/`screenshot{png_b64, w, h}` → `action{kind, args}`/`action_result{ok, error}` → `task_status`. The executor connects outbound only (no inbound ports on the controlled machine). Safety on the executor side: panic key that kills the session (Linux: a read-only evdev keyboard watch — never grab the device, so it can't disable the user's input; Win/Mac: a global hotkey), max-actions-per-minute rate limit, a visible "being controlled" indicator, and the brain confirms destructive intents by voice before dispatch.

**Networking (research prototype):** the gateway binds `0.0.0.0:<port>` on the compute node, then is exposed publicly via a reverse-proxy chain: an **SSH tunnel from a laptop through the login node to the compute node** (`ssh -f -N -L 8080:<compute-node>:<port> <user>@<cluster-login-host>`), then **`ngrok http 8080`** for a public HTTPS URL. The **static webapp deploys on a static host (e.g. Vercel)** with an `API_BASE_URL` env var pointed at the ngrok URL — the static host gives the TLS/secure-context `getUserMedia` requires, ngrok carries it to the job. The job only needs to bind a known host:port and print the compute-node name (for the tunnel). Keep a STUN server in the WebRTC config (`stun:stun.l.google.com:19302`); only add TURN if media fails through ngrok in testing. Research/demo only, not hardened for public exposure. (A persistent reverse proxy on a dedicated server substitutes for the SSH-tunnel+ngrok hop unchanged.)

**VRAM budget (48 GB):** UI-TARS ~20 GB (0.42) + Qwen3.5-9B-FP8 ~10 GB (0.22) + Kyutai TTS ~5 GB + Nemotron ONNX ~3.4 GB ≈ **38 GB**, ~10 GB headroom — all four share the **single GPU** in one job allocation. Start order matters (vLLM claims fractions of _total_ memory at startup): vllm-uitars → vllm-brain → tts → asr → gateway. No systemd/root on compute nodes, so a **launcher backgrounds these in order inside one sbatch job**: track child PIDs in a `BG_PIDS` array, `wait_for` (curl-poll) each service's health/port before starting the next, and `trap cleanup EXIT INT TERM` to kill all children when the job ends.

## Server environment (SLURM / HPC cluster — the real L40S host)

The L40S is reached through a **SLURM batch cluster** (university HPC), not a persistent always-on box. This is the single biggest deployment constraint. Key facts:

- **Scheduler:** SLURM. Workflow: `git clone` on the login node → put secrets in encrypted `.env` → `sbatch`. Partition: **`gpu-l40s` (48 GB)** — the only GPU we have access to. Request `--gres=gpu:1 --cpus-per-task=… --time=24:00:00`. **~24 h max wall time** — everything in the job dies at the limit; resubmit. No persistent daemon; always-on is out of scope.
- **Container runtime = Apptainer (Singularity), not bare pip.** This is the cluster-blessed way to run as non-root and dodge `$HOME` cache-quota limits. Run vLLM straight from the official image: `apptainer run --nv docker://vllm/vllm-openai:latest --model … --port …`. Run moshi (TTS) from a tiny custom image built once from a `.def` (pytorch base + git, then `pip install git+…/moshi` at launch). `--nv` for GPU, `--fakeroot` to build, `--bind` for caches. Build/cache the `.sif` once in the submit dir; subsequent jobs skip the build. Avoid a shared-FS pip venv (concurrent installs race and corrupt it).
- **Node-local fast SSD for caches:** bind `/local/$SLURM_JOB_ID` into the container as the HF + Apptainer cache (`APPTAINER_BIND=/local/$SLURM_JOB_ID:/root/.cache/huggingface`, `APPTAINER_CACHEDIR`/`APPTAINER_TMPDIR` likewise). Using `$HOME` for these freezes the shared FS. Pre-warm weights into a persistent `HF_HOME` (e.g. `$HOME/hf_cache` or submit-dir) on the **login node** so jobs start fast and survive restricted compute-node internet. Set `HF_TOKEN` for gated repos.
- **Modules:** `module load apptainer`; for any host-side Python, `module load python/3.11.10 cuda12.8/toolkit/12.8.1`, `unset PYTHONHOME`. No `sudo`/systemd — every service is a user process inside the allocation.
- **Secrets via dotenvx:** encrypted `.env`/`.env.production` committed; private key kept out of git in `~/.secrets/<name>.key` and `source`d at job start; standalone `dotenvx` binary in `~/.local/bin`, wrapping service launches (`dotenvx run -- apptainer run …`). Reuse verbatim.
- **Multi-node option:** SLURM also supports split topologies (`scontrol show hostnames`, pin services to nodes with `srun -w <node>`, cross-node URLs over the cluster fabric). Not needed for the single-L40S budget, but available if a model is later moved to its own node.

**Implication for `server/deploy/`:** ship **(1)** a login-node `prefetch_weights.sh` + a `build_images.sh` (build/cache the moshi `.sif`, pull the vLLM image) and **(2)** `callpc.sbatch` — one job that loads `apptainer`, sources the dotenvx key, sets up `/local/$JOBID` caches, launches the five services (two `vllm-openai` containers, the moshi-TTS container, the in-process ASR, the Pipecat gateway) in dependency order with `wait_for` health polls and a `cleanup` trap, prints the compute-node name + gateway port (for the SSH tunnel), and `wait`s. Research/demo loop: submit a 24 h job, open the tunnel + ngrok, call in.

## Phases

**Phase 0 — provisioning (SLURM + Apptainer):** login-node one-time setup — `build_images.sh` (build/cache the moshi `.sif`, pull `docker://vllm/vllm-openai:latest`), `prefetch_weights.sh` (UI-TARS, Qwen3.5-9B-FP8, Kyutai TTS+voices into `HF_HOME`), the vendored ASR `download_onnx.py` + its `__main__` self-test, and the dotenvx key in `~/.secrets/`. Then a first `callpc.sbatch` bringing up the five services in order, printing the compute-node + port. Verify each standalone inside the job (curl the two vLLM endpoints, a TTS `/synthesize` smoke test, a WS client against the ASR engine).

**Phase 1 — voice loop:** Pipecat pipeline with the two custom services + prebuilt WebRTC UI; then the custom phone-call webapp (dial/ringing/in-call screens, mute, hang up, call timer; mobile-first). Milestone: full conversation with sub-second-feeling latency and working barge-in from a phone on speaker, and from Windows/Mac desktop browsers.

**Phase 2 — control plane:** `callpc-executor` (Rust, all three OS backends) + pairing flow, control hub on the server, UI-TARS task runner. Drive it with _typed_ instructions first (debug web page showing screenshot trail + Thought/Action per step). Milestone: "open firefox and search for X" works on Windows, macOS, and Linux; coordinate calibration test passes on each; panic key works; download-and-pair takes under a minute on a fresh machine.

**Phase 3 — wire voice → tasks:** brain tool-calling dispatches to the task runner; narration events spoken between steps; barge-in cancels/redirects tasks; voice confirmation for destructive actions. Milestone: the full demo — call from phone, ask "send a message to John on WhatsApp Web saying I'll be late", watch it happen, hear progress.

**Phase 4 — polish:** call UI sounds/animations, auth hardening (per-device tokens), transcript + screenshot-trail logging for every task, executor login-launch toggle, macOS notarization + Windows code-signing, and (if moving off the cluster) a persistent always-on host so calls don't need a live sbatch job + tunnel.

## Verification

- Per-service smoke tests in Phase 0 (each model service has a standalone test script).
- Voice loop: measure end-to-end latency (speak→hear); target < 1.5 s; barge-in cuts TTS < 300 ms.
- Coordinate calibration: render a full-screen grid target page, ask UI-TARS to click named cells, assert hit positions (catches smart_resize mapping errors immediately).
- Scripted task suite via the Phase-2 debug page: open app, type in field, browser search, window switch.
- Executor matrix: pair + run the scripted task suite on Windows, macOS, Linux-X11, Linux-Wayland.
- Cross-platform caller check: Chrome/Android, Safari/iOS, Chrome/Windows, Safari/macOS.

## Top risks

1. UI-TARS coordinate mapping (smart_resize) — mitigated by calibration test before any real task.
2. TTS streaming on the cluster — the proven moshi-in-Apptainer server is HTTP/non-streaming; making it stream low-latency is the main TTS risk. Fallback: Rust `moshi-server` WS, or chunk the HTTP output finely.
3. Linux-Wayland input/screenshot quirks — enigo/libei first, dotool-style absolute-axis uinput fallback; Xorg session as last resort.
4. VRAM startup ordering on the single 48 GB L40S — fixed `--gpu-memory-utilization` fractions + ordered launch with health-checks; if the ~38/48 GB budget gets tight, drop the brain to a smaller quant (Qwen3.5-4B-FP8) or enable vLLM sleep-mode on UI-TARS between tasks.
5. Unsigned binaries — SmartScreen/Gatekeeper warnings until signing/notarization lands (Phase 4); document right-click-open for macOS.
6. SLURM constraints — 24 h job cap (resubmit per session; research prototype, not always-on). Reachability handled by the user's own reverse proxy; only add TURN if direct WebRTC media fails in testing.

## Vendored ASR engine (`server/asr/`)

Working, tested code — already copied into this repo (adapt imports/paths during Phase 0):

- `onnx_engine.py` — `OnnxASR` (offline) + `OnnxStreamingSession` (real-time). Torch/NeMo-free: onnxruntime + numpy + sentencepiece only; ~0.6–0.9 GB host RAM, ~3.4 GB VRAM. Has a `__main__` self-test (offline vs streaming on a wav).
- `onnx_features.py` — numpy-only 128-mel log frontend replicating NeMo's preprocessor exactly (preemph 0.97, n_fft 512, win 400, hop 160, slaney mel 0–8 kHz, natural log with 2^-24 guard); mel filterbank + Hann window precomputed in `assets/mel_fb.npy` / `assets/window.npy`. Do not swap in librosa — bit-compatibility with the export matters.
- `download_onnx.py` — fetches the pre-exported ONNX checkpoint (~2.5 GB, gitignored) from HF `altunenes/parakeet-rs`, subfolder `nemotron-3.5-asr-streaming-0.6b-onnx/`: `encoder.onnx(+.data)`, `decoder_joint.onnx`, `tokenizer.model`, `config.json`.
- `ws_engine.py` — WebSocket wrapper (protocol spec in `docs/ASR_PROTOCOL.md`): `start{lang}` → `ready{chunk_bytes, sample_rate}` → binary int16 16 kHz mono PCM frames → incremental `{"type":"text"}` deltas → `stop`. Serializes to one streamer at a time.

How the streaming works (hard-won — don't rediscover): the exported encoder consumes one 65-mel-frame window per call (9 lookback + 56 new frames → 7 encoder output frames), threading cache tensors (`cache_last_channel/time/len`) between calls; greedy RNNT decoding threads LSTM states (`input_states_1/2`) and the last emitted token across chunks; `finalize()` flushes the final partial window at utterance end. One encoder step = 56 × 160 samples = **560 ms of audio** (`chunk_bytes` = 17,920) — interim text deltas therefore arrive at ~0.56 s cadence; that's the export's fixed chunk geometry, not a tunable. The model is prompt-conditioned per language (`prompt_index` from `config.json`'s `prompt_dictionary`, 40 locales); emitted language tags (`<en-US>`) are stripped by the session.

Integration choice: the Pipecat `STTService` can either speak the WS protocol to `ws_engine.py` as a separate process, or import `OnnxStreamingSession` in-process — prefer in-process (one fewer service) unless GPU process isolation proves useful.

### STT choice: Nemotron over Kyutai STT (decided)

Kyutai STT is the convenient option (same family as the TTS), but Nemotron wins for a real-time "call":

| | Nemotron 3.5 ASR 0.6B (vendored) | Kyutai STT 2.6b-en |
|---|---|---|
| Params / VRAM | 0.6B / ~3.4 GB | 2.6B / ~6 GB |
| Latency to text | **~80–560 ms** (configurable; our export = 560 ms) | **~2.5 s** design delay |
| WER (English) | ~7.2% | ~6.4% (slightly better) |
| Languages | 40 locales | English |

The ~2.5 s Kyutai STT delay would gut the conversational feel; Nemotron's lower latency + lower VRAM + multilingual coverage outweigh Kyutai's ~0.8-point WER edge. Kyutai stays for **TTS**, where its streaming quality is the draw. STT is behind the Pipecat service interface, so revisiting later is a ~150-line swap, not an architecture change.
