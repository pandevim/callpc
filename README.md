# callpc — call your computer and tell it what to do

**Phone your own computer like a person, talk to it, and watch it do things** —
open apps, browse, type, click — narrating progress back to you over the call.
The phone side is just a web page styled like a phone call; all the intelligence
runs on a server; the only thing installed on the controlled machine is a tiny
(~4.5 MB) agent that injects input and returns screenshots.

> Status: **working end-to-end** — call from a phone → say *"open the calculator"*
> → it opens on the real machine and tells you it's done. Built as a research
> prototype on a university HPC (single L40S GPU). See [PLAN.md](PLAN.md) for the
> full design, decisions, and phase-by-phase build log.

---

## How it works

```
 Phone / any browser            Server (1 GPU)                     Your computer
┌──────────────────┐  WebRTC  ┌────────────────────────────┐  WSS  ┌──────────────────┐
│ call web page    │◀────────▶│ Voice gateway (Pipecat)    │◀─────▶│ callpc-executor  │
│ (looks like a    │ (audio)  │  STT  Nemotron ONNX        │ (out- │  screenshots     │
│  phone call,     │          │  Brain Qwen3.5-9B (vLLM)   │ bound)│   (xcap/portal)  │
│  push-to-talk)   │          │  TTS  Kyutai 1.6B          │       │  input injection │
└──────────────────┘          │  Tasks UI-TARS-1.5 (vLLM)  │       │   (enigo/uinput) │
                              └────────────────────────────┘       └──────────────────┘
```

1. **You speak** → browser sends mic audio over WebRTC → **Nemotron STT** transcribes.
2. The **brain** (Qwen3.5-9B) replies conversationally, and when you ask for
   something to be done it calls the `computer_task` tool.
3. The **task runner** loops: screenshot from your computer → **UI-TARS** vision
   model picks the next action (click/type/scroll…) → the **executor** performs it
   → repeat until done. Each action is **spoken back** to you via **Kyutai TTS**.
4. Your computer runs the **executor** — a dumb I/O stub with zero smarts: it
   receives an action, injects it, and returns the next screenshot. It connects
   **outbound only**, so nothing is exposed on the controlled machine.

The voice loop has real **barge-in** (interrupt the bot mid-sentence) and the call
page is **push-to-talk** by default. A **shared passcode** gates every action so
not just anyone with the URL can call or drive a paired computer.

## Repo layout

```
callpc/
  server/
    gateway/        FastAPI + Pipecat voice pipeline (STT→brain→TTS), tool
                    handlers (brain.py), Kyutai TTS server + client, control hub
    tasks/          UI-TARS task runner + OpenAI-compatible client (coord mapping)
    control/        executor control hub (pairing, device tokens, action relay)
    asr/            vendored Nemotron streaming ONNX engine (STT)
    deploy/         SLURM + Apptainer: build_images / prefetch_weights / callpc.sbatch
    preflight.py    Phase-0 health checks (GPU, deps, models, STUN, …)
  executor/         Rust agent installed on the controlled computer (the only install)
    src/{main,screen,input,uinput}.rs
  webapp/           call UI (index/style/app.js) + connect.html + debug.html
  docs/             PROTOCOL.md (control channel), ASR_PROTOCOL.md
  PLAN.md           full design + build log
```

## Models (all self-hosted on one 48 GB L40S)

| Role            | Model                       | Notes |
|-----------------|-----------------------------|-------|
| Speech-to-text  | Nemotron 3.5 ASR 0.6B (ONNX)| streaming, in-process, ~3.4 GB |
| Brain / dialog  | Qwen3.5-9B (FP8, vLLM)      | tool-calling; `qwen3_coder` parser |
| GUI agent       | UI-TARS-1.5-7B (vLLM)       | screenshot → action; abs-pixel coords |
| Text-to-speech  | Kyutai TTS 1.6B (moshi)     | streaming 24 kHz, barge-in |

Co-resident at ~42/45 GB. Brain alternatives (ternary/1-bit) were evaluated and
rejected for now — see PLAN.md "Top risks" / brain notes.

---

## Running it

The server runs on a SLURM cluster as one ~24 h job (Apptainer containers). The
controlled computer runs the executor. A phone calls in over a public HTTPS URL.

### 0. One-time setup (login node)

```bash
git clone https://github.com/pandevim/callpc && cd callpc/server
mkdir -p logs

# intra-cluster ssh (so the job can reverse-tunnel to the login node)
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519        # if you don't have one
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys

# ngrok for a public URL (free static domain from dashboard.ngrok.com)
mkdir -p ~/.local/bin && cd ~/.local/bin
curl -sLO https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar xzf ngrok-v3-stable-linux-amd64.tgz && cd -

# build the container images + prefetch model weights (each is an sbatch)
sbatch deploy/build_images.sbatch        # vllm.sif, moshi.sif, gateway.sif
sbatch deploy/prefetch_weights.sbatch    # UI-TARS, Qwen, Kyutai TTS + voices

# put your session env in one file
cat > ~/.callpc.env <<'ENV'
export ACCESS_CODE=<choose-a-passcode>
export NGROK_AUTHTOKEN=<your-token>
export NGROK_DOMAIN=<your>.ngrok-free.dev      # bare hostname
export TURN_URL="turn:global.relay.metered.ca:80,turn:global.relay.metered.ca:443,turns:global.relay.metered.ca:443?transport=tcp"
export TURN_USER=<turn-user>
export TURN_PASS=<turn-pass>
export GATEWAY_PORT=8723
ENV
chmod 600 ~/.callpc.env
```

### 1. Each session — bring the server up

```bash
cd ~/projects/callpc/server && git pull
source ~/.callpc.env && sbatch deploy/callpc.sbatch
tail -f logs/callpc_*.out          # wait for "callpc UP"; it prints the public URL
```

The banner prints `PHONE / ANY DEVICE: https://<your>.ngrok-free.dev` — that's the
link you share. (For laptop dev access the job also reverse-forwards the port to the
login node, so `ssh -f -N -L 8723:localhost:8723 <login>` + `http://localhost:8723`
works and never changes between jobs.)

### 2. Connect the computer you want to control

Open `https://<your-url>` **on the computer you want to control**. The page is
device-aware: with nothing connected yet it shows only the download + pairing
instructions (no call button). Enter the passcode, grab the pairing code, and run
the executor on that machine:

```bash
# prebuilt binary (Releases) — or build: cargo build --release --manifest-path executor/Cargo.toml
./callpc-executor --url wss://<your-url>/control/ws --code <PAIRING_CODE>
```

It pairs once, stores a token, and reconnects automatically afterwards. The page
then flips to **"✓ This computer is connected"** on its own — it knows because the
executor runs a loopback beacon (`127.0.0.1:8724`) the page quietly probes. Open the
same URL on your phone and you'll instead see **"📞 \<name\> connected — call it"**,
plus a *"Set up a different computer →"* link if you want to pair another machine.

- **Windows / macOS / Linux-X11:** download-and-run, no fuss (unsigned-binary
  warning on Win/Mac for now).
- **Linux Wayland (GNOME/KDE):** screenshots use the desktop portal (approve the
  one-time prompt); input uses a `uinput` virtual device (be in the `input` group).

### 3. Call in

Open `https://<your-url>` on your phone (keep it on speaker), enter the passcode,
tap the green button, **hold to talk**, and say something — *"open Firefox and
search for trains to Boston."* Watch it happen; hear it narrate.

### Debugging

- `https://<your-url>/debug.html` — type a task, watch the screenshot/Thought/Action
  trail step by step; "Calibrate" measures click accuracy.
- `python preflight.py` (inside the job) — per-service health checks.
- Per-task screenshots + transcript are written under `server/.state/tasks/`.

---

## Security note

This is a **research prototype**, not hardened for hostile exposure. The passcode
is a single shared secret over ngrok's TLS — it stops "anyone with the URL" but is
not multi-user auth. The executor can control your real machine; only pair machines
you own, and use the panic path (kill the executor process) to stop a session.
Per-device tokens, signing/notarization, and an always-on host are tracked in
PLAN.md Phase 4.

## License

Code: MIT (executor). Model weights carry their own licenses (UI-TARS Apache-2.0,
Kyutai TTS CC-BY-4.0, Qwen Apache-2.0). The vendored Nemotron ASR export is fetched
separately; see `server/asr/`.
