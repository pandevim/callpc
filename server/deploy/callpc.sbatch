#!/bin/bash
#SBATCH --job-name=callpc
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/callpc_%j.out
#SBATCH --error=logs/callpc_%j.err

# The callpc launcher — ONE sbatch job that brings up all services on the single
# L40S in dependency order, health-gating each before starting the next, and
# kills them all when the job ends. vLLM claims a FRACTION of TOTAL VRAM at
# startup, so order + fixed fractions matter (PLAN.md "VRAM budget"):
#
#   vllm-uitars (0.42) -> vllm-brain (0.22) -> moshi-tts -> asr -> gateway
#
# Prereqs: deploy/build_images.sbatch (images) + deploy/prefetch_weights.sbatch
# (model weights staged in HF_HOME). Then:  sbatch deploy/callpc.sbatch
#
# STATUS: vLLMs + gateway + moshi-tts are built and wired (standalone ASR stays
# a blueprint — STT runs in-process in the gateway). Default ENABLE_* set =
# the Phase-1 voice loop: brain + tts + gateway, UI-TARS OFF for VRAM headroom
# (it's a Phase-2 service; the full 5-service stack needs the fractions trimmed
# — see the VRAM note below).

set -uo pipefail   # NOT -e: a single service failing should run cleanup, not die silently
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -d server ] && cd server
mkdir -p logs
# shellcheck disable=SC1091
source deploy/_env.sh

[ -f "$VLLM_SIF" ] || { echo "ERROR: $VLLM_SIF missing — run deploy/build_images.sbatch first" >&2; exit 1; }

# --- config (all overridable at submit: `UITARS_PORT=9000 sbatch ...`) ---
UITARS_PORT=${UITARS_PORT:-8000};  UITARS_MODEL=${UITARS_MODEL:-ByteDance-Seed/UI-TARS-1.5-7B}
BRAIN_PORT=${BRAIN_PORT:-8001};    BRAIN_MODEL=${BRAIN_MODEL:-Qwen/Qwen3.5-9B}
TTS_PORT=${TTS_PORT:-8089}
ASR_PORT=${ASR_PORT:-8765}
GATEWAY_PORT=${GATEWAY_PORT:-8080}

# VRAM fractions + context lengths (override at submit to tune the budget).
# UI-TARS weights ~15.7 GB; at 0.42 its vision-encoder cache (sized to context)
# + KV went negative -> 16k ctx needed 0.50. Qwen3.5-9B is a HYBRID GDN
# (mamba) + vision model: FP8 weights ~10.8 GB plus a mamba-state cache AND a
# vision-encoder cache, so at 0.26 KV went negative too -> needs 0.32 + 8k ctx.
# FULL-STACK budget (Phase 2+): uitars 0.46@8k (~20.7) + brain 0.30 (~13.5)
# + TTS(~5) + in-gateway ASR(~3.4) ≈ 42.6/45 GB. The 8k ctx bounds the task
# runner to ~3 screenshots of history (1280x800 ≈ 1.3k tokens each). If
# vllm-uitars fails with a negative-KV error, raise UITARS_UTIL and drop
# BRAIN_MAXLEN to 4096. Verify with `preflight.py vram`.
UITARS_UTIL=${UITARS_UTIL:-0.46};  UITARS_MAXLEN=${UITARS_MAXLEN:-8192}
BRAIN_UTIL=${BRAIN_UTIL:-0.30};    BRAIN_MAXLEN=${BRAIN_MAXLEN:-8192}

# Service toggles. Defaults = full Phase-2 stack. ASR stays 0: STT runs
# in-process inside the gateway (the standalone block below is the alternative).
# Voice-only (more headroom): ENABLE_UITARS=0 sbatch deploy/callpc.sbatch
ENABLE_UITARS=${ENABLE_UITARS:-1}
ENABLE_TTS=${ENABLE_TTS:-1}
ENABLE_ASR=${ENABLE_ASR:-0}
ENABLE_GATEWAY=${ENABLE_GATEWAY:-1}

[ -n "${HF_TOKEN:-}" ] && export APPTAINERENV_HF_TOKEN="$HF_TOKEN"

# --- service lifecycle: track child PIDs, kill them all on exit ---
BG_PIDS=()
cleanup() {
    echo
    echo "[cleanup] stopping ${#BG_PIDS[@]} service(s)..."
    for pid in "${BG_PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
    echo "[cleanup] done"
}
trap cleanup EXIT INT TERM

# wait_for_http URL NAME PID [TIMEOUT_S] — poll an HTTP health endpoint, but bail
# early if the backing process dies (so we surface its log instead of hanging).
wait_for_http() {
    local url=$1 name=$2 pid=$3 timeout=${4:-900} start=$SECONDS
    echo "[wait] $name -> $url (timeout ${timeout}s)"
    until curl -sf -o /dev/null "$url"; do
        kill -0 "$pid" 2>/dev/null || { echo "[FAIL] $name process exited early — see its log"; return 1; }
        (( SECONDS - start > timeout )) && { echo "[FAIL] $name timed out after ${timeout}s"; return 1; }
        sleep 5
    done
    echo "[ready] $name in $(( SECONDS - start ))s"
}

# wait_for_port HOST PORT NAME PID [TIMEOUT_S] — same, for plain TCP services (ASR ws).
wait_for_port() {
    local host=$1 port=$2 name=$3 pid=$4 timeout=${5:-300} start=$SECONDS
    echo "[wait] $name -> $host:$port (timeout ${timeout}s)"
    until (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; do
        kill -0 "$pid" 2>/dev/null || { echo "[FAIL] $name process exited early — see its log"; return 1; }
        (( SECONDS - start > timeout )) && { echo "[FAIL] $name timed out after ${timeout}s"; return 1; }
        sleep 3
    done
    exec 3<&- 3>&- 2>/dev/null || true
    echo "[ready] $name in $(( SECONDS - start ))s"
}

LOG="logs/${SLURM_JOB_ID:-local}"

echo "================ callpc launcher ================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ============================ 1. UI-TARS (vLLM) ============================
# Model is a POSITIONAL arg to `vllm serve` (the image entrypoint); --model is
# deprecated. The task runner will later need a multi-image limit, but its JSON
# value gets quote-mangled through `apptainer run` — re-add it then with dot
# syntax instead of JSON:  --limit-mm-per-prompt.image 5
if [ "$ENABLE_UITARS" = 1 ]; then
    echo; echo "=== [1/5] vllm-uitars ($UITARS_MODEL) on :$UITARS_PORT ==="
    # --limit-mm-per-prompt.image: dot syntax on purpose — the JSON form gets
    # quote-mangled through apptainer run. 5 = the task runner's max history.
    apptainer run --nv --bind "$CALLPC_BIND" "$VLLM_SIF" \
        "$UITARS_MODEL" --served-model-name ui-tars \
        --port "$UITARS_PORT" --gpu-memory-utilization "$UITARS_UTIL" \
        --max-model-len "$UITARS_MAXLEN" --enforce-eager \
        --limit-mm-per-prompt.image 5 \
        > "${LOG}_vllm-uitars.log" 2>&1 &
    BG_PIDS+=($!)
    wait_for_http "http://127.0.0.1:$UITARS_PORT/health" vllm-uitars "${BG_PIDS[-1]}" 900 \
        || { echo "vllm-uitars failed — tail ${LOG}_vllm-uitars.log"; exit 1; }
else
    echo; echo "=== [1/5] vllm-uitars: DISABLED (Phase-2 service; set ENABLE_UITARS=1 + trim fractions) ==="
fi

# ============================ 2. Qwen brain (vLLM) =========================
# No official 9B FP8 repo, so quantize the BF16 weights to FP8 at load (Ada
# supports FP8) — this is the documented single-L40S pattern for Qwen3.5-9B.
# Util bumped 0.22 -> 0.26 to cover the higher load-time peak of dynamic quant
# (BF16 read -> FP8). If it OOMs, drop to BRAIN_MODEL=Qwen/Qwen3.5-4B.
# NB: Qwen3.5's hybrid GDN arch needs vLLM >=0.17 — if the brain errors with an
# unknown-architecture message, the vLLM image is too old; re-pull :latest.
echo; echo "=== [2/5] vllm-brain ($BRAIN_MODEL, fp8) on :$BRAIN_PORT ==="
apptainer run --nv --bind "$CALLPC_BIND" "$VLLM_SIF" \
    "$BRAIN_MODEL" --served-model-name brain --quantization fp8 \
    --port "$BRAIN_PORT" --gpu-memory-utilization "$BRAIN_UTIL" \
    --max-model-len "$BRAIN_MAXLEN" --enforce-eager \
    > "${LOG}_vllm-brain.log" 2>&1 &
BG_PIDS+=($!)
wait_for_http "http://127.0.0.1:$BRAIN_PORT/health" vllm-brain "${BG_PIDS[-1]}" 900 \
    || { echo "vllm-brain failed — tail ${LOG}_vllm-brain.log"; exit 1; }

# ============================ 3. moshi TTS =================================
# Streaming Kyutai TTS (server/gateway/tts_server.py) in the moshi image.
# /health answers only after model load + warmup synthesis, so the gate below
# means "ready for a real call" (first load also downloads the voice -> the
# 900 s timeout). TTS_VOICE overrides the voice (kyutai/tts-voices names).
if [ "$ENABLE_TTS" = 1 ]; then
    [ -f "$MOSHI_SIF" ] || { echo "ERROR: $MOSHI_SIF missing — build it (deploy/build_images.sbatch)"; exit 1; }
    echo; echo "=== [3/5] moshi-tts on :$TTS_PORT ==="
    [ -n "${TTS_VOICE:-}" ] && export APPTAINERENV_TTS_VOICE="$TTS_VOICE"
    apptainer run --nv --bind "$CALLPC_BIND" "$MOSHI_SIF" \
        "$SERVER_DIR/gateway/tts_server.py" --port "$TTS_PORT" \
        > "${LOG}_moshi-tts.log" 2>&1 &
    BG_PIDS+=($!)
    wait_for_http "http://127.0.0.1:$TTS_PORT/health" moshi-tts "${BG_PIDS[-1]}" 900 \
        || { echo "moshi-tts failed — tail ${LOG}_moshi-tts.log"; exit 1; }
else
    echo; echo "=== [3/5] moshi-tts: DISABLED (set ENABLE_TTS=1; gateway runs text-only) ==="
fi

# ============================ 4. ASR (ws_engine) ===========================
# BLUEPRINT — server/asr/ws_engine.py needs onnxruntime-gpu + the CUDA-12 libs
# (same setup the preflight uses). In prod the ASR likely runs IN-PROCESS inside
# the gateway (PLAN.md) rather than as its own service; this standalone block is
# the alternative. Flip ENABLE_ASR=1 to use it.
if [ "$ENABLE_ASR" = 1 ]; then
    echo; echo "=== [4/5] asr ws on :$ASR_PORT ==="
    apptainer exec --nv --writable-tmpfs --bind "$CALLPC_BIND" "$VLLM_SIF" bash -c '
        python3 -m pip install --quiet onnxruntime-gpu sentencepiece soundfile \
            nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cudnn-cu12 \
            nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusparse-cu12 \
            nvidia-cuda-nvrtc-cu12 nvidia-nvjitlink-cu12
        export LD_LIBRARY_PATH="$(find "$HOME/.local" /usr /opt -path "*/nvidia/*/lib" -type d 2>/dev/null | sort -u | tr "\n" ":")/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
        export ORT_INTRA_OP_THREADS=$(nproc)
        cd '"$SERVER_DIR"'/asr
        ASR_HOST=0.0.0.0 ASR_PORT='"$ASR_PORT"' python3 ws_engine.py
    ' > "${LOG}_asr.log" 2>&1 &
    BG_PIDS+=($!)
    wait_for_port 127.0.0.1 "$ASR_PORT" asr "${BG_PIDS[-1]}" 600 \
        || { echo "asr failed — tail ${LOG}_asr.log"; exit 1; }
else
    echo; echo "=== [4/5] asr: DISABLED (set ENABLE_ASR=1; or runs in-process in the gateway) ==="
fi

# ============================ 5. gateway ===================================
# The Pipecat FastAPI gateway (server/gateway/app.py): /api/offer signaling +
# the voice pipeline. Phase 1a = mic -> Nemotron STT -> log transcripts. Runs in
# the CUDA-12 gateway image (onnxruntime-gpu CUDA EP works there with no lib hack).
# Port + ORT thread count passed via env (APPTAINERENV_* -> container env).
if [ "$ENABLE_GATEWAY" = 1 ]; then
    [ -f "$GATEWAY_SIF" ] || { echo "ERROR: $GATEWAY_SIF missing — build it (deploy/build_images.sbatch)"; exit 1; }
    echo; echo "=== [5/5] gateway on :$GATEWAY_PORT (brain @ :$BRAIN_PORT) ==="
    # WebRTC media needs TURN across the cluster NAT — pass creds through if set:
    #   TURN_URL=... TURN_USER=... TURN_PASS=... ENABLE_GATEWAY=1 sbatch deploy/callpc.sbatch
    [ -n "${TURN_URL:-}" ]  && export APPTAINERENV_TURN_URL="$TURN_URL"
    [ -n "${TURN_USER:-}" ] && export APPTAINERENV_TURN_USER="$TURN_USER"
    [ -n "${TURN_PASS:-}" ] && export APPTAINERENV_TURN_PASS="$TURN_PASS"
    [ -z "${TURN_URL:-}" ] && echo "WARN: TURN_URL unset — WebRTC media will likely fail over the cluster NAT"
    # TTS: point the gateway at the moshi server, or run text-only when off.
    if [ "$ENABLE_TTS" = 1 ]; then
        APPTAINERENV_TTS_WS_URL="ws://127.0.0.1:$TTS_PORT"
    else
        APPTAINERENV_TTS_WS_URL="off"
    fi
    export APPTAINERENV_TTS_WS_URL
    APPTAINERENV_GATEWAY_PORT="$GATEWAY_PORT" \
    APPTAINERENV_ORT_INTRA_OP_THREADS="$(nproc)" \
    APPTAINERENV_BRAIN_BASE_URL="http://127.0.0.1:$BRAIN_PORT/v1" \
    APPTAINERENV_BRAIN_MODEL="brain" \
    APPTAINERENV_UITARS_BASE_URL="http://127.0.0.1:$UITARS_PORT/v1" \
    apptainer run --nv --bind "$CALLPC_BIND" "$GATEWAY_SIF" \
        "$SERVER_DIR/gateway/app.py" \
        > "${LOG}_gateway.log" 2>&1 &
    BG_PIDS+=($!)
    wait_for_http "http://127.0.0.1:$GATEWAY_PORT/health" gateway "${BG_PIDS[-1]}" 300 \
        || { echo "gateway failed — tail ${LOG}_gateway.log"; exit 1; }
else
    echo; echo "=== [5/5] gateway: DISABLED (set ENABLE_GATEWAY=1) ==="
fi

# ===================== reachability (no per-node tunnels) ==================
NODE=$(hostname)
# The submit host IS the login node — no placeholder to edit.
LOGIN_HOST=${LOGIN_HOST:-${SLURM_SUBMIT_HOST:-}}

# (a) Reverse-forward the gateway port to the LOGIN node. The laptop then
# always tunnels to localhost on the login node — the laptop command never
# changes with the compute node again. Needs intra-cluster ssh (shared $HOME:
# one-time `ssh-keygen -t ed25519 -N "" && cat ~/.ssh/id_ed25519.pub >>
# ~/.ssh/authorized_keys`). A stale listener from a crashed job is cleared
# best-effort first; ours dies with the job via the cleanup trap.
SSH_TUNNEL_UP=0
if [ "${ENABLE_SSH_TUNNEL:-1}" = 1 ] && [ -n "$LOGIN_HOST" ] && [ "$ENABLE_GATEWAY" = 1 ]; then
    echo; echo "=== reverse tunnel -> $LOGIN_HOST:$GATEWAY_PORT ==="
    ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes "$LOGIN_HOST" \
        "fuser -k -n tcp $GATEWAY_PORT 2>/dev/null; true" 2>/dev/null || true
    ssh -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
        -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
        -N -R "127.0.0.1:$GATEWAY_PORT:127.0.0.1:$GATEWAY_PORT" "$LOGIN_HOST" \
        > "${LOG}_sshtunnel.log" 2>&1 &
    BG_PIDS+=($!)
    sleep 2
    if kill -0 "${BG_PIDS[-1]}" 2>/dev/null; then
        SSH_TUNNEL_UP=1
        echo "[tunnel] up: $LOGIN_HOST:$GATEWAY_PORT -> $NODE:$GATEWAY_PORT"
    else
        echo "WARN: reverse tunnel failed (intra-cluster ssh keys set up? see ${LOG}_sshtunnel.log)"
    fi
fi

# (b) Optional ngrok straight from the job: a public HTTPS URL with NO laptop
# in the loop (phone-callable). Needs the ngrok binary (~/.local/bin) and
# NGROK_AUTHTOKEN; NGROK_DOMAIN (a free reserved static domain) makes the URL
# identical across jobs. Skipped silently when not configured.
NGROK_URL=""
if [ -n "${NGROK_AUTHTOKEN:-}" ] && command -v ngrok >/dev/null 2>&1 && [ "$ENABLE_GATEWAY" = 1 ]; then
    echo; echo "=== ngrok ==="
    # --domain wants a BARE hostname; tolerate a pasted URL (scheme/trailing
    # slash => ERR_NGROK_9038 "Invalid URL").
    NGROK_DOMAIN="${NGROK_DOMAIN:-}"
    NGROK_DOMAIN="${NGROK_DOMAIN#https://}"; NGROK_DOMAIN="${NGROK_DOMAIN#http://}"
    NGROK_DOMAIN="${NGROK_DOMAIN%%/*}"
    NGROK_ARGS=(http "$GATEWAY_PORT" --log stdout --log-format json)
    [ -n "$NGROK_DOMAIN" ] && NGROK_ARGS+=(--domain "$NGROK_DOMAIN")
    NGROK_AUTHTOKEN="$NGROK_AUTHTOKEN" ngrok "${NGROK_ARGS[@]}" > "${LOG}_ngrok.log" 2>&1 &
    BG_PIDS+=($!)
    for _ in $(seq 1 15); do
        # `|| true`: grep finds nothing until ngrok's API is up — must not
        # trip pipefail/errexit (this exact line once killed the whole job).
        NGROK_URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null \
            | grep -o '"public_url":"https:[^"]*"' | head -1 | cut -d'"' -f4 || true)
        [ -n "$NGROK_URL" ] && break
        sleep 2
    done
    [ -n "$NGROK_URL" ] && echo "[ngrok] public URL: $NGROK_URL" \
        || echo "WARN: ngrok started but no URL yet — tail ${LOG}_ngrok.log"
fi

# ============================ up — banner + wait ===========================
echo
echo "================== callpc UP =================="
echo "compute node : $NODE"
echo "vLLM (in-job): $([ "$ENABLE_UITARS" = 1 ] && echo "ui-tars http://127.0.0.1:$UITARS_PORT   ")brain http://127.0.0.1:$BRAIN_PORT"
echo "moshi-tts    : $([ "$ENABLE_TTS" = 1 ] && echo "ws://127.0.0.1:$TTS_PORT" || echo "disabled")"
echo "gateway port : $GATEWAY_PORT"
echo
if [ -n "$NGROK_URL" ]; then
    echo "PHONE / ANY DEVICE:  $NGROK_URL"
fi
if [ "$SSH_TUNNEL_UP" = 1 ]; then
    echo "LAPTOP (same command every job — target is the login node, not $NODE):"
    echo "  ssh -f -N -L $GATEWAY_PORT:localhost:$GATEWAY_PORT \$USER@$LOGIN_HOST"
    echo "  then open http://localhost:$GATEWAY_PORT"
    echo "  (or add to ~/.ssh/config under the login host: LocalForward $GATEWAY_PORT localhost:$GATEWAY_PORT)"
else
    echo "Expose the gateway from your laptop (per-node fallback):"
    echo "  ssh -f -N -L $GATEWAY_PORT:$NODE:$GATEWAY_PORT \$USER@$LOGIN_HOST"
fi
echo
echo "Verify services with the preflight, pointed at this job:"
echo "  UITARS_URL=http://127.0.0.1:$UITARS_PORT BRAIN_URL=http://127.0.0.1:$BRAIN_PORT \\"
echo "    python3 preflight.py vllm vram"
echo "==============================================="
echo
echo "[main] services up; holding until walltime or service exit (Ctrl-C / scancel to stop)"
wait
