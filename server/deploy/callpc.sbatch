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
# STATUS: the two vLLM services are fully wired and runnable now. moshi-tts /
# asr / gateway are scaffolded as ready-to-fill blueprints (the services aren't
# built yet) — enable each by flipping ENABLE_* below once its code lands. The
# orchestration framework (health gates, PID tracking, cleanup trap, tunnel
# banner) is complete and works with whatever subset is enabled.

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
# + KV went negative -> needs 0.50 + shorter context. Qwen3.5-9B is a HYBRID GDN
# (mamba) + vision model: FP8 weights ~10.8 GB plus a mamba-state cache AND a
# vision-encoder cache, so at 0.26 KV went negative too -> needs 0.32 + 8k ctx.
# Budget: uitars 0.50 + brain 0.32 = 0.82 (~37 GB); trim both before enabling
# TTS(5)+ASR(3.4) for the full stack.
UITARS_UTIL=${UITARS_UTIL:-0.50};  UITARS_MAXLEN=${UITARS_MAXLEN:-16384}
BRAIN_UTIL=${BRAIN_UTIL:-0.32};    BRAIN_MAXLEN=${BRAIN_MAXLEN:-8192}

# Flip to 1 as each service gets built (see the blueprint blocks below).
ENABLE_TTS=${ENABLE_TTS:-0}
ENABLE_ASR=${ENABLE_ASR:-0}
ENABLE_GATEWAY=${ENABLE_GATEWAY:-0}

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
echo; echo "=== [1/5] vllm-uitars ($UITARS_MODEL) on :$UITARS_PORT ==="
apptainer run --nv --bind "$CALLPC_BIND" "$VLLM_SIF" \
    "$UITARS_MODEL" --served-model-name ui-tars \
    --port "$UITARS_PORT" --gpu-memory-utilization "$UITARS_UTIL" \
    --max-model-len "$UITARS_MAXLEN" --enforce-eager \
    > "${LOG}_vllm-uitars.log" 2>&1 &
BG_PIDS+=($!)
wait_for_http "http://127.0.0.1:$UITARS_PORT/health" vllm-uitars "${BG_PIDS[-1]}" 900 \
    || { echo "vllm-uitars failed — tail ${LOG}_vllm-uitars.log"; exit 1; }

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
# BLUEPRINT — needs the streaming TTS server (Phase 1, server/gateway/tts_*.py).
# When built, flip ENABLE_TTS=1. Runs in the moshi image; binds the repo so it
# can import the server module.
if [ "$ENABLE_TTS" = 1 ]; then
    echo; echo "=== [3/5] moshi-tts on :$TTS_PORT ==="
    apptainer run --nv --bind "$CALLPC_BIND" "$MOSHI_SIF" \
        "$SERVER_DIR/gateway/tts_server.py" --port "$TTS_PORT" \
        > "${LOG}_moshi-tts.log" 2>&1 &
    BG_PIDS+=($!)
    wait_for_http "http://127.0.0.1:$TTS_PORT/health" moshi-tts "${BG_PIDS[-1]}" 600 \
        || { echo "moshi-tts failed"; exit 1; }
else
    echo; echo "=== [3/5] moshi-tts: DISABLED (set ENABLE_TTS=1 once tts_server.py exists) ==="
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
# BLUEPRINT — the Pipecat FastAPI gateway (server/gateway/app.py): /api/offer
# signaling, static call UI, the voice pipeline (Nemotron STT + brain + Kyutai
# TTS). Needs a gateway image (CUDA-12 base, onnxruntime-gpu + pipecat baked in).
# Flip ENABLE_GATEWAY=1 when it exists.
if [ "$ENABLE_GATEWAY" = 1 ]; then
    echo; echo "=== [5/5] gateway on :$GATEWAY_PORT ==="
    # apptainer run --nv --bind "$CALLPC_BIND" "$GATEWAY_SIF" \
    #     "$SERVER_DIR/gateway/app.py" --port "$GATEWAY_PORT" \
    #     > "${LOG}_gateway.log" 2>&1 &
    # BG_PIDS+=($!)
    # wait_for_http "http://127.0.0.1:$GATEWAY_PORT/" gateway "${BG_PIDS[-1]}" 300 || exit 1
    echo "(gateway image not built yet — see blueprint)"
else
    echo; echo "=== [5/5] gateway: DISABLED (set ENABLE_GATEWAY=1 once app.py + image exist) ==="
fi

# ============================ up — banner + wait ===========================
NODE=$(hostname)
LOGIN_HOST=${LOGIN_HOST:-<cluster-login-host>}
echo
echo "================== callpc UP =================="
echo "compute node : $NODE"
echo "vLLM (in-job): ui-tars http://127.0.0.1:$UITARS_PORT   brain http://127.0.0.1:$BRAIN_PORT"
echo "gateway port : $GATEWAY_PORT (enable once built)"
echo
echo "Expose the gateway from your laptop:"
echo "  ssh -f -N -L $GATEWAY_PORT:$NODE:$GATEWAY_PORT \$USER@$LOGIN_HOST"
echo "  ngrok http $GATEWAY_PORT"
echo
echo "Verify services with the preflight, pointed at this job:"
echo "  UITARS_URL=http://127.0.0.1:$UITARS_PORT BRAIN_URL=http://127.0.0.1:$BRAIN_PORT \\"
echo "    python3 preflight.py vllm vram"
echo "==============================================="
echo
echo "[main] services up; holding until walltime or service exit (Ctrl-C / scancel to stop)"
wait
