#!/bin/bash
#SBATCH --job-name=callpc-preflight
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/preflight_%j.out
#SBATCH --error=logs/preflight_%j.err

# Phase-0 preflight, Apptainer edition. Validates the things that will hurt us
# before we wire the full stack: that the apptainer module + GPU passthrough
# (--nv) work, that the real vLLM image runs and sees the L40S, that the moshi
# image imports, that the vendored ASR engine loads + runs (and its RAM/RTF),
# the VRAM budget vs. the live 48 GB card, and WebRTC UDP egress (STUN).
#
# Requires: ./deploy/build_images.sbatch already run (sbatch deploy/build_images.sbatch).
# Thin by design — all shared setup is in deploy/_env.sh (NeuralSINDy_v2 style).

cd "${SLURM_SUBMIT_DIR:-$PWD}"
# the submit dir may be repo root or server/; normalise to server/.
[ -d server ] && cd server
mkdir -p logs
# shellcheck disable=SC1091
source deploy/_env.sh

[ -f "$VLLM_SIF" ] || { echo "ERROR: $VLLM_SIF missing — run ./deploy/build_images.sbatch first" >&2; exit 1; }

echo
echo "=================== 1. apptainer + GPU passthrough (--nv) ==================="
# If the container can't see the GPU, nothing downstream works. Fail loud here.
apptainer exec --nv "$VLLM_SIF" nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
apptainer exec --nv "$VLLM_SIF" python -c \
    "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

echo
echo "=================== 2. moshi image import ==================="
if [ -f "$MOSHI_SIF" ]; then
    apptainer exec --nv "$MOSHI_SIF" python -c "import moshi, msgpack; print('moshi import OK')" \
        || echo "WARN: moshi image import failed"
else
    echo "SKIP: $MOSHI_SIF not built yet"
fi

echo
echo "=================== 3. preflight harness (inside vLLM image) ==================="
# Run the harness in the real vLLM container. The image has python3.12 + numpy +
# torch; the ASR engine additionally needs onnxruntime/sentencepiece/soundfile,
# installed ephemerally into a writable tmpfs overlay (preflight only — the
# production gateway image bakes these in). The vllm/tts/asr_ws/gateway probes
# SKIP because those services aren't up yet; that's expected on a first run.
#
# To also test the reverse-proxy chain, `export PUBLIC_URL=https://<id>.ngrok.app`
# before sbatch and add `proxy` to the check list.
apptainer exec --nv --writable-tmpfs "$VLLM_SIF" bash -c '
    set -e
    pip install --quiet onnxruntime-gpu sentencepiece soundfile huggingface_hub
    cd '"$SERVER_DIR"'
    python preflight.py env imports asr vram stun --download
'

echo
echo "DONE — preflight complete"
