#!/bin/bash
# One-time login-node image build. Run ONCE before submitting any job (and
# again after changing moshi.def). Pulls the official vLLM image and builds the
# moshi TTS image into deploy/images/*.sif. Jobs then just `apptainer exec` them
# — no pulls/builds at runtime (concurrent pulls on a shared FS corrupt the cache).
#
#   ./deploy/build_images.sh
#
# Set HF_TOKEN first if you also want to prefetch gated weights (UI-TARS is
# Apache, Qwen/Kyutai may need it):  export HF_TOKEN=hf_...

set -euo pipefail
cd "$(dirname "$0")/.."          # -> server/
# shellcheck disable=SC1091
source deploy/_env.sh

echo "=== vLLM image ==="
if [ -f "$VLLM_SIF" ]; then
    echo "exists: $VLLM_SIF (delete to refresh)"
else
    # The single image used for BOTH vLLM services (UI-TARS + Qwen brain).
    apptainer pull "$VLLM_SIF" docker://vllm/vllm-openai:latest
    echo "built: $VLLM_SIF"
fi

echo "=== moshi (Kyutai TTS) image ==="
if [ -f "$MOSHI_SIF" ]; then
    echo "exists: $MOSHI_SIF (delete to refresh)"
else
    apptainer build --fakeroot "$MOSHI_SIF" deploy/moshi.def
    echo "built: $MOSHI_SIF"
fi

echo
echo "images ready in $IMAGES_DIR:"
ls -lh "$IMAGES_DIR"/*.sif 2>/dev/null || true
echo
echo "next: sbatch deploy/preflight.sbatch"
