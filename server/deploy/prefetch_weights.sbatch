#!/bin/bash
#SBATCH --job-name=callpc-prefetch
#SBATCH --partition=compute-hf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prefetch_%j.out
#SBATCH --error=logs/prefetch_%j.err

# Stage the vLLM model weights into HF_HOME on a CPU node BEFORE running
# callpc.sbatch, so the GPU job starts fast and doesn't burn L40S time (or hit
# restricted internet) downloading. Downloads run inside the vLLM container,
# which already has huggingface_hub; they land in $SERVER_DIR/.hf_cache (bound).
#
#   export HF_TOKEN=hf_...                 # if any repo is gated
#   sbatch deploy/prefetch_weights.sbatch
#
# The ASR ONNX export is fetched separately by the preflight (asr --download);
# the Kyutai TTS weights are pulled the first time the moshi server runs.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -d server ] && cd server
mkdir -p logs
# shellcheck disable=SC1091
source deploy/_env.sh

[ -f "$VLLM_SIF" ] || { echo "ERROR: $VLLM_SIF missing — run deploy/build_images.sbatch first" >&2; exit 1; }

UITARS_MODEL=${UITARS_MODEL:-ByteDance-Seed/UI-TARS-1.5-7B}
BRAIN_MODEL=${BRAIN_MODEL:-Qwen/Qwen3.5-9B}   # no official 9B FP8 repo; vLLM quantizes at load

# Pass HF_TOKEN into the container if set (gated repos).
[ -n "${HF_TOKEN:-}" ] && export APPTAINERENV_HF_TOKEN="$HF_TOKEN"

echo "prefetching into HF_HOME=$HF_HOME"
echo "  UI-TARS: $UITARS_MODEL"
echo "  brain:   $BRAIN_MODEL"

apptainer exec --bind "$CALLPC_BIND" "$VLLM_SIF" python3 - "$UITARS_MODEL" "$BRAIN_MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
for repo in sys.argv[1:]:
    print(f"=== {repo} ===", flush=True)
    p = snapshot_download(repo)
    print(f"  -> {p}", flush=True)
print("prefetch done", flush=True)
PY

echo "DONE — weights staged. Next: sbatch deploy/callpc.sbatch"
