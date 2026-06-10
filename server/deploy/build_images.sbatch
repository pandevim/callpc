#!/bin/bash
#SBATCH --job-name=callpc-build-images
#SBATCH --partition=compute-hf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err

# One-time image build. Pulls the official vLLM image and builds the moshi TTS
# image into deploy/images/*.sif. Jobs then just `apptainer exec` them — no
# pulls/builds at runtime (concurrent pulls on a shared FS corrupt the cache).
#
# Run it on a COMPUTE node, NOT the login node (the SIF pack is CPU/IO-heavy and
# login nodes forbid it). No GPU needed — pick a CPU partition. Two ways:
#
#   mkdir -p logs                       # once: SLURM needs logs/ to exist
#   sbatch deploy/build_images.sbatch       # preferred — logs to logs/build_<jobid>.out
#       # override the partition if compute-hf is busy:
#       sbatch --partition=compute-short deploy/build_images.sbatch
#
#   # or interactively:
#   srun --partition=compute-hf --cpus-per-task=8 --mem=64G --time=00:40:00 --pty bash
#   cd ~/projects/callpc/server && ./deploy/build_images.sbatch
#
# Set HF_TOKEN first if you also want gated weights later (UI-TARS is Apache,
# Qwen/Kyutai may need it):  export HF_TOKEN=hf_... ; sbatch ...

set -euo pipefail

# Locate server/. Under `sbatch`, SLURM runs a COPY of this script from a spool
# dir, so `dirname "$0"` is useless — use SLURM_SUBMIT_DIR. Under direct/srun
# execution, fall back to the script's own location.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
    [ -d server ] && cd server      # submitted from repo root rather than server/
else
    cd "$(dirname "$0")/.."         # ./deploy/build_images.sbatch  ->  server/
fi
mkdir -p logs
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
