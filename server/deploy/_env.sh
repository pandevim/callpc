#!/bin/bash
# Shared Apptainer env for all callpc cluster jobs. Source this from each
# .sbatch (and from build_images.sbatch). It loads the apptainer module, points the
# caches at node-local SSD, and exports the image + cache paths — it does NOT
# build or pull anything (that is build_images.sbatch, a one-time compute-node step (sbatch);
# concurrent image pulls on a shared FS race and corrupt the cache).
#
#   ./deploy/build_images.sbatch        # run ONCE on a compute node first
#   sbatch deploy/preflight.sbatch  # jobs just source this file
#
# Mirrors the NeuralSINDy_v2 sbatch/_env.sh split, but the runtime is Apptainer
# (vLLM + moshi ship as containers per PLAN.md), not a venv.
#
# Edit the module name below to match `module avail apptainer` on your cluster.

set -euo pipefail

# --- Paths (server/ is the working base; deploy/ lives under it) ---
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SERVER_DIR="$(dirname "$DEPLOY_DIR")"
export IMAGES_DIR="$DEPLOY_DIR/images"
export VLLM_SIF="${VLLM_SIF:-$IMAGES_DIR/vllm.sif}"
export MOSHI_SIF="${MOSHI_SIF:-$IMAGES_DIR/moshi.sif}"
mkdir -p "$IMAGES_DIR"

# --- Apptainer module ---
module purge 2>/dev/null || true
module load apptainer 2>/dev/null || module load singularity 2>/dev/null || \
    echo "WARN: no apptainer/singularity module — relying on it being on PATH"
command -v apptainer >/dev/null 2>&1 || command -v singularity >/dev/null 2>&1 || {
    echo "ERROR: apptainer not found. Check 'module avail apptainer'." >&2
    return 1 2>/dev/null || exit 1
}

# --- Caches on node-local fast SSD (PLAN.md: never use $HOME — freezes the
#     shared FS). Inside a job we get /local/$SLURM_JOB_ID; on the login node we
#     fall back to a workspace dir. ---
if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/local/$SLURM_JOB_ID" ]; then
    SCRATCH="/local/$SLURM_JOB_ID"
else
    SCRATCH="${SCRATCH:-$SERVER_DIR/.scratch}"
fi
mkdir -p "$SCRATCH"
export APPTAINER_CACHEDIR="$SCRATCH/apptainer_cache"
export APPTAINER_TMPDIR="$SCRATCH/apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

# --- HuggingFace cache: persistent (prefetch weights here on the login node so
#     compute-node jobs start fast and survive restricted internet) ---
export HF_HOME="${HF_HOME:-$SERVER_DIR/.hf_cache}"
mkdir -p "$HF_HOME"

# Bind the HF cache into containers at the path HF libraries expect.
export APPTAINER_BIND="${APPTAINER_BIND:-}${APPTAINER_BIND:+,}$HF_HOME:/root/.cache/huggingface,$SERVER_DIR:$SERVER_DIR"

# --- GPU sanity (no-op on the login node) ---
echo "apptainer: $(apptainer --version 2>/dev/null || singularity --version)"
echo "SCRATCH=$SCRATCH  HF_HOME=$HF_HOME"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
