#!/bin/bash
#SBATCH --job-name=callpc-preflight
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Single L40S (48 GB) — the only GPU we have. The full callpc stack
# (UI-TARS ~20 GB + Qwen3.5-9B-FP8 ~10 GB + Kyutai ~5 GB + Nemotron ASR
# ~3.4 GB ≈ 38 GB) must fit in this one card. This preflight only loads the
# ASR engine; the `vram` check reports the budget vs. the live card so we
# know the headroom before wiring the heavy services.
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/preflight_%j.out
#SBATCH --error=logs/preflight_%j.err

set -euo pipefail
mkdir -p logs

# --- Secrets (optional for preflight) ---
# The preflight only pulls the public Nemotron ONNX export
# (altunenes/parakeet-rs — not gated), so HF_TOKEN is NOT required here.
# It WILL be needed by the real launcher (UI-TARS / Qwen / Kyutai). Export it
# now if you want to validate gated access too:  export HF_TOKEN=hf_...
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN present — gated repos reachable"
else
    echo "HF_TOKEN not set — fine for preflight (ASR export is public)"
fi

# --- Python + CUDA modules (mirrors the TriMem cluster setup) ---
module purge
module load python/3.11.10
# Pick the cuda toolkit your cluster exposes (`module avail cuda`). onnxruntime-gpu
# bundles its own CUDA provider libs but still wants a matching driver; if the
# CUDA provider fails to load, the engine falls back to CPU and preflight still
# passes (a 0.6B model runs fine on CPU for a smoke test).
module load cuda12.8/toolkit/12.8.1 || echo "WARN: cuda module load failed — ASR will use CPU provider"

# Module sets PYTHONHOME, which conflicts with the venv. Drop it.
unset PYTHONHOME

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# --- Virtualenv (recreate if built with a different interpreter) ---
if [ -d "venv" ] && ! venv/bin/python -c 'import sys; assert sys.version_info[:2]==(3,11)' 2>/dev/null; then
    rm -rf venv
fi
if [ ! -d "venv" ]; then
    python3.11 -m venv venv
fi
source venv/bin/activate
pip install --quiet --upgrade pip

# --- Preflight dependencies ---
# Everything the ASR engine + harness needs. The infra probes (stun, proxy,
# vllm, tts, gateway) use only the stdlib, so this list is just the ASR stack.
pip install --quiet \
    onnxruntime-gpu \
    numpy \
    sentencepiece \
    soundfile \
    huggingface_hub

# Keep the HF cache on the workspace, not $HOME (HPC quota).
export HF_HOME=${HF_HOME:-$PWD/.hf_cache}
mkdir -p "$HF_HOME"

# --- Run the preflight ---
# First run: validate the environment, import groups, load + benchmark the ASR
# engine (downloads the ~2.5 GB ONNX export once into asr/models/onnx), report
# the VRAM budget vs. the live card, and confirm WebRTC UDP egress (STUN).
#
# The vllm/tts/asr_ws/gateway probes SKIP here because those services aren't up
# yet — they become meaningful once callpc.sbatch launches them. To also test
# the reverse-proxy chain, export PUBLIC_URL (your ngrok HTTPS URL) and add
# `proxy` to the check list below.
python preflight.py env imports asr vram stun --download

echo "DONE — preflight complete"
