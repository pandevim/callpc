#!/usr/bin/env python3
"""Download the pre-exported Nemotron 3.5 ONNX model into models/onnx/.
Run once before using the ONNX backend:  ./.venv/bin/python download_onnx.py
"""
import os, shutil
from huggingface_hub import hf_hub_download
REPO, SUB = "altunenes/parakeet-rs", "nemotron-3.5-asr-streaming-0.6b-onnx"
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "onnx")
os.makedirs(dest, exist_ok=True)
for f in ["encoder.onnx", "encoder.onnx.data", "decoder_joint.onnx", "tokenizer.model", "config.json"]:
    p = hf_hub_download(REPO, filename=f"{SUB}/{f}")
    shutil.copy(p, os.path.join(dest, f))
    print("got", f)
print("ONNX model ready in models/onnx/")
