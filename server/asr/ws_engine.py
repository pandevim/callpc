#!/usr/bin/env python3
"""Platform-agnostic streaming ASR engine — WebSocket API.

This is THE boundary between the speech engine and any platform frontend. It
has no Linux/GNOME/ydotool dependencies: it accepts audio over a WebSocket and
returns recognized text. A Linux tray app, a browser extension (getUserMedia +
WebSocket), or a Windows app can all drive it identically. See PROTOCOL.md.

Protocol (ws://127.0.0.1:PORT):
    client -> {"type":"start","lang":"en-US"}
    server -> {"type":"ready","chunk_bytes":5120,"sample_rate":16000}
            or {"type":"error","error":"busy"}
    client -> <binary frame: int16-LE mono PCM, chunk_bytes per frame> ...
    server -> {"type":"text","text":"<incremental text>"}   (per chunk)
    client -> {"type":"stop"}        (end utterance; connection may reuse)

Single active streamer at a time (one GPU); others get {"error":"busy"}.
"""

import asyncio
import json
import os
import sys

import websockets

# Engine internals live at the repo root; import them LAZILY (per backend) so the
# onnx backend never pulls in PyTorch/NeMo (~1 GB) — that's the whole RAM point.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HOST = os.environ.get("ASR_HOST", "127.0.0.1")
PORT = int(os.environ.get("ASR_PORT", "8765"))
DEFAULT_LANG = os.environ.get("DICTATE_LANG", "en-US")


def _log(m):
    print(m, file=sys.stderr, flush=True)


def _build_session(backend):
    """Construct the streaming session for the chosen backend. Both expose the
    same interface (set_lang/reset/feed/chunk_bytes/sample_rate)."""
    if backend == "onnx":
        import onnx_engine  # onnxruntime + numpy, ~0.9 GB RAM, ~3.3 GB VRAM
        if not os.path.exists(os.path.join(onnx_engine.MODEL_DIR, "encoder.onnx")):
            _log("[engine] onnx model missing (run download_onnx.py); falling back to pytorch")
        else:
            asr = onnx_engine.OnnxASR(lang=DEFAULT_LANG)
            return onnx_engine.OnnxStreamingSession(asr)
    # default: PyTorch fp16 (~1.7 GB RAM, ~1.37 GB VRAM) — imports torch+NeMo here
    import asr_core
    import stream_core
    model = asr_core.load_model(log=_log)
    stream_core.configure_for_streaming(model)
    return stream_core.StreamingSession(model, lang=DEFAULT_LANG)


async def main() -> int:
    loop = asyncio.get_running_loop()
    backend = os.environ.get("ASR_BACKEND", "onnx").lower()
    _log(f"[engine] backend={backend}")
    # Heavy, blocking load on a worker thread so the event loop starts cleanly.
    session = await loop.run_in_executor(None, lambda: _build_session(backend))
    busy = asyncio.Lock()
    _log(f"[engine] ready (chunk_bytes={session.chunk_bytes})")

    async def handler(ws):
        try:
            first = await ws.recv()
            req = json.loads(first) if isinstance(first, str) else {}
        except Exception:
            return
        if req.get("type") != "start":
            await ws.send(json.dumps({"type": "error", "error": "expected start"}))
            return
        if busy.locked():
            await ws.send(json.dumps({"type": "error", "error": "busy"}))
            return

        async with busy:
            lang = req.get("lang", DEFAULT_LANG)
            await loop.run_in_executor(None, session.set_lang, lang)
            await ws.send(json.dumps({
                "type": "ready",
                "chunk_bytes": session.chunk_bytes,
                "sample_rate": session.sample_rate,
            }))
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    import numpy as np
                    pcm = np.frombuffer(bytes(msg), dtype=np.int16)
                    try:
                        delta = await loop.run_in_executor(None, session.feed, pcm)
                    except Exception as e:  # noqa: BLE001
                        await ws.send(json.dumps({"type": "error", "error": f"{type(e).__name__}: {e}"}))
                        break
                    await ws.send(json.dumps({"type": "text", "text": delta}))
                else:
                    try:
                        ctl = json.loads(msg)
                    except Exception:
                        continue
                    if ctl.get("type") == "stop":
                        # flush any final partial window (ONNX backend), then reset
                        if hasattr(session, "finalize"):
                            tail = await loop.run_in_executor(None, session.finalize)
                            if tail:
                                await ws.send(json.dumps({"type": "text", "text": tail}))
                        await loop.run_in_executor(None, session.reset)
                        await ws.send(json.dumps({"type": "stopped"}))
                        break

    async with websockets.serve(handler, HOST, PORT, max_size=None):
        _log(f"[engine] listening on ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        _log("\n[engine] shutting down")
