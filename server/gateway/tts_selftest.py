#!/usr/bin/env python3
"""Smoke test for tts_server.py — speaks one sentence, measures latency.

Client-only (stdlib + websockets; no torch/moshi), so it runs from any image or
the login node through a tunnel:

    python tts_selftest.py [--url ws://127.0.0.1:8089] [--text "..."] [--wav out.wav]

Reports time-to-first-audio (the metric that matters for the call feel; target
~0.5 s incl. the model's 1.28 s audio-delay generated faster than real time),
total wall time, and the audio duration. Writes a wav to listen to if asked.
"""
import argparse
import asyncio
import json
import sys
import time
import wave


async def main() -> int:
    ap = argparse.ArgumentParser(description="TTS server smoke test")
    ap.add_argument("--url", default="ws://127.0.0.1:8089")
    ap.add_argument("--text", default="Hello! This is your computer speaking. The voice loop is working.")
    ap.add_argument("--voice", default=None, help="kyutai/tts-voices name (server default if unset)")
    ap.add_argument("--wav", default=None, help="write the synthesized audio here")
    args = ap.parse_args()

    from websockets.asyncio.client import connect

    async with connect(args.url, max_size=None) as ws:
        start = {"type": "start"}
        if args.voice:
            start["voice"] = args.voice
        await ws.send(json.dumps(start))
        ready = json.loads(await ws.recv())
        if ready.get("type") != "ready":
            print(f"FAIL: expected ready, got {ready}", file=sys.stderr)
            return 1
        sr = ready["sample_rate"]
        print(f"ready: {sr} Hz, {ready['frame_samples']} samples/frame")

        await ws.send(json.dumps({"type": "text", "text": args.text}))
        await ws.send(json.dumps({"type": "flush"}))
        t0 = time.monotonic()
        ttfa = None
        chunks: list[bytes] = []
        while True:
            msg = await ws.recv()
            if isinstance(msg, (bytes, bytearray)):
                if ttfa is None:
                    ttfa = time.monotonic() - t0
                chunks.append(bytes(msg))
            elif json.loads(msg).get("type") == "end":
                break
        total = time.monotonic() - t0

    if not chunks:
        print("FAIL: no audio received", file=sys.stderr)
        return 1
    audio_s = sum(len(c) for c in chunks) / 2 / sr
    rtf = total / audio_s if audio_s else 0
    print(f"OK: TTFA {ttfa*1000:.0f} ms, total {total:.2f} s for {audio_s:.2f} s audio (RTF {rtf:.2f})")
    if args.wav:
        with wave.open(args.wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(b"".join(chunks))
        print(f"wrote {args.wav}")
    return 0 if (rtf < 1.0) else 2  # slower than real time = not call-usable


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
