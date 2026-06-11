"""Nemotron streaming ASR as a Pipecat STTService — in-process, no extra socket.

Wraps the vendored server/asr/onnx_engine.OnnxStreamingSession. Pipecat resamples
the caller's mic audio to the sample_rate we request (16 kHz) and hands run_stt()
raw 16-bit PCM bytes; we batch it to the engine's native 560 ms window and emit
each text delta IMMEDIATELY as a final TranscriptionFrame fragment — Pipecat's
universal aggregator assembles the turn with its own stop strategies. Holding
text until VAD-stop is a trap: on noisy phone-speaker calls Silero may never
emit UserStoppedSpeaking (seen live — whole utterances swallowed).

Pipecat API (verified June 2026): STTService.run_stt(audio: bytes) ->
AsyncGenerator[Frame]; frames + processor base from pipecat.*. If a future
Pipecat reorg moves an import, the gateway image's %test (python -c import)
surfaces it at build time, not at call time.
"""
import asyncio
import concurrent.futures
import os
import sys

import numpy as np

from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.stt_service import STTService
from pipecat.utils.time import time_now_iso8601

# the vendored engine lives one dir up, in asr/
_ASR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "asr")
sys.path.insert(0, _ASR_DIR)
import onnx_engine  # noqa: E402


# ONE engine per process, loaded lazily on the first call. The weights are
# ~3.4 GB VRAM and ~4 s to load — per-call instances accumulated VRAM across
# reconnects. ORT InferenceSession.run() is thread-safe; per-call state lives
# in each service's own OnnxStreamingSession.
_ENGINE: "onnx_engine.OnnxASR | None" = None


def _get_engine(lang: str, model_dir: str | None):
    global _ENGINE
    if _ENGINE is None:
        kw = {"lang": lang}
        if model_dir:
            kw["model_dir"] = model_dir
        _ENGINE = onnx_engine.OnnxASR(**kw)
    return _ENGINE


class NemotronSTTService(STTService):
    """Streaming STT over the ONNX Nemotron engine. One session per service
    instance (one caller at a time — matches the single-GPU engine)."""

    SAMPLE_RATE = 16000

    def __init__(self, *, lang: str = "en-US", model_dir: str | None = None,
                 user_id: str = "caller", require_gpu: bool | None = None, **kwargs):
        super().__init__(sample_rate=self.SAMPLE_RATE, **kwargs)
        self._lang = lang
        self._user_id = user_id
        self._asr = _get_engine(lang, model_dir)

        # Guard against a silent CPU fallback: get_providers() reflects the
        # provider actually ACTIVE on the session (CUDA drops off here if its
        # libs fail to load at runtime, even when compiled in). On a real call
        # ASR must run on the GPU. Default REQUIRE_GPU=1 -> hard fail; set
        # REQUIRE_GPU=0 to allow CPU (e.g. a GPU-less dev box).
        active = self._asr.enc.get_providers()
        self.on_gpu = any(("CUDA" in p) or ("Tensorrt" in p) for p in active)
        if require_gpu is None:
            require_gpu = os.environ.get("REQUIRE_GPU", "1") not in ("0", "false", "False", "")
        info = f"[NemotronSTT] onnxruntime providers={active} on_gpu={self.on_gpu}"
        if not self.on_gpu:
            if require_gpu:
                raise RuntimeError(
                    info + " — ASR fell back to CPU but REQUIRE_GPU=1. Check the CUDA-12 "
                    "libs / LD_LIBRARY_PATH in the gateway image, or set REQUIRE_GPU=0 to allow CPU.")
            print(info + "  WARNING: running on CPU", flush=True)
        else:
            print(info, flush=True)

        self._session = onnx_engine.OnnxStreamingSession(self._asr, sample_rate=self.SAMPLE_RATE)
        self._buf = bytearray()  # audio pending until one full engine window
        # SINGLE-thread executor: feed/finalize/reset mutate one session; on the
        # default pool they ran on different threads concurrently (audio keeps
        # flowing while a VAD stop is being finalized) and raced. One worker
        # serializes every engine call in arrival order.
        self._exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nemotron-stt")

    async def _in_thread(self, fn, *args):
        # the engine is blocking numpy/onnx — keep it off the event loop
        return await asyncio.get_running_loop().run_in_executor(self._exec, fn, *args)

    # Hard cap on session length between resets: feed() recomputes the mel over
    # the whole since-reset audio (O(n^2) over a call), and resets normally
    # happen at VAD boundaries — which may NEVER fire on noisy phone-speaker
    # calls. 45 s of speech without a boundary forces a flush+reset.
    MAX_SESSION_S = 45

    def _final(self, text: str) -> TranscriptionFrame:
        return TranscriptionFrame(text.strip(), self._user_id, time_now_iso8601(),
                                  language=self._lang)

    async def run_stt(self, audio: bytes):
        """Buffer to the engine's native 560 ms window, then emit each text
        delta IMMEDIATELY as a final fragment (Pipecat's aggregator assembles
        the turn with its own stop strategies). We deliberately do NOT hold
        text until VAD stop: on a noisy phone-speaker call Silero can fire
        'user started' but never 'user stopped', which used to swallow whole
        utterances — VAD stop is only a tail-flush bonus now."""
        self._buf += audio
        if len(self._buf) < self._session.chunk_bytes:
            return
        pcm = np.frombuffer(bytes(self._buf), dtype=np.int16)
        self._buf.clear()
        try:
            delta = await self._in_thread(self._session.feed, pcm)
        except Exception as e:  # noqa: BLE001 — keep the call alive
            print(f"[NemotronSTT] feed failed ({type(e).__name__}: {e})", flush=True)
            await self._in_thread(self._session.reset)
            return
        if delta and delta.strip():
            yield self._final(delta)
        if len(self._session._audio) > self.SAMPLE_RATE * self.MAX_SESSION_S:
            async for f in self._flush("session cap"):
                yield f

    async def _flush(self, why: str):
        """Flush the engine's pending window (tail text) and reset the session."""
        try:
            tail = await self._in_thread(self._session.finalize)
        except Exception as e:  # noqa: BLE001
            print(f"[NemotronSTT] finalize failed on {why} ({type(e).__name__}: {e})", flush=True)
            tail = ""
        try:
            await self._in_thread(self._session.reset)
        except Exception as e:  # noqa: BLE001
            print(f"[NemotronSTT] reset failed ({type(e).__name__}: {e})", flush=True)
        self._buf.clear()
        if tail and tail.strip():
            yield self._final(tail)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # VAD boundaries are tail-flush + reset points when they DO fire
        # (utterance text already streamed out as fragments either way).
        if isinstance(frame, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame)):
            boundary = "vad start" if isinstance(frame, UserStartedSpeakingFrame) else "vad stop"
            async for f in self._flush(boundary):
                await self.push_frame(f)
