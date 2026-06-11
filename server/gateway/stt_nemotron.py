"""Nemotron streaming ASR as a Pipecat STTService — in-process, no extra socket.

Wraps the vendored server/asr/onnx_engine.OnnxStreamingSession. Pipecat resamples
the caller's mic audio to the sample_rate we request (16 kHz) and hands run_stt()
raw 16-bit PCM bytes; we feed it to the engine, emit interim transcripts as text
accrues (~0.56 s cadence — the export's fixed chunk geometry), and flush the final
TranscriptionFrame when the user stops speaking (VAD).

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
    InterimTranscriptionFrame,
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
        self._acc = ""  # accumulated text for the in-flight utterance
        # SINGLE-thread executor: feed/finalize/reset mutate one session; on the
        # default pool they ran on different threads concurrently (audio keeps
        # flowing while a VAD stop is being finalized) and raced. One worker
        # serializes every engine call in arrival order.
        self._exec = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nemotron-stt")

    async def _in_thread(self, fn, *args):
        # the engine is blocking numpy/onnx — keep it off the event loop
        return await asyncio.get_running_loop().run_in_executor(self._exec, fn, *args)

    async def run_stt(self, audio: bytes):
        """Feed one PCM chunk; emit a cumulative interim transcript if text grew."""
        pcm = np.frombuffer(audio, dtype=np.int16)
        if pcm.size == 0:
            return
        delta = await self._in_thread(self._session.feed, pcm)
        if delta:
            self._acc += delta
            yield InterimTranscriptionFrame(
                self._acc.strip(), self._user_id, time_now_iso8601(), language=self._lang)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        # VAD boundaries drive utterance reset + finalization.
        if isinstance(frame, UserStartedSpeakingFrame):
            self._acc = ""
            await self._in_thread(self._session.reset)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # NEVER lose the turn: without a final TranscriptionFrame the
            # aggregator won't run the LLM and the call goes silent. If the
            # engine flush fails, the accumulated interims ARE the transcript.
            try:
                tail = await self._in_thread(self._session.finalize)
            except Exception as e:  # noqa: BLE001 — degrade to interims, keep the call alive
                print(f"[NemotronSTT] finalize failed ({type(e).__name__}: {e}); "
                      "using interim text", flush=True)
                tail = ""
            text = (self._acc + (tail or "")).strip()
            self._acc = ""
            try:
                await self._in_thread(self._session.reset)
            except Exception as e:  # noqa: BLE001
                print(f"[NemotronSTT] reset failed ({type(e).__name__}: {e})", flush=True)
            if text:
                await self.push_frame(
                    TranscriptionFrame(text, self._user_id, time_now_iso8601(), language=self._lang))
