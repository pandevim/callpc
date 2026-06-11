#!/usr/bin/env python3
"""Streaming Kyutai TTS server — WebSocket API (runs inside moshi.sif).

The PLAN's "main TTS task": wrap the moshi Python TTS in a server that yields
24 kHz PCM frames AS THEY GENERATE (low time-to-first-audio), not after the
whole clip. We use the DSM streaming pattern (TTSGen, vendored below from
kyutai-labs/delayed-streams-modeling scripts/tts_pytorch_streaming.py): we own
the per-frame step loop, so barge-in can abort generation at any 80 ms frame
boundary — `TTSModel.generate()` has no cancellation hook.

Protocol (ws://HOST:PORT, mirrors asr/ws_engine.py's style; one utterance at a
time per connection, connection reusable):
    client -> {"type":"start","voice":"<optional voice name>"}
    server -> {"type":"ready","sample_rate":24000,"frame_samples":1920}
    client -> {"type":"text","text":"<sentence or fragment>"} ...
    client -> {"type":"flush"}     (end utterance -> synthesize)
    server -> <binary: int16-LE mono PCM @ 24 kHz, one 80 ms frame per message>
    server -> {"type":"end"}       (utterance fully flushed)
Cancellation = client closes the socket (the Pipecat InterruptibleTTSService
contract); the step loop notices on the next frame and stops.

`GET /health` on the same port answers 200 once the model is loaded + warmed,
so the launcher's wait_for_http gates on actual readiness.

Model facts (moshi 0.2.13, kyutai/tts-1.6b-en_fr — verified June 2026):
  - mimi: 24 kHz, 12.5 fps -> 1920 samples / 80 ms per frame.
  - audio_delay 1.28 s -> the first ~16 frames carry -1 sentinels; only frames
    with no -1 are decodable, so first audio lands after ~16 GPU steps (well
    under real time on the L40S).
  - CFG-distilled: keep TTSModel.cfg_coef at 1.0 and pass cfg_coef (1.0..4.0,
    step 0.5) to make_condition_attributes instead.
"""
import argparse
import asyncio
import http
import json
import os
import sys
import threading
import typing as tp
from dataclasses import dataclass

import numpy as np
import torch
from websockets.exceptions import ConnectionClosed

from moshi.conditioners import dropout_all_conditions
from moshi.models.lm import LMGen
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import (
    DEFAULT_DSM_TTS_REPO,
    DEFAULT_DSM_TTS_VOICE_REPO,
    ConditionAttributes,
    TTSModel,
    script_to_entries,
)

HOST = os.environ.get("TTS_HOST", "0.0.0.0")
PORT = int(os.environ.get("TTS_PORT", "8089"))
HF_REPO = os.environ.get("TTS_REPO", DEFAULT_DSM_TTS_REPO)            # kyutai/tts-1.6b-en_fr
VOICE_REPO = os.environ.get("TTS_VOICE_REPO", DEFAULT_DSM_TTS_VOICE_REPO)  # kyutai/tts-voices
DEFAULT_VOICE = os.environ.get("TTS_VOICE", "expresso/ex03-ex01_happy_001_channel1_334s.wav")
CFG_COEF = float(os.environ.get("TTS_CFG_COEF", "2.0"))


def _log(m):
    print(m, file=sys.stderr, flush=True)


class _Cancelled(Exception):
    """Raised inside the step loop to abort generation on barge-in."""


# --------------------------------------------------------------------------- #
# TTSGen — vendored from kyutai-labs/delayed-streams-modeling
# scripts/tts_pytorch_streaming.py (MIT). The moshi package itself only ships
# the all-at-once TTSModel.generate(); this is the incremental step-loop
# variant. Do not "simplify" the hooks — they replicate generate() exactly
# (text state machine, audio-delay zeroing, padding bonus).
# --------------------------------------------------------------------------- #

def _prepare_script(model: TTSModel, script: str, first_turn: bool):
    multi_speaker = first_turn and model.multi_speaker
    return script_to_entries(
        model.tokenizer,
        model.machine.token_ids,
        model.mimi.frame_rate,
        [script],
        multi_speaker=multi_speaker,
        padding_between=1,
    )


def _make_null(all_attributes: tp.Sequence[ConditionAttributes]) -> list[ConditionAttributes]:
    # When using CFG, returns the null conditions.
    return dropout_all_conditions(all_attributes)


@dataclass
class TTSGen:
    tts_model: TTSModel
    attributes: tp.Sequence[ConditionAttributes]
    on_frame: tp.Optional[tp.Callable[[torch.Tensor], None]] = None

    def __post_init__(self):
        tts_model = self.tts_model
        attributes = self.attributes
        self.offset = 0
        self.state = self.tts_model.machine.new_state([])
        if tts_model.cfg_coef != 1.0:
            if tts_model.valid_cfg_conditionings:
                raise ValueError(
                    "This model does not support direct CFG, but was trained with "
                    "CFG distillation. Pass instead `cfg_coef` to `make_condition_attributes`."
                )
            nulled = _make_null(attributes)
            attributes = list(attributes) + nulled

        assert tts_model.lm.condition_provider is not None
        prepared = tts_model.lm.condition_provider.prepare(attributes)
        condition_tensors = tts_model.lm.condition_provider(prepared)

        def _on_text_logits_hook(text_logits):
            if tts_model.padding_bonus:
                text_logits[..., tts_model.machine.token_ids.pad] += tts_model.padding_bonus
            return text_logits

        def _on_audio_hook(audio_tokens):
            audio_offset = tts_model.lm.audio_offset
            delays = tts_model.lm.delays
            for q in range(audio_tokens.shape[1]):
                delay = delays[q + audio_offset]
                if self.offset < delay + tts_model.delay_steps:
                    audio_tokens[:, q] = tts_model.machine.token_ids.zero

        def _on_text_hook(text_tokens):
            tokens = text_tokens.tolist()
            out_tokens = []
            for token in tokens:
                out_token, _ = tts_model.machine.process(self.offset, self.state, token)
                out_tokens.append(out_token)
            text_tokens[:] = torch.tensor(out_tokens, dtype=torch.long, device=text_tokens.device)

        tts_model.lm.dep_q = tts_model.n_q
        self.lm_gen = LMGen(
            tts_model.lm,
            temp=tts_model.temp,
            temp_text=tts_model.temp,
            cfg_coef=tts_model.cfg_coef,
            condition_tensors=condition_tensors,
            on_text_logits_hook=_on_text_logits_hook,
            on_text_hook=_on_text_hook,
            on_audio_hook=_on_audio_hook,
            cfg_is_masked_until=None,
            cfg_is_no_text=True,
        )
        # DEVIATION from the vendored script: no streaming_forever(1) here. It
        # enters streaming on the SHARED lm module and never exits, so the next
        # utterance's fresh TTSGen asserts "<module> is already streaming!".
        # The caller wraps the step loop in `with lm_gen.streaming(1)` instead
        # (per-utterance enter/exit, unwinds on errors/barge-in too).

    def process_last(self):
        while len(self.state.entries) > 0 or self.state.end_step is not None:
            self._step()
        additional_steps = self.tts_model.delay_steps + max(self.tts_model.lm.delays) + 8
        for _ in range(additional_steps):
            self._step()

    def process(self):
        while len(self.state.entries) > self.tts_model.machine.second_stream_ahead:
            self._step()

    def _step(self):
        missing = self.tts_model.lm.n_q - self.tts_model.lm.dep_q
        input_tokens = torch.full(
            (1, missing, 1),
            self.tts_model.machine.token_ids.zero,
            dtype=torch.long,
            device=self.tts_model.lm.device,
        )
        frame = self.lm_gen.step(input_tokens)
        self.offset += 1
        if frame is not None:
            if self.on_frame is not None:
                self.on_frame(frame)

    def append_entry(self, entry):
        self.state.entries.append(entry)


# --------------------------------------------------------------------------- #
# engine: blocking synthesis, one utterance at a time (single GPU)
# --------------------------------------------------------------------------- #

class KyutaiEngine:
    def __init__(self, hf_repo: str, voice_repo: str, device: str = "cuda"):
        _log(f"[tts] loading {hf_repo} on {device} (voices: {voice_repo})...")
        info = CheckpointInfo.from_hf_repo(hf_repo)
        self.model = TTSModel.from_checkpoint_info(
            info, n_q=32, temp=0.6, device=device, voice_repo=voice_repo)
        self.sample_rate = int(self.model.mimi.sample_rate)                 # 24000
        self.frame_samples = int(self.sample_rate / self.model.mimi.frame_rate)  # 1920
        self._attrs: dict[str, ConditionAttributes] = {}                    # voice -> conditioning
        _log(f"[tts] model ready ({self.sample_rate} Hz, {self.frame_samples} samples/frame, "
             f"delay {self.model.delay_steps} frames)")

    def _attributes(self, voice: str) -> ConditionAttributes:
        if voice not in self._attrs:
            path = self.model.get_voice_path(voice)
            self._attrs[voice] = self.model.make_condition_attributes([path], cfg_coef=CFG_COEF)
            _log(f"[tts] voice loaded: {voice}")
        return self._attrs[voice]

    @torch.no_grad()  # TTSModel.generate() has this built in; our TTSGen step
    # loop runs outside it, and mimi.decode() output would otherwise carry a
    # grad graph (".numpy() on Tensor that requires grad").
    def synthesize(self, texts: list[str], voice: str,
                   emit: tp.Callable[[bytes], None],
                   cancelled: threading.Event) -> int:
        """Blocking. Calls emit(int16 PCM bytes) per 80 ms frame as it generates.
        Returns frames emitted; aborts (raising nothing) if `cancelled` is set."""
        attrs = self._attributes(voice)
        n_frames = 0

        def _on_frame(frame):
            nonlocal n_frames
            if cancelled.is_set():
                raise _Cancelled
            if (frame != -1).all():
                pcm = self.model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                pcm = np.clip(pcm[0, 0], -1, 1)
                emit((pcm * 32767.0).astype(np.int16).tobytes())
                n_frames += 1

        # Fresh TTSGen per utterance: conditioning + the text state machine are
        # baked in at construction. Streaming state (LM + mimi decoder) is
        # entered HERE and exited at the end of the with-block — leaving it
        # entered would break the next utterance (see note in __post_init__).
        gen = TTSGen(self.model, [attrs], on_frame=_on_frame)
        try:
            with gen.lm_gen.streaming(1), self.model.mimi.streaming(1):
                for text in texts:
                    for entry in _prepare_script(self.model, text, first_turn=True):
                        gen.append_entry(entry)
                    gen.process()
                gen.process_last()
        except _Cancelled:
            _log(f"[tts] cancelled after {n_frames} frames")
        return n_frames


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #

async def main() -> int:
    ap = argparse.ArgumentParser(description="streaming Kyutai TTS server")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    from websockets.asyncio.server import serve  # websockets >= 13 (in moshi.sif)

    loop = asyncio.get_running_loop()
    engine = await loop.run_in_executor(
        None, lambda: KyutaiEngine(HF_REPO, VOICE_REPO,
                                   "cuda" if torch.cuda.is_available() else "cpu"))

    # Warmup before binding the port: first synthesis pays voice download +
    # cuda-graph capture; /health must mean "ready for a real call".
    t0 = loop.time()
    n = await loop.run_in_executor(
        None, engine.synthesize, ["Warm up."], DEFAULT_VOICE, lambda b: None, threading.Event())
    _log(f"[tts] warmup: {n} frames in {loop.time() - t0:.1f}s")

    gpu_lock = asyncio.Lock()  # one synthesis at a time (single GPU)

    async def speak(ws, texts: list[str], voice: str):
        """Run one utterance: stream PCM frames to the socket, then send end."""
        q: asyncio.Queue[bytes | None] = asyncio.Queue()
        cancelled = threading.Event()

        def _emit(b: bytes):
            loop.call_soon_threadsafe(q.put_nowait, b)

        async with gpu_lock:
            fut = loop.run_in_executor(None, engine.synthesize, texts, voice, _emit, cancelled)
            fut.add_done_callback(lambda _: q.put_nowait(None))  # runs on the loop
            try:
                while (chunk := await q.get()) is not None:
                    await ws.send(chunk)
                await fut  # surface engine errors
                await ws.send(json.dumps({"type": "end"}))
            except BaseException as e:
                cancelled.set()  # stop the GPU loop on disconnect/cancel
                try:
                    await fut
                except Exception:
                    pass
                if not isinstance(e, ConnectionClosed):
                    # engine error with the socket still up — tell the client
                    # before the connection drops (selftest/gateway print it)
                    try:
                        await ws.send(json.dumps(
                            {"type": "error", "error": f"{type(e).__name__}: {e}"}))
                    except Exception:
                        pass
                raise

    async def handler(ws):
        voice = DEFAULT_VOICE
        texts: list[str] = []
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    continue  # no binary input on this protocol
                try:
                    ctl = json.loads(msg)
                except Exception:
                    continue
                kind = ctl.get("type")
                if kind == "start":
                    voice = ctl.get("voice") or voice
                    texts = []
                    await ws.send(json.dumps({
                        "type": "ready",
                        "sample_rate": engine.sample_rate,
                        "frame_samples": engine.frame_samples,
                    }))
                elif kind == "text":
                    if ctl.get("text", "").strip():
                        texts.append(ctl["text"].strip())
                elif kind == "flush":
                    if texts:
                        await speak(ws, texts, voice)
                    else:
                        await ws.send(json.dumps({"type": "end"}))
                    texts = []
        except ConnectionClosed as e:  # routine: barge-in / client hangup
            _log(f"[tts] connection closed: {e}")
        except Exception as e:  # noqa: BLE001 — real errors: full stack to the log
            import traceback
            traceback.print_exc(file=sys.stderr)
            _log(f"[tts] connection ended on error: {type(e).__name__}: {e}")

    def _process_request(connection, request):
        # plain-HTTP health probe on the WS port (for curl / wait_for_http)
        if request.path == "/health":
            return connection.respond(http.HTTPStatus.OK, '{"status":"ok","service":"kyutai-tts"}\n')
        return None

    async with serve(handler, args.host, args.port,
                     process_request=_process_request, max_size=None):
        _log(f"[tts] listening on ws://{args.host}:{args.port} (voice={DEFAULT_VOICE})")
        await asyncio.Future()  # run forever
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except KeyboardInterrupt:
        _log("\n[tts] shutting down")
