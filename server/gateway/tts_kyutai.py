"""Kyutai TTS as a Pipecat InterruptibleTTSService — WS client to tts_server.py.

Why InterruptibleTTSService: a Kyutai stream can only be cancelled by closing
the socket. This base does exactly that on barge-in — when an InterruptionFrame
arrives while the bot is speaking it calls our _disconnect() then _connect()
(verified pipecat 1.3.0), and tts_server.py aborts generation when the socket
drops. We just implement the connection lifecycle.

Pipecat 1.3 websocket-TTS contract (verified June 2026, from the LMNT/Gradium
reference services): run_tts(text, context_id) sends the text over the WS and
yields None ("audio arrives via the receive loop"); the receive task delivers
TTSAudioRawFrames through append_to_audio_context() — never push_frame directly.
With push_start_frame/push_stop_frames=True the base emits TTSStarted/Stopped
itself (our explicit TTSStoppedFrame on the server's "end" message just beats
the idle-timeout fallback).

Protocol (see tts_server.py): {"type":"start","voice":...} -> "ready";
{"type":"text"} + {"type":"flush"} -> binary int16 24 kHz PCM frames -> "end".
"""
import json
import os
from typing import AsyncGenerator, Optional

from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import InterruptibleTTSService


class KyutaiTTSService(InterruptibleTTSService):
    """Streams speech from our Kyutai tts_server.py. One utterance per
    run_tts() call (Pipecat hands us sentence-sized chunks); 24 kHz output
    matches the pipeline's default output rate — no resampling."""

    SAMPLE_RATE = 24000

    def __init__(self, *, url: Optional[str] = None, voice: Optional[str] = None,
                 sample_rate: Optional[int] = None, **kwargs):
        super().__init__(
            push_stop_frames=True,
            push_start_frame=True,
            pause_frame_processing=True,
            sample_rate=sample_rate or self.SAMPLE_RATE,
            **kwargs,
        )
        self._url = url or os.environ.get("TTS_WS_URL", "ws://127.0.0.1:8089")
        self._voice = voice or os.environ.get("TTS_VOICE")  # None -> server default
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        return True

    # --- lifecycle: ws + receive task (the InterruptibleTTSService contract) ---

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            self._websocket = await websocket_connect(self._url)
            start: dict = {"type": "start"}
            if self._voice:
                start["voice"] = self._voice
            await self._websocket.send(json.dumps(start))  # server answers "ready" (receive loop ignores it)
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Kyutai TTS connect failed ({self._url}): {e}", exception=e)
            self._websocket = None
            await self._call_event_handler("on_connection_error", f"{e}")

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            await self.push_error(error_msg=f"Kyutai TTS disconnect error: {e}", exception=e)
        finally:
            self._websocket = None
            await self._call_event_handler("on_disconnected")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Kyutai TTS websocket not connected")

    # --- audio in: server -> TTSAudioRawFrames into the active audio context ---

    async def _receive_messages(self):
        async for message in self._get_websocket():
            context_id = self.get_active_audio_context_id()
            if isinstance(message, (bytes, bytearray)):
                if not context_id:
                    continue  # stale audio after an interruption — drop
                await self.stop_ttfb_metrics()
                await self.append_to_audio_context(context_id, TTSAudioRawFrame(
                    audio=bytes(message),
                    sample_rate=self.sample_rate,
                    num_channels=1,
                    context_id=context_id,
                ))
            else:
                msg = json.loads(message)
                if msg.get("type") == "end" and context_id:
                    # explicit end-of-utterance beats the push_stop_frames idle timeout
                    await self.append_to_audio_context(
                        context_id, TTSStoppedFrame(context_id=context_id))
                elif msg.get("type") == "error":
                    await self.push_error(error_msg=f"Kyutai TTS server error: {msg.get('error')}")

    # --- text out ---

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                await self._connect()
            try:
                await self._get_websocket().send(json.dumps({"type": "text", "text": text}))
                await self._get_websocket().send(json.dumps({"type": "flush"}))
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                yield ErrorFrame(error=f"Kyutai TTS send failed: {e}")
                yield TTSStoppedFrame(context_id=context_id)
                await self._disconnect()
                await self._connect()
                return
            yield None  # audio arrives via _receive_messages
        except Exception as e:
            yield ErrorFrame(error=f"Kyutai TTS error: {e}")
