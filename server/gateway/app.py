"""callpc voice gateway — FastAPI + Pipecat SmallWebRTC.

Phase 1a (this file): caller mic -> Nemotron STT -> log transcripts -> audio out.
The brain (Qwen tool-calling) and Kyutai TTS splice into this same pipeline next.

Signaling is one `POST /api/offer` (the small-webrtc prebuilt UI, mounted at /,
posts here). ICE is STUN-only — add TURN only if media fails through ngrok.

Run: `python app.py` inside the gateway image (GATEWAY_PORT from env). The ASR
model must be present at server/asr/models/onnx (the preflight `asr --download`
already staged it).
"""
import json
import os

from aiortc import RTCIceServer
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from stt_nemotron import NemotronSTTService

HOST = os.environ.get("GATEWAY_HOST", "0.0.0.0")
PORT = int(os.environ.get("GATEWAY_PORT", "8080"))
STT_LANG = os.environ.get("STT_LANG", "en-US")

# Brain: the Qwen orchestrator served by vLLM (OpenAI-compatible). In the
# launcher it's on the same node at :8001; vLLM ignores the api_key.
BRAIN_BASE_URL = os.environ.get("BRAIN_BASE_URL", "http://127.0.0.1:8001/v1")
BRAIN_MODEL = os.environ.get("BRAIN_MODEL", "brain")
BRAIN_API_KEY = os.environ.get("BRAIN_API_KEY", "x")

# Phone-style persona (Phase 1b: conversational only — the computer_task tool is
# wired in Phase 3, so don't claim tasks are done yet).
PERSONA = os.environ.get("PERSONA", (
    "You are callpc, a helpful assistant someone reached by calling their computer "
    "on the phone. You are speaking out loud, so keep replies short and natural — "
    "one or two sentences, no lists, no markdown, no emoji. If asked to do something "
    "on the computer, respond conversationally and say you'll be able to once you're "
    "fully set up; don't pretend a task is finished. Ask a brief clarifying question "
    "when you need to."
))


def _ice_config():
    """STUN always; add TURN if TURN_URL is set (needed when media can't traverse
    the cluster NAT — the browser shows 'ICE failed, add a TURN server'). Returns
    plain dicts for the browser; the server converts them to RTCIceServer."""
    servers = [{"urls": "stun:stun.l.google.com:19302"}]
    turn = os.environ.get("TURN_URL")
    if turn:
        # TURN_URL may be comma-separated (UDP :80 / TCP :443 / TLS turns:443) —
        # offering all transports maximizes traversal through strict firewalls.
        urls = [u.strip() for u in turn.split(",") if u.strip()]
        s = {"urls": urls if len(urls) > 1 else urls[0]}
        if os.environ.get("TURN_USER"):
            s["username"] = os.environ["TURN_USER"]
        if os.environ.get("TURN_PASS"):
            s["credential"] = os.environ["TURN_PASS"]
        servers.append(s)
    return servers


ICE_CONFIG = _ice_config()                                  # dicts -> injected into the page
ICE_SERVERS = [RTCIceServer(**s) for s in ICE_CONFIG]        # -> aiortc (server side)
print(f"[gateway] ICE servers: {[s['urls'] for s in ICE_CONFIG]}", flush=True)

app = FastAPI()
_connections: dict[str, SmallWebRTCConnection] = {}  # pc_id -> connection


class TranscriptLogger(FrameProcessor):
    """Prints the caller's interim + final transcripts (placed after STT)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            print(f"[FINAL]   {frame.text!r}", flush=True)
        elif isinstance(frame, InterimTranscriptionFrame):
            print(f"[interim] {frame.text!r}", flush=True)
        await self.push_frame(frame, direction)


class BotLogger(FrameProcessor):
    """Accumulates the brain's streamed text and prints the full reply on
    response end (placed after the LLM). Until TTS lands, this is how we see the
    bot 'talk'. TranscriptionFrame subclasses TextFrame, so exclude it."""

    def __init__(self):
        super().__init__()
        self._buf = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame) and not isinstance(
                frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            self._buf += frame.text
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._buf.strip():
                print(f"[bot]     {self._buf.strip()!r}", flush=True)
            self._buf = ""
        await self.push_frame(frame, direction)


async def _run_bot(connection: SmallWebRTCConnection):
    """One pipeline per call. transport.input() -> STT -> logger -> output."""
    transport = SmallWebRTCTransport(
        webrtc_connection=connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )
    stt = NemotronSTTService(lang=STT_LANG)
    llm = OpenAILLMService(api_key=BRAIN_API_KEY, base_url=BRAIN_BASE_URL, model=BRAIN_MODEL)
    context = LLMContext(messages=[{"role": "system", "content": PERSONA}])
    agg = LLMContextAggregatorPair(context)

    # caller mic -> STT -> (log) -> user-context -> brain -> (log) -> out -> assistant-context
    # allow_interruptions=True gives barge-in once TTS is producing audio.
    pipeline = Pipeline([
        transport.input(),
        stt,
        TranscriptLogger(),
        agg.user(),
        llm,
        BotLogger(),
        transport.output(),
        agg.assistant(),
    ])
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.get("/health")
async def health():
    return {"status": "ok", "service": "callpc-gateway", "active_calls": len(_connections)}


@app.post("/api/offer")
async def offer(req: dict, background_tasks: BackgroundTasks):
    """WebRTC signaling: reuse an existing pc_id (renegotiate) or start a new call."""
    pc_id = req.get("pc_id")
    if pc_id and pc_id in _connections:
        conn = _connections[pc_id]
        await conn.renegotiate(sdp=req["sdp"], type=req["type"],
                               restart_pc=req.get("restart_pc", False))
    else:
        conn = SmallWebRTCConnection(ice_servers=ICE_SERVERS)
        await conn.initialize(sdp=req["sdp"], type=req["type"])

        @conn.event_handler("closed")
        async def _on_closed(c: SmallWebRTCConnection):
            _connections.pop(c.pc_id, None)
            print(f"[gateway] call {c.pc_id} closed ({len(_connections)} active)", flush=True)

        _connections[conn.pc_id] = conn
        background_tasks.add_task(_run_bot, conn)
        print(f"[gateway] new call {conn.pc_id} ({len(_connections)} active)", flush=True)

    return JSONResponse(conn.get_answer())


# Minimal self-contained test page. The pipecat prebuilt UI is an RTVI client
# that POSTs to /start with its own handshake — incompatible with our raw
# /api/offer SDP endpoint. This page does standard WebRTC offer/answer directly
# against /api/offer (aiortc speaks vanilla WebRTC), so we control both ends.
# The real phone-call UI ships separately (Vercel webapp) later.
_CALL_UI = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>callpc</title></head>
<body style="font-family:system-ui;max-width:480px;margin:48px auto;text-align:center">
<h2>callpc &mdash; voice test</h2>
<button id="go" style="font-size:18px;padding:12px 24px">Connect &amp; talk</button>
<p id="status">idle</p>
<audio id="remote" autoplay></audio>
<script>
const log = m => { document.getElementById('status').textContent = m; console.log(m); };
document.getElementById('go').onclick = async () => {
  try {
    log('requesting mic...');
    const stream = await navigator.mediaDevices.getUserMedia({audio:true, video:false});
    const pc = new RTCPeerConnection({iceServers: __ICE_SERVERS__});
    stream.getTracks().forEach(t => pc.addTrack(t, stream));
    pc.ontrack = e => { document.getElementById('remote').srcObject = e.streams[0]; };
    pc.onconnectionstatechange = () => log('peer: ' + pc.connectionState);
    await pc.setLocalDescription(await pc.createOffer());
    // non-trickle: wait for ICE gathering to finish before sending the offer
    await new Promise(res => {
      if (pc.iceGatheringState === 'complete') return res();
      const t = setTimeout(res, 3000);
      pc.addEventListener('icegatheringstatechange', () => {
        if (pc.iceGatheringState === 'complete') { clearTimeout(t); res(); }
      });
    });
    log('signaling...');
    const r = await fetch('/api/offer', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({sdp: pc.localDescription.sdp, type: pc.localDescription.type})});
    if (!r.ok) { log('offer rejected: HTTP ' + r.status); return; }
    await pc.setRemoteDescription(await r.json());
    log('connected — speak!');
  } catch (e) { log('error: ' + e.message); console.error(e); }
};
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return _CALL_UI.replace("__ICE_SERVERS__", json.dumps(ICE_CONFIG))


if __name__ == "__main__":
    import uvicorn

    print(f"[gateway] starting on {HOST}:{PORT} (STT lang={STT_LANG})", flush=True)
    uvicorn.run(app, host=HOST, port=PORT)
