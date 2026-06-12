"""Chat gateway — WhatsApp-style messaging with the computer (Phase 5).

One WebSocket (`/api/chat/ws?code=<passcode>`) carries the whole conversation;
HTTP middleware doesn't cover WebSockets, so the passcode is checked here (the
socket is closed with 4401 when it's wrong). All connected clients (phone +
laptop) share ONE conversation, mirroring the single-tenant call line, and the
last HISTORY_MAX items are replayed on connect so reloads keep the thread.

client -> server:
    {"type": "text",  "text": ...}                      a typed message
    {"type": "voice", "b64": ..., "mime": "audio/..."}  a recorded voice note
    {"type": "screenshot"}                              quick action (no LLM)
server -> client:
    {"type": "history", "items": [...], "status": ...}  replay on connect
    {"type": "msg", "role": "user"|"assistant", "text"} durable bubble
    {"type": "image", "b64": ..., "caption": ...}       screenshot bubble
    {"type": "transcript", "text": ...}                 what a voice note said
    {"type": "status", "text": ...}                     in-place progress ("" hides)
    {"type": "error", "text": ...}                      transient failure note

Update policy (the Operator/Devin pattern — durable milestones, ephemeral
progress): step-by-step task activity streams into ONE status bubble updated
in place; real messages land only for milestones — finished (summary + a fresh
final screenshot), stuck/needs-input (+ screenshot), failed (+ screenshot).
The bridge is tasks.runner.EVENT, registered at import; it also fires for
tasks started from a voice call, so the chat doubles as the call's paper trail.

Voice notes: the browser records webm/opus (Safari: mp4/aac), we decode with
PyAV (already present via aiortc) to 16 kHz mono and run the same Nemotron
engine offline (OnnxASR.transcribe — stateless, safe next to a live call).
Replies are text; TTS voice-note replies are a possible later addition.
"""
import asyncio
import base64
import concurrent.futures
import io
import json
import os
import sys
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control.hub import hub                # noqa: E402
from tasks.runner import runner            # noqa: E402
import tasks.runner as task_runner         # noqa: E402
import stt_nemotron                        # noqa: E402  (shared ASR engine)
import onnx_engine                         # noqa: E402  (on path via stt_nemotron)

STT_LANG = os.environ.get("STT_LANG", "en-US")
BRAIN_BASE_URL = os.environ.get("BRAIN_BASE_URL", "http://127.0.0.1:8001/v1")
BRAIN_MODEL = os.environ.get("BRAIN_MODEL", "brain")
BRAIN_API_KEY = os.environ.get("BRAIN_API_KEY", "x")

HISTORY_MAX = 150   # replayed items kept in memory
IMAGE_KEEP = 8      # only the newest screenshots keep their pixels (b64 is big)
CTX_MAX = 48        # brain context entries kept beyond the system prompt

CHAT_PERSONA = os.environ.get("CHAT_PERSONA", (
    "You are callpc, the assistant for the user's own computer, chatting with them "
    "in a messaging app. Keep replies short and conversational — one to three plain "
    "sentences, no markdown, no bullet lists; an emoji here and there is fine.\n\n"
    "CRITICAL: You control the computer ONLY by calling the computer_task tool, and "
    "you see the screen ONLY by calling take_screenshot. You cannot open apps, click, "
    "type, browse, or look at the screen by yourself. NEVER say you are doing "
    "something, that it's done, or describe the screen unless you called the matching "
    "tool in the same turn.\n\n"
    "computer_task takes ONE self-contained instruction, e.g. 'Open Firefox and "
    "search for tickets to Boston'. While it runs, the chat automatically shows the "
    "user live progress, and when it finishes, gets stuck, or fails the chat posts a "
    "message plus a screenshot by itself — so after starting a task just acknowledge "
    "briefly and do NOT invent or repeat progress. Use task_status if asked how it's "
    "going, cancel_task to stop it. When the user asks to see the screen, call "
    "take_screenshot — the image is posted into the chat for you. For destructive or "
    "irreversible actions (deleting, sending messages, payments) get a clear yes "
    "first. Voice notes arrive as transcribed text. If no computer is connected, say "
    "so plainly."
))

TOOLS = [
    {"type": "function", "function": {
        "name": "computer_task",
        "description": ("Perform a task on the user's computer (GUI automation: open "
                        "apps, click, type, browse). Provide one complete, "
                        "self-contained instruction describing the end goal."),
        "parameters": {"type": "object", "properties": {
            "instruction": {"type": "string", "description":
                "The full task, e.g. 'Open Slack and read the latest message in the "
                "development channel.'"}}, "required": ["instruction"]},
    }},
    {"type": "function", "function": {
        "name": "take_screenshot",
        "description": ("Capture the computer's screen right now and post it into "
                        "the chat. Use when the user asks to see the screen or what "
                        "something looks like."),
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "task_status",
        "description": "Check how the current (or last) computer task is going.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "cancel_task",
        "description": "Stop the computer task that is currently running.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

# voice-note ASR runs off the event loop; one worker serializes engine calls
_STT_EXEC = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="chat-stt")


def _decode_voice(data: bytes) -> np.ndarray:
    """Browser voice note (webm/opus or mp4/aac) -> float32 mono 16 kHz."""
    import av  # present via aiortc; imported lazily to keep module import light
    chunks = []
    with av.open(io.BytesIO(data)) as container:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(audio=0):
            frame.pts = None  # browsers emit sparse/odd pts; resample by content
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().reshape(-1))
        try:  # flush the resampler's tail (PyAV >= 9)
            for rf in resampler.resample(None):
                chunks.append(rf.to_ndarray().reshape(-1))
        except Exception:  # noqa: BLE001
            pass
    if not chunks:
        return np.zeros(0, np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _transcribe(pcm: np.ndarray) -> str:
    text = stt_nemotron._get_engine(STT_LANG, None).transcribe(pcm)
    return onnx_engine._LANG_TAG.sub("", text or "").strip()


class ChatHub:
    def __init__(self):
        self.sockets: set[WebSocket] = set()
        self.history: list[dict] = []          # wire items, replayed on connect
        self.messages: list[dict] = [{"role": "system", "content": CHAT_PERSONA}]
        self._lock = asyncio.Lock()            # one brain turn at a time
        self._client = None
        self._status = ""                      # last status-bubble text
        self._notes: list[str] = []            # task milestones the brain hasn't
                                               # seen yet (folded into the next
                                               # user turn — Qwen3.5's template
                                               # 400s on mid-convo system msgs)

    def client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(base_url=BRAIN_BASE_URL, api_key=BRAIN_API_KEY)
        return self._client

    # ---------------- fan-out ----------------

    async def broadcast(self, item: dict, save: bool = True,
                        exclude: WebSocket | None = None):
        if save:
            item = dict(item)
            item.setdefault("ts", int(time.time()))
            self.history.append(item)
            self._trim_history()
        data = json.dumps(item)
        for ws in list(self.sockets):
            if ws is exclude:
                continue
            try:
                await ws.send_text(data)
            except Exception:  # noqa: BLE001 — dead socket; the receive loop reaps it
                self.sockets.discard(ws)

    def _trim_history(self):
        if len(self.history) > HISTORY_MAX:
            del self.history[:len(self.history) - HISTORY_MAX]
        imgs = [i for i in self.history if i.get("type") == "image" and i.get("b64")]
        for i in imgs[:-IMAGE_KEEP]:
            i.pop("b64", None)
            i["expired"] = True

    async def set_status(self, text: str):
        self._status = text
        await self.broadcast({"type": "status", "text": text}, save=False)

    # ---------------- inbound ----------------

    async def on_text(self, text: str, ws: WebSocket | None = None):
        # the sender already rendered its own bubble; sync the other devices
        await self.broadcast({"type": "msg", "role": "user", "text": text}, exclude=ws)
        await self._think(text)

    async def on_voice(self, b64: str, ws: WebSocket | None = None):
        loop = asyncio.get_running_loop()
        try:
            pcm = await loop.run_in_executor(_STT_EXEC, _decode_voice, base64.b64decode(b64))
        except Exception as e:  # noqa: BLE001
            await self.broadcast({"type": "error",
                                  "text": f"couldn't decode that voice note ({type(e).__name__})"},
                                 save=False)
            return
        if len(pcm) < 16000 * 0.3:
            await self.broadcast({"type": "error", "text": "that voice note was too short"},
                                 save=False)
            return
        await self.set_status("transcribing…")
        try:
            text = await loop.run_in_executor(_STT_EXEC, _transcribe, pcm)
        except Exception as e:  # noqa: BLE001
            await self.set_status("")
            await self.broadcast({"type": "error",
                                  "text": f"transcription failed ({type(e).__name__})"}, save=False)
            return
        if not text:
            await self.set_status("")
            await self.broadcast({"type": "error",
                                  "text": "I couldn't make out any words in that"}, save=False)
            return
        # durable: every device (and the replay) sees what the note said
        await self.broadcast({"type": "transcript", "text": text})
        await self._think(text)

    async def send_screenshot_now(self):
        """Composer quick action: screen -> chat, no LLM round-trip."""
        try:
            shot = await hub.screenshot()
        except Exception as e:  # noqa: BLE001 (RuntimeError when no executor online)
            await self.broadcast({"type": "error", "text": str(e)}, save=False)
            return
        await self.broadcast({"type": "image", "b64": shot["png_b64"],
                              "caption": "current screen"})

    # ---------------- the brain turn ----------------

    async def _think(self, user_text: str):
        async with self._lock:
            if self._notes:
                # ground truth about runner activity, inlined so the message
                # list stays template-legal (system only at the very start)
                user_text = ("[context — task updates the user already saw in "
                             "the chat: " + "; ".join(self._notes) + "]\n" + user_text)
                self._notes = []
            self.messages.append({"role": "user", "content": user_text})
            self._trim_ctx()
            await self.set_status("thinking…")
            try:
                await self._brain_loop()
            except Exception as e:  # noqa: BLE001 — surface, don't kill the socket
                print(f"[chat] brain turn failed: {type(e).__name__}: {e}", flush=True)
                await self.broadcast({"type": "error",
                                      "text": f"brain error: {type(e).__name__}: {e}"}, save=False)
            finally:
                # a freshly-started task owns the status bubble from here
                if runner.state != "running":
                    await self.set_status("")

    async def _brain_loop(self):
        for _ in range(6):
            resp = await self.client().chat.completions.create(
                model=BRAIN_MODEL,
                messages=self.messages,
                tools=TOOLS,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            m = resp.choices[0].message
            if m.tool_calls:
                self.messages.append({"role": "assistant", "content": m.content or "",
                                      "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name,
                                  "arguments": tc.function.arguments}}
                    for tc in m.tool_calls]})
                for tc in m.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except Exception:  # noqa: BLE001
                        args = {}
                    result = await self._tool(tc.function.name, args)
                    self.messages.append({"role": "tool", "tool_call_id": tc.id,
                                          "content": json.dumps(result)})
                continue
            text = (m.content or "").strip()
            if text:
                self.messages.append({"role": "assistant", "content": text})
                await self.broadcast({"type": "msg", "role": "assistant", "text": text})
            return
        await self.broadcast({"type": "error",
                              "text": "the brain looped on tools without answering"}, save=False)

    async def _tool(self, name: str, args: dict) -> dict:
        print(f"[chat] tool {name} {args}", flush=True)
        if name == "computer_task":
            instruction = str(args.get("instruction", "")).strip()
            if not instruction:
                return {"error": "no instruction provided"}
            if not hub.online():
                return {"error": "no computer is connected — the user needs to pair one first"}
            result = runner.start(instruction)
            if result.get("error"):
                return result
            return {"status": "started",
                    "note": ("progress, the outcome, and a final screenshot are posted "
                             "into the chat automatically — acknowledge briefly")}
        if name == "take_screenshot":
            try:
                shot = await hub.screenshot()
            except Exception as e:  # noqa: BLE001
                return {"error": str(e)}
            await self.broadcast({"type": "image", "b64": shot["png_b64"],
                                  "caption": "current screen"})
            return {"status": "sent",
                    "note": "the screenshot is now visible in the chat — refer to it briefly"}
        if name == "task_status":
            st = runner.status()
            last = st["steps"][-1] if st["steps"] else None
            return {"state": st["state"], "detail": st["detail"],
                    "steps_done": len(st["steps"]),
                    "last_action": last["action_text"] if last else None}
        if name == "cancel_task":
            return await runner.cancel()
        return {"error": f"unknown tool {name!r}"}

    def _trim_ctx(self):
        """Bound the brain context. Cut only at a user-message boundary so an
        assistant tool_calls entry is never orphaned from its tool results."""
        if len(self.messages) <= 1 + CTX_MAX:
            return
        body = self.messages[1:]
        cut = len(body) - CTX_MAX
        while cut < len(body) and body[cut].get("role") != "user":
            cut += 1
        self.messages = [self.messages[0]] + body[cut:]

    # ---------------- task-runner bridge ----------------

    async def on_task_event(self, kind: str, text: str, png_b64: str | None):
        if kind == "progress":
            await self.set_status(text)
            return
        label = {"finished": "✅ Done", "stuck": "🤚 I need your help",
                 "failed": "⚠️ Task failed"}.get(kind, kind)
        await self.set_status("")
        await self.broadcast({"type": "msg", "role": "assistant",
                              "text": f"{label} — {text}"})
        if png_b64:
            caption = {"finished": "how it ended up", "stuck": "where I'm stuck",
                       "failed": "the screen when it failed"}.get(kind, "")
            await self.broadcast({"type": "image", "b64": png_b64, "caption": caption})
        # ground truth for later turns (the brain never saw the runner act).
        # NOT appended as a system message — Qwen3.5's chat template rejects
        # system messages that aren't at the beginning ("System message must
        # be at the beginning", HTTP 400) and would poison every later turn.
        self._notes = self._notes[-4:] + [f"{kind}: {text}"]


chat = ChatHub()
task_runner.EVENT = chat.on_task_event  # call- and chat-started tasks both land here
router = APIRouter()


@router.websocket("/api/chat/ws")
async def chat_ws(ws: WebSocket):
    await ws.accept()
    # per-device passcode (shown by the executor) or the operator master key
    if not hub.code_ok(ws.query_params.get("code", "")):
        await ws.close(code=4401)   # the page shows the passcode prompt on this
        return
    chat.sockets.add(ws)
    try:
        await ws.send_text(json.dumps({
            "type": "history", "items": chat.history,
            "status": chat._status if runner.state == "running" else "",
        }))
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "text":
                text = (msg.get("text") or "").strip()
                if text:
                    await chat.on_text(text, ws)
            elif kind == "voice":
                if msg.get("b64"):
                    await chat.on_voice(msg["b64"], ws)
            elif kind == "screenshot":
                await chat.send_screenshot_now()
    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        print(f"[chat] socket error: {type(e).__name__}: {e}", flush=True)
    finally:
        chat.sockets.discard(ws)
