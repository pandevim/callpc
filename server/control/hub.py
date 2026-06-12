"""Control hub — server side of the executor control channel (docs/PROTOCOL.md).

Lives INSIDE the gateway process (FastAPI router included by gateway/app.py),
so executors connect outbound to the same public URL as the call webapp —
wss://<host>/control/ws — no extra port, tunnel, or service.

Responsibilities: pairing (single-use 6-char codes -> long-lived device
tokens, persisted in server/.state/devices.json), connection registry, and a
request/response correlator (`screenshot()` / `action()`) that the Phase-2
task runner calls. Debug REST endpoints let curl exercise the whole path
before the task runner exists.

Single-tenant by design for now: requests go to the one online executor
(token argument reserved for multi-device later).
"""
import asyncio
import base64
import json
import os
import secrets
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

STATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".state")
DEVICES_PATH = os.path.join(STATE_DIR, "devices.json")

PAIR_TTL_S = 600
REQUEST_TIMEOUT_S = 20.0


class ControlHub:
    def __init__(self):
        os.makedirs(STATE_DIR, exist_ok=True)
        self._devices: dict[str, dict] = self._load()  # token -> device record
        self._online: dict[str, dict] = {}              # token -> {"ws": WebSocket, "info": dict}
        self._pair_codes: dict[str, float] = {}         # code -> expiry monotonic
        self._pending: dict[int, asyncio.Future] = {}   # request id -> reply future
        self._next_id = 1

    # ---------------- persistence ----------------

    def _load(self) -> dict:
        try:
            with open(DEVICES_PATH) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        tmp = DEVICES_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self._devices, f, indent=2)
        os.replace(tmp, DEVICES_PATH)

    # ---------------- pairing ----------------

    def new_pair_code(self) -> str:
        # unambiguous alphabet (no 0/O/1/I)
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        code = "".join(secrets.choice(alphabet) for _ in range(6))
        self._pair_codes[code] = time.monotonic() + PAIR_TTL_S
        return code

    def _redeem_pair_code(self, code: str) -> bool:
        expiry = self._pair_codes.pop((code or "").strip().upper(), None)
        return expiry is not None and time.monotonic() < expiry

    # ---------------- registry ----------------

    def online(self) -> list[dict]:
        return [o["info"] for o in self._online.values()]

    def paired(self) -> list[dict]:
        return [{k: v for k, v in d.items() if k != "token"} for d in self._devices.values()]

    def _pick_online_token(self, token: str | None) -> str:
        if token:
            if token not in self._online:
                raise RuntimeError("requested device is not online")
            return token
        if not self._online:
            raise RuntimeError("no executor online")
        return next(iter(self._online))

    # ---------------- request/response to the executor ----------------

    async def request(self, msg: dict, token: str | None = None,
                      timeout: float = REQUEST_TIMEOUT_S) -> dict:
        """Send one correlated request to the online executor, await its reply."""
        token = self._pick_online_token(token)
        msg = dict(msg)
        msg["id"] = self._next_id
        self._next_id += 1
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[msg["id"]] = fut
        try:
            await self._online[token]["ws"].send_text(json.dumps(msg))
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(msg["id"], None)

    async def screenshot(self, token: str | None = None) -> dict:
        """-> {"png_b64":..., "w":..., "h":...} from the controlled computer."""
        reply = await self.request({"type": "screenshot_request"}, token)
        if reply.get("type") != "screenshot":
            raise RuntimeError(f"unexpected reply: {reply.get('type')}")
        return reply

    async def action(self, kind: str, args: dict, token: str | None = None) -> dict:
        """Execute one input action; -> {"ok": bool, "error": str|None}."""
        reply = await self.request({"type": "action", "kind": kind, "args": args}, token)
        if reply.get("type") != "action_result":
            raise RuntimeError(f"unexpected reply: {reply.get('type')}")
        return reply

    # ---------------- the websocket endpoint ----------------

    async def handle(self, ws: WebSocket):
        await ws.accept()
        token = None
        try:
            first = json.loads(await asyncio.wait_for(ws.receive_text(), 15))
            kind = first.get("type")

            if kind == "pair":
                if not self._redeem_pair_code(first.get("code", "")):
                    await ws.send_text(json.dumps({"type": "error", "error": "bad pair code"}))
                    return
                token = secrets.token_urlsafe(32)
                self._devices[token] = {
                    "name": first.get("name", "?"),
                    "os": first.get("os", "?"),
                    "distro": first.get("distro"),
                    "paired_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self._save()
                await ws.send_text(json.dumps({"type": "paired", "token": token}))
                print(f"[hub] paired new device: {self._devices[token]['name']} "
                      f"({self._devices[token]['os']})", flush=True)
            elif kind == "hello":
                token = first.get("token")
                if token not in self._devices:
                    await ws.send_text(json.dumps({"type": "error", "error": "unknown token"}))
                    return
                await ws.send_text(json.dumps({"type": "ready"}))
            else:
                await ws.send_text(json.dumps({"type": "error", "error": "expected pair or hello"}))
                return

            info = {
                "name": first.get("name", self._devices[token]["name"]),
                "os": first.get("os", self._devices[token]["os"]),
                "distro": first.get("distro", self._devices[token].get("distro")),
                "screen_w": first.get("screen_w"),
                "screen_h": first.get("screen_h"),
            }
            # one live connection per device: a reconnect replaces the old one
            self._online[token] = {"ws": ws, "info": info}
            print(f"[hub] executor online: {info['name']} ({info['os']}, "
                  f"{info['screen_w']}x{info['screen_h']}) — {len(self._online)} online", flush=True)

            while True:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                    continue
                fut = self._pending.get(msg.get("id"))
                if fut and not fut.done():
                    fut.set_result(msg)
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        except Exception as e:  # noqa: BLE001
            print(f"[hub] connection error: {type(e).__name__}: {e}", flush=True)
        finally:
            if token and self._online.get(token, {}).get("ws") is ws:
                self._online.pop(token, None)
                print(f"[hub] executor offline ({len(self._online)} online)", flush=True)


hub = ControlHub()
router = APIRouter()


@router.websocket("/control/ws")
async def control_ws(ws: WebSocket):
    await hub.handle(ws)


@router.post("/api/pair/new")
async def pair_new():
    return {"code": hub.new_pair_code(), "expires_in": PAIR_TTL_S}


@router.get("/api/devices")
async def devices():
    return {"online": hub.online(), "paired": hub.paired()}


# ---- debug endpoints: exercise the control path with curl (pre-task-runner) ----

@router.get("/api/control/screenshot")
async def control_screenshot():
    try:
        shot = await hub.screenshot()
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "w": shot.get("w"), "h": shot.get("h"),
            "png_bytes": len(base64.b64decode(shot.get("png_b64", "")))}


@router.post("/api/control/action")
async def control_action(req: dict):
    try:
        result = await hub.action(req.get("kind", ""), req.get("args", {}))
    except RuntimeError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": result.get("ok"), "error": result.get("error")}
