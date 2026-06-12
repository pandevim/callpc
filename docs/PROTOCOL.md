# callpc control protocol (server hub ⇄ executor)

JSON over a single WebSocket, platform-agnostic — mirrors the ASR protocol's
style (docs/ASR_PROTOCOL.md). The executor (the only thing installed on the
controlled computer) connects **outbound only** to the gateway's public URL:

    wss://<gateway-host>/control/ws        (same port/URL as the call webapp)

No inbound ports on the controlled machine. All intelligence is server-side;
the executor is a dumb I/O stub: receive an action → inject input / return a
screenshot.

## Pairing (first run, no token yet)

1. The user gets a 6-char pairing code from the server:
   `POST /api/pair/new` → `{"code": "AB3K9Q", "expires_in": 600}`
   (the landing page calls this and shows the code). This endpoint is OPEN —
   pairing offers up your *own* computer; it grants no control over anyone
   else's. Control is gated by the per-device passcode below.
2. The executor connects and sends:
   ```json
   {"type": "pair", "code": "AB3K9Q", "name": "<hostname>", "os": "windows|macos|linux",
    "distro": "Ubuntu 24.04.1 LTS", "screen_w": 2560, "screen_h": 1440}
   ```
   (`distro` is a human-readable OS string, mainly to show which Linux distro the
   controlled machine is; `/api/devices` surfaces it to the landing page.)
3. Server validates the code (single-use, TTL 600 s) and replies:
   `{"type": "paired", "token": "<long-lived device token>", "access_code": "XK4M2P"}`
   then treats the connection as online (no separate hello needed).
   Invalid/expired code → `{"type": "error", "error": "bad pair code"}` + close.

`access_code` is THIS device's passcode (TeamViewer model): the executor
persists it and **displays it on the controlled computer** — the only place
it's shown. Anyone wanting to call/chat/drive the computer must type it on
the web page. It is never served by `/api/devices` or the loopback beacon
(any local webpage could read that origin). The operator may additionally
set `ACCESS_CODE` on the gateway as a master key.

The executor persists the token locally and never pairs again.

## Session (every connection after pairing)

```json
-> {"type": "hello", "token": "...", "name": "<hostname>", "os": "...",
    "distro": "...", "screen_w": 2560, "screen_h": 1440}
<- {"type": "ready", "access_code": "XK4M2P"}   (or {"type":"error",...} + close)
```

`ready` re-sends the device passcode so executors paired before the feature
(or after a wiped config) pick theirs up and display it.

## Local presence beacon (executor → loopback HTTP, not the gateway)

So the call web page can tell it is open **on** the controlled computer (this
binary is running here) rather than on a phone, the executor also runs a tiny
HTTP server bound to **loopback only**: `127.0.0.1:8724` (override with
`--local-port` / `CALLPC_LOCAL_PORT`; `0` disables). The page fetches it:

```
GET http://127.0.0.1:8724/whoami
-> {"callpc": true, "name": "...", "os": "...", "distro": "...",
    "screen_w": 1920, "screen_h": 1080,
    "paired": true, "configured": true, "online": true}
```

`online` mirrors the live gateway WebSocket; `paired` is whether a device token
is stored; `configured` is whether a connect target (URL) is known. Every
response (and the `OPTIONS` preflight) carries CORS plus
`Access-Control-Allow-Private-Network: true`, so an HTTPS page may reach it
(loopback is a "potentially trustworthy" origin, so this is not mixed content).
The `/whoami` signal only refines the landing UI — it gates nothing — so a
browser that blocks it simply falls back to `/api/devices`.

### Auto-configure (so the user never types a URL or code)

If the agent is running but **unpaired** (`paired:false`), the page hands it the
gateway URL + a fresh pairing code; the agent then connects and pairs itself:

```
POST http://127.0.0.1:8724/configure   {"url": "wss://<host>/control/ws", "code": "AB3K9Q"}
-> {"ok": true}                         (or {"ok": false, "error": "already paired"})
```

The agent only honours `/configure` while it has no stored token, so a loopback
page can't re-target an already-paired agent. On success the URL is persisted, so
later runs reconnect with **no arguments at all**. Run with `--url`/`--code` to
skip auto-configure (scripted/headless), or `--local-port 0` to disable the beacon.

## Requests (server → executor; every request carries an `id`,
## every reply echoes it)

Screenshot:
```json
<- {"type": "screenshot_request", "id": 7}
-> {"type": "screenshot", "id": 7, "png_b64": "...", "w": 2560, "h": 1440}
```

Action (kinds match UI-TARS's action space; coordinates are **physical
screen pixels** — the task runner does the smart_resize mapping):
```json
<- {"type": "action", "id": 8, "kind": "click",        "args": {"x": 100, "y": 200}}
<- {"type": "action", "id": 9, "kind": "left_double",  "args": {"x": 100, "y": 200}}
<- {"type": "action", "id": 10, "kind": "right_single","args": {"x": 100, "y": 200}}
<- {"type": "action", "id": 11, "kind": "drag",        "args": {"x1":1,"y1":2,"x2":3,"y2":4}}
<- {"type": "action", "id": 12, "kind": "hotkey",      "args": {"keys": ["ctrl", "t"]}}
<- {"type": "action", "id": 13, "kind": "type",        "args": {"text": "hello"}}
<- {"type": "action", "id": 14, "kind": "scroll",      "args": {"x":1,"y":2,"direction":"down","amount":5}}
<- {"type": "action", "id": 15, "kind": "wait",        "args": {"ms": 1000}}
-> {"type": "action_result", "id": 8, "ok": true, "error": null}
```

Keepalive (either direction): `{"type":"ping"}` → `{"type":"pong"}`.

## Chat channel (webapp ⇄ gateway, `/api/chat/ws`)

Separate from the control channel above: this is the *user-facing* WhatsApp-style
thread served by `gateway/chat.py`. Auth is the device passcode (or the
operator master key) as a query param
(`/api/chat/ws?code=...`; HTTP middleware can't see WebSockets) — a wrong code
closes the socket with **4401** and the page shows the passcode prompt. All open
sockets share one conversation; the last ~150 items are replayed on connect as
`{"type":"history","items":[...],"status":...}` (older screenshots have their
`b64` pruned and carry `"expired": true`).

```jsonc
// client -> server
{"type": "text",  "text": "open firefox"}
{"type": "voice", "b64": "<webm/opus or mp4/aac>", "mime": "audio/webm"}
{"type": "screenshot"}                       // quick action, no LLM round-trip

// server -> client
{"type": "msg", "role": "assistant", "text": "...", "ts": 1718180000}
{"type": "image", "b64": "<png>", "caption": "how it ended up", "ts": ...}
{"type": "transcript", "text": "what the voice note said", "ts": ...}
{"type": "status", "text": "step 3 · clicking"}   // in-place bubble; "" hides it
{"type": "error", "text": "..."}                   // transient, not stored
```

Update policy: task progress streams into the single `status` bubble; durable
messages (+ a fresh screenshot) land only on milestones — finished, stuck
(needs user input), or failed. The bridge is `tasks.runner.EVENT`, so tasks
started from a voice call report into the chat too. Voice notes are decoded
with PyAV and transcribed by the same Nemotron engine (`OnnxASR.transcribe`).

## Safety (executor side, per PLAN.md)

Panic key kills the session; max-actions-per-minute rate limit; visible
"being controlled" indicator. The brain confirms destructive intents by voice
before dispatching.

## Testing without the Rust binary

`executor/fake_executor.py` speaks this protocol completely: pairs, answers
screenshots (a synthetic grid image), and prints actions instead of injecting
them. Debug endpoints on the gateway:
`GET /api/devices` (paired + online), `GET /api/control/screenshot`,
`POST /api/control/action {"kind":..., "args":...}`.
