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
   (later: the "Connect this computer" page calls this and shows the code).
2. The executor connects and sends:
   ```json
   {"type": "pair", "code": "AB3K9Q", "name": "<hostname>", "os": "windows|macos|linux",
    "distro": "Ubuntu 24.04.1 LTS", "screen_w": 2560, "screen_h": 1440}
   ```
   (`distro` is a human-readable OS string, mainly to show which Linux distro the
   controlled machine is; `/api/devices` surfaces it to the landing page.)
3. Server validates the code (single-use, TTL 600 s) and replies:
   `{"type": "paired", "token": "<long-lived device token>"}`
   then treats the connection as online (no separate hello needed).
   Invalid/expired code → `{"type": "error", "error": "bad pair code"}` + close.

The executor persists the token locally and never pairs again.

## Session (every connection after pairing)

```json
-> {"type": "hello", "token": "...", "name": "<hostname>", "os": "...",
    "distro": "...", "screen_w": 2560, "screen_h": 1440}
<- {"type": "ready"}                       (or {"type":"error",...} + close)
```

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
