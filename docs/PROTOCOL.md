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
    "screen_w": 2560, "screen_h": 1440}
   ```
3. Server validates the code (single-use, TTL 600 s) and replies:
   `{"type": "paired", "token": "<long-lived device token>"}`
   then treats the connection as online (no separate hello needed).
   Invalid/expired code → `{"type": "error", "error": "bad pair code"}` + close.

The executor persists the token locally and never pairs again.

## Session (every connection after pairing)

```json
-> {"type": "hello", "token": "...", "name": "<hostname>", "os": "...",
    "screen_w": 2560, "screen_h": 1440}
<- {"type": "ready"}                       (or {"type":"error",...} + close)
```

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
