# Engine protocol (the portable boundary)

The speech engine is **platform-agnostic** and speaks one protocol: a local
**WebSocket**. Any frontend — the Linux tray app here, a future browser
extension, a Windows app, another distro — drives the same engine by
implementing this contract. The engine has no knowledge of keyboards, ydotool,
GNOME, or the OS.

```
┌────────────────────────────┐         ┌──────────────────────────────┐
│ FRONTEND (platform adapter)│  ws://  │ ENGINE (core/engine.py)       │
│  - capture microphone      │ ──────▶ │  Nemotron streaming ASR       │
│  - inject recognized text  │ ◀────── │  audio in → text out          │
└────────────────────────────┘         └──────────────────────────────┘
```

## Endpoint

`ws://127.0.0.1:8765` (port configurable via `ASR_PORT`).

## Messages

Text frames are JSON; binary frames are raw audio.

### 1. Start an utterance — client → engine (JSON)
```json
{ "type": "start", "lang": "en-US" }
```
`lang` is a prompt key the model knows (`en-US`, `en-GB`, `es-ES`, `de-DE`,
`fr-FR`, `hi-IN`, `zh-CN`, `ja-JP`, … 40 locales).

### 2. Ready — engine → client (JSON)
```json
{ "type": "ready", "chunk_bytes": 5120, "sample_rate": 16000 }
```
or, if another client is already streaming (single GPU):
```json
{ "type": "error", "error": "busy" }
```
`chunk_bytes` = the exact size of each audio frame the engine expects
(`chunk_bytes / 2` int16 samples = one streaming step).

### 3. Stream audio — client → engine (BINARY)
Each binary frame is **raw signed 16-bit little-endian, mono, 16 kHz PCM**,
`chunk_bytes` long (pad the final short frame with zeros). Send frames as fast
as the mic produces them.

### 4. Incremental text — engine → client (JSON), one per audio frame
```json
{ "type": "text", "text": " curiosity" }
```
`text` is the **new** text recognized from that frame (may be empty, a sub-word,
or several words). Concatenate in order to build the transcript. Append it to the
target input as it arrives. Language tags (`<en-US>`) are already stripped.

### 5. End the utterance — client → engine (JSON)
```json
{ "type": "stop" }
```
Engine resets its streaming state and replies `{"type":"stopped"}`. The same
connection may start another utterance.

### Errors — engine → client (JSON)
```json
{ "type": "error", "error": "<message>" }
```

## Notes for porting

- **Browser extension:** capture with `getUserMedia` → an `AudioWorklet`
  downsamples to 16 kHz mono int16 → `ws.send(buffer)`; insert `text` into the
  focused field via the DOM / `execCommand`/`InputEvent`.
- **Windows / other OS:** capture via the OS audio API; inject via the OS input
  API (e.g. SendInput on Windows). The WebSocket exchange is identical.
- The engine serializes to **one streamer at a time** (one GPU). Frontends should
  handle `{"error":"busy"}`.
- Reference frontend: `linux/client.py` (≈150 lines) — keyboard + `pw-record` +
  `ydotool`, nothing engine-specific.
