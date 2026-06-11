#!/usr/bin/env python3
"""Fake callpc-executor — protocol-complete stand-in for the Rust binary.

Speaks docs/PROTOCOL.md end to end: pairs (or hellos with a stored token),
answers screenshot requests with a synthetic desktop image, and PRINTS actions
instead of injecting them. Use it to build/test the control hub and the
UI-TARS task runner before (and alongside) the real executor.

    pip install websockets            # (+ pillow for a useful grid screenshot)
    python fake_executor.py --url wss://<public-host>/control/ws --code AB3K9Q
    python fake_executor.py --url ws://127.0.0.1:8723/control/ws   # token saved from last pair

The synthetic screenshot is a labeled grid (needs pillow) — also the basis of
the coordinate-calibration test (PLAN.md verification): ask UI-TARS to click a
named cell and assert where the click lands. Without pillow it returns a tiny
solid PNG (protocol still exercised).

Token persists in ~/.config/callpc/executor.json. Reconnects with backoff.
"""
import argparse
import asyncio
import base64
import io
import json
import os
import platform
import socket
import sys

CONF_DIR = os.path.join(os.path.expanduser("~"), ".config", "callpc")
CONF_PATH = os.path.join(CONF_DIR, "executor.json")

SCREEN_W, SCREEN_H = 1280, 800

# 1x1 dark-blue PNG — fallback when pillow isn't installed
_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkqGeoBwAB"
    "hAD/Y9Y0nQAAAABJRU5ErkJggg==")


def _log(m):
    print(m, file=sys.stderr, flush=True)


def load_conf() -> dict:
    try:
        with open(CONF_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def save_conf(conf: dict):
    os.makedirs(CONF_DIR, exist_ok=True)
    with open(CONF_PATH, "w") as f:
        json.dump(conf, f, indent=2)


def make_screenshot() -> bytes:
    """Labeled grid 'desktop' (pillow), or a tiny PNG without it."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return _TINY_PNG
    img = Image.new("RGB", (SCREEN_W, SCREEN_H), (24, 28, 44))
    d = ImageDraw.Draw(img)
    step = 100
    for x in range(0, SCREEN_W, step):
        d.line([(x, 0), (x, SCREEN_H)], fill=(60, 70, 100))
    for y in range(0, SCREEN_H, step):
        d.line([(0, y), (SCREEN_W, y)], fill=(60, 70, 100))
    for x in range(0, SCREEN_W, step):
        for y in range(0, SCREEN_H, step):
            d.text((x + 6, y + 4), f"{chr(65 + x // step)}{y // step}", fill=(160, 175, 210))
    d.text((SCREEN_W // 2 - 80, SCREEN_H // 2), "callpc fake desktop", fill=(110, 231, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


async def run(url: str, code: str | None):
    from websockets.asyncio.client import connect

    conf = load_conf()
    if code and conf.get("token"):
        # already paired — a reconnect must NOT replay the (single-use) code:
        # seen live, a job restart mid-session led to "bad pair code" + exit.
        _log("[exec] token already stored — ignoring --code (delete "
             f"{CONF_PATH} to re-pair)")
        code = None
    info = {
        "name": socket.gethostname(),
        "os": {"Darwin": "macos", "Windows": "windows"}.get(platform.system(), "linux"),
        "screen_w": SCREEN_W,
        "screen_h": SCREEN_H,
    }

    async with connect(url, max_size=None) as ws:
        if code:
            await ws.send(json.dumps({"type": "pair", "code": code, **info}))
            reply = json.loads(await ws.recv())
            if reply.get("type") != "paired":
                _log(f"[exec] pairing failed: {reply}")
                return 1
            conf["token"] = reply["token"]
            save_conf(conf)
            _log(f"[exec] paired! token stored in {CONF_PATH}")
        else:
            if not conf.get("token"):
                _log("[exec] no stored token — run once with --code <pairing code>")
                return 1
            await ws.send(json.dumps({"type": "hello", "token": conf["token"], **info}))
            reply = json.loads(await ws.recv())
            if reply.get("type") != "ready":
                _log(f"[exec] hello rejected: {reply} — re-pair with --code")
                return 1
        _log(f"[exec] online as {info['name']} ({info['os']}) — waiting for requests")

        async for raw in ws:
            msg = json.loads(raw)
            kind = msg.get("type")
            if kind == "screenshot_request":
                png = make_screenshot()
                await ws.send(json.dumps({
                    "type": "screenshot", "id": msg["id"],
                    "png_b64": base64.b64encode(png).decode(),
                    "w": SCREEN_W, "h": SCREEN_H,
                }))
                _log(f"[exec] screenshot -> {len(png)} bytes")
            elif kind == "action":
                _log(f"[exec] ACTION {msg.get('kind')} {json.dumps(msg.get('args', {}))}")
                if msg.get("kind") == "wait":
                    await asyncio.sleep(min(msg.get("args", {}).get("ms", 0), 5000) / 1000)
                await ws.send(json.dumps({
                    "type": "action_result", "id": msg["id"], "ok": True, "error": None}))
            elif kind == "pong":
                pass
    return 0


async def main():
    ap = argparse.ArgumentParser(description="fake callpc executor (protocol test double)")
    ap.add_argument("--url", default=os.environ.get("CALLPC_URL", "ws://127.0.0.1:8723/control/ws"))
    ap.add_argument("--code", default=None, help="pairing code (first run only)")
    ap.add_argument("--once", action="store_true", help="exit instead of reconnecting")
    args = ap.parse_args()

    backoff = 2
    while True:
        try:
            rc = await run(args.url, args.code)
            args.code = None  # paired — reconnects use the stored token
            if rc:
                return rc
            backoff = 2  # clean disconnect: server restarted etc.
        except Exception as e:  # noqa: BLE001
            _log(f"[exec] connection lost ({type(e).__name__}: {e})")
        if args.once:
            return 0
        _log(f"[exec] reconnecting in {backoff}s...")
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2, 60)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
