"""UI-TARS task runner — the Phase-2 agent loop + REST API.

Loop: screenshot (via the control hub, from the paired executor) -> UI-TARS
(vLLM) -> parse Thought/Action -> map coords to screen pixels -> dispatch the
action back through the hub — until `finished()`, max steps, cancel, or error.

One task at a time (single GPU, single controlled computer). State + the full
step trail (screenshot, thought, action, dispatch result) are kept in memory
for the debug page (webapp/debug.html) and, later, the Phase-3 narration hook.

REST (router included by gateway/app.py):
    POST   /api/task {"instruction": ...}     start (409-style error if running)
    GET    /api/task                          state + step trail (no images)
    GET    /api/task/screenshot/{n}           step n's raw screenshot (png)
    DELETE /api/task                          cancel the running task
    POST   /api/calibrate {"cells": [...]}    coordinate-mapping accuracy test
                                              against the fake executor's grid

The calibrate check is PLAN.md's top-risk mitigation: ask UI-TARS to click
named 100-px grid cells on the fake executor's synthetic desktop and measure
how far the mapped pixel lands from each cell's true center — catches any
smart_resize mismatch immediately, before a real desktop is involved.
"""
import asyncio
import base64
import json
import os
import time

from fastapi import APIRouter
from fastapi.responses import Response

from control.hub import hub
from tasks.uitars_client import UITarsClient, box_center_px, to_protocol

MAX_STEPS = 15
# Per-task audit trail: screenshots + a transcript.json under server/.state/tasks
# (gitignored). Set TASK_LOG=0 to disable. Each run gets its own timestamped dir.
TASK_LOG = os.environ.get("TASK_LOG", "1") not in ("0", "false", "False", "")
LOG_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".state", "tasks")
NARRATE = None  # Phase 3: app.py sets this per call to a coroutine(text) that
                # queues a TTSSpeakFrame — spoken narration between actions.


async def _say(text: str):
    fn = NARRATE
    if fn:
        try:
            await fn(text)
        except Exception as e:  # noqa: BLE001 — narration must never kill a task
            print(f"[task] narration failed: {e}", flush=True)


def _phrase(kind: str, args: dict) -> str:
    """Short spoken phrase for an action (English — UI-TARS's own summaries
    sometimes come back in Chinese, which our TTS can't speak)."""
    if kind == "click":
        return "clicking"
    if kind == "left_double":
        return "double clicking"
    if kind == "right_single":
        return "right clicking"
    if kind == "type":
        text = args.get("text", "").strip()
        return f"typing {text[:40]}" if text else "typing"
    if kind == "hotkey":
        return "pressing " + " ".join(args.get("keys", []))
    if kind == "scroll":
        return "scrolling " + args.get("direction", "")
    if kind == "drag":
        return "dragging"
    if kind == "wait":
        return "waiting a moment"
    return kind


class TaskRunner:
    def __init__(self):
        self.client = UITarsClient()
        self.state = "idle"          # idle|running|finished|failed|cancelled|needs_user
        self.instruction = ""
        self.detail = ""
        self.steps: list[dict] = []  # {n, thought, action_text, dispatched, result, png_b64}
        self._turns: list[dict] = []  # model history: {"sent_b64", "response"}
        self._task: asyncio.Task | None = None
        self._last_phrase = ""       # narration dedupe
        self._suppressed = 0
        self._log_dir = None         # this run's audit dir (server/.state/tasks/<ts>)

    # ---------------- public API ----------------

    def status(self) -> dict:
        return {
            "state": self.state,
            "instruction": self.instruction,
            "detail": self.detail,
            "ts": int(time.time()),
            "steps": [{k: s[k] for k in
                       ("n", "thought", "action_text", "dispatched", "result")}
                      for s in self.steps],
        }

    def start(self, instruction: str) -> dict:
        if self._task and not self._task.done():
            return {"error": "a task is already running — cancel it first"}
        self.state, self.instruction, self.detail = "running", instruction, ""
        self.steps, self._turns = [], []
        self._last_phrase, self._suppressed = "", 0
        self._log_dir = None
        if TASK_LOG:
            try:
                self._log_dir = os.path.join(LOG_ROOT, time.strftime("%Y%m%d-%H%M%S"))
                os.makedirs(self._log_dir, exist_ok=True)
            except Exception as e:  # noqa: BLE001 — logging must never block a task
                print(f"[task] log dir failed: {e}", flush=True)
                self._log_dir = None
        self._task = asyncio.get_running_loop().create_task(self._run(instruction))
        return {"ok": True}

    def _write_log(self):
        """Persist the step trail (thoughts/actions/results) + screenshots for
        this run. Called as steps complete and at the end."""
        if not self._log_dir:
            return
        try:
            for s in self.steps:
                p = os.path.join(self._log_dir, f"step{s['n']:02d}.png")
                if "png_b64" in s and not os.path.exists(p):
                    with open(p, "wb") as f:
                        f.write(base64.b64decode(s["png_b64"]))
            with open(os.path.join(self._log_dir, "transcript.json"), "w") as f:
                json.dump({
                    "instruction": self.instruction,
                    "state": self.state,
                    "detail": self.detail,
                    "steps": [{k: s[k] for k in
                               ("n", "thought", "action_text", "dispatched", "result")}
                              for s in self.steps],
                }, f, indent=2)
        except Exception as e:  # noqa: BLE001
            print(f"[task] log write failed: {e}", flush=True)

    async def cancel(self) -> dict:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self.state, self.detail = "cancelled", "cancelled by user"
        return {"ok": True, "state": self.state}

    async def _narrate_action(self, pkind: str, args: dict):
        """Narrate signal, not noise: announce a phrase once when it changes
        (typing/keys always carry content so they always differ), and emit a
        'still working' heartbeat instead of repeating — the model may retry
        the same click several times (e.g. when the screen gives no feedback),
        which used to come out as 'clicking clicking clicking'."""
        phrase = _phrase(pkind, args)
        if phrase != self._last_phrase:
            self._last_phrase = phrase
            self._suppressed = 0
            await _say(phrase)
            return
        self._suppressed += 1
        if self._suppressed % 4 == 0:
            await _say("Still working on it.")

    # ---------------- the loop ----------------

    async def _run(self, instruction: str):
        try:
            for n in range(1, MAX_STEPS + 1):
                shot = await hub.screenshot()
                screen_w, screen_h = shot["w"], shot["h"]
                text, actions, sent_b64, _, _ = await self.client.predict(
                    instruction, self._turns, shot["png_b64"])

                step = {
                    "n": n,
                    "thought": (actions[0].get("thought") if actions else "") or "",
                    "action_text": "Action: " + text.split("Action:")[-1].strip()
                                   if "Action:" in text else text,
                    "dispatched": "",
                    "result": "",
                    "png_b64": shot["png_b64"],
                }
                self.steps.append(step)
                self._turns.append({"sent_b64": sent_b64, "response": text})
                print(f"[task] step {n}: {step['action_text'][:120]}", flush=True)
                self._write_log()  # persist incrementally so a crash still leaves a trail

                for action in actions:
                    kind = action["action_type"]
                    if kind == "finished":
                        self.state = "finished"
                        self.detail = action["action_inputs"].get("content", "done")
                        await _say("All done.")
                        return
                    if kind == "call_user":
                        self.state = "needs_user"
                        self.detail = "the agent asked for user input"
                        await _say("I need more direction to continue.")
                        return
                    mapped = to_protocol(action, screen_w, screen_h)
                    if mapped is None:
                        step["result"] += f" [unknown action {kind!r} skipped]"
                        continue
                    pkind, args = mapped
                    step["dispatched"] = f"{pkind} {args}"
                    await self._narrate_action(pkind, args)
                    if pkind == "wait":
                        await asyncio.sleep(args["ms"] / 1000)
                        step["result"] = "waited"
                        continue
                    result = await hub.action(pkind, args)
                    step["result"] = "ok" if result.get("ok") else f"FAILED: {result.get('error')}"
                    if not result.get("ok"):
                        self.state, self.detail = "failed", f"executor error: {result.get('error')}"
                        await _say("Something went wrong with the computer, I had to stop.")
                        return
                    await asyncio.sleep(0.8)  # let the UI settle before the next shot

            self.state, self.detail = "failed", f"no finished() after {MAX_STEPS} steps"
            await _say("I couldn't finish that task.")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — surface in the status, don't die silently
            self.state, self.detail = "failed", f"{type(e).__name__}: {e}"
            print(f"[task] failed: {self.detail}", flush=True)
            await _say("I hit a problem and had to stop the task.")
        finally:
            self._write_log()  # final state (finished/failed/needs_user) + all shots
            if self._log_dir:
                print(f"[task] trail saved -> {self._log_dir}", flush=True)


runner = TaskRunner()
router = APIRouter()


@router.post("/api/task")
async def task_start(req: dict):
    instruction = (req.get("instruction") or "").strip()
    if not instruction:
        return {"error": "instruction required"}
    return runner.start(instruction)


@router.get("/api/task")
async def task_status():
    return runner.status()


@router.delete("/api/task")
async def task_cancel():
    return await runner.cancel()


@router.get("/api/task/screenshot/{n}")
async def task_screenshot(n: int):
    for s in runner.steps:
        if s["n"] == n:
            return Response(base64.b64decode(s["png_b64"]), media_type="image/png")
    return Response(status_code=404)


# ---------------- coordinate calibration (top-risk check) ----------------

@router.post("/api/calibrate")
async def calibrate(req: dict):
    """For each grid cell (fake executor's 100-px labeled grid), ask UI-TARS to
    click its center with NO history, map to pixels, and measure the error
    against the true center. err <= 50 px = landed inside the cell."""
    cells = req.get("cells") or ["B1", "F3", "K6", "D7"]
    client = UITarsClient(history_n=1)
    results = []
    for cell in cells:
        col, row = ord(cell[0].upper()) - 65, int(cell[1:])
        expect = (col * 100 + 50, row * 100 + 50)
        try:
            shot = await hub.screenshot()
            text, actions, _, _, _ = await client.predict(
                f"Click the exact center of the grid cell labeled {cell}.", [], shot["png_b64"])
            clicks = [a for a in actions if a["action_type"] in
                      ("click", "left_double", "right_single")]
            if not clicks:
                results.append({"cell": cell, "error": f"no click action ({text[:80]!r})"})
                continue
            got = box_center_px(clicks[0]["action_inputs"], "start_box", shot["w"], shot["h"])
            err = round(((got[0] - expect[0]) ** 2 + (got[1] - expect[1]) ** 2) ** 0.5, 1)
            results.append({"cell": cell, "expected": expect, "got": got,
                            "err_px": err, "pass": err <= 50})
        except Exception as e:  # noqa: BLE001
            results.append({"cell": cell, "error": f"{type(e).__name__}: {e}"})
    passed = [r for r in results if r.get("pass")]
    return {"pass": len(passed) == len(cells),
            "passed": f"{len(passed)}/{len(cells)}", "cells": results}
