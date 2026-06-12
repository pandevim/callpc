"""Brain wiring (Phase 3) — persona + tool schemas + handlers.

The Qwen brain drives the computer through three tools bridged to the Phase-2
task runner: computer_task (start), task_status (poll), cancel_task (stop).
vLLM needs `--enable-auto-tool-choice --tool-call-parser qwen3_coder`
server-side (callpc.sbatch sets it).

Pipecat 1.3 contract (verified): tools go into LLMContext as a ToolsSchema of
FunctionSchemas; handlers receive FunctionCallParams and report through
params.result_callback(result) — the LLM is automatically re-run with the
result appended, so it acknowledges out loud ("Okay, I'm opening Slack now")
without extra plumbing. Step-by-step narration is the task runner's job
(tasks/runner.py NARRATE hook -> TTSSpeakFrame), not the LLM's.
"""
import os

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

from control.hub import hub
from tasks.runner import runner

BRAIN_BASE_URL = os.environ.get("BRAIN_BASE_URL", "http://127.0.0.1:8001/v1")
BRAIN_MODEL = os.environ.get("BRAIN_MODEL", "brain")
BRAIN_API_KEY = os.environ.get("BRAIN_API_KEY", "x")

PERSONA = os.environ.get("PERSONA", (
    "You are callpc, a friendly assistant the user reached by phoning their computer. "
    "You are speaking out loud: keep replies to one or two short, natural sentences — "
    "no lists, no markdown, no emoji.\n\n"
    "CRITICAL: You control the computer ONLY by calling the computer_task tool. You "
    "cannot open apps, click, type, or browse by yourself. Whenever the user asks for "
    "ANYTHING to be done on the computer — open an app, go to a website, click, type, "
    "search, play something — you MUST call computer_task with one clear instruction. "
    "NEVER say you are doing it, opening it, or that it's done unless you actually "
    "called the tool in the same turn. Do not narrate or claim an action you did not "
    "dispatch. If you catch yourself about to say 'I'm opening...' or 'done', call "
    "computer_task instead.\n\n"
    "computer_task takes ONE self-contained instruction, e.g. 'Open Firefox and search "
    "for tickets to Boston'. Answer ordinary questions (not about the computer) yourself "
    "without the tool. If the user asks what's on the screen, what's open or running, or "
    "what's going on on the computer, call look_at_screen with their question and tell "
    "them what it returns — NEVER use computer_task just to inspect, and never guess. "
    "While a task runs, progress is narrated to the user automatically "
    "— don't invent progress; use task_status if they ask how it's going, cancel_task if "
    "they say stop. For destructive or irreversible actions — deleting, sending messages, "
    "payments — say what you'll do and get a clear yes before calling computer_task. If "
    "no computer is connected, say so plainly."
))


def build_tools() -> ToolsSchema:
    return ToolsSchema(standard_tools=[
        FunctionSchema(
            name="computer_task",
            description=(
                "Perform a task on the user's computer (GUI automation: open apps, "
                "click, type, browse). Provide one complete, self-contained "
                "instruction describing the end goal."),
            properties={
                "instruction": {
                    "type": "string",
                    "description": "The full task, e.g. 'Open Slack and read the latest "
                                   "message in the development channel.'",
                },
            },
            required=["instruction"],
        ),
        FunctionSchema(
            name="look_at_screen",
            description=("Look at the computer's screen right now and describe it. Use "
                         "for any question about what's open, running, or visible — it "
                         "does not click or change anything."),
            properties={
                "question": {
                    "type": "string",
                    "description": "What the user wants to know about the screen.",
                },
            },
            required=[],
        ),
        FunctionSchema(
            name="task_status",
            description="Check how the current (or last) computer task is going.",
            properties={},
            required=[],
        ),
        FunctionSchema(
            name="cancel_task",
            description="Stop the computer task that is currently running.",
            properties={},
            required=[],
        ),
    ])


async def _computer_task(params: FunctionCallParams):
    instruction = str(params.arguments.get("instruction", "")).strip()
    if not instruction:
        await params.result_callback({"error": "no instruction provided"})
        return
    if not hub.online():
        await params.result_callback(
            {"error": "no computer is connected — the user needs to pair one first"})
        return
    result = runner.start(instruction)
    if result.get("error"):
        await params.result_callback(result)
        return
    print(f"[brain] computer_task started: {instruction!r}", flush=True)
    await params.result_callback({
        "status": "started",
        "note": ("The task is running; progress is narrated to the user "
                 "automatically. Acknowledge briefly."),
    })


_VISION = None  # lazy AsyncOpenAI client for one-shot screen description


async def _look_at_screen(params: FunctionCallParams):
    """Voice-call eyes: screenshot -> one-shot vision call against the same
    Qwen vLLM -> short spoken-style description returned to the call brain.
    The raw screenshot also lands in the chat, as the call's paper trail."""
    question = str(params.arguments.get("question", "") or "what is on the screen")
    if not hub.online():
        await params.result_callback(
            {"error": "no computer is connected — the user needs to pair one first"})
        return
    try:
        shot = await hub.screenshot()
    except Exception as e:  # noqa: BLE001
        await params.result_callback({"error": f"couldn't capture the screen: {e}"})
        return
    import chat as chat_mod  # imported lazily: brain is imported before chat in app.py
    try:
        await chat_mod.chat.broadcast({"type": "image", "b64": shot["png_b64"],
                                       "caption": "current screen (asked on the call)"})
    except Exception:  # noqa: BLE001 — chat mirror is best-effort
        pass
    global _VISION
    if _VISION is None:
        from openai import AsyncOpenAI
        _VISION = AsyncOpenAI(base_url=BRAIN_BASE_URL, api_key=BRAIN_API_KEY)
    try:
        b64 = chat_mod._shrink_for_brain(shot["png_b64"])
        resp = await _VISION.chat.completions.create(
            model=BRAIN_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text":
                    "This is the user's computer screen. Answer for speech (two or "
                    "three short plain sentences, no lists): " + question},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ]}],
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            max_tokens=200,
        )
        desc = (resp.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        await params.result_callback({"error": f"couldn't analyze the screen: {e}"})
        return
    await params.result_callback({
        "screen_description": desc,
        "note": "a screenshot was also posted into the user's chat thread",
    })


async def _task_status(params: FunctionCallParams):
    st = runner.status()
    last = st["steps"][-1] if st["steps"] else None
    await params.result_callback({
        "state": st["state"],
        "detail": st["detail"],
        "steps_done": len(st["steps"]),
        "last_action": last["action_text"] if last else None,
    })


async def _cancel_task(params: FunctionCallParams):
    await params.result_callback(await runner.cancel())


def register_handlers(llm):
    llm.register_function("computer_task", _computer_task)
    llm.register_function("look_at_screen", _look_at_screen)
    llm.register_function("task_status", _task_status)
    llm.register_function("cancel_task", _cancel_task)
