"""UI-TARS-1.5 client — OpenAI-compatible calls to vLLM + coordinate mapping.

Implements the OFFICIAL drive pattern for the open-weights UI-TARS-1.5-7B
(verified June 2026 from bytedance/UI-TARS + OSWorld's mm_agents/uitars15_v1):

- Prompt: the qwen25vl_normal action space (start_box='<|box_start|>(x,y)
  <|box_end|>'); the model emits bare `click(start_box='(607,647)')`.
- Coordinates: ABSOLUTE pixels in the smart_resize'd image (Qwen2.5-VL).
  `parse_action_to_structure_output(model_type="qwen25vl")` recomputes
  smart_resize over the dims of the image WE SENT and returns boxes
  normalized 0-1; we map to screen pixels ourselves:
  px = (x1+x2)/2 * screen_w. (The package's pyautogui helper has a
  scale_factor mismatch in 0.5.x — do not use it.)
- History: per-turn screenshot (user) + raw Thought/Action (assistant, with
  add_box_token re-applied at send time); turns older than the image window
  keep their text only. Last `history_n` images total, current included.
- Sampling (benchmark-reproducing): temperature 1.0, top_p 0.9,
  frequency_penalty 1, max_tokens 500; retry up to 3x on parse failure.
- Images are pre-shrunk to UITARS_MAX_PIXELS (default 1920x1080-equivalent,
  ~660 vision tokens) so 5-image history fits the 8k serving context;
  aspect is preserved, so the normalized coords stay valid on the original.
"""
import ast
import base64
import io
import math
import os

from openai import AsyncOpenAI
from PIL import Image
from ui_tars.action_parser import add_box_token, parse_action_to_structure_output

UITARS_BASE_URL = os.environ.get("UITARS_BASE_URL", "http://127.0.0.1:8000/v1")
UITARS_MODEL = os.environ.get("UITARS_MODEL", "ui-tars")
# A 1080p desktop screenshot is ~2700 vision tokens (factor-28 smart_resize), so
# two of them already blew past UI-TARS's 8k serving context. Cap to 720p-equiv
# (~1200 tokens/image): 4-image history + prompt ≈ 6k, comfortably under 8192.
# Plenty of resolution for UI targeting; raise if you also raise UITARS_MAXLEN.
MAX_PIXELS = int(os.environ.get("UITARS_MAX_PIXELS", str(1280 * 720)))
MIN_PIXELS = 100 * 28 * 28
HISTORY_N = int(os.environ.get("UITARS_HISTORY_N", "4"))
LANGUAGE = os.environ.get("UITARS_LANGUAGE", "English")

# OSWorld uitars15_v1 qwen25vl_normal prompt, verbatim (language param ours).
PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use \"\\n\" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format.

## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""


def prepare_image(png_b64: str) -> tuple[str, int, int]:
    """RGB-convert + cap to MAX_PIXELS (sqrt scale, aspect preserved).
    Returns (png_b64_sent, sent_w, sent_h)."""
    img = Image.open(io.BytesIO(base64.b64decode(png_b64)))
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.width * img.height > MAX_PIXELS:
        f = math.sqrt(MAX_PIXELS / (img.width * img.height))
        img = img.resize((int(img.width * f), int(img.height * f)))
    if img.width * img.height < MIN_PIXELS:
        f = math.sqrt(MIN_PIXELS / (img.width * img.height))
        img = img.resize((math.ceil(img.width * f), math.ceil(img.height * f)))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), img.width, img.height


def box_center_px(action_inputs: dict, key: str, screen_w: int, screen_h: int):
    """Parsed boxes are stringified [x1,y1,x2,y2] normalized 0-1 -> screen px."""
    box = ast.literal_eval(action_inputs[key])
    return (round((box[0] + box[2]) / 2 * screen_w),
            round((box[1] + box[3]) / 2 * screen_h))


def to_protocol(action: dict, screen_w: int, screen_h: int):
    """Map one parsed UI-TARS action -> (kind, args) for the control protocol
    (docs/PROTOCOL.md). Returns None for non-dispatchable actions
    (finished / call_user — the runner handles those)."""
    t = action["action_type"]
    ai = action["action_inputs"]
    if t in ("click", "left_double", "right_single"):
        x, y = box_center_px(ai, "start_box", screen_w, screen_h)
        return t, {"x": x, "y": y}
    if t == "drag":
        x1, y1 = box_center_px(ai, "start_box", screen_w, screen_h)
        x2, y2 = box_center_px(ai, "end_box", screen_w, screen_h)
        return "drag", {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    if t == "hotkey":
        return "hotkey", {"keys": ai.get("key", "").split()}
    if t == "type":
        return "type", {"text": ai.get("content", "")}
    if t == "scroll":
        if "start_box" in ai:
            x, y = box_center_px(ai, "start_box", screen_w, screen_h)
        else:
            x, y = screen_w // 2, screen_h // 2
        return "scroll", {"x": x, "y": y,
                          "direction": ai.get("direction", "down"), "amount": 5}
    if t == "wait":
        return "wait", {"ms": 5000}
    return None  # finished / call_user / unknown


class UITarsClient:
    """One conversation turn = predict(instruction, turns, current screenshot).
    `turns` = [{"sent_b64": ..., "response": raw Thought/Action text}, ...]."""

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 history_n: int = HISTORY_N):
        self._client = AsyncOpenAI(base_url=base_url or UITARS_BASE_URL, api_key="x")
        self._model = model or UITARS_MODEL
        self._history_n = history_n

    @staticmethod
    def _img_msg(b64: str) -> dict:
        return {"role": "user", "content": [{"type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}}]}

    def _messages(self, instruction: str, turns: list[dict], cur_b64: str) -> list[dict]:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
            {"role": "user", "content": [{"type": "text", "text": PROMPT.format(
                language=LANGUAGE, instruction=instruction)}]},
        ]
        # last (history_n - 1) turn images + the current one = history_n images;
        # older turns keep their assistant text only (official scheme).
        first_with_image = len(turns) - (self._history_n - 1)
        for i, t in enumerate(turns):
            if i >= first_with_image:
                messages.append(self._img_msg(t["sent_b64"]))
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": add_box_token(t["response"])}]})
        messages.append(self._img_msg(cur_b64))
        return messages

    async def predict(self, instruction: str, turns: list[dict], png_b64: str):
        """-> (raw_text, parsed_actions, sent_b64, sent_w, sent_h). Raises
        RuntimeError after 3 unparseable responses."""
        sent_b64, w, h = prepare_image(png_b64)
        messages = self._messages(instruction, turns, sent_b64)
        last_err = None
        for _ in range(3):
            resp = await self._client.chat.completions.create(
                model=self._model, messages=messages,
                temperature=1.0, top_p=0.9, frequency_penalty=1, max_tokens=500)
            text = (resp.choices[0].message.content or "").strip()
            try:
                actions = parse_action_to_structure_output(
                    text, 1000, origin_resized_height=h, origin_resized_width=w,
                    model_type="qwen25vl")
                return text, actions, sent_b64, w, h
            except Exception as e:  # noqa: BLE001 — retry with fresh sampling
                last_err = f"{type(e).__name__}: {e} (response: {text[:200]!r})"
        raise RuntimeError(f"UI-TARS response unparseable after 3 tries — {last_err}")
