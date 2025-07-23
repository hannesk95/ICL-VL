"""
model.py – unified VLM backend for Gemini + (single-GPU) LLaVA-HF
last updated 2025-07-23 (c patch 8)

What's new in this patch:
• Robust JSON extraction: regex grabs the last {...} block, basic repairs.
• Hard guard: for binary tasks, never return "Unknown"; pick a class.
• Default scores repaired to be in [0,1] and sum≈1 when missing.
• Prompt (LLaVA) reminds: never output Unknown.

Gemini path unchanged.
"""

from __future__ import annotations

import io, os, re, json, time
from collections import deque
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch
import PIL.Image as Image


# ──────────────────────────────────────────────────────────────────────
# Helpers & globals
# ──────────────────────────────────────────────────────────────────────
def _snake(text: str) -> str:
    return "_".join(text.lower().split())


class VLMBackend(str, Enum):
    GEMINI = "gemini"
    LLAVA_HF = "llava_hf"


_BACKEND: VLMBackend | None = None
_GENERATION_CFG: Dict[str, Any] = {}
_HF_PROC_MODEL = None
_MODEL_NAME = "gemini-2.0-flash"

# Gemini token bucket
_MAX_CALLS, _PERIOD = 16, 80.0
_CALLS: deque[float] = deque()
_LOCK = Lock()
RL_VERBOSE = bool(os.getenv("RL_VERBOSE", "1"))


def _acquire_token() -> None:
    while True:
        with _LOCK:
            now = time.time()
            while _CALLS and now - _CALLS[0] > _PERIOD:
                _CALLS.popleft()
            if len(_CALLS) < _MAX_CALLS:
                _CALLS.append(now)
                if RL_VERBOSE:
                    print(f"[rate-limit] token ok → {_MAX_CALLS-len(_CALLS)} left")
                return
            wait = _PERIOD - (now - _CALLS[0]) + 0.05
        time.sleep(wait)


def _rate_limited(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        _acquire_token()
        return fn(*a, **kw)
    return wrapper


def configure_gemini(model_cfg: dict | None = None) -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("set GEMINI_API_KEY in your environment")
    genai.configure(api_key=api_key)

    global _MODEL_NAME, _GENERATION_CFG
    if model_cfg:
        _MODEL_NAME = model_cfg.get("model_name", _MODEL_NAME)
        _GENERATION_CFG = model_cfg.get("model_kwargs", {})


def load_prompt_text() -> str:
    with open(os.getenv("PROMPT_PATH", "./prompts/few_shot.txt"), encoding="utf-8") as fp:
        return fp.read()


@_rate_limited
def gemini_api_call(contents,
                    classification_type: str = "binary",
                    label_list: List[str] | None = None):
    model = genai.GenerativeModel(_MODEL_NAME,
                                  generation_config=_GENERATION_CFG)
    raw = model.generate_content(contents).text.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.I).strip()
    try:
        pred = json.loads(raw)
    except Exception:
        pred = {"thoughts": "Unable to parse response", "answer": "Unknown"}

    if classification_type == "binary" and label_list and len(label_list) == 2:
        pos_key = f"score_{_snake(label_list[0])}"
        neg_key = f"score_{_snake(label_list[1])}"
        pred.setdefault(pos_key, -1)
        pred.setdefault(neg_key, -1)
        pred.setdefault("location", None)
    else:
        pred.setdefault("score", -1.0)
    return pred


# ──────────────────────────────────────────────────────────────────────
# LLaVA helpers
# ──────────────────────────────────────────────────────────────────────
def _setup_llava_hf(model_id: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for LLaVA backend")

    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map={"": 0},
        quantization_config=qcfg,
        trust_remote_code=True,
    )
    model.eval()
    return proc, model


def _map_label(free_text: str, label_list: List[str] | None):
    if not label_list:
        return free_text
    text = re.sub(r"[^a-z0-9]+", " ", free_text.lower()).strip()
    for lbl in label_list:
        if text == lbl.lower():
            return lbl
    if len(label_list) == 2:
        tum, no = label_list
        patterns = [
            (r"\bclass\s*1\b", tum),
            (r"\blow\s*grade\b", tum),
            (r"\blgg\b", tum),
            (r"\bclass\s*2\b", no),
            (r"\bhigh\s*grade\b", no),
            (r"\bhgg\b", no),
            (r"\bno\s*tumou?r\b", no),
            (r"\bbenign\b", no),
        ]
        for pat, mapped in patterns:
            if re.search(pat, text):
                return mapped
    return "Unknown"


def _attempt_json_fix(text: str) -> Dict[str, Any] | None:
    """
    Try harder to extract JSON:
    • take the last {...} block
    • replace single quotes with double if needed
    • remove trailing commas
    """
    m = None
    # find all JSON-like blocks
    for match in re.finditer(r"\{[\s\S]*\}", text):
        m = match  # keep last
    if not m:
        return None
    candidate = m.group(0).strip()

    # Quick repairs
    cand = candidate
    # fix single quotes (only on keys) – naive but works often
    cand = re.sub(r"(?<=\{|,)\s*'([^']+)'\s*:", r'"\1":', cand)
    cand = re.sub(r":\s*'([^']+)'", lambda m: ':"{}"'.format(m.group(1).replace('"', '\\"')), cand)
    # remove trailing commas before } or ]
    cand = re.sub(r",\s*(\}|\])", r"\1", cand)

    try:
        return json.loads(cand)
    except Exception:
        return None


def _repair_binary_output(data: Dict[str, Any],
                          label_list: List[str]) -> Dict[str, Any]:
    """Ensure answer in {label_list}, scores in [0,1], sum≈1."""
    pos, neg = label_list
    pk, nk = f"score_{_snake(pos)}", f"score_{_snake(neg)}"

    # fix scores
    s_pos = float(data.get(pk, -1))
    s_neg = float(data.get(nk, -1))
    if not (0.0 <= s_pos <= 1.0) or not (0.0 <= s_neg <= 1.0):
        if 0 <= s_pos <= 1 and s_neg == -1:
            s_neg = 1 - s_pos
        elif 0 <= s_neg <= 1 and s_pos == -1:
            s_pos = 1 - s_neg
        else:
            # default to 0.5/0.5
            s_pos = 0.5
            s_neg = 0.5
    # normalise
    tot = s_pos + s_neg
    if tot > 0:
        s_pos, s_neg = s_pos / tot, s_neg / tot

    data[pk], data[nk] = round(s_pos, 4), round(s_neg, 4)

    # fix answer
    ans = data.get("answer", "")
    if ans not in label_list:
        data["answer"] = pos if s_pos >= s_neg else neg

    # ensure thoughts exists
    data.setdefault("thoughts", "")
    data.setdefault("location", None)
    return data


def _extract_llava_json(resp: str,
                        classification_type: str,
                        label_list: List[str] | None):
    # strip code fences
    cleaned = re.sub(r"^```(?:json)?|```$", "", resp, flags=re.I).strip()
    cleaned = cleaned.replace(r"\_", "_")

    data = None
    try:
        data = json.loads(cleaned)
    except Exception:
        data = _attempt_json_fix(cleaned)

    if data is None:
        if os.getenv("DEBUG_LAVA", "0") == "1":
            print("[LLaVA raw no-json]", cleaned)
        data = {
            "thoughts": "Free-form reply; JSON parse failed.",
            "answer": _map_label(cleaned, label_list) if label_list else "Unknown",
        }

    # Guarantees / defaults
    if classification_type == "binary" and label_list and len(label_list) == 2:
        data = _repair_binary_output(data, label_list)
    else:
        data.setdefault("score", -1.0)
        if "answer" not in data:
            data["answer"] = "Unknown"

    return data


def _llava_hf_call(contents,
                   classification_type: str = "binary",
                   label_list: List[str] | None = None):
    """
    Robust LLaVA call.
    """
    processor, model = _HF_PROC_MODEL

    chat, images = [], []
    for msg in contents:
        role = msg["role"].lower()
        role = "assistant" if role.startswith("assistant") or role.startswith("model") else "user"
        parts = []
        for part in msg["parts"]:
            if isinstance(part, dict) and "mime_type" in part:
                img = Image.open(io.BytesIO(part["data"])).convert("RGB")
                images.append(img)
                parts.append({"type": "image"})
            else:
                txt = part["text"] if isinstance(part, dict) else str(part)
                txt = txt.replace("<image>", "[IMG]")
                parts.append({"type": "text", "text": txt})
        chat.append({"role": role, "content": parts})

    prompt_text = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = processor(
        text=prompt_text,
        images=images or None,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to("cuda:0") if torch.is_tensor(v) else v for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=_GENERATION_CFG.get("max_new_tokens", 512),
        do_sample=True,
        temperature=0.4,
        top_p=0.9,
        repetition_penalty=1.2,
    )

    out = model.generate(**inputs, **gen_kwargs)
    resp = processor.decode(out[0], skip_special_tokens=True).strip()
    if os.getenv("DEBUG_LAVA", "0") == "1":
        print("── RAW LLaVA ──\n", resp, "\n──────────────")
    resp = resp.split("ASSISTANT:")[-1].strip()
    return _extract_llava_json(resp, classification_type, label_list)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────
def configure_vlm(model_cfg: dict | None = None):
    global _BACKEND, _HF_PROC_MODEL, _GENERATION_CFG, _MODEL_NAME

    model_cfg = model_cfg or {}
    _BACKEND = VLMBackend(model_cfg.get("backend", "gemini").lower())
    _GENERATION_CFG = model_cfg.get("model_kwargs", {})

    if _BACKEND is VLMBackend.GEMINI:
        configure_gemini(model_cfg)
        model_name = _MODEL_NAME
    elif _BACKEND is VLMBackend.LLAVA_HF:
        model_name = model_cfg.get("model_name", "llava-hf/llava-1.5-7b-hf")
        _HF_PROC_MODEL = _setup_llava_hf(model_name)
    else:
        raise ValueError(f"Unsupported backend: {_BACKEND}")

    print(f"[VLM] backend={_BACKEND.value} | model={model_name}")


def vlm_api_call(contents,
                 classification_type: str = "binary",
                 label_list: List[str] | None = None):
    if _BACKEND is VLMBackend.GEMINI:
        return gemini_api_call(contents, classification_type, label_list)
    if _BACKEND is VLMBackend.LLAVA_HF:
        return _llava_hf_call(contents, classification_type, label_list)
    raise RuntimeError("VLM backend not configured")


# ──────────────────────────────────────────────────────────────────────
# Prompt builders
# ──────────────────────────────────────────────────────────────────────
def build_gemini_prompt(
    few_shot_samples,
    test_image,
    classification_type: str = "binary",
    label_list: List[str] | None = None,
):
    # Unchanged
    instruction = load_prompt_text()
    contents = [{"role": "user", "parts": [{"text": instruction}]}]

    if classification_type == "binary":
        if not label_list or len(label_list) != 2:
            raise ValueError("binary classification expects two labels")
        pos_lbl, neg_lbl = label_list
        pos_items = few_shot_samples.get(pos_lbl, [])
        neg_items = few_shot_samples.get(neg_lbl, [])
        max_len = max(len(pos_items), len(neg_items))

        for i in range(max_len):
            if i < len(pos_items):
                img_part, _ = pos_items[i]
                contents += [
                    {"role": "user", "parts": [
                        img_part,
                        {"text": f"[Positive Example {i+1}] Please classify:"},
                    ]},
                    {"role": "model", "parts": [{
                        "text": json.dumps({
                            "thoughts": f"Consistent with {pos_lbl.lower()}",
                            "answer":  pos_lbl,
                            f"score_{_snake(pos_lbl)}": 0.95,
                            f"score_{_snake(neg_lbl)}": 0.05,
                            "location": "suspected region",
                        })
                    }]},
                ]
            if i < len(neg_items):
                img_part, _ = neg_items[i]
                contents += [
                    {"role": "user", "parts": [
                        img_part,
                        {"text": f"[Negative Example {i+1}] Please classify:"},
                    ]},
                    {"role": "model", "parts": [{
                        "text": json.dumps({
                            "thoughts": f"No evidence of {pos_lbl.lower()}",
                            "answer":  neg_lbl,
                            f"score_{_snake(pos_lbl)}": 0.05,
                            f"score_{_snake(neg_lbl)}": 0.95,
                            "location": None,
                        })
                    }]},
                ]

        contents.append({"role": "user", "parts": [
            test_image,
            {"text": "[Test] Please classify:"},
        ]})

    else:
        for lbl, items in few_shot_samples.items():
            for i, (img_part, _) in enumerate(items, 1):
                contents += [
                    {"role": "user", "parts": [
                        img_part,
                        f"[{lbl} Example {i}] Please classify:",
                    ]},
                    {"role": "model", "parts": [{
                        "text": json.dumps({
                            "thoughts": f"Representative of {lbl}",
                            "answer":  lbl,
                            "score":   0.95,
                        })
                    }]},
                ]
        contents.append({"role": "user", "parts": [
            test_image,
            "[Test] Please classify:",
        ]})
    return contents


def _read_text(path: str, default: str = "") -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


def build_llava_prompt(
    few_shot_samples,  # unused now (zero-shot)
    test_image,
    classification_type: str = "binary",
    label_list: List[str] | None = None,
    prompt_text_path: str | None = None,
):
    """
    ZERO-SHOT prompt for LLaVA with explicit instructions:
    • Never output Unknown
    • JSON must be valid
    • Dual-branch reasoning
    """
    if prompt_text_path is None:
        prompt_text_path = os.getenv("PROMPT_PATH", "./prompts/few_shot_llava.txt")
    base = _read_text(prompt_text_path).strip().replace("<image>", "[IMG]")

    extra = (
        "Never answer 'Unknown'. Choose the most likely of the two classes.\n"
        "In the JSON:\n"
        " - scores must be in [0,1] and sum to ~1.\n"
        " - Do not add extra keys.\n"
        "Thoughts must follow the 4-step structure provided."
    )
    sys_prompt = f"{base}\n\n{extra}"

    contents: List[Dict[str, Any]] = [
        {"role": "system", "parts": [{"text": sys_prompt}]},
        {"role": "user",   "parts": [test_image, {"text": "Classify this image now."}]}
    ]
    return contents