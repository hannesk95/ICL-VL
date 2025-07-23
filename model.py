"""
model.py – unified VLM backend for Gemini + (single-GPU) LLaVA-HF
last updated 2025-07-01 (b patch 3)

Key additions versus your original:
• Robust JSON extraction for LLaVA
• Regex-based free-form–answer → canonical-label mapping
• trust_remote_code=True when loading LLaVA
• DEBUG_LAVA=1 env-var prints raw model output for debugging
Gemini behaviour is unchanged.

Public API expected by main.py:
    configure_vlm(model_cfg: dict|None)
    vlm_api_call(contents, classification_type="binary", label_list=None)
    build_gemini_prompt(...)
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────
# standard lib
# ──────────────────────────────────────────────────────────────────────
import io, os, re, json, time
from collections import deque
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Dict, List

# ──────────────────────────────────────────────────────────────────────
# third-party
# ──────────────────────────────────────────────────────────────────────
import google.generativeai as genai
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch
import PIL.Image as Image


# ╭──────────────────────────────────────────────────────────────────╮
# │ Shared helpers & globals                                        │
# ╰──────────────────────────────────────────────────────────────────╯
def _snake(text: str) -> str:
    """lower-snake helper – “No Tumor” → “no_tumor”."""
    return "_".join(text.lower().split())


class VLMBackend(str, Enum):
    GEMINI = "gemini"
    LLAVA_HF = "llava_hf"


_BACKEND: VLMBackend | None = None
_GENERATION_CFG: Dict[str, Any] = {}
_HF_PROC_MODEL = None             # (processor, model)
_MODEL_NAME = "gemini-2.0-flash"  # default Gemini


# ╭──────────────────────────────────────────────────────────────────╮
# │ Gemini helpers  (+ simple token-bucket rate limiter)            │
# ╰──────────────────────────────────────────────────────────────────╯
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
            time.sleep(_PERIOD - (now - _CALLS[0]) + 0.05)


def _rate_limited(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        _acquire_token()
        return fn(*a, **kw)

    return wrapper


def configure_gemini(model_cfg: dict | None = None) -> None:
    """Initialise Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("set GEMINI_API_KEY in your environment")
    genai.configure(api_key=api_key)

    global _MODEL_NAME, _GENERATION_CFG
    if model_cfg:
        _MODEL_NAME = model_cfg.get("model_name", _MODEL_NAME)
        _GENERATION_CFG = model_cfg.get("model_kwargs", {})


def load_prompt_text() -> str:
    """Reads the prompt template specified via PROMPT_PATH env."""
    with open(os.getenv("PROMPT_PATH", "./prompts/few_shot.txt"),
              encoding="utf-8") as fp:
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


# ╭──────────────────────────────────────────────────────────────────╮
# │ LLaVA HF helpers                                                │
# ╰──────────────────────────────────────────────────────────────────╯
def _setup_llava_hf(model_id: str):
    """Load checkpoint in 4-bit on **cuda:0**."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for LLaVA backend")

    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    qcfg = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_compute_dtype=torch.float16)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        device_map={"": 0},                      # pin every sub-module
        quantization_config=qcfg,
        trust_remote_code=True,
    )
    model.eval()
    return proc, model


# ── regex-based synonym mapper ─────────────────────────────────────
def _map_label(free_text: str, label_list: List[str] | None):
    if not label_list:
        return free_text
    text = re.sub(r"[^a-z0-9]+", " ", free_text.lower()).strip()

    # exact match
    for lbl in label_list:
        if text == lbl.lower():
            return lbl

    if len(label_list) == 2:
        tum, no = label_list
        patterns = [
            (r"\bclass\s*1\b", tum),
            (r"\bhigh\s*grade\b", tum),
            (r"\bhgg\b", tum),
            (r"\bclass\s*2\b", no),
            (r"\blow\s*grade\b", no),
            (r"\blgg\b", no),
            (r"\bno\s*tumou?r\b", no),
            (r"\bbenign\b", no),
        ]
        for pat, mapped in patterns:
            if re.search(pat, text):
                return mapped
    return "Unknown"


def _extract_llava_json(resp: str,
                        classification_type: str,
                        label_list: List[str] | None):
    cleaned = re.sub(r"^```(?:json)?|```$", "", resp, flags=re.I).strip()
    # 2) **NEW** – un-escape Markdown underscores (and nothing else)
    cleaned = cleaned.replace(r"\_", "_")
    try:
        data = json.loads(cleaned)
    except Exception:
        if os.getenv("DEBUG_LAVA", "0") == "1":
            print("[LLaVA raw]", cleaned)
        data = {
            "thoughts": "Free-form reply; JSON parse failed.",
            "answer":   _map_label(cleaned, label_list),
        }

    if classification_type == "binary" and label_list and len(label_list) == 2:
        pos_key = f"score_{_snake(label_list[0])}"
        neg_key = f"score_{_snake(label_list[1])}"
        data.setdefault(pos_key, -1)
        data.setdefault(neg_key, -1)
        data.setdefault("location", None)
    else:
        data.setdefault("score", -1.0)
    return data


def _llava_hf_call(contents,
                   classification_type: str = "binary",
                   label_list: List[str] | None = None):
    """Run LLaVA 1.5-HF on a single GPU and parse JSON robustly."""
    processor, model = _HF_PROC_MODEL

    prompt_lines, images = [], []
    for msg in contents:
        role = msg["role"].upper().replace("MODEL", "ASSISTANT")
        parts = []
        for part in msg["parts"]:
            if isinstance(part, dict) and "mime_type" in part:
                images.append(Image.open(io.BytesIO(part["data"])))
                parts.append("<image>")
            elif isinstance(part, Image.Image):
                images.append(part)
                parts.append("<image>")
            else:
                txt = part["text"] if isinstance(part, dict) else str(part)
                parts.append(txt.strip())
        joined = " ".join(parts).strip()
        prompt_lines.append(joined if role == "SYSTEM"
                            else f"{role}: {joined}")

    prompt_lines.append(
        "USER: Respond ONLY with valid JSON as per the template – no markdown.")
    prompt_lines.append("ASSISTANT:")

    prompt_text = "\n".join(prompt_lines)
    inputs = processor(text=prompt_text,
                       images=images or None,
                       return_tensors="pt")
    inputs = {k: v.to("cuda:0") if torch.is_tensor(v) else v
              for k, v in inputs.items()}

    out = model.generate(**inputs,
                         max_new_tokens=_GENERATION_CFG.get("max_new_tokens",
                                                             512),
                         top_p=_GENERATION_CFG.get("top_p", 0.9),
                         temperature=_GENERATION_CFG.get("temperature", 0.2))
    resp = processor.decode(out[0], skip_special_tokens=True)

    if os.getenv("DEBUG_LAVA", "0") == "1":
        print("── RAW LLaVA ──\n", resp, "\n──────────────")

    resp = re.split(r"ASSISTANT:", resp, flags=re.I)[-1].strip()
    return _extract_llava_json(resp, classification_type, label_list)


# ╭──────────────────────────────────────────────────────────────────╮
# │ Public configure + unified inference API                        │
# ╰──────────────────────────────────────────────────────────────────╯
def configure_vlm(model_cfg: dict | None = None):
    """Choose backend (Gemini or LLaVA) and load weights."""
    global _BACKEND, _HF_PROC_MODEL, _GENERATION_CFG, _MODEL_NAME

    model_cfg = model_cfg or {}
    _BACKEND = VLMBackend(model_cfg.get("backend", "gemini").lower())
    _GENERATION_CFG = model_cfg.get("model_kwargs", {})

    if _BACKEND is VLMBackend.GEMINI:
        configure_gemini(model_cfg)
        model_name = _MODEL_NAME
    elif _BACKEND is VLMBackend.LLAVA_HF:
        model_name = model_cfg.get("model_name",
                                   "llava-hf/llava-1.5-7b-hf")
        _HF_PROC_MODEL = _setup_llava_hf(model_name)
    else:
        raise ValueError(f"Unsupported backend: {_BACKEND}")

    print(f"[VLM] backend={_BACKEND.value} | model={model_name}")


def vlm_api_call(contents,
                 classification_type: str = "binary",
                 label_list: List[str] | None = None):
    """Back-end agnostic inference dispatch."""
    if _BACKEND is VLMBackend.GEMINI:
        return gemini_api_call(contents, classification_type, label_list)
    if _BACKEND is VLMBackend.LLAVA_HF:
        return _llava_hf_call(contents, classification_type, label_list)
    raise RuntimeError("VLM backend not configured")


# ╭──────────────────────────────────────────────────────────────────╮
# │ Few-shot prompt builder – unchanged logic                       │
# ╰──────────────────────────────────────────────────────────────────╯
def build_gemini_prompt(
    few_shot_samples,
    test_image,
    classification_type: str = "binary",
    label_list: List[str] | None = None,
):
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

    else:  # multi-class
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