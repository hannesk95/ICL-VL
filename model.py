"""
model.py  –  Unified backend for Gemini AND Hugging-Face LLaVA / MedLLaVA
            (Option 1: in-process torch inference, Python 3.13-compatible)
"""

from __future__ import annotations
import os, io, json, re, time, base64
from collections import deque
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Dict, List

# ========== GEMINI IMPORTS / RATE-LIMITER (unchanged) ========== #
import google.generativeai as genai

_MAX_CALLS, _PERIOD = 16, 80.0
_CALLS: deque[float] = deque(); _LOCK = Lock()
RL_VERBOSE = bool(os.getenv("RL_VERBOSE", "1"))

def _acquire_token():
    while True:
        with _LOCK:
            now = time.time()
            while _CALLS and now - _CALLS[0] > _PERIOD:
                _CALLS.popleft()
            if len(_CALLS) < _MAX_CALLS:
                _CALLS.append(now)
                if RL_VERBOSE:
                    print(f"[rate-limit] token OK → { _MAX_CALLS-len(_CALLS)} left")
                return
            wait = _PERIOD - (now - _CALLS[0]) + 0.01
            if RL_VERBOSE:
                print(f"[rate-limit] bucket empty; sleeping {wait:.2f}s")
        time.sleep(wait)

# ========== BACKEND ENUM ========== #
class VLMBackend(str, Enum):
    GEMINI   = "gemini"
    LLAVA_HF = "llava_hf"

# ========== GLOBAL BACKEND STATE ========== #
_BACKEND        : VLMBackend | None = None
_GENERATION_CFG : Dict[str, Any] = {}
_HF_PROC_MODEL  = None        # (processor, model) tuple when LLaVA is active

# ───────────────────────────────────────────────────────────────── #
# 1) GEMINI helpers (unchanged)                                   #
# ───────────────────────────────────────────────────────────────── #
_MODEL_NAME = "gemini-2.0-flash"

def configure_gemini(model_cfg: dict | None = None):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    global _MODEL_NAME, _GENERATION_CFG
    if model_cfg:
        _MODEL_NAME     = model_cfg.get("model_name", _MODEL_NAME)
        _GENERATION_CFG = model_cfg.get("model_kwargs", {})

def load_prompt_text() -> str:
    with open(os.getenv("PROMPT_PATH", "./prompts/few_shot.txt"), "r") as f:
        return f.read()

def _rate_limited(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        _acquire_token()
        return fn(*a, **kw)
    return wrapper

def _normalise(lbl: str) -> str:    # helper for binary post-proc
    return "_".join(lbl.lower().split())

@_rate_limited
def gemini_api_call(contents, classification_type="binary", label_list=None):
    if os.getenv("DEBUG_PROMPT", "0") == "1":
        cfg_show = {**_GENERATION_CFG} or {"temperature": "model-default"}
        print(f"[gen] model={_MODEL_NAME}  params={cfg_show}")

    model = genai.GenerativeModel(_MODEL_NAME, generation_config=_GENERATION_CFG)
    response = model.generate_content(contents)
    raw_text = response.text.strip()
    raw_text = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.I).strip()
    try:
        pred: Dict[str, Any] = json.loads(raw_text)
    except Exception:
        pred = {"thoughts": "Unable to parse response.", "answer": "Unknown"}

    # ----- binary post-processing (unchanged) -----
    if classification_type == "binary" and label_list and len(label_list) == 2:
        pos_lbl, neg_lbl = label_list
        pos_key = f"score_{_normalise(pos_lbl)}"
        neg_key = f"score_{_normalise(neg_lbl)}"
        pred.setdefault(pos_key, -1); pred.setdefault(neg_key, -1)
        pred.setdefault("location", None)
    else:
        pred.setdefault("score", -1.0)
    return pred

# ───────────────────────────────────────────────────────────────── #
# 2)  LLaVA (HF) SETUP AND CALL                                    #
# ───────────────────────────────────────────────────────────────── #
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch, PIL.Image as Image

def _setup_llava_hf(model_name: str):
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    dtype    = torch.float16 if device == "cuda" else torch.float32
    proc     = AutoProcessor.from_pretrained(model_name)
    model    = LlavaForConditionalGeneration.from_pretrained(
                   model_name,
                   torch_dtype=dtype,
                   load_in_4bit=True,           # ← uses bitsandbytes if available
                   device_map="auto",
               )
    model.to(device).eval()
    return proc, model

def _llava_hf_call(contents, classification_type="binary", label_list=None):
    processor, model = _HF_PROC_MODEL
    # convert Gemini-style list[dict] → HF multimodal chat messages
    messages = []
    for msg in contents:
        role = msg["role"]
        parts_out = []
        for part in msg["parts"]:
            if isinstance(part, dict) and "mime_type" in part:
                img = Image.open(io.BytesIO(part["data"]))
                parts_out.append({"type": "image", "image": img})
            else:   # plain text
                txt = part["text"] if isinstance(part, dict) else part
                parts_out.append({"type": "text", "text": txt})
        messages.append({"role": role, "content": parts_out})

    prompt_inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    gen_cfg = dict(max_new_tokens=_GENERATION_CFG.get("max_new_tokens", 512),
                   temperature=_GENERATION_CFG.get("temperature", 0.2),
                   top_p=_GENERATION_CFG.get("top_p", 0.9))
    output = model.generate(**prompt_inputs, **gen_cfg)
    resp   = processor.decode(output[0], skip_special_tokens=True)

    # The assistant’s JSON is after the last “ASSISTANT:” token (HF template)
    resp = resp.split("ASSISTANT:")[-1].strip()
    resp = re.sub(r"^```(?:json)?|```$", "", resp, flags=re.I).strip()
    try:
        pred = json.loads(resp)
    except Exception:
        pred = {"thoughts": "Parse failure", "answer": "Unknown"}

    # ensure mandatory keys so that downstream code doesn’t crash
    if classification_type == "binary" and label_list and len(label_list) == 2:
        pos_key = f"score_{_normalise(label_list[0])}"
        neg_key = f"score_{_normalise(label_list[1])}"
        pred.setdefault(pos_key, -1); pred.setdefault(neg_key, -1)
        pred.setdefault("location", None)
    else:
        pred.setdefault("score", -1.0)
    return pred

# ───────────────────────────────────────────────────────────────── #
# 3)  ONE public setup + ONE unified call                           #
# ───────────────────────────────────────────────────────────────── #
def configure_vlm(model_cfg: dict | None = None):
    """
    Call once at start-up.  Chooses backend, initialises models.
    """
    global _BACKEND, _HF_PROC_MODEL, _GENERATION_CFG

    if model_cfg is None:
        model_cfg = {}

    _BACKEND        = VLMBackend(model_cfg.get("backend", "gemini").lower())
    _GENERATION_CFG = model_cfg.get("model_kwargs", {})

    if _BACKEND is VLMBackend.GEMINI:
        configure_gemini(model_cfg)

    elif _BACKEND is VLMBackend.LLAVA_HF:
        model_name       = model_cfg.get("model_name", "llava-hf/llava-1.5-7b-hf")
        _HF_PROC_MODEL   = _setup_llava_hf(model_name)

    else:
        raise ValueError(f"Unsupported backend: {_BACKEND}")

def vlm_api_call(contents, classification_type="binary", label_list=None):
    if _BACKEND is VLMBackend.GEMINI:
        return gemini_api_call(contents, classification_type, label_list)
    if _BACKEND is VLMBackend.LLAVA_HF:
        return _llava_hf_call(contents, classification_type, label_list)
    raise RuntimeError("VLM backend not configured")