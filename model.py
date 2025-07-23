"""
model.py – unified VLM backend for Gemini + (single-GPU) LLaVA-HF
last updated 2025-07-23 (c patch 6)

Key changes from previous patch:
• build_llava_prompt: no JSON in demos; only brief NL descriptions. JSON schema given once.
• Strong anti-copy instruction added.
• _llava_hf_call: do_sample=False + repetition_penalty to reduce verbatim copying.
• Still robust to HF apply_chat_template output types.
• Gemini path untouched.
"""

from __future__ import annotations

import io, os, re, json, time
from collections import deque
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Dict, List

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

# Gemini rate limiter
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
    cleaned = cleaned.replace(r"\_", "_")
    try:
        data = json.loads(cleaned)
    except Exception:
        if os.getenv("DEBUG_LAVA", "0") == "1":
            print("[LLaVA raw]", cleaned)
        data = {
            "thoughts": "Free-form reply; JSON parse failed.",
            "answer": _map_label(cleaned, label_list),
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
    """
    Robust LLaVA call:
      1) Convert our contents → HF chat dict (no '<image>' literal).
      2) Get prompt string via apply_chat_template(tokenize=False).
      3) Tokenize with processor(...).
      4) Greedy decode w/ repetition_penalty to avoid copy.
    """
    processor, model = _HF_PROC_MODEL

    chat, images = [], []
    for msg in contents:
        role = msg["role"].lower()
        role = "assistant" if role.startswith("assistant") or role.startswith("model") else "user"
        parts = []
        for p in msg["parts"]:
            if isinstance(p, dict) and "mime_type" in p:
                img = Image.open(io.BytesIO(p["data"])).convert("RGB")
                images.append(img)
                parts.append({"type": "image"})
            else:
                txt = p["text"] if isinstance(p, dict) else str(p)
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
        top_p=_GENERATION_CFG.get("top_p", 1.0),
        do_sample=False,
        repetition_penalty=1.15,
    )
    # temperature ignored when do_sample=False, so skip it

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
                    {"role": "user", "parts": [img_part,
                                               {"text": f"[Positive Example {i+1}] Please classify:"}]},
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
                    {"role": "user", "parts": [img_part,
                                               {"text": f"[Negative Example {i+1}] Please classify:"}]},
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

        contents.append({"role": "user", "parts": [test_image,
                                                   {"text": "[Test] Please classify:"}]})

    else:
        for lbl, items in few_shot_samples.items():
            for i, (img_part, _) in enumerate(items, 1):
                contents += [
                    {"role": "user", "parts": [img_part,
                                               f"[{lbl} Example {i}] Please classify:"]},
                    {"role": "model", "parts": [{
                        "text": json.dumps({
                            "thoughts": f"Representative of {lbl}",
                            "answer":  lbl,
                            "score":   0.95,
                        })
                    }]},
                ]
        contents.append({"role": "user", "parts": [test_image,
                                                   "[Test] Please classify:"]})
    return contents


def _read_text(path: str, default: str = "") -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return default


def build_llava_prompt(
    few_shot_samples,
    test_image,
    classification_type: str = "binary",
    label_list: List[str] | None = None,
    prompt_text_path: str | None = None,
):
    """
    LLaVA-friendly: no JSON examples; concise schema; anti-copy instructions.
    """
    if prompt_text_path is None:
        prompt_text_path = os.getenv("PROMPT_PATH", "./prompts/few_shot_llava.txt")
    base_prompt = _read_text(prompt_text_path).strip().replace("<image>", "[IMG]")

    # Add anti-copy note
    anti_copy = (
        "IMPORTANT: Do NOT reuse wording or numbers from any example. "
        "Your JSON must reflect THIS image only."
    )
    base_prompt = f"{base_prompt}\n\n{anti_copy}"

    contents: List[Dict[str, Any]] = [
        {"role": "system", "parts": [{"text": base_prompt}]}
    ]

    if classification_type == "binary":
        if not label_list or len(label_list) != 2:
            raise ValueError("binary classification expects two labels")
        pos_lbl, neg_lbl = label_list

        # Give NL-only demos (no JSON) – keep it super short
        shots = []
        max_len = max(len(few_shot_samples.get(pos_lbl, [])),
                      len(few_shot_samples.get(neg_lbl, [])))
        for i in range(max_len):
            if i < len(few_shot_samples.get(pos_lbl, [])):
                shots.append((few_shot_samples[pos_lbl][i][0],
                              "Homogeneous bright lesion, sharp borders, minimal edema → suggest class1."))
            if i < len(few_shot_samples.get(neg_lbl, [])):
                shots.append((few_shot_samples[neg_lbl][i][0],
                              "Heterogeneous signal, necrotic center, marked edema → suggest class2."))
            if len(shots) >= 2:
                break

        for img_part, nl_hint in shots:
            contents.append({"role": "user", "parts": [img_part, {"text": "Example image. Briefly describe key T2 features."}]})
            contents.append({"role": "assistant", "parts": [{"text": nl_hint}]})

    else:
        # Multi-class: same idea – NL hints
        for lbl, items in few_shot_samples.items():
            if not items:
                continue
            img_part, _ = items[0]
            contents.append({"role": "user", "parts": [img_part, {"text": "Example image. Briefly describe key features."}]})
            contents.append({"role": "assistant", "parts": [{"text": f"Representative of {lbl} based on visual features."}]})

    # Final image to classify
    contents.append({"role": "user", "parts": [test_image, {"text": "Now classify this image. Return ONLY valid JSON as specified."}]})
    return contents