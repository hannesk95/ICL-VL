"""
model.py – Gemini helper + prompt builder (flexible binary labels)
           + interleaved positive/negative few-shot pairs
           (DEBUG: prints prompt size, token count, and generation parameters)
"""

from __future__ import annotations

import os
import json
import re
import time
from collections import deque
from functools import wraps
from threading import Lock
from typing import Dict, List, Any

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Simple token-bucket rate-limiter (16 requests / 80-s window)
# ---------------------------------------------------------------------------
_MAX_CALLS = 16
_PERIOD = 80.0
_CALLS: deque[float] = deque()
_LOCK = Lock()

RL_VERBOSE = bool(os.getenv("RL_VERBOSE", "1"))

# ---------------------------------------------------------------------------
# Global model configuration (set by configure_gemini)
# ---------------------------------------------------------------------------
_MODEL_NAME: str = "gemini-2.0-flash"
_GENERATION_CFG: Dict[str, Any] = {}   # temperature, top_p, max_tokens, …


def _acquire_token() -> None:
    while True:
        with _LOCK:
            now = time.time()
            while _CALLS and now - _CALLS[0] > _PERIOD:
                _CALLS.popleft()

            if len(_CALLS) < _MAX_CALLS:
                _CALLS.append(now)
                if RL_VERBOSE:
                    remaining = _MAX_CALLS - len(_CALLS)
                    print(f"[rate-limit] token OK → {remaining} left")
                return

            wait = _PERIOD - (now - _CALLS[0]) + 0.01
            if RL_VERBOSE:
                print(f"[rate-limit] bucket empty; sleeping {wait:.2f} s")
        time.sleep(wait)


# ---------------------------------------------------------------------------
# Gemini configuration & prompt helpers
# ---------------------------------------------------------------------------
def configure_gemini(model_cfg: dict | None = None) -> None:
    """Set API key and remember model parameters (name, generation config)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY – set it in .env or env.")
    genai.configure(api_key=api_key)

    global _MODEL_NAME, _GENERATION_CFG
    if model_cfg:
        _MODEL_NAME       = model_cfg.get("model_name", _MODEL_NAME)
        _GENERATION_CFG   = model_cfg.get("model_kwargs", {})


def load_prompt_text() -> str:
    path = os.getenv("PROMPT_PATH", "./prompts/few_shot.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Prompt builder (unchanged logic + DEBUG instrumentation)
# ---------------------------------------------------------------------------
def build_gemini_prompt(
    few_shot_samples: Dict[str, List[Any]],
    test_image,
    classification_type: str = "binary",
    label_list: List[str] | None = None,
):
    instruction = load_prompt_text()
    contents = [{"role": "user", "parts": [{"text": instruction}]}]

    # ---------- BINARY ---------- #
    if classification_type == "binary":
        if not label_list or len(label_list) != 2:
            raise ValueError("Binary classification requires exactly two labels.")

        pos_label, neg_label = label_list
        pos_items = few_shot_samples.get(pos_label, [])
        neg_items = few_shot_samples.get(neg_label, [])

        max_len = max(len(pos_items), len(neg_items))
        for i in range(max_len):
            # positive
            if i < len(pos_items):
                img_part, _ = pos_items[i]
                idx = i + 1
                contents.append({
                    "role": "user",
                    "parts": [
                        img_part,
                        {"text": f"[Positive Example {idx}] Please classify this scan:"},
                    ],
                })
                contents.append({
                    "role": "model",
                    "parts": [{"text": json.dumps({
                        "thoughts": f"Findings consistent with {pos_label.lower()}.",
                        "answer":   pos_label,
                        f"score_{pos_label.lower().replace(' ', '_')}": 0.95,
                        f"score_{neg_label.lower().replace(' ', '_')}": 0.05,
                        "location": "suspected region",
                    })}],
                })

            # negative
            if i < len(neg_items):
                img_part, _ = neg_items[i]
                idx = i + 1
                contents.append({
                    "role": "user",
                    "parts": [
                        img_part,
                        {"text": f"[Negative Example {idx}] Please classify this scan:"},
                    ],
                })
                contents.append({
                    "role": "model",
                    "parts": [{"text": json.dumps({
                        "thoughts": f"No evidence of {pos_label.lower()}.",
                        "answer":   neg_label,
                        f"score_{pos_label.lower().replace(' ', '_')}": 0.05,
                        f"score_{neg_label.lower().replace(' ', '_')}": 0.95,
                        "location": None,
                    })}],
                })

        # test case
        contents.append({
            "role": "user",
            "parts": [
                test_image,
                {"text": "[Test] Please classify this scan:"},
            ],
        })

    # ---------- MULTICLASS ---------- #
    else:
        for label_name, items in few_shot_samples.items():
            for i, (img, _) in enumerate(items, 1):
                contents.append({"role": "user", "parts": [img, f"[{label_name} Example {i}] Please classify this slide:"]})
                contents.append({"role": "model", "parts": [json.dumps({
                    "thoughts": f"Representative of {label_name}",
                    "answer":   label_name,
                    "score":    0.95,
                })]})

        contents.append({"role": "user", "parts": [test_image, "[Test] Please classify this histopathology slide:"]})

    # ------------------------------------------------------------------
    # DEBUG instrumentation: count messages & tokens (optional)
    # ------------------------------------------------------------------
    if os.getenv("DEBUG_PROMPT", "0") == "1":
        pos_ct = len(few_shot_samples.get(label_list[0], [])) if classification_type == "binary" and label_list else 0
        neg_ct = len(few_shot_samples.get(label_list[1], [])) if classification_type == "binary" and label_list else 0
        print(f"[prompt] {len(contents)} total messages "
              f"(instr+test+2·shots → pos:{pos_ct} neg:{neg_ct})")

        try:
            model_tmp = genai.GenerativeModel(_MODEL_NAME)
            toks = model_tmp.count_tokens(contents).total_tokens
            print(f"[prompt] {toks} tokens (Flash hard limit ≈ 32 k)")
        except Exception as exc:
            print(f"[prompt] token count unavailable: {exc}")

    return contents


# ---------------------------------------------------------------------------
# Decorator for rate-limited Gemini calls
# ---------------------------------------------------------------------------
def _rate_limited(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        _acquire_token()
        return fn(*a, **kw)
    return wrapper


# ---------------------------------------------------------------------------
# Helper: normalise label → snake_case
# ---------------------------------------------------------------------------
def _normalise(label: str) -> str:
    return "_".join(label.lower().split())


# ---------------------------------------------------------------------------
# Main call helper
# ---------------------------------------------------------------------------
@_rate_limited
def gemini_api_call(
    contents: List[Dict[str, Any]],
    classification_type: str = "binary",
    label_list: List[str] | None = None,
) -> Dict[str, Any]:

    # DEBUG: show generation parameters
    if os.getenv("DEBUG_PROMPT", "0") == "1":
        cfg_show = {**_GENERATION_CFG} or {"temperature": "model‑default"}
        print(f"[gen] model={_MODEL_NAME}  params={cfg_show}")

    model = genai.GenerativeModel(
        _MODEL_NAME,
        generation_config=_GENERATION_CFG,
    )

    response = model.generate_content(contents)
    raw_text = response.text.strip()
    raw_text = re.sub(r"^```(?:json)?|```$", "", raw_text, flags=re.I).strip()

    try:
        prediction_data: Dict[str, Any] = json.loads(raw_text)
    except Exception:
        prediction_data = {"thoughts": "Unable to parse response.", "answer": "Unknown"}

    # -------- binary post-processing --------
    if classification_type == "binary" and label_list and len(label_list) == 2:
        pos_lbl, neg_lbl = label_list
        pos_key = f"score_{_normalise(pos_lbl)}"
        neg_key = f"score_{_normalise(neg_lbl)}"

        alias_map = {
            "score_tumor":     pos_key,
            "score_no_tumor":  neg_key,
            "score_class1":    pos_key,
            "score_class2":    neg_key,
        }
        for old, new in alias_map.items():
            if old in prediction_data and new not in prediction_data:
                prediction_data[new] = prediction_data.pop(old)

        s_pos = prediction_data.get(pos_key)
        s_neg = prediction_data.get(neg_key)
        if s_pos is None and s_neg is not None:
            try:
                prediction_data[pos_key] = round(1 - float(s_neg), 3)
            except Exception:
                pass
        if s_neg is None and s_pos is not None:
            try:
                prediction_data[neg_key] = round(1 - float(s_pos), 3)
            except Exception:
                pass

        prediction_data.setdefault(pos_key, -1)
        prediction_data.setdefault(neg_key, -1)
        prediction_data.setdefault("location", None)

    else:   # multiclass safeguard
        prediction_data.setdefault("score", -1.0)

    return prediction_data
