import os
import json
import google.generativeai as genai
# --- rate-limiter with optional prints --------------------------------------
import os, time
from collections import deque
from functools import wraps
from threading import Lock

_MAX_CALLS = 15          # tokens in a rolling window
_PERIOD    = 90.0        # seconds
_CALLS     = deque()     # timestamps of recent calls
_LOCK      = Lock()

# toggle with env-var or constant
RL_VERBOSE = True

def _acquire_token():
    """Block until a token is free; optionally print waiting stats."""
    while True:
        with _LOCK:
            now = time.time()
            # drop calls older than 60 s
            while _CALLS and now - _CALLS[0] > _PERIOD:
                _CALLS.popleft()

            if len(_CALLS) < _MAX_CALLS:
                _CALLS.append(now)
                if RL_VERBOSE:
                    remaining = _MAX_CALLS - len(_CALLS)
                    print(f"[rate-limit] token OK → {remaining} left")
                return                      # proceed!

            # otherwise we must wait
            wait = _PERIOD - (now - _CALLS[0]) + 0.01
            if RL_VERBOSE:
                print(f"[rate-limit] bucket empty; sleeping {wait:.2f} s")

        time.sleep(wait)                    # outside the lock


def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing GEMINI_API_KEY. Please set it in .env or your environment."
        )
    genai.configure(api_key=api_key)


def load_prompt_text():
    path = os.getenv("PROMPT_PATH", "./prompts/few_shot.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_gemini_prompt(few_shot_samples, test_image, classification_type="binary"):
    instruction = load_prompt_text()
    contents = [{"role": "user", "parts": [instruction]}]

    if classification_type == "binary":
        pos_items = few_shot_samples.get("Tumor", [])
        neg_items = few_shot_samples.get("No Tumor", [])

        # Positive examples
        for i, (img, _) in enumerate(pos_items, 1):
            contents.append(
                {
                    "role": "user",
                    "parts": [img, f"[Positive Example {i}] Please classify this scan:"],
                }
            )
            contents.append(
                {
                    "role": "model",
                    "parts": [
                        json.dumps(
                            {
                                "thoughts": "Suspicious lesion observed.",
                                "answer": "Tumor",
                                "score_tumor": 0.95,
                                "score_no_tumor": 0.05,
                                "location": "left region",
                            }
                        )
                    ],
                }
            )

        # Negative examples
        for i, (img, _) in enumerate(neg_items, 1):
            contents.append(
                {
                    "role": "user",
                    "parts": [img, f"[Negative Example {i}] Please classify this scan:"],
                }
            )
            contents.append(
                {
                    "role": "model",
                    "parts": [
                        json.dumps(
                            {
                                "thoughts": "No abnormality observed.",
                                "answer": "No Tumor",
                                "score_tumor": 0.05,
                                "score_no_tumor": 0.95,
                                "location": None,
                            }
                        )
                    ],
                }
            )

        # Test image
        contents.append(
            {"role": "user", "parts": [test_image, "[Test] Please classify this scan:"]}
        )

    else:  # multiclass
        for label_name, items in few_shot_samples.items():
            for i, (img, _) in enumerate(items, 1):
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            img,
                            f"[{label_name} Example {i}] Please classify this slide:",
                        ],
                    }
                )
                contents.append(
                    {
                        "role": "model",
                        "parts": [
                            json.dumps(
                                {
                                    "thoughts": f"Representative of {label_name}",
                                    "answer": label_name,
                                    "score": 0.95,
                                }
                            )
                        ],
                    }
                )

        contents.append(
            {
                "role": "user",
                "parts": [test_image, "[Test] Please classify this histopathology slide:"],
            }
        )

    return contents

def _rate_limited(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        _acquire_token()
        return fn(*a, **kw)
    return wrapper


@_rate_limited
def gemini_api_call(contents, classification_type="binary"):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(contents)
    raw_text = response.text.strip()

    # Strip markdown fences if present
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:].strip()
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:].strip()
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3].strip()

    # Attempt to parse JSON
    try:
        prediction_data = json.loads(raw_text)
    except Exception:
        if classification_type == "binary":
            prediction_data = {
                "thoughts": "Unable to parse response.",
                "answer": "Unknown",
                "score_tumor": -1,
                "score_no_tumor": -1,
                "location": None,
            }
        else:
            prediction_data = {
                "thoughts": "Unable to parse response.",
                "answer": "Unknown",
                "score": -1.0,
            }

    # Ensure expected keys exist
    if classification_type == "binary":
        prediction_data.setdefault("score_tumor", -1)
        prediction_data.setdefault("score_no_tumor", -1)
        prediction_data.setdefault("location", None)
    else:
        prediction_data.setdefault("score", -1.0)

    return prediction_data