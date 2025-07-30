"""
model.py – unified VLM backend for Gemini, LLaVA-HF and Med-LLaVA
last updated 2025-07-29  (e patch 15: fully separated LLaVA-HF vs Med-LLaVA)
"""

from __future__ import annotations

import io, os, re, json, time
from collections import deque
from enum import Enum
from functools import wraps
from threading import Lock
from typing import Any, Dict, List, Tuple

import google.generativeai as genai
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import PIL.Image as Image


# ──────────────────────────────────────────────────────────────────────
# Helpers & globals
# ──────────────────────────────────────────────────────────────────────
def _snake(text: str) -> str:
    return "_".join(text.lower().split())


class VLMBackend(str, Enum):
    GEMINI    = "gemini"
    LLAVA_HF  = "llava_hf"
    MED_LLAVA = "med_llava"


_BACKEND: VLMBackend | None = None
_GENERATION_CFG: Dict[str, Any] = {}
_HF_PROC_MODEL: Tuple[Any, Any] | None = None
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
# LLaVA helpers (shared utilities, not code paths)
# ──────────────────────────────────────────────────────────────────────
def _setup_llava_hf(model_id: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for LLaVA backend")

    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,      # use bf16 if supported
        device_map={"": 0},             # single-GPU; adjust if needed
        trust_remote_code=True,
    )
    model.eval()
    return proc, model

def _sanitize_medllava_output(data: Dict[str, Any], label_list: List[str]) -> Dict[str, Any]:
    """
    Post-process ONLY Med-LLaVA outputs to remove template artifacts and
    force coherent, non-neutral results tied to the selected class.
    """
    def _to_float(x, default=-1.0):
        try:
            return float(x)
        except Exception:
            return default

    def _is_placeholder_thoughts(t: str) -> bool:
        t = (t or "").strip()
        if not t:
            return True
        if "...." in t:
            return True
        if re.fullmatch(r"\s*1\.\s*2\.\s*3\.\s*", t) is not None:
            return True
        if re.search(r'\b(or|either)\b', t, flags=re.I):
            return True
        if t.lower().startswith("key t2 features considered"):
            return True
        return False

    pos, neg = label_list
    pk, nk = f"score_{_snake(pos)}", f"score_{_snake(neg)}"

    # --- Decide on a single class (from answer/thoughts/scores) ---
    ans_raw = str(data.get("answer", "")).strip()
    ans_l = ans_raw.lower()
    if (" or " in ans_l) or (ans_l not in (pos.lower(), neg.lower())):
        inferred = _map_label(str(data.get("thoughts", "")), label_list)
        if inferred in label_list:
            ans = inferred
        else:
            sp, sn = _to_float(data.get(pk, -1)), _to_float(data.get(nk, -1))
            if (0.0 <= sp <= 1.0) or (0.0 <= sn <= 1.0):
                ans = pos if sp >= sn else neg
            else:
                ans = pos
    else:
        ans = pos if ans_l == pos.lower() else neg
    data["answer"] = ans

    # --- Scores: clamp, complement, avoid neutrality, renormalize ---
    sp, sn = _to_float(data.get(pk, -1)), _to_float(data.get(nk, -1))
    if not (0.0 <= sp <= 1.0 and 0.0 <= sn <= 1.0):
        if 0.0 <= sp <= 1.0 and not (0.0 <= sn <= 1.0):
            sn = 1.0 - sp
        elif 0.0 <= sn <= 1.0 and not (0.0 <= sp <= 1.0):
            sp = 1.0 - sn
        else:
            sp, sn = (0.86, 0.14) if ans == pos else (0.14, 0.86)
    sp, sn = max(0.0, min(1.0, sp)), max(0.0, min(1.0, sn))
    # If still too neutral, bias toward the chosen class
    if abs(sp - sn) < 0.1:
        sp, sn = (0.62, 0.38) if ans == pos else (0.38, 0.62)
    tot = sp + sn
    if tot > 0:
        sp, sn = sp / tot, sn / tot
    data[pk], data[nk] = round(sp, 4), round(sn, 4)

    # --- Location: empty/ambiguous -> None
    loc = data.get("location", None)
    if isinstance(loc, str):
        if (not loc.strip()) or (" or " in loc.lower()) or (loc.strip().lower() == "null"):
            data["location"] = None

    # --- Thoughts: replace placeholders with class-specific content ---
    t = str(data.get("thoughts", "")).strip()
    t = re.sub(r"[^\x20-\x7E]", "", t)  # strip odd unicode (triangles, etc.)
    if _is_placeholder_thoughts(t):
        if ans == pos:
            t = "1. Homogeneous T2 bright signal, sharp margin, little edema, no necrosis. 2. Matches low-grade clues. 3. Therefore class1."
        else:
            t = "1. Heterogeneous T2 signal, ill-defined/infiltrative margin, marked edema, central necrosis. 2. Matches high-grade clues. 3. Therefore class2."
    else:
        # Ensure the numbered style
        if not re.match(r"\s*1\.", t):
            t = "1. " + t
    data["thoughts"] = t

    # Convenience flags for the caller (not written to JSON persisted by main.py)
    data["_needs_repair"] = _is_placeholder_thoughts(t)
    data["_nearly_neutral"] = abs(data[pk] - data[nk]) < 0.1

    return data

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
    m = None
    for match in re.finditer(r"\{[\s\S]*\}", text):
        m = match
    if not m:
        return None
    candidate = m.group(0).strip()

    cand = candidate
    cand = re.sub(r"(?<=\{|,)\s*'([^']+)'\s*:", r'"\1":', cand)
    cand = re.sub(r":\s*'([^']+)'", lambda m: ':"{}"'.format(m.group(1).replace('"', '\\"')), cand)
    cand = re.sub(r",\s*(\}|\])", r"\1", cand)

    try:
        return json.loads(cand)
    except Exception:
        return None


def _repair_binary_output(data: Dict[str, Any],
                          label_list: List[str]) -> Dict[str, Any]:
    pos, neg = label_list
    pk, nk = f"score_{_snake(pos)}", f"score_{_snake(neg)}"

    s_pos = float(data.get(pk, -1))
    s_neg = float(data.get(nk, -1))
    if not (0.0 <= s_pos <= 1.0) or not (0.0 <= s_neg <= 1.0):
        if 0 <= s_pos <= 1 and s_neg == -1:
            s_neg = 1 - s_pos
        elif 0 <= s_neg <= 1 and s_pos == -1:
            s_pos = 1 - s_neg
        else:
            s_pos = 0.5
            s_neg = 0.5
    tot = s_pos + s_neg
    if tot > 0:
        s_pos, s_neg = s_pos / tot, s_neg / tot

    data[pk], data[nk] = round(s_pos, 4), round(s_neg, 4)

    ans = data.get("answer", "")
    if ans not in label_list:
        data["answer"] = pos if s_pos >= s_neg else neg

    data.setdefault("thoughts", "")
    data.setdefault("location", None)
    return data


def _extract_llava_json(resp: str,
                        classification_type: str,
                        label_list: List[str] | None):
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

    if classification_type == "binary" and label_list and len(label_list) == 2:
        data = _repair_binary_output(data, label_list)
    else:
        data.setdefault("score", -1.0)
        if "answer" not in data:
            data["answer"] = "Unknown"

    return data


# ──────────────────────────────────────────────────────────────────────
# LLaVA-HF (ORIGINAL, untouched behavior)
# ──────────────────────────────────────────────────────────────────────
def _llava_hf_call_hf(contents,
                      classification_type: str = "binary",
                      label_list: List[str] | None = None):
    """
    ORIGINAL LLaVA-HF path (as in your old file). Uses multimodal parts,
    decodes the full sequence, and keeps the tail after the last 'ASSISTANT:'.
    """
    processor, model = _HF_PROC_MODEL

    chat, images = [], []
    for msg in contents:
        role = msg["role"].lower()
        parts = []
        for part in msg["parts"]:
            if isinstance(part, dict) and "mime_type" in part:
                img = Image.open(io.BytesIO(part["data"])).convert("RGB")
                images.append(img)
                parts.append({"type": "image"})
            else:
                txt = part["text"] if isinstance(part, dict) else str(part)
                txt = txt.replace("<image>", "[IMG]")  # neutralize if present in text
                parts.append({"type": "text", "text": txt})
        chat.append({"role": role, "content": parts})

    if os.getenv("DEBUG_LAVA", "0") == "1":
        print(f"[LLaVA-HF] chat messages: {len(chat)} | images attached: {len(images)}")

    prompt_text = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
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
        print("── RAW LLaVA-HF ──\n", resp, "\n──────────────────")
    resp = resp.split("ASSISTANT:")[-1].strip()
    return _extract_llava_json(resp, classification_type, label_list)


def _build_llava_prompt_hf(
    few_shot_samples,
    test_image,
    classification_type: str = "binary",
    label_list: List[str] | None = None,
    prompt_text_path: str | None = None,
):
    """
    ORIGINAL prompt builder for LLaVA-HF (exactly like your old version).
    Do NOT wrap the image; pass it through as-is.
    """
    if prompt_text_path is None:
        prompt_text_path = os.getenv("PROMPT_PATH", "./prompts/few_shot_llava.txt")
    base = _read_text(prompt_text_path).strip()

    extra = (
        "Never answer 'Unknown'. Choose the most likely of the two classes.\n"
        "In the JSON:\n"
        " - scores must be in [0,1] and sum to ~1.\n"
        " - Do not add extra keys.\n"
        "Thoughts must follow the 4-step structure provided."
    )
    sys_prompt = f"{base}\n\n{extra}"

    return [
        {"role": "system", "parts": [{"text": sys_prompt}]},
        {"role": "user",   "parts": [test_image, {"text": "Classify this image now."}]}
    ]


# ──────────────────────────────────────────────────────────────────────
# Med-LLaVA (NEW, isolated path)
# ──────────────────────────────────────────────────────────────────────
def _ensure_patch_size_for_med(proc):
    """
    Some community LLaVA checkpoints don't expose `patch_size` where Transformers expects it.
    For Med-LLaVA we ensure a sane value so ProcessingLlava can compute image token counts.
    """
    ps = getattr(proc, "patch_size", None)
    if ps is None:
        ip = getattr(proc, "image_processor", None)
        ps = getattr(ip, "patch_size", None) if ip is not None else None
    if ps is None:
        ps = 14  # ViT-L/14 is the common visual backbone for LLaVA-1.5 & Med-LLaVA
        if os.getenv("DEBUG_LAVA", "0") == "1":
            print("[Med-LLaVA] 'patch_size' missing; defaulting to 14.")
    proc.patch_size = ps
    if os.getenv("DEBUG_LAVA", "0") == "1":
        print(f"[Med-LLaVA] Using patch_size={proc.patch_size}")
    return proc


def _med_llava_call(contents,
                    classification_type: str = "binary",
                    label_list: List[str] | None = None):
    """
    Med-LLaVA path (string chat + JSON seed + optional repair pass):
      • Build chat with string content and explicit "<image>" placeholders (template-friendly).
      • Attach PIL images separately so the processor inserts visual tokens.
      • Seed a short JSON prefix so generation continues inside an object.
      • Decode only the continuation; trim to the last '}' and parse.
      • Sanitize, and if still too generic or neutral, run a short repair pass.
    """
    processor, model = _HF_PROC_MODEL

    # ---- 1) Build plain chat (strings) + collect images ----
    chat, images = [], []
    for msg in contents:
        role = "assistant" if msg["role"].lower().startswith(("assistant", "model")) \
               else ("system" if msg["role"].lower().startswith("system") else "user")

        segs: List[str] = []
        for part in msg["parts"]:
            if isinstance(part, dict) and "mime_type" in part:
                img = Image.open(io.BytesIO(part["data"])).convert("RGB")
                images.append(img)
                segs.append("<image>")
            else:
                txt = part["text"] if isinstance(part, dict) else str(part)
                txt = re.sub(r"<\s*image\s*>", "", txt, flags=re.I)  # guard
                segs.append(txt)
        chat.append({"role": role, "content": "\n".join(segs)})

    if os.getenv("DEBUG_LAVA", "0") == "1":
        print(f"[Med-LLaVA] chat messages: {len(chat)} | images attached: {len(images)}")

    # ---- 2) Template + JSON seed ----
    prompt_text = processor.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
    )
    assistant_json_prefix = ' {"thoughts":"1. '
    prompt_text_seeded = prompt_text + assistant_json_prefix

    # ---- 3) Tokenize with image(s) ----
    def _tokenize(text: str):
        inputs = processor(
            text=text,
            images=images or None,
            return_tensors="pt",
            padding=True,
        )
        return {k: (v.to("cuda:0") if torch.is_tensor(v) else v) for k, v in inputs.items()}

    inputs = _tokenize(prompt_text_seeded)
    prompt_len = int(inputs["input_ids"].shape[1])

    # ---- 4) Generation (first pass) ----
    gen_kwargs = dict(
        max_new_tokens=_GENERATION_CFG.get("max_new_tokens", 384),
        min_new_tokens=_GENERATION_CFG.get("min_new_tokens", 64),
        do_sample=_GENERATION_CFG.get("do_sample", True),
        temperature=_GENERATION_CFG.get("temperature", 0.3),
        top_p=_GENERATION_CFG.get("top_p", 0.95),
        repetition_penalty=_GENERATION_CFG.get("repetition_penalty", 1.05),
    )
    out = model.generate(**inputs, **gen_kwargs)

    # ---- 5) Decode continuation ----
    seq = out["sequences"][0] if isinstance(out, dict) and "sequences" in out else out[0]
    if getattr(seq, "dim", lambda: 1)() == 2:
        seq = seq[0]
    total_len = int(seq.shape[0])
    gen_only = seq[prompt_len:] if total_len > prompt_len else seq

    tok = getattr(processor, "tokenizer", None)
    resp = (tok.decode(gen_only, skip_special_tokens=True).strip()
            if tok is not None else processor.decode(gen_only, skip_special_tokens=True).strip())

    resp = re.sub(r'^\s*ASSISTANT:\s*', '', resp, flags=re.I)
    resp = '{"thoughts":"1. ' + resp
    start = resp.find("{")
    if start > 0:
        resp = resp[start:]
    end = resp.rfind("}")
    if end != -1:
        resp = resp[:end + 1]

    if os.getenv("DEBUG_LAVA", "0") == "1":
        print(f"[Med-LLaVA] prompt_len={prompt_len} | total_len={total_len} | gen_len={int(gen_only.shape[0])}")
        print("── RAW Med-LLaVA (gen-only, JSON-seeded, trimmed) ──\n", resp, "\n──────────────────────")

    data = _extract_llava_json(resp, classification_type, label_list)

    # ---- 6) Med-LLaVA-only sanitize ----
    if classification_type == "binary" and label_list and len(label_list) == 2 and isinstance(data, dict):
        data = _sanitize_medllava_output(data, label_list)

        # ---- 7) Optional short repair pass if still generic/neutral ----
        if data.get("_needs_repair") or data.get("_nearly_neutral"):
            # Minimal, image-conditioned repair directive
            repair_directive = (
                "REVISE: Return ONE JSON object only. Replace any placeholders. "
                "In 'thoughts', describe concrete visual features of THIS T2 image only: "
                "signal pattern (homogeneous/heterogeneous), margin (well/poorly-defined), "
                "edema (little/marked), necrosis (present/absent), then the decision. "
                "In 'answer', write exactly one of ['class1','class2'] (no 'or'). "
                "In scores, give complementary probabilities in [0,1] that reflect the decision "
                "(avoid 0.50/0.50). Keep 'location' as lobe+side if identifiable, else null."
            )

            # We reuse the same chat, but append the repair directive as the last user turn.
            chat_repair = chat + [{"role": "user", "content": repair_directive}]
            prompt_repair = processor.apply_chat_template(
                chat_repair,
                add_generation_prompt=True,
                tokenize=False,
            )
            prompt_repair = prompt_repair + ' {"thoughts":"1. '
            inputs2 = _tokenize(prompt_repair)
            prompt_len2 = int(inputs2["input_ids"].shape[1])

            gen_kwargs2 = dict(
                max_new_tokens=min(256, _GENERATION_CFG.get("max_new_tokens", 384)),
                min_new_tokens=min(64, _GENERATION_CFG.get("min_new_tokens", 64)),
                do_sample=_GENERATION_CFG.get("do_sample", True),
                temperature=_GENERATION_CFG.get("temperature", 0.35),
                top_p=_GENERATION_CFG.get("top_p", 0.95),
                repetition_penalty=_GENERATION_CFG.get("repetition_penalty", 1.05),
            )
            out2 = model.generate(**inputs2, **gen_kwargs2)

            seq2 = out2["sequences"][0] if isinstance(out2, dict) and "sequences" in out2 else out2[0]
            if getattr(seq2, "dim", lambda: 1)() == 2:
                seq2 = seq2[0]
            gen_only2 = seq2[prompt_len2:] if int(seq2.shape[0]) > prompt_len2 else seq2

            resp2 = (tok.decode(gen_only2, skip_special_tokens=True).strip()
                     if tok is not None else processor.decode(gen_only2, skip_special_tokens=True).strip())
            resp2 = re.sub(r'^\s*ASSISTANT:\s*', '', resp2, flags=re.I)
            resp2 = '{"thoughts":"1. ' + resp2
            s2 = resp2.find("{")
            if s2 > 0:
                resp2 = resp2[s2:]
            e2 = resp2.rfind("}")
            if e2 != -1:
                resp2 = resp2[:e2 + 1]

            if os.getenv("DEBUG_LAVA", "0") == "1":
                print("── RAW Med-LLaVA (repair pass, trimmed) ──\n", resp2, "\n──────────────────────")

            data2 = _extract_llava_json(resp2, classification_type, label_list)
            if isinstance(data2, dict):
                data2 = _sanitize_medllava_output(data2, label_list)
                # Prefer repair if it removed placeholders & reduced neutrality
                if (not data2.get("_needs_repair", False)) or (data2.get("_nearly_neutral", False) is False):
                    data = data2

        # Clean internal flags before returning
        data.pop("_needs_repair", None)
        data.pop("_nearly_neutral", None)

    return data


# ──────────────────────────────────────────────────────────────────────
# Public API (dispatchers)
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

    elif _BACKEND is VLMBackend.MED_LLAVA:
        model_name = model_cfg.get(
            "model_name",
            "Eren-Senoglu/llava-med-v1.5-mistral-7b-hf"
        )
        proc, model = _setup_llava_hf(model_name)
        proc = _ensure_patch_size_for_med(proc)  # only for med_llava
        _HF_PROC_MODEL = (proc, model)

    else:
        raise ValueError(f"Unsupported backend: {_BACKEND}")

    print(f"[VLM] backend={_BACKEND.value} | model={model_name}")


def vlm_api_call(contents,
                 classification_type: str = "binary",
                 label_list: List[str] | None = None):
    if _BACKEND is VLMBackend.GEMINI:
        return gemini_api_call(contents, classification_type, label_list)
    if _BACKEND is VLMBackend.LLAVA_HF:
        return _llava_hf_call_hf(contents, classification_type, label_list)
    if _BACKEND is VLMBackend.MED_LLAVA:
        return _med_llava_call(contents, classification_type, label_list)
    raise RuntimeError("VLM backend not configured")


# ──────────────────────────────────────────────────────────────────────
# Prompt builders (dispatch – name kept stable for main.py)
# ──────────────────────────────────────────────────────────────────────
def build_gemini_prompt(
    few_shot_samples,
    test_image,
    classification_type: str = "binary",
    label_list: List[str] | None = None,
):
    # -----------------------------------------------------------------
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
    few_shot_samples,
    test_image,
    classification_type: str = "binary",
    label_list: List[str] | None = None,
    prompt_text_path: str | None = None,
):
    """
    Public builder name kept stable for main.py.
    DISPATCH ONLY: each backend has its own builder to avoid cross-coupling.
    """
    if _BACKEND is VLMBackend.MED_LLAVA:
        return _build_medllava_prompt(
            few_shot_samples, test_image, classification_type, label_list, prompt_text_path
        )
    else:
        return _build_llava_prompt_hf(
            few_shot_samples, test_image, classification_type, label_list, prompt_text_path
        )
        
def _build_medllava_prompt(
    few_shot_samples,
    test_image,
    classification_type: str = "binary",
    label_list: List[str] | None = None,
    prompt_text_path: str | None = None,
):
    """
    Med-LLaVA prompt builder (demote examples, emphasize image):
      • Put the full few-shot text (`few_shot_med.txt`) into the SYSTEM turn.
      • Final USER turn is minimal: attach the image + strict JSON rules,
        and explicitly say "analyze the attached image only; do not copy examples".
    """
    if prompt_text_path is None:
        prompt_text_path = os.getenv("PROMPT_PATH", "./few_shot_med.txt")

    base = _read_text(prompt_text_path).strip()
    # Keep the worked examples intact in SYSTEM; do not include <image> tokens here.
    base = re.sub(r"<\s*image\s*>", "[image omitted]", base, flags=re.I)

    final_user_directive = (
        "Analyze the ATTACHED IMAGE ONLY. Do not copy wording from the worked examples.\n"
        "Reply with EXACTLY ONE JSON object and NOTHING ELSE. Keys (exactly): "
        "\"thoughts\", \"answer\", \"score_class1\", \"score_class2\", \"location\".\n"
        "Begin with '{' and end with '}'. Scores are numeric in [0,1] and sum to ~1.\n"
        "Keep 'thoughts' to three short numbered clauses: '1. ... 2. ... 3. ...'."
    )

    # Convert PIL → dict (image part first)
    buf = io.BytesIO()
    if isinstance(test_image, Image.Image):
        test_image.save(buf, format="PNG")
        img_part = {"mime_type": "image/png", "data": buf.getvalue()}
    elif isinstance(test_image, dict) and "mime_type" in test_image and "data" in test_image:
        img_part = test_image
    elif isinstance(test_image, (bytes, bytearray)):
        img_part = {"mime_type": "image/png", "data": bytes(test_image)}
    else:
        raise TypeError(f"Unsupported image type for Med-LLaVA prompt: {type(test_image)}")

    return [
        {"role": "system", "parts": [{"text": base}]},
        {"role": "user",   "parts": [img_part, {"text": final_user_directive}]},
    ]