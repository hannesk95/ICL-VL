import os
import json
import google.generativeai as genai


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