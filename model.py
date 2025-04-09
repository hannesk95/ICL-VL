import os
import json
import google.generativeai as genai

def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Please set it in .env or your environment.")
    genai.configure(api_key=api_key)

def load_prompt_text():
    path = os.getenv("PROMPT_PATH", "/u/home/obt/ICL-VL/prompts/few_shot.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def build_gemini_prompt(pos_images, neg_images, test_images):
    instruction = load_prompt_text()
    contents = [{"role": "user", "parts": [instruction]}]

    # Positive examples
    for i, img in enumerate(pos_images, 1):
        contents.append({
            "role": "user",
            "parts": [img, f"[Positive Example {i}] Please classify this MRI scan:"]
        })
        contents.append({
            "role": "model",
            "parts": [json.dumps({
                "thoughts": "Mass detected in the left temporal lobe, irregular enhancement noted.",
                "answer": "Tumor",
                "score_tumor": 0.95,
                "score_no_tumor": 0.05,
                "location": "left temporal lobe"
            })]
        })

    # Negative examples
    for i, img in enumerate(neg_images, 1):
        contents.append({
            "role": "user",
            "parts": [img, f"[Negative Example {i}] Please classify this MRI scan:"]
        })
        contents.append({
            "role": "model",
            "parts": [json.dumps({
                "thoughts": "Brain structures appear normal with no abnormal growths.",
                "answer": "No Tumor",
                "score_tumor": 0.05,
                "score_no_tumor": 0.95,
                "location": None
            })]
        })

    # Test images
    for i, img in enumerate(test_images, 1):
        contents.append({
            "role": "user",
            "parts": [img, f"[Test {i}] Please classify this MRI scan:"]
        })

    return contents

def gemini_api_call(contents):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(contents)

    try:
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()
        prediction_data = json.loads(raw_text)
    except Exception:
        # Fallback if parsing fails
        prediction_data = {
            "thoughts": "Unable to parse response.",
            "answer": "Unknown",
            "score_tumor": -1,
            "score_no_tumor": -1,
            "location": None
        }

    # Ensure missing keys are set to default
    if "score_tumor" not in prediction_data:
        prediction_data["score_tumor"] = -1
    if "score_no_tumor" not in prediction_data:
        prediction_data["score_no_tumor"] = -1
    if "location" not in prediction_data:
        prediction_data["location"] = None

    return prediction_data