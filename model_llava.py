import os
import requests
import base64
from dotenv import load_dotenv

# Load .env file
load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def hf_llava_call(model_name, image_path, prompt):
    # Read and encode image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "inputs": {
            "prompt": prompt,
            "image": img_b64
        }
    }

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }

    api_url = f"https://api-inference.huggingface.co/models/{model_name}"

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"[ERROR] {response.status_code}: {response.text}")
        return {"thoughts": "API error", "answer": "Unknown"}

    try:
        return response.json()
    except Exception:
        return {"thoughts": "Invalid JSON", "answer": "Unknown"}

def classify_with_model(model_type, image_path, prompt):
    model_map = {
        "llava": "liuhaotian/llava-v1.5-7b",
        "llava-med": "llava-hf/llava-med-7b",
    }
    model_name = model_map.get(model_type)
    if not model_name:
        raise ValueError(f"Unknown model type: {model_type}")

    return hf_llava_call(model_name, image_path, prompt)