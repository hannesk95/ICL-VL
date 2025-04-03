import os
import base64
import requests
from dotenv import load_dotenv
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

# 1) Load API key from .env
load_dotenv() 
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY. Please set it in .env or your environment.")

# 2) Define Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with images.
            transform (callable, optional): Optional transform to be applied.
        """
        self.root_dir = root_dir
        self.transform = transform
        # List out only images
        self.image_paths = [
            os.path.join(self.root_dir, f)
            for f in os.listdir(self.root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        print(f"[DEBUG] Attempting to load: {img_path}")
        try:
            image = Image.open(img_path)
            print(f"[DEBUG] Original mode: {image.mode}")
            image = image.convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"[ERROR] Could not load {img_path}. Error: {e}")
            raise

# 3) Define transforms and instantiate the Datasets + DataLoaders
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

pos_dataset = CustomImageDataset(root_dir='data/positive', transform=transform)
neg_dataset = CustomImageDataset(root_dir='data/negative', transform=transform)
test_dataset = CustomImageDataset(root_dir='data/test', transform=transform)

pos_loader = DataLoader(pos_dataset, batch_size=1, shuffle=True)
neg_loader = DataLoader(neg_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 4) Helper to convert a single image tensor -> base64
def encode_image_to_base64(image_tensor):
    """
    image_tensor: torch.Tensor of shape [1, C, H, W], batch_size=1
    We'll:
      - Convert to PIL,
      - Save to buffer,
      - Base64-encode result.
    """
    # Remove batch dimension -> shape [C, H, W]
    image_tensor = image_tensor.squeeze(0)

    # Convert from Tensor -> PIL Image
    to_pil = T.ToPILImage()
    pil_img = to_pil(image_tensor)

    import io
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG')
    buffer.seek(0)

    img_bytes = buffer.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64

# 5) Build the in-context prompt
def build_in_context_prompt(pos_img_b64, neg_img_b64, test_img_b64):
    """
    Example of a text-based prompt structure that references images. 
    The exact schema depends on Gemini's API for multi-image in-context input.
    """
    prompt = {
        "messages": [
            {"role": "system", "content": "You are a medical image classification AI."},
            {"role": "user", "content": "Here are two examples of MRI images labeled by tumor presence."},
            {"role": "user", "content": f"Positive example:\n<image>{pos_img_b64}</image>\nLabel: Positive"},
            {"role": "user", "content": f"Negative example:\n<image>{neg_img_b64}</image>\nLabel: Negative"},
            {"role": "user", "content": "Now classify this new image:"},
            {"role": "user", "content": f"<image>{test_img_b64}</image>"}
        ]
    }
    return prompt

# 6) Gemini API call
def gemini_api_call(prompt):
    """
    Sends the prompt to Gemini's endpoint and returns the raw JSON response.
    Adjust as needed for Gemini's actual endpoint and request structure.
    """
    endpoint_url = "https://api.gemini.ai/v1/chat/completions"  # Replace with real
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json",
    }
    # Send the request
    response = requests.post(endpoint_url, headers=headers, json=prompt)
    # Raise if error
    response.raise_for_status()
    return response.json()

# 7) Parsing the Gemini response
def parse_gemini_response(response):
    """
    Extract classification from the model's text.
    The exact JSON structure depends on Gemini's response format.
    """
    try:
        model_text = response["choices"][0]["message"]["content"]
        return model_text.strip()
    except KeyError:
        return "ERROR: Unexpected response format"

def main():
    for pos_img, neg_img, test_img in zip(pos_loader, neg_loader, test_loader):
        # 1) Convert to base64
        pos_b64 = encode_image_to_base64(pos_img)
        neg_b64 = encode_image_to_base64(neg_img)
        test_b64 = encode_image_to_base64(test_img)

        # 2) Build prompt
        prompt = build_in_context_prompt(pos_b64, neg_b64, test_b64)

        # 3) Call Gemini
        response = gemini_api_call(prompt)

        # 4) Parse prediction
        prediction_text = parse_gemini_response(response)
        print("Gemini Model Prediction:", prediction_text)
        print("----------------------------------------------------")

if __name__ == "__main__":
    main()