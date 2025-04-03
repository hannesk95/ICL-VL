import os
import base64
import io
import torch
from dotenv import load_dotenv
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import re


import google.generativeai as genai

#######################################
# 1) Load and Configure
#######################################
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY. Please set it in .env or your environment.")

# Configure the google.generativeai library
genai.configure(api_key=GEMINI_API_KEY)

#######################################
# 2) Dataset & DataLoader
#######################################
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(self.root_dir, f)
            for f in os.listdir(self.root_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, img_path
        except Exception as e:
            print(f"[ERROR] Could not load {img_path}. Error: {e}")
            raise

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

pos_dataset = CustomImageDataset(root_dir='data/positive', transform=transform)
neg_dataset = CustomImageDataset(root_dir='data/negative', transform=transform)
test_dataset = CustomImageDataset(root_dir='data/test', transform=transform)

pos_loader = DataLoader(pos_dataset, batch_size=1, shuffle=True)
neg_loader = DataLoader(neg_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

#######################################
# 3) Helper Functions
#######################################
def encode_image_to_base64(image_tensor):
    """Converts a single image tensor to a base64-encoded JPEG string."""
    image_tensor = image_tensor.squeeze(0)  # remove batch dimension
    pil_img = T.ToPILImage()(image_tensor)

    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG')
    buffer.seek(0)

    img_bytes = buffer.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_b64

def build_in_context_prompt_str(pos_img_b64, neg_img_b64, test_imgs_b64):
    """
    Create a single text prompt containing:
    - Quick instructions
    - One positive and one negative example
    - The new test images
    """
    test_images_str = ""
    for i, test_b64 in enumerate(test_imgs_b64, start=1):
        test_images_str += f"Image {i}:\n<image>{test_b64}</image>\n"

    prompt_str = (
        "System instructions: You are a helpful medical image classification AI. "
        "Classify each test image as Positive (tumor) or Negative.\n\n"
        "User context: Here are two examples of MRI images labeled by tumor presence.\n\n"
        "Positive example:\n"
        f"<image>{pos_img_b64}</image>\nLabel: Positive\n\n"
        "Negative example:\n"
        f"<image>{neg_img_b64}</image>\nLabel: Negative\n\n"
        "Now classify these new images:\n\n"
        f"{test_images_str}"
    )
    return prompt_str

def gemini_api_call(prompt_str):
    """
    Use generate_content() with the older google.generativeai library,
    which doesn't accept 'temperature' or 'generate_text()'.
    """
    # Instantiates the model interface
    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    # generate_content() without any temperature argument
    # The library's version in your environment likely doesn't support advanced params.
    response = model.generate_content(prompt_str)

    # In many older versions, the response text is in `response.text`
    # If that doesn't exist, try `response.result` or `response.generations[0].content`.
    # Adjust if needed:
    return response.text

#######################################
# 4) Main
#######################################
def main():
    # Grab a single positive and negative example
    pos_iter = iter(pos_loader)
    neg_iter = iter(neg_loader)

    try:
        pos_img, _ = next(pos_iter)
        neg_img, _ = next(neg_iter)
    except StopIteration:
        print("[ERROR] Not enough images in 'positive' or 'negative' dataset.")
        return

    # Encode them to base64
    pos_b64 = encode_image_to_base64(pos_img)
    neg_b64 = encode_image_to_base64(neg_img)

    # Iterate over test data in batches (batch_size=4 by default)
    for test_batch in test_loader:
        test_imgs, test_paths = test_batch  # shape: [batch_size, 3, 224, 224]

        # Convert each test image to base64
        test_imgs_b64 = [
            encode_image_to_base64(img.unsqueeze(0)) for img in test_imgs
        ]

        # Build the combined text prompt
        prompt_str = build_in_context_prompt_str(pos_b64, neg_b64, test_imgs_b64)

        # Call the model via generate_content()
        prediction_text = gemini_api_call(prompt_str)

        print("Gemini Model Prediction:")
        print(prediction_text)
        print("----------------------------------------------------")

        # Extract only classification lines using regex
        classification_lines = [
            line.strip()
            for line in prediction_text.strip().split("\n")
            if re.match(r"[*\-]\s+\*\*Image\s+\d+.*", line)
        ]

        # Fallback if the model uses different formatting (e.g., "Classification: ...")
        if not classification_lines:
            classification_lines = [
                line.strip()
                for line in prediction_text.strip().split("\n")
                if "Classification" in line
            ]

        # Match predictions to test image paths
        for path, line in zip(test_paths, classification_lines):
            print(f"Image: {path} => Prediction: {line}")

if __name__ == "__main__":
    main()