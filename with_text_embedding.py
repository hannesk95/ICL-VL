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

# For demonstration, keep batch_size=1 for the positive/negative so we can iterate exactly as many as we need.
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

def build_in_context_prompt_str(pos_imgs_b64, neg_imgs_b64, test_imgs_b64):
    """
    Create a prompt with:
    - Clear instructions
    - Labeled positive/negative examples
    - Unlabeled test images for classification
    """

    # Introductory instructions
    prompt_str = (
        "System instructions:\n"
        "You are a medical image classification model. Each MRI scan is labeled as either:\n"
        "- Positive: Tumor is present\n"
        "- Negative: No tumor\n\n"
        "Use the examples below to learn the format, then classify the test images.\n\n"
    )

    # Add labeled examples (positives and negatives combined with consistent format)
    all_examples = [(b64, "Positive") for b64 in pos_imgs_b64] + [(b64, "Negative") for b64 in neg_imgs_b64]
    
    for idx, (b64_str, label) in enumerate(all_examples, start=1):
        prompt_str += (
            f"Example {idx}:\n"
            f"<image>{b64_str}</image>\n"
            f"Classification: {label}\n\n"
        )

    # Add test images for classification
    prompt_str += "---\n\nNow classify these test images:\n\n"

    for i, test_b64 in enumerate(test_imgs_b64, start=1):
        prompt_str += (
            f"Image {i}:\n"
            f"<image>{test_b64}</image>\n"
            f"Classification:\n\n"
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
    response = model.generate_content(prompt_str)

    return response.text

#######################################
# 4) Main
#######################################
def main(num_pos_examples=3, num_neg_examples=3):
    """
    Main function:
    num_pos_examples => how many positive examples to show
    num_neg_examples => how many negative examples to show
    """

    print(f"[INFO] Attempting to load {num_pos_examples} positive and {num_neg_examples} negative examples.")
    print(f"[INFO] Found {len(pos_dataset)} images in 'data/positive' directory.")
    print(f"[INFO] Found {len(neg_dataset)} images in 'data/negative' directory.")
    print(f"[INFO] Found {len(test_dataset)} images in 'data/test' directory.")

    # Gather multiple positive examples
    pos_iter = iter(pos_loader)
    pos_examples_b64 = []
    pos_paths = []
    for _ in range(num_pos_examples):
        try:
            pos_img, pos_path = next(pos_iter)
            pos_examples_b64.append(encode_image_to_base64(pos_img))
            pos_paths.append(pos_path)
        except StopIteration:
            print(f"[ERROR] Not enough images in 'positive' dataset to gather {num_pos_examples} examples.")
            break

    # Gather multiple negative examples
    neg_iter = iter(neg_loader)
    neg_examples_b64 = []
    neg_paths = []
    for _ in range(num_neg_examples):
        try:
            neg_img, neg_path = next(neg_iter)
            neg_examples_b64.append(encode_image_to_base64(neg_img))
            neg_paths.append(neg_path)
        except StopIteration:
            print(f"[ERROR] Not enough images in 'negative' dataset to gather {num_neg_examples} examples.")
            break

    # Debug prints: show which images we ended up loading for context
    print("\n[DEBUG] Positive example paths:")
    for path in pos_paths:
        print(f"    - {path}")
    print("\n[DEBUG] Negative example paths:")
    for path in neg_paths:
        print(f"    - {path}")

    if len(pos_examples_b64) < num_pos_examples or len(neg_examples_b64) < num_neg_examples:
        print("[WARNING] Fewer examples than requested were loaded. Proceeding anyway.\n")

    # Iterate over test data in batches (batch_size=4 by default)
    for test_batch in test_loader:
        test_imgs, test_paths = test_batch  # shape: [batch_size, 3, 224, 224]

        # Convert each test image to base64
        test_imgs_b64 = [
            encode_image_to_base64(img.unsqueeze(0)) for img in test_imgs
        ]

        # Build the combined text prompt
        prompt_str = build_in_context_prompt_str(pos_examples_b64, neg_examples_b64, test_imgs_b64)

        # Print the entire (or partial) prompt for debugging
        print("\n[DEBUG] Final Prompt (truncated to 1,000 characters):")
        print(prompt_str[:1000] + "...\n" if len(prompt_str) > 1000 else prompt_str)

        # Call the model via generate_content()
        prediction_text = gemini_api_call(prompt_str)

        print("[INFO] Gemini Model Prediction:")
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
        # (The length of classification_lines may be fewer than test_paths, so watch indexes)
        for path, line in zip(test_paths, classification_lines):
            print(f"Image: {path} => Prediction: {line}")

if __name__ == "__main__":
    # Example usage: request 5 positive and 5 negative examples
    main(num_pos_examples=5, num_neg_examples=5)