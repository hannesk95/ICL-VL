import google.generativeai as genai
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import os

# ------------------------------
# 1. Configure Gemini API
# ------------------------------
API_KEY = "XX"  # Replace with actual API key
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite")

# ------------------------------
# 2. Define Data Loaders
# ------------------------------
class MedicalImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = sorted([
            f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))
        ])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

# Transformations
transform = T.Compose([
    T.Resize((256, 256)), 
    T.ToTensor()
])

# Create datasets
pos_dataset = MedicalImageDataset("data/positive", transform)
neg_dataset = MedicalImageDataset("data/negative", transform)
test_dataset = MedicalImageDataset("data/test", transform)

# Create data loaders
batch_size = 1
pos_loader = DataLoader(pos_dataset, batch_size=batch_size, shuffle=True)
neg_loader = DataLoader(neg_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# ------------------------------
# 3. Process Images Using In-Context Learning
# ------------------------------
def format_in_context_prompt(pos_img, neg_img, test_img):
    """
    Formats the images and text prompts for Gemini's in-context learning.
    """
    return [
        pos_img, "This is a medical scan with a tumor present.",
        neg_img, "This is a medical scan with no tumor.",
        test_img, "Does this image contain a tumor?"
    ]

# Iterate over datasets in triplets
for (test_img, pos_img, neg_img) in zip(test_loader, pos_loader, neg_loader):
    # Convert PyTorch tensors back to PIL images
    test_pil = T.ToPILImage()(test_img.squeeze(0))
    pos_pil = T.ToPILImage()(pos_img.squeeze(0))
    neg_pil = T.ToPILImage()(neg_img.squeeze(0))

    # Prepare the in-context prompt
    prompt = format_in_context_prompt(pos_pil, neg_pil, test_pil)

    # Send request to Gemini
    try:
        print("🟡 Sending images to Gemini...")
        response = model.generate_content(prompt)
        print(f"✅ Model Prediction: {response.text}")
    except Exception as e:
        print(f"❌ API Error: {e}")