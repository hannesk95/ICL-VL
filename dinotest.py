import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from tqdm import tqdm
import faiss

# === Step 1: Load CSV with image paths ===
csv_path = "data/CRC100K/binary/CRC100K_dataset.csv"
df = pd.read_csv(csv_path)
image_paths = df["path"].tolist()

# === Step 2: Load DINOv2 model from Hugging Face ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
model.eval()

# === Step 3: Preprocessing helper ===
def preprocess_image(path):
    image = Image.open(path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    return inputs["pixel_values"]

# === Step 4: Extract embeddings ===
embeddings = []
valid_paths = []

for path in tqdm(image_paths, desc="Extracting DINOv2 embeddings"):
    try:
        pixel_values = preprocess_image(path).to(device)
        with torch.no_grad():
            outputs = model(pixel_values)
            feat = outputs.last_hidden_state[:, 0]  # CLS token
        embeddings.append(feat.squeeze().cpu().numpy())
        valid_paths.append(path)
    except Exception as e:
        print(f"Skipping {path}: {e}")

embeddings = np.stack(embeddings)

# === Step 5: Save embeddings and meta info ===
os.makedirs("embeddings", exist_ok=True)
np.save("embeddings/CRC100K_binary_dinov2_224.npy", embeddings)
np.save("embeddings/CRC100K_binary_dinov2_224.faiss.meta.npy", np.array(valid_paths))

# === Step 6: Build FAISS index ===
features = embeddings.astype("float32")
index = faiss.IndexFlatL2(features.shape[1])
index.add(features)
faiss.write_index(index, "embeddings/CRC100K_binary_dinov2_224.faiss")

print("✅ Embedding extraction and indexing complete!")