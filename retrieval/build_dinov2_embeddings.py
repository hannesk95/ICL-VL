#!/usr/bin/env python
# retrieval/build_dinov2_embeddings.py
"""
Embed training images with a DINOv2 ViT.

Example (native size):
    python retrieval/build_dinov2_embeddings.py \
        --csv data/CRC100K/binary/CRC100K_dataset.csv \
        --out embeddings/CRC100K_binary_dinov2.npz \
        --model vit_small_patch14_dinov2 \
        --img-size 518        # <-- native

Example (fast 224 px):
    python retrieval/build_dinov2_embeddings.py \
        --csv data/CRC100K/binary/CRC100K_dataset.csv \
        --out embeddings/CRC100K_binary_dinov2_224.npz \
        --model vit_small_patch14_dinov2 \
        --img-size 224
"""

import argparse, numpy as np, pandas as pd, timm, torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from pathlib import Path

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # force desired input size
    model = timm.create_model(
        args.model,
        pretrained=True,
        img_size=args.img_size,          # <- key line
        dynamic_img_size=True            # makes future size changes safe
    ).to(device).eval()

    tf = T.Compose([
        T.Resize(args.img_size),
        T.CenterCrop(args.img_size),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])

    df = pd.read_csv(args.csv)
    paths, labels, feats = [], [], []

    for path, lbl in tqdm(zip(df["path"], df["label"]), total=len(df), ncols=80):
        try:
            img = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        except FileNotFoundError:
            print(f"[WARN] missing: {path}"); continue
        with torch.no_grad(): feat = model(img).cpu().numpy()
        paths.append(path); labels.append(lbl); feats.append(feat)
        
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(
        args.out,
        paths=np.array(paths),
        labels=np.array(labels),
        feats=np.vstack(feats).astype("float32"),
    )
    print(f"[OK] saved {len(paths):,} embeddings ➜ {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="vit_small_patch14_dinov2")
    p.add_argument("--img-size", type=int, default=518, help="518 (native) or 224, etc.")
    main(p.parse_args())