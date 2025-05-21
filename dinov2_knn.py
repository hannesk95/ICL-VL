#!/usr/bin/env python
"""
dinov2_knn.py  –  top-32 neighbours for each histology tile (CSV: fname,label,path)
"""

# ---------------------------- CONFIG -------------------------------- #
CSV_FILE   = "data/CRC100K/binary/CRC100K_dataset.csv"   # your CSV
ROOT_DIR   = "."            # repo root that contains the “data/” folder
K_NEIGH    = 32
MODEL_NAME = "vit_large_patch14_dinov2.lvd142m"          # 224-px DINOv2-L
BATCH_SIZE = 32
DEVICE     = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
OUT_CSV    = "knn_32.csv"
# -------------------------------------------------------------------- #

import csv, sys
from pathlib import Path
import pandas as pd
import torch, numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm, faiss
from tqdm import tqdm


# ----------------------- helpers ------------------------------------ #
def load_model(name, device):
    try:
        return timm.create_model(name, pretrained=True).eval().to(device)
    except RuntimeError as e:
        if "Invalid pretrained tag" in str(e):
            return timm.create_model(name.split(".")[0], pretrained=True).eval().to(device)
        raise


def build_transform(target_px):
    resize = int(target_px * 256 / 224)
    return transforms.Compose([
        transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(target_px),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std =(0.229, 0.224, 0.225)),
    ])


@torch.inference_mode()
def extract_feats(paths, model, tfm, device, bs):
    feats = []
    for i in tqdm(range(0, len(paths), bs), desc="Extracting features"):
        imgs = [tfm(Image.open(p).convert("RGB")) for p in paths[i:i+bs]]
        x    = torch.stack(imgs).to(device)
        out  = model.forward_features(x)
        if isinstance(out, dict):          # timm ≥0.9
            cls = out["x_norm_clstoken"]   # (B,D)
        else:                              # older timm fallback
            cls = out[:, 0] if out.ndim == 3 else out
        feats.append(cls.cpu())
    feats = torch.cat(feats).numpy()
    faiss.normalize_L2(feats)
    return feats


def knn_topk(feats, k, gpu=True):
    N, D = feats.shape
    index = faiss.IndexFlatIP(D)
    if gpu and faiss.get_num_gpus():
        index = faiss.index_cpu_to_all_gpus(index)
    index.add(feats.astype(np.float32))
    _, idx = index.search(feats, k+1)
    return idx[:, 1:]                      # drop self-match


# ----------------------- pipeline ----------------------------------- #
def main():
    df = pd.read_csv(CSV_FILE)

    # sanity-check columns
    for col in ["fname", "path"]:
        if col not in df.columns:
            print(f"❌ CSV is missing column “{col}” (columns present: {list(df.columns)})")
            sys.exit(1)

    csv_dir  = Path(CSV_FILE).resolve().parent
    root_dir = Path(ROOT_DIR).resolve()

    # make absolute paths
    def resolve(p):
        p = Path(p)
        if p.is_absolute(): return p
        a = csv_dir / p
        b = root_dir / p
        return a if a.exists() else b
    df["abs_path"] = df["path"].apply(resolve).apply(lambda p: p.resolve())

    missing = [p for p in df["abs_path"] if not p.exists()]
    if missing:
        print("❌ missing file:", missing[0]); sys.exit(1)

    device = torch.device(DEVICE)
    model  = load_model(MODEL_NAME, device)
    img_px = model.patch_embed.img_size
    img_px = img_px[0] if isinstance(img_px, (tuple,list)) else img_px
    tfm    = build_transform(img_px)

    feats = extract_feats(df["abs_path"].tolist(), model, tfm, device, BATCH_SIZE)
    idx   = knn_topk(feats, K_NEIGH, gpu="cuda" in DEVICE)

    # ---------------- write output ---------------- #
    fnames = df["fname"].tolist()
    header = ["query"] + [f"neighbor_{i}" for i in range(1, K_NEIGH+1)]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for q, nbr in zip(fnames, idx):
            writer.writerow([q] + [fnames[j] for j in nbr])

    print(f"✓ Saved {K_NEIGH} neighbours per tile → {OUT_CSV}")


if __name__ == "__main__":
    main()