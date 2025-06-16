#!/usr/bin/env python
"""
dinov2_knn.py
=============

Build a neighbour table (CSV) that lists the **top-K most similar _training_ tiles**
for **every query tile** that might appear at inference time.

Key changes
-----------
* Accept **two** CSVs:
    • `--train-csv`  – images that form the neighbour pool (usually the train split)  
    • `--query-csv`  – images that will ever be looked up (train + test)

* Normalise every filename to lower-case and strip whitespace → avoids ".PNG" vs
  ".png" or trailing-space mismatches.

* Early sanity checks: duplicates, missing files, and 1-to-1 coverage.

* Same output format as before:
        query,neighbor_1,…,neighbor_K
        
        
python dinov2_knn.py \
    --train-csv data/CRC100K/binary/CRC100K_dataset.csv \
    --query-csv data/CRC100K/binary/CRC100K_all_data.csv \
    --out-csv  data/CRC100K/binary/knn_32.csv
"""

# ───────────────────────────────────────────────────────── CLI ──
from __future__ import annotations
import argparse, csv, sys, os
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--train-csv", required=True,
                    help="CSV that provides the neighbour pool (must contain columns fname,label,path)")
parser.add_argument("--query-csv", required=True,
                    help="CSV that provides every possible query image (train + test)")
parser.add_argument("--root-dir",  default=".", help="Repo root – used to resolve relative paths")
parser.add_argument("--k",         type=int, default=32, help="Neighbours to keep per query")
parser.add_argument("--model",     default="vit_large_patch14_dinov2.lvd142m",
                    help="Any timm vision transformer; must accept 224×224 images")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--device",    default="cuda:0" if __import__("torch").cuda.is_available() else "cpu")
parser.add_argument("--out-csv",   default="knn_32.csv")
args = parser.parse_args()

# ───────────────────────────────────────────────────────── Imports ──
import pandas as pd
import torch, numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import timm, faiss

# ───────────────────────────────────────────────────────── Helpers ──
def load_model(name: str, device: torch.device):
    model = timm.create_model(name, pretrained=True)
    model.eval().to(device)
    return model

def build_transform(target_px: int):
    resize = int(target_px * 256 / 224)
    return transforms.Compose([
        transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(target_px),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std =(0.229, 0.224, 0.225)),
    ])

@torch.inference_mode()
def extract_feats(paths: list[Path], model, tfm, device, bs: int):
    feats = []
    for i in tqdm(range(0, len(paths), bs), desc="Extracting features"):
        imgs = [tfm(Image.open(p).convert("RGB")) for p in paths[i:i+bs]]
        x    = torch.stack(imgs).to(device)
        out  = model.forward_features(x)
        cls  = out["x_norm_clstoken"] if isinstance(out, dict) else \
               (out[:, 0] if out.ndim == 3 else out)
        feats.append(cls.cpu())
    feats = torch.cat(feats).numpy()
    faiss.normalize_L2(feats)
    return feats

def to_abs(p: str | Path, csv_dir: Path, root_dir: Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    a = csv_dir / p
    b = root_dir / p
    return a if a.exists() else b

# ───────────────────────────────────────────────────────── Load data ──
root_dir  = Path(args.root_dir).resolve()

def load_csv(csv_path: str) -> pd.DataFrame:
    """Read a CRC100K-style CSV and return df with absolute paths."""
    csv_path = Path(csv_path).resolve()
    df = pd.read_csv(csv_path)
    for col in ("fname", "path"):
        if col not in df.columns:
            print(f"❌ CSV {csv_path} missing “{col}” column", file=sys.stderr)
            sys.exit(1)

    csv_dir = csv_path.parent
    df["fname"] = df["fname"].astype(str).str.strip().str.lower()
    df["abs_path"] = df["path"].apply(lambda p: to_abs(p, csv_dir, root_dir))
    missing = [p for p in df["abs_path"] if not p.exists()]
    if missing:
        print(f"❌ {len(missing)} files listed in {csv_path} do not exist, e.g. {missing[0]}", file=sys.stderr)
        sys.exit(1)
    return df

print("[INFO] Reading CSVs …")
train_df  = load_csv(args.train_csv)
query_df  = load_csv(args.query_csv)

# duplicate-check
dup = query_df["fname"].duplicated(keep=False)
if dup.any():
    dups = query_df.loc[dup, "fname"].unique()[:5]
    print(f"❌ Duplicate query filenames found: {dups} …", file=sys.stderr)
    sys.exit(1)

# Sanity: every query must have label/path
print(f"[INFO] {len(train_df)} training tiles  → neighbour pool")
print(f"[INFO] {len(query_df)} query   tiles  → will appear at inference")

# ───────────────────────────────────────────────────────── Feature extraction ──
device     = torch.device(args.device)
model      = load_model(args.model, device)
img_px     = model.patch_embed.img_size
img_px     = img_px[0] if isinstance(img_px, (tuple, list)) else img_px
transform  = build_transform(img_px)

print("[INFO] Extracting features for TRAIN (index) …")
train_feats = extract_feats(train_df["abs_path"].tolist(), model, transform, device, args.batch_size)

print("[INFO] Building Faiss index …")
index = faiss.IndexFlatIP(train_feats.shape[1])
if "cuda" in args.device and faiss.get_num_gpus():
    index = faiss.index_cpu_to_all_gpus(index)
index.add(train_feats.astype(np.float32))

print("[INFO] Extracting features for QUERY images …")
query_feats = extract_feats(query_df["abs_path"].tolist(), model, transform, device, args.batch_size)

print("[INFO] Running search …")
_, idx = index.search(query_feats, args.k + 1)      # +1 because first neighbour is itself **if** query ⊂ train

# Map neighbours to filenames
train_fnames = train_df["fname"].tolist()
rows = []
for q, neigh in zip(query_df["fname"], idx):
    neighbours = [train_fnames[i] for i in neigh if train_fnames[i] != q][:args.k]
    rows.append([q] + neighbours)

# ───────────────────────────────────────────────────────── Save ──
header = ["query"] + [f"neighbor_{i}" for i in range(1, args.k + 1)]
out_csv = Path(args.out_csv)
with out_csv.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"✓ Saved {len(rows)} rows → {out_csv.resolve()}")