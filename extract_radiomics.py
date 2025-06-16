#!/usr/bin/env python3
"""extract_radiomics.py 

Extract 2-D PyRadiomics features from the PNGs produced by `image_strategies.py`.

Usage
-----
python extract_radiomics.py \
  --input_dir  data/sarcoma/train/T1_png \
  --output_csv radiomics_features.csv \
  --spacing    0.8 0.8             # in-plane mm (optional)
  --settings_yaml radiomics_params.yaml  # optional PyRadiomics YAML

"""
from __future__ import annotations

import argparse
import pathlib
import re
from typing import List, Dict

import numpy as np
import pandas as pd
from PIL import Image
import SimpleITK as sitk
from radiomics import featureextractor  # type: ignore
from tqdm import tqdm

# ----------------------------------------------------------------------------
# Helper – map image png → expected mask png
# ----------------------------------------------------------------------------

def derive_mask_path(img_path: pathlib.Path) -> pathlib.Path:
    stem = img_path.stem
    if stem.endswith("_same_rgb"):
        mask_stem = stem.replace("_same_rgb", "_mask")
    elif re.search(r"_neigh\d+$", stem):
        mask_stem = re.sub(r"_neigh\d+$", "_neigh_mask", stem)
    elif re.search(r"_mipax_slab\d+$", stem):
        mask_stem = stem + "_mask"
    else:
        # Fallback to generic _mask
        mask_stem = stem + "_mask"
    return img_path.with_name(mask_stem + ".png")


def list_pairs(input_dir: pathlib.Path) -> List[Dict[str, pathlib.Path]]:
    pairs: List[Dict[str, pathlib.Path]] = []
    for img_path in input_dir.rglob("*.png"):
        if "_mask" in img_path.stem:
            continue  # skip mask files themselves
        mask_path = derive_mask_path(img_path)
        if mask_path.exists():
            # parse metadata (case, strategy)
            m = re.match(r"(?P<case>.+?)_z\d+_(?P<strategy>.+)$", img_path.stem)
            pairs.append({
                "image": img_path,
                "mask": mask_path,
                "case": m.group("case") if m else img_path.parent.name,
                "strategy": m.group("strategy") if m else "unknown",
            })
        else:
            print(f"[WARN] mask missing for {img_path.name} – skipped.")
    return pairs

# ----------------------------------------------------------------------------
# I/O – PNG → SimpleITK (grayscale)
# ----------------------------------------------------------------------------

def png_to_sitk(path: pathlib.Path, spacing_xy: tuple[float, float]) -> sitk.Image:
    arr = np.array(Image.open(path).convert("L"), dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing_xy + (1.0,))  # dummy z-spacing for 2-D
    return img

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Extract 2-D radiomic features from PNG + mask pairs")
    ap.add_argument("--input_dir", type=pathlib.Path, required=True)
    ap.add_argument("--output_csv", type=pathlib.Path, required=True)
    ap.add_argument("--spacing", type=float, nargs=2, default=(1.0, 1.0), metavar=("SX", "SY"), help="Pixel spacing (mm) in X and Y")
    ap.add_argument("--settings_yaml", type=pathlib.Path, default=None, help="Optional PyRadiomics YAML config")
    args = ap.parse_args()

    pairs = list_pairs(args.input_dir)
    if not pairs:
        ap.error("No image/mask pairs found. Check input_dir path and filename patterns.")

    # Configure PyRadiomics extractor
    if args.settings_yaml and args.settings_yaml.is_file():
        extractor = featureextractor.RadiomicsFeatureExtractor(str(args.settings_yaml))
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()
    extractor.settings.update({"force2D": True, "label": 255})

    rows = []
    for p in tqdm(pairs, desc="Radiomics"):
        img_itk = png_to_sitk(p["image"], spacing_xy=tuple(args.spacing))
        mask_itk = png_to_sitk(p["mask"], spacing_xy=tuple(args.spacing))
        res = extractor.execute(img_itk, mask_itk)
        features = {k: v for k, v in res.items() if k.startswith("original")}  # keep only feature keys
        features.update({"case": p["case"], "strategy": p["strategy"]})
        rows.append(features)

    df = pd.DataFrame(rows).sort_values(["case", "strategy"])
    df.to_csv(args.output_csv, index=False)
    print(f"\n✓ Saved {len(df)} rows to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
