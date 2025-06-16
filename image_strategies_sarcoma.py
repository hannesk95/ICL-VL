#!/usr/bin/env python3
"""image_strategies.py

Export 2-D **images *and* masks** from 3-D medical volumes while keeping enough
spatial context for both vision-language models (Gemini Flash, Med-Flamingo …)
and 2-D radiomics pipelines.

For each `(image, mask)` pair the script selects the axial slice that contains
**the largest tumour cross-section** and can emit up to three representations
— each with a corresponding mask file:

| Strategy              | RGB channels                                          | Extra flags        | Mask that is saved                         |
|-----------------------|--------------------------------------------------------|--------------------|--------------------------------------------|
| `same_slice_rgb`      | `[z , z , z ]` – duplicate the slice into R,G,B        | –                  | Central slice                              |
| `neighbor_stack`      | `[z-Δ, z, z+Δ]` – pseudo-2½-D local context            | `--delta Δ`        | Central slice                              |
| `mip_axial`           | Axial MIP of slab `z ± S`                             | `--slab S`         | Axial MIP of mask slab `z ± S`             |

---

# 1) All three representations (default)
python image_strategies.py \
  --input_dir  data/sarcoma/sarcoma/train/T1 \
  --output_dir data/sarcoma/train/T1_png

# 2) Only the duplicated single slice (RGB + mask)
python image_strategies.py \
  --input_dir  …/train/T1 \
  --output_dir …/T1_png \
  --strategies same_slice_rgb

# 3) Neighbour stack with a larger offset (Δ = 2)
python image_strategies.py \
  --input_dir  …/train/T1 \
  --output_dir …/T1_png \
  --strategies neighbor_stack \
  --delta 2

# 4) Axial MIP of a thicker slab (S = 7 ⇒ 15-slice slab)
python image_strategies.py \
  --input_dir  …/train/T1 \
  --output_dir …/T1_png \
  --strategies mip_axial \
  --slab 7
```

Optional global flags:
* `--spacing 0.8 0.8 0.8`   # resample voxels to 0.8 mm isotropic
* `--ext .nii`              # if volumes are plain `.nii` (not gzipped)

Install dependencies: `pip install torchio numpy imageio tqdm`
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Tuple

import numpy as np
import imageio.v2 as imageio  # legacy API for imwrite
import torchio as tio

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:  # fallback when tqdm is missing
    def tqdm(iterable, *args, **kwargs):
        return iterable

# ==============================================================================
# (image, mask) pair discovery – mask name inserts "-label" right before EXT
# ==============================================================================

def find_image_mask_pairs(folder: pathlib.Path, ext: str) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    pairs: List[Tuple[pathlib.Path, pathlib.Path]] = []
    for img_path in folder.rglob(f"*{ext}"):
        if "-label" in img_path.stem:
            continue  # skip mask files
        mask_path = img_path.with_name(img_path.name.replace(ext, f"-label{ext}"))
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            print(f"[WARN] mask missing for {img_path.name} – skipped.")
    return pairs

# ==============================================================================
# Canonical orientation + isotropic resampling
# ==============================================================================

def preprocess_volume(img_p: pathlib.Path, mask_p: pathlib.Path, spacing: Tuple[float, float, float]):
    subj = tio.Compose([tio.ToCanonical(), tio.Resample(spacing)])(
        tio.Subject(image=tio.ScalarImage(img_p), mask=tio.LabelMap(mask_p))
    )
    img_np = subj.image.data.squeeze().numpy()           # (Z, Y, X)
    mask_np = subj.mask.data.squeeze().numpy().astype(np.uint8)
    return img_np, mask_np

# ==============================================================================
# Utilities – pick slice & convert to uint8
# ==============================================================================

def largest_tumour_slice(mask_vol: np.ndarray) -> int:
    """Return the z-index where the tumour area (# non-zero voxels) is maximal."""
    return int((mask_vol > 0).sum(axis=(1, 2)).argmax())


def _norm_to_uint8(arr2d: np.ndarray) -> np.ndarray:
    arr = arr2d.astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return (arr * 255).astype(np.uint8)


def _mask_to_uint8(mask2d: np.ndarray) -> np.ndarray:
    return (mask2d > 0).astype(np.uint8) * 255

# ==============================================================================
# Strategy builders
# ==============================================================================

# 1) same_slice_rgb -----------------------------------------------------------------

def make_same_slice_rgb(vol: np.ndarray, z: int) -> np.ndarray:
    g = _norm_to_uint8(vol[z])
    return np.stack([g, g, g], axis=-1)

# 2) neighbor_stack -----------------------------------------------------------------

def make_neighbor_stack(vol: np.ndarray, z: int, delta: int = 1) -> np.ndarray:
    z_min, z_max = 0, vol.shape[0] - 1
    zs = [np.clip(z - delta, z_min, z_max), z, np.clip(z + delta, z_min, z_max)]
    chans = [_norm_to_uint8(vol[z_]) for z_ in zs]
    return np.stack(chans, axis=-1)

# 3) mip_axial ----------------------------------------------------------------------

def make_mip_axial(vol: np.ndarray, z: int, slab: int = 5) -> np.ndarray:
    z0, z1 = max(z - slab, 0), min(z + slab + 1, vol.shape[0])
    mip = vol[z0:z1].max(axis=0)
    g = _norm_to_uint8(mip)
    return np.stack([g, g, g], axis=-1)

# ==============================================================================
# Writers – images & masks
# ==============================================================================

def save_png(rgb: np.ndarray, path: pathlib.Path):
    imageio.imwrite(path, rgb)


def save_mask(mask2d: np.ndarray, path: pathlib.Path):
    imageio.imwrite(path, _mask_to_uint8(mask2d))

# ==============================================================================
# CLI
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description="Export 2-D context-rich images *and* masks from 3-D volumes")
    ap.add_argument("--input_dir", type=pathlib.Path, required=True, help="Folder with *.nii.gz volumes & *-label.nii.gz masks")
    ap.add_argument("--output_dir", type=pathlib.Path, required=True, help="Destination folder for PNGs")
    ap.add_argument("--spacing", type=float, nargs=3, default=(1, 1, 1), metavar=("SX", "SY", "SZ"), help="Isotropic voxel size after resampling (mm)")
    ap.add_argument("--strategies", nargs="+", default=["all"], choices=["same_slice_rgb", "neighbor_stack", "mip_axial", "all"], help="Which representations to generate")
    ap.add_argument("--delta", type=int, default=1, help="Neighbour offset for neighbor_stack")
    ap.add_argument("--slab", type=int, default=5, help="Half-width of axial-MIP slab (slices)")
    ap.add_argument("--ext", type=str, default=".nii.gz", help="File extension of volumes (default .nii.gz)")
    args = ap.parse_args()

    # Resolve strategy list
    strategies = ["same_slice_rgb", "neighbor_stack", "mip_axial"] if "all" in args.strategies else args.strategies

    # Basic checks
    if not args.input_dir.is_dir():
        sys.exit(f"Input directory {args.input_dir} not found.")

    pairs = find_image_mask_pairs(args.input_dir, args.ext)
    if not pairs:
        sys.exit("No image/mask pairs discovered – check naming convention or --ext flag.")

    # ----------------------------------------------------------------------------
    for img_p, mask_p in tqdm(pairs, desc="Processing cases"):
        case_id = img_p.stem  # filename without extension
        img_vol, mask_vol = preprocess_volume(img_p, mask_p, spacing=tuple(args.spacing))
        z = largest_tumour_slice(mask_vol)

        out_dir = args.output_dir / case_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # ------- Strategy: same_slice_rgb --------------------------------------
        if "same_slice_rgb" in strategies:
            save_png(make_same_slice_rgb(img_vol, z), out_dir / f"{case_id}_z{z:03d}_same_rgb.png")
            save_mask(mask_vol[z], out_dir / f"{case_id}_z{z:03d}_mask.png")

        # ------- Strategy: neighbor_stack --------------------------------------
        if "neighbor_stack" in strategies:
            save_png(make_neighbor_stack(img_vol, z, delta=args.delta), out_dir / f"{case_id}_z{z:03d}_neigh{args.delta}.png")
            save_mask(mask_vol[z], out_dir / f"{case_id}_z{z:03d}_neigh_mask.png")

        # ------- Strategy: mip_axial -------------------------------------------
        if "mip_axial" in strategies:
            save_png(make_mip_axial(img_vol, z, slab=args.slab), out_dir / f"{case_id}_z{z:03d}_mipax_slab{args.slab}.png")
            z0, z1 = max(z - args.slab, 0), min(z + args.slab + 1, mask_vol.shape[0])
            mip_mask = (mask_vol[z0:z1] > 0).max(axis=0).astype(np.uint8)
            save_mask(mip_mask, out_dir / f"{case_id}_z{z:03d}_mipax_slab{args.slab}_mask.png")

    # ----------------------------------------------------------------------------
    print(f"\nDone – outputs saved under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
