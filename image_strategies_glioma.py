#!/usr/bin/env python3
"""
image_strategies.py  —  export 2-D context-rich images *and* masks from 3-D volumes
────────────────────────────────────────────────────────────────────────────────────
Works out-of-the-box with the UCSF-PDGM glioma dataset, whose files look like

    UCSF-PDGM-0004_FLAIR_bias.nii.gz            ← image we want
    UCSF-PDGM-0004_tumor_segmentation_merged.nii.gz  ← its mask

The script pairs      <PATIENT><IMG_SUFFIX><EXT>               with
                      <PATIENT><MASK_SUFFIX><EXT>

so the filenames never have to contain “-label”.

Examples
────────
# 1) default settings: process                                        *_FLAIR_bias
python image_strategies.py \
  --input_dir  data/glioma/glioma_four_sequences \
  --output_dir data/glioma/png

# 2) change which sequence is exported (e.g. T1c)
python image_strategies_glioma.py \
  --input_dir  data/glioma/glioma_four_sequences \
  --output_dir data/glioma/png_T1c \
  --img_suffix _T1c_bias

# 3) full control
python image_strategies.py \
  --input_dir  data/glioma/glioma_four_sequences \
  --output_dir data/glioma/png \
  --img_suffix  _FLAIR_bias \
  --mask_suffix _tumor_segmentation_merged \
  --spacing 0.8 0.8 0.8 \
  --strategies mip_axial neighbor_stack \
  --delta 2 \
  --slab 7
  
  python image_strategies_glioma.py \
  --input_dir  data/glioma/glioma_four_sequences \
  --output_dir data/glioma/png_T1 \
  --img_suffix _T1_bias
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Tuple

import numpy as np
import imageio.v2 as imageio           # legacy API for imwrite
import torchio as tio

# ─── Optional progress bar ──────────────────────────────────────────────────────
try:
    from tqdm import tqdm
except ImportError:                      # fallback when tqdm is missing
    def tqdm(iterable, *args, **kwargs):
        return iterable


# ════════════════════════════════════════════════════════════════════════════════
# (image, mask) discovery — any naming pattern you want
# ════════════════════════════════════════════════════════════════════════════════
def find_image_mask_pairs(
    folder: pathlib.Path,
    ext: str,
    img_suffix: str = "_FLAIR_bias",
    mask_suffix: str = "_tumor_segmentation_merged",
) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    """
    Pair volumes and masks based on a *shared* patient ID and configurable
    suffixes.

    Example:  *_FLAIR_bias.nii.gz     → image
              *_tumor_segmentation_merged.nii.gz → mask
    """
    pairs: List[Tuple[pathlib.Path, pathlib.Path]] = []
    glob_pattern = f"*{img_suffix}{ext}"           # e.g. "*_FLAIR_bias.nii.gz"

    for img_path in folder.rglob(glob_pattern):
        # Patient ID is everything before the first "_"
        pid = img_path.stem.split("_")[0]          # "UCSF-PDGM-0004"
        mask_name = f"{pid}{mask_suffix}{ext}"
        mask_path = img_path.with_name(mask_name)

        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            print(f"[WARN] mask missing for {img_path.name} – skipped.")

    return pairs


# ════════════════════════════════════════════════════════════════════════════════
# Canonical orientation + isotropic resampling
# ════════════════════════════════════════════════════════════════════════════════
def preprocess_volume(
    img_p: pathlib.Path,
    mask_p: pathlib.Path,
    spacing: Tuple[float, float, float],
):
    subj = tio.Compose([tio.ToCanonical(), tio.Resample(spacing)])(
        tio.Subject(image=tio.ScalarImage(img_p), mask=tio.LabelMap(mask_p))
    )
    img_np  = np.transpose(subj.image.data.squeeze().numpy(),  (2, 1, 0))  # (Z, Y, X)
    mask_np = np.transpose(
        subj.mask.data.squeeze().numpy().astype(np.uint8), (2, 1, 0)
    ) 
    return img_np, mask_np


# ════════════════════════════════════════════════════════════════════════════════
# Utilities — pick slice & convert to uint8
# ════════════════════════════════════════════════════════════════════════════════
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


# ════════════════════════════════════════════════════════════════════════════════
# Strategy builders
# ════════════════════════════════════════════════════════════════════════════════
def make_same_slice_rgb(vol: np.ndarray, z: int) -> np.ndarray:
    g = _norm_to_uint8(vol[z])
    return np.stack([g, g, g], axis=-1)


def make_neighbor_stack(vol: np.ndarray, z: int, delta: int = 1) -> np.ndarray:
    z_min, z_max = 0, vol.shape[0] - 1
    zs = [np.clip(z - delta, z_min, z_max), z, np.clip(z + delta, z_min, z_max)]
    chans = [_norm_to_uint8(vol[z_]) for z_ in zs]
    return np.stack(chans, axis=-1)


def make_mip_axial(vol: np.ndarray, z: int, slab: int = 5) -> np.ndarray:
    z0, z1 = max(z - slab, 0), min(z + slab + 1, vol.shape[0])
    mip = vol[z0:z1].max(axis=0)
    g = _norm_to_uint8(mip)
    return np.stack([g, g, g], axis=-1)


# ════════════════════════════════════════════════════════════════════════════════
# Writers
# ════════════════════════════════════════════════════════════════════════════════
def save_png(rgb: np.ndarray, path: pathlib.Path):
    imageio.imwrite(path, rgb)


def save_mask(mask2d: np.ndarray, path: pathlib.Path):
    imageio.imwrite(path, _mask_to_uint8(mask2d))


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export 2-D context-rich images *and* masks from 3-D volumes",
    )

    # I/O -----------------------------------------------------------------------
    ap.add_argument("--input_dir",  type=pathlib.Path, required=True,
                    help="Folder containing the *.nii(.gz) volumes and masks")
    ap.add_argument("--output_dir", type=pathlib.Path, required=True,
                    help="Destination folder for the PNGs")
    ap.add_argument("--ext", type=str, default=".nii.gz",
                    help="Filename extension of the volumes/masks")

    # Naming pattern ------------------------------------------------------------
    ap.add_argument("--img_suffix",  type=str, default="_FLAIR_bias",
                    help="Suffix that identifies the image volume")
    ap.add_argument("--mask_suffix", type=str, default="_tumor_segmentation_merged",
                    help="Suffix that identifies the mask")

    # Pre-processing ------------------------------------------------------------
    ap.add_argument("--spacing", type=float, nargs=3, default=(1, 1, 1),
                    metavar=("SX", "SY", "SZ"),
                    help="Isotropic voxel size after resampling (mm)")

    # Strategies ----------------------------------------------------------------
    ap.add_argument("--strategies", nargs="+", default=["all"],
                    choices=["same_slice_rgb", "neighbor_stack", "mip_axial", "all"],
                    help="Which representations to generate")
    ap.add_argument("--delta", type=int, default=1,
                    help="Neighbour offset for neighbor_stack")
    ap.add_argument("--slab",  type=int, default=5,
                    help="Half-width of axial-MIP slab (slices)")

    args = ap.parse_args()

    strategies = (["same_slice_rgb", "neighbor_stack", "mip_axial"]
                  if "all" in args.strategies else args.strategies)

    # Basic checks --------------------------------------------------------------
    if not args.input_dir.is_dir():
        sys.exit(f"Input directory {args.input_dir} not found.")

    pairs = find_image_mask_pairs(
        args.input_dir,
        args.ext,
        img_suffix=args.img_suffix,
        mask_suffix=args.mask_suffix,
    )
    if not pairs:
        sys.exit("No image/mask pairs discovered – check suffixes or --ext flag.")

    # Processing loop -----------------------------------------------------------
    for img_p, mask_p in tqdm(pairs, desc="Processing cases"):
        case_id = img_p.stem                       # e.g. UCSF-PDGM-0004_FLAIR_bias
        img_vol, mask_vol = preprocess_volume(img_p, mask_p,
                                              spacing=tuple(args.spacing))
        z = largest_tumour_slice(mask_vol)

        out_dir = args.output_dir / case_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Strategy: same_slice_rgb ---------------------------------------------
        if "same_slice_rgb" in strategies:
            save_png(make_same_slice_rgb(img_vol, z),
                     out_dir / f"{case_id}_z{z:03d}_same_rgb.png")
            save_mask(mask_vol[z],
                      out_dir / f"{case_id}_z{z:03d}_mask.png")

        # Strategy: neighbor_stack --------------------------------------------
        if "neighbor_stack" in strategies:
            save_png(make_neighbor_stack(img_vol, z, delta=args.delta),
                     out_dir / f"{case_id}_z{z:03d}_neigh{args.delta}.png")
            save_mask(mask_vol[z],
                      out_dir / f"{case_id}_z{z:03d}_neigh_mask.png")

        # Strategy: mip_axial --------------------------------------------------
        if "mip_axial" in strategies:
            save_png(make_mip_axial(img_vol, z, slab=args.slab),
                     out_dir / f"{case_id}_z{z:03d}_mipax_slab{args.slab}.png")
            z0, z1 = max(z - args.slab, 0), min(z + args.slab + 1, mask_vol.shape[0])
            mip_mask = (mask_vol[z0:z1] > 0).max(axis=0).astype(np.uint8)
            save_mask(mip_mask,
                      out_dir / f"{case_id}_z{z:03d}_mipax_slab{args.slab}_mask.png")

    print(f"\nDone – outputs saved under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()