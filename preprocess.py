"""preprocess.py
Automated preprocessing pipeline for 3-D medical image volumes and their corresponding
segmentation masks.  For every image/mask pair inside *input_dir* the script:
    1. Loads the image volume and its mask.
    2. Converts both to the canonical RAS+ orientation (TorchIO `ToCanonical`).
    3. Resamples them to isotropic *spacing* (default: **1 mm³**).
    4. Finds the axial (z-axis) slice with the largest tumour area.
    5. Saves the selected image slice and mask slice as aligned 8-bit PNG files.

Typical invocation – **only the train/T1 folder**:
    python preprocess.py \
        --input_dir data/sarcoma/sarcoma/train/T1 \
        --output_dir data/sarcoma/sarcoma/train/T1_png

Dependencies
------------
* torchio   – orientation + resampling
* numpy
* imageio   – PNG saving
* tqdm      – progress bar (optional)

Install with  
    pip install torchio numpy imageio tqdm
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Tuple

import numpy as np
import imageio.v2 as imageio  # legacy API (imwrite, etc.)
import torchio as tio
# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:  # graceful fallback if tqdm is absent
    def tqdm(iterable, *args, **kwargs):
        return iterable


# =====================================================================================
# Helper functions
# =====================================================================================

def find_image_mask_pairs(input_dir: pathlib.Path, ext: str) -> List[Tuple[pathlib.Path, pathlib.Path]]:
    """Find (image, mask) pairs assuming *mask* filenames insert "-label" before *ext*.

    Example
    -------
    Image : ``Sar001T1.nii.gz``   → Mask : ``Sar001T1-label.nii.gz``
    Works for names such as ``Sar020_updatedT1.nii.gz`` as well.
    """
    pairs: List[Tuple[pathlib.Path, pathlib.Path]] = []

    for img_path in input_dir.rglob(f"*{ext}"):
        if "-label" in img_path.stem:  # skip mask files
            continue

        mask_name = img_path.name.replace(ext, f"-label{ext}")
        mask_path = img_path.with_name(mask_name)

        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            print(f"[WARN] Mask not found for {img_path.name} – case skipped.")
    return pairs


def preprocess_volume(
    img_path: pathlib.Path,
    mask_path: pathlib.Path,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Load → canonical → resample.  Returns numpy arrays shaped (Z, Y, X)."""

    subject = tio.Subject(
        image=tio.ScalarImage(img_path),
        mask=tio.LabelMap(mask_path),
    )

    # Canonical orientation (RAS+) then resample
    pipeline = tio.Compose([tio.ToCanonical(), tio.Resample(spacing)])
    subject = pipeline(subject)

    image_np = subject.image.data.squeeze().numpy()
    mask_np = subject.mask.data.squeeze().numpy().astype(np.uint8)

    return image_np, mask_np


def largest_tumour_slice(mask_vol: np.ndarray) -> int:
    """Return axial index (z) where mask has the most non-zero voxels."""
    areas = (mask_vol > 0).sum(axis=(1, 2))
    return int(areas.argmax())


def save_slice_png(
    img_slice: np.ndarray,
    mask_slice: np.ndarray,
    out_dir: pathlib.Path,
    case_id: str,
    slice_idx: int,
) -> None:
    """Save *aligned* image and mask slices as 8-bit PNGs."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalise intensities → [0, 255]
    img = img_slice.astype(np.float32)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()
    img_uint8 = (img * 255).astype(np.uint8)

    mask_uint8 = (mask_slice > 0).astype(np.uint8) * 255

    imageio.imwrite(out_dir / f"{case_id}_z{slice_idx:03d}.png", img_uint8)
    imageio.imwrite(out_dir / f"{case_id}_z{slice_idx:03d}_mask.png", mask_uint8)


# =====================================================================================
# Command-line interface
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="Extract the axial slice with biggest tumour area.")
    parser.add_argument("--input_dir", type=pathlib.Path, required=True, help="Folder that contains *.nii.gz volumes (e.g. test/T1)")
    parser.add_argument("--output_dir", type=pathlib.Path, required=True, help="Where PNG slices will be written")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0), metavar=("SX", "SY", "SZ"), help="Isotropic voxel size (mm). Default = 1 1 1")
    parser.add_argument("--ext", type=str, default=".nii.gz", help="File extension (default .nii.gz)")

    args = parser.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"Input directory '{args.input_dir}' not found.")

    pairs = find_image_mask_pairs(args.input_dir, args.ext)
    if not pairs:
        sys.exit("No image/mask pairs discovered. Check folder or naming convention.")

    for img_path, mask_path in tqdm(pairs, desc="Processing cases"):
        case_id = img_path.name.replace(args.ext, "")  # e.g. Sar001T1

        image_vol, mask_vol = preprocess_volume(img_path, mask_path, spacing=tuple(args.spacing))
        z_idx = largest_tumour_slice(mask_vol)

        save_slice_png(image_vol[z_idx], mask_vol[z_idx], args.output_dir, case_id, z_idx)

    print(f"\nDone – slices saved under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
