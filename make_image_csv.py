#!/usr/bin/env python3
"""
make_image_csv.py
─────────────────
Create a CSV catalogue of PNGs produced by *image_strategies.py*.

Example
───────
# same-slice RGB images
python make_image_csv.py \
    --root  data/glioma/png_T1           \
    --keyword same_rgb                \
    --out  same_rgb_images.csv

# axial MIP images (keyword is 'mipax')
python make_image_csv.py \
    --root  data/glioma/png_T1c           \
    --keyword mipax                   \
    --out  mip_images.csv
    
python make_image_csv.py \
    --root  data/glioma/png_T2           \
    --keyword same_rgb                \
    --out  same_rgb_images.csv
"""
import argparse
import csv
import pathlib
from typing import List, Tuple


def collect_images(root: pathlib.Path,
                   keyword: str,
                   ext: str = ".png") -> List[Tuple[str, str]]:
    """
    Return a list of (filename, absolute_path) tuples for every file whose
    *name* contains ``keyword`` and ends with ``ext`` (default: ".png").
    """
    imgs = []
    glob = f"*{keyword}*{ext}"
    for p in root.rglob(glob):
        if p.is_file():
            imgs.append((p.name, str(p.resolve())))
    return sorted(imgs)            # deterministic order


def write_csv(rows: List[Tuple[str, str]], out_csv: pathlib.Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["fname", "path"])   # header
        wr.writerows(rows)
    print(f"CSV written → {out_csv.resolve()}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--root", type=pathlib.Path, required=True,
                    help="Folder that contains the PNGs (searches recursively)")
    ap.add_argument("--keyword", type=str, default="same_rgb",
                    help="Substring that must appear in the filename")
    ap.add_argument("--ext", type=str, default=".png",
                    help="Image filename extension")
    ap.add_argument("--out", type=pathlib.Path, required=True,
                    help="Destination CSV file")
    args = ap.parse_args()

    if not args.root.is_dir():
        ap.error(f"root folder {args.root} does not exist or is not a directory")

    rows = collect_images(args.root, args.keyword, args.ext)
    if not rows:
        print("No matching images found – check --root, --keyword or --ext.")
        return

    write_csv(rows, args.out)


if __name__ == "__main__":
    main()