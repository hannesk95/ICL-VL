# dinov2.py  –  MNIST × DINOv2 visualisation (fixed indentation)
"""
Simply run:

    python dinov2.py

The script:
* tight‑crops each MNIST digit (optional),
* resizes to a square 518 × 518 canvas (ViT‑B/14’s native resolution),
* extracts DINOv2 CLS embeddings,
* reduces to 2‑D via PCA→t‑SNE (better separation than raw PCA),
* writes **mnist_tsne.png** and opens a window if a GUI backend exists.

Install dependencies once:

```bash
pip install torch torchvision timm matplotlib scikit-learn umap-learn
```

You can experiment by editing the CONFIG section below (switch to UMAP, turn
cropping/inversion on‑off, etc.).
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torchvision.transforms as T
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import functional as F

# -----------------------------------------------------------------------------
# CONFIGURATION – tweak as you like
# -----------------------------------------------------------------------------
MODEL_NAME = "vit_base_patch14_dinov2.lvd142m"   # DINOv2 backbone
BATCH_SIZE = 256                                 # images per batch
SAMPLE_BATCHES: int | None = None                # None = full test set
SEED = 42
OUTPUT_PNG = "mnist_tsne.png"

METHOD = "tsne"          # "pca", "tsne", or "umap"
PERPLEXITY = 30           # t‑SNE hyper‑parameter
N_PCA = 50                # PCA dims before t‑SNE/UMAP

CROP_TIGHT = True         # remove black borders
INVERT = False            # flip colours (black→white background)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# Pre‑processing helpers
# -----------------------------------------------------------------------------
class TightCrop:
    """Crop a PIL image to the minimal non‑zero bounding box."""

    def __call__(self, img):
        np_img = np.array(img)
        coords = np.argwhere(np_img > 0)
        if coords.size == 0:
            return img  # blank – shouldn't happen
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return img.crop((x0, y0, x1, y1))

class Invert:
    def __call__(self, img):
        return F.invert(img)

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def gui_available() -> bool:
    import matplotlib, os
    backend = matplotlib.get_backend().lower()
    if any(b in backend for b in ("agg", "pdf", "svg")):
        return False
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def get_backbone(name: str = MODEL_NAME) -> torch.nn.Module:
    model = timm.create_model(name, pretrained=True, num_classes=0)
    model.eval().to(DEVICE)
    return model


def extract_features(model: torch.nn.Module, loader: DataLoader, max_batches: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    feats, labels = [], []
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if DEVICE == "cuda" else nullcontext()
    with torch.no_grad(), amp_ctx:
        for i, (x, y) in enumerate(loader):
            feats.append(model(x.to(DEVICE)).cpu())
            labels.append(y)
            if max_batches is not None and i >= max_batches - 1:
                break
    return torch.cat(feats).numpy(), torch.cat(labels).numpy()

# -----------------------------------------------------------------------------
# Main routine
# -----------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)

    # 1) backbone & resolution -------------------------------------------------
    model = get_backbone()
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "img_size"):
        input_res = model.patch_embed.img_size[0]  # 518 for ViT‑B/14
    else:
        input_res = model.pretrained_cfg.get("input_size", (3, 224, 224))[-1]

    # 2) transform pipeline ----------------------------------------------------
    transform_list: list[T.Compose | T.Resize | TightCrop | Invert] = []
    if CROP_TIGHT:
        transform_list.append(TightCrop())
    if INVERT:
        transform_list.append(Invert())
    transform_list.extend([
        T.Resize((input_res, input_res), interpolation=T.InterpolationMode.BILINEAR),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = T.Compose(transform_list)

    # 3) dataset ---------------------------------------------------------------
    ds = MNIST(root=Path("./data"), train=False, download=True, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4) feature extraction ----------------------------------------------------
    feats, targets = extract_features(model, loader, SAMPLE_BATCHES)
    print(f"Collected {feats.shape[0]} samples with {feats.shape[1]}‑D features")

    # 5) dimensionality reduction --------------------------------------------
    if METHOD == "pca":
        reducer = PCA(n_components=2, random_state=SEED)
        pts_2d = reducer.fit_transform(feats)
        subtitle = f"PCA (var {reducer.explained_variance_ratio_.sum()*100:.1f} %)"
    elif METHOD == "tsne":
        pca_lite = PCA(n_components=N_PCA, random_state=SEED).fit_transform(feats)
        reducer = TSNE(n_components=2, init="pca", random_state=SEED, perplexity=PERPLEXITY)
        pts_2d = reducer.fit_transform(pca_lite)
        subtitle = "t‑SNE after PCA"
    elif METHOD == "umap":
        import umap
        pca_lite = PCA(n_components=N_PCA, random_state=SEED).fit_transform(feats)
        reducer = umap.UMAP(n_components=2, random_state=SEED)
        pts_2d = reducer.fit_transform(pca_lite)
        subtitle = "UMAP after PCA"
    else:
        raise ValueError("METHOD must be 'pca', 'tsne', or 'umap'")

    # 6) plot ------------------------------------------------------------------
    plt.figure(figsize=(8, 8))
    sc = plt.scatter(pts_2d[:, 0], pts_2d[:, 1], c=targets, s=6, cmap="tab10")
    plt.legend(*sc.legend_elements(), title="Digit", fontsize="small", loc="best")
    plt.title(f"MNIST – DINOv2 ViT‑B/14 • {subtitle}")
    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(OUTPUT_PNG, dpi=300)
    print(f"Saved plot to {OUTPUT_PNG}")

    if gui_available():
        plt.show()
    else:
        print("(No GUI backend detected – figure saved only.)")

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
