# query_knn.py – build few-shot pool via *query-aware* cosine K-NN
#                (one mini-batch per test image)

from typing import List, Dict, Tuple, Optional
from PIL import Image
import os
import numpy as np
import torch

from sampler import (
    _embedder,          # loads backbone + attaches ._inference_transform
    _feat,              # path → embedding (with caching)
    _to_part,           # PIL → Gemini multipart dict
    _canonical,         # label mapping helper
    CSVDataset,         # simple CSV dataset wrapper (imported in sampler)
)

# Optional: only needed for radiomics
try:
    from radiomics import featureextractor
    import SimpleITK as sitk
    _HAS_RAD = True
except Exception:
    _HAS_RAD = False

_RAD_EXTRACTOR = None            # lazy-initialized pyradiomics extractor
_RAD_CACHE: Dict[Tuple[str, Optional[str]], torch.Tensor] = {}  # (img,mask) -> vec


def _embed_pil(img: Image.Image, model, device: str) -> torch.Tensor:
    """Return L2-normalised feature vector for a PIL image."""
    x = model._inference_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        v = model(x).squeeze(0)
    return torch.nn.functional.normalize(v, dim=0).cpu()

def _get_radiomics_extractor(params_yaml: Optional[str], force2D: bool):
    """
    Build a PyRadiomics extractor. We rely on the params YAML to select
    image types and features (no 'enableAll*' calls to avoid noisy logs).
    """
    global _RAD_EXTRACTOR
    if _RAD_EXTRACTOR is None:
        if not _HAS_RAD:
            raise ImportError("Install 'pyradiomics' and 'SimpleITK' to use radiomics.")

        params_path = None
        if params_yaml:
            cand = os.path.expanduser(params_yaml)
            if not os.path.isabs(cand):
                cand = os.path.normpath(os.path.join(os.getcwd(), cand))
            params_path = cand if os.path.isfile(cand) else None
            if params_yaml and not params_path:
                print(f"[radiomics] WARNING: params file not found: {params_yaml} "
                      f"(resolved: {cand}); using PyRadiomics defaults.")

        _RAD_EXTRACTOR = (featureextractor.RadiomicsFeatureExtractor(params_path)
                          if params_path else featureextractor.RadiomicsFeatureExtractor())

        # enforce 2D behavior for PNG slices
        _RAD_EXTRACTOR.settings['force2D'] = bool(force2D)
        _RAD_EXTRACTOR.settings['force2Ddimension'] = 0

        # Optional: quiet the radiomics logger
        import logging
        logging.getLogger("radiomics").setLevel(logging.ERROR)
    return _RAD_EXTRACTOR


def _sitk_from_pil(img: Image.Image) -> "sitk.Image":
    """Convert PIL → SimpleITK image (grayscale float)."""
    arr = np.array(img.convert("L"), dtype=np.float32)
    return sitk.GetImageFromArray(arr)


def _radiomics_vec_from_pil(img: Image.Image,
                            mask_img: Optional[Image.Image],
                            extractor) -> torch.Tensor:
    """Extract numeric PyRadiomics features and L2-normalize to a vector.
    Ensures mask is binary {0,1} so that label=1 is present.
    """
    # image as float
    img_itk = _sitk_from_pil(img)

    # mask → binary {0,1}
    if mask_img is None:
        # whole-image ROI
        mask_arr = np.ones_like(np.array(img.convert("L")), dtype=np.uint8)
    else:
        # convert any grayscale mask (0/255 etc.) to {0,1}
        mask_arr = (np.array(mask_img.convert("L")) > 0).astype(np.uint8)

    # guard against empty mask
    if mask_arr.sum() == 0:
        raise ValueError("Radiomics mask is empty after binarization.")

    mask_itk = sitk.GetImageFromArray(mask_arr)

    # IMPORTANT: label must be 1 because mask is {0,1}
    res = extractor.execute(img_itk, mask_itk, label=1)

    # keep numeric outputs and normalize
    vals = [float(v) for v in res.values() if isinstance(v, (int, float))]
    v = torch.tensor(vals, dtype=torch.float32)
    if v.numel() == 0:
        v = torch.zeros(1, dtype=torch.float32)
    return torch.nn.functional.normalize(v, dim=0)


def _mask_from_image_path(img_path: str,
                          rgb_suffix: str,
                          mask_suffix: str) -> Optional[str]:
    """
    Infer mask path from an image path using suffix rule:
      ..._same_rgb.png  ->  ..._mask.png
    Falls back to <root> + mask_suffix if the rgb_suffix doesn't match.
    Returns path only if it exists.
    """
    if img_path.endswith(rgb_suffix):
        cand = img_path[:-len(rgb_suffix)] + mask_suffix
    else:
        root, _ = os.path.splitext(img_path)
        cand = root + mask_suffix
    return cand if os.path.exists(cand) else None


def build_query_knn_samples(
    query_img: Image.Image,
    train_csv: str,
    classification_type: str,
    label_list: List[str],
    num_shots: int,
    embedder_name: str = "vit_base_patch14_dinov2.lvd142m",
    device: str = "cpu",
    radiomics_cfg: dict | None = None,   # NEW: radiomics settings
    query_path: Optional[str] = None,    # NEW: to infer query mask
) -> Dict[str, List[Tuple[Dict, str]]]:
    """
    Return, per label, the `num_shots` most-similar training images
    (cosine similarity on L2-normalized features).

    Backends:
      • embedder_name == "radiomics" → PyRadiomics features (masked)
      • otherwise                   → CNN/ViT embeddings (e.g., DINOv2)
    """

    # ───────── Radiomics path ─────────
    if embedder_name.lower() == "radiomics":
        rcfg = radiomics_cfg or {}
        extractor = _get_radiomics_extractor(
            rcfg.get("params_yaml"),
            rcfg.get("force2D", True),
        )
        rgb_suffix         = rcfg.get("rgb_suffix", "_same_rgb.png")
        mask_suffix        = rcfg.get("mask_suffix", "_mask.png")
        require_mask       = bool(rcfg.get("require_mask", True))
        whole_img_fallback = bool(rcfg.get("whole_image_fallback", False))

        # 1) Query feature (use mask if we can infer it by suffix)
        q_mask_img = None
        if query_path:
            q_mask_path = _mask_from_image_path(query_path, rgb_suffix, mask_suffix)
            if q_mask_path and os.path.exists(q_mask_path):
                q_mask_img = Image.open(q_mask_path).convert("L")
            elif require_mask and not whole_img_fallback:
                raise FileNotFoundError(f"[radiomics] Query mask not found for: {query_path}")

        q_feat = _radiomics_vec_from_pil(query_img, q_mask_img, extractor)

        # 2) Train pool (masked)
        ds = CSVDataset(train_csv, transform=None)
        pools: Dict[str, List[str]] = {lbl: [] for lbl in label_list}
        feats: Dict[str, List[torch.Tensor]] = {lbl: [] for lbl in label_list}

        for _, path, raw_lbl in ds:
            lbl = _canonical(raw_lbl, classification_type)
            if lbl not in label_list:
                continue

            mask_path = _mask_from_image_path(path, rgb_suffix, mask_suffix)
            if mask_path is None and require_mask and not whole_img_fallback:
                raise FileNotFoundError(f"[radiomics] Mask not found for: {path}")

            key = (path, mask_path if (mask_path and os.path.exists(mask_path)) else None)
            if key not in _RAD_CACHE:
                img = Image.open(path).convert("RGB")
                mask_img = Image.open(mask_path).convert("L") if key[1] else None
                _RAD_CACHE[key] = _radiomics_vec_from_pil(img, mask_img, extractor)

            pools[lbl].append(path)
            feats[lbl].append(_RAD_CACHE[key])

        # 3) Top-k per label
        few: Dict[str, List[Tuple[Dict, str]]] = {}
        for lbl in label_list:
            f_stack = torch.stack(feats[lbl])            # (N,D)
            sims    = torch.mv(f_stack, q_feat)          # (N,)
            idx_top = sims.topk(num_shots).indices.tolist()
            paths   = [pools[lbl][i] for i in idx_top]
            few[lbl] = [(_to_part(Image.open(p).convert("RGB")), p) for p in paths]
        return few

    # ───────── CNN/ViT path (DINOv2 etc.) ─────────
    model  = _embedder(embedder_name, device)
    q_feat = _embed_pil(query_img, model, device)

    ds     = CSVDataset(train_csv, transform=None)
    pools  = {lbl: [] for lbl in label_list}
    feats  = {lbl: [] for lbl in label_list}

    for _, path, raw_lbl in ds:
        lbl = _canonical(raw_lbl, classification_type)
        if lbl in label_list:
            pools[lbl].append(path)
            feats[lbl].append(_feat(path, model, device))

    few: Dict[str, List[Tuple[Dict, str]]] = {}
    for lbl in label_list:
        f_stack = torch.stack(feats[lbl])              # (N,D)
        sims    = torch.mv(f_stack, q_feat)            # (N,)
        idx_top = sims.topk(num_shots).indices.tolist()
        paths   = [pools[lbl][i] for i in idx_top]
        few[lbl] = [(_to_part(Image.open(p).convert("RGB")), p) for p in paths]

    return few