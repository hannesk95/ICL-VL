# query_knn.py – build few-shot pool via *query-aware* cosine K-NN
#                (one mini-batch per test image)

from typing import List, Dict, Tuple
from PIL import Image
import torch

from sampler import (
    _embedder,          # loads backbone + attaches ._inference_transform
    _feat,              # path → embedding (with caching)
    _to_part,           # PIL → Gemini multipart dict
    _canonical,         # label mapping helper
    CSVDataset,         # simple CSV dataset wrapper
)


def _embed_pil(img: Image.Image, model, device: str) -> torch.Tensor:
    """Return L2-normalised feature vector for a PIL image."""
    x = model._inference_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        v = model(x).squeeze(0)
    return torch.nn.functional.normalize(v, dim=0).cpu()


def build_query_knn_samples(
    query_img: Image.Image,
    train_csv: str,
    classification_type: str,
    label_list: List[str],
    num_shots: int,
    embedder_name: str = "vit_base_patch14_dinov2.lvd142m",
    device: str = "cpu",
) -> Dict[str, List[Tuple[Dict, str]]]:
    """
    For the given *query_img* return, per label, the `num_shots`
    most-similar training images (cosine similarity in feature space).
    """

    # 1) backbone + query embedding
    model  = _embedder(embedder_name, device)
    q_feat = _embed_pil(query_img, model, device)

    # 2) load & embed the training pool (cached across calls via _feat)
    ds     = CSVDataset(train_csv, transform=None)
    pools  = {lbl: [] for lbl in label_list}
    feats  = {lbl: [] for lbl in label_list}

    for _, path, raw_lbl in ds:
        lbl = _canonical(raw_lbl, classification_type)
        if lbl in label_list:
            pools[lbl].append(path)
            feats[lbl].append(_feat(path, model, device))

    # 3) pick top-k neighbours for each label
    few: Dict[str, List[Tuple[Dict, str]]] = {}
    for lbl in label_list:
        f_stack = torch.stack(feats[lbl])              # (N,D)
        sims    = torch.mv(f_stack, q_feat)            # (N,)
        idx_top = sims.topk(num_shots).indices.tolist()
        paths   = [pools[lbl][i] for i in idx_top]
        few[lbl] = [(_to_part(Image.open(p).convert("RGB")), p) for p in paths]

    return few