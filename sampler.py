"""
sampler.py

• build_balanced_indices   – balanced test subset
• build_few_shot_samples   – random / KNN dispatch
• _random_samples          – random few-shot routine
• _knn_samples             – KNN few-shot routine
• _to_webp_part            – resize + WebP encode → Gemini-ready dict
"""
import csv
import io
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from PIL import Image
import torchvision.transforms as T

from dataset import CSVDataset

Part = Dict[str, bytes]            # alias for readability


# --------------------------------------------------------------------------- #
# helper: balanced evaluation subset                                          #
# --------------------------------------------------------------------------- #
def build_balanced_indices(
    dataset: CSVDataset,
    classification_type: str,
    label_list: List[str],
    total_images: int,
    randomize: bool = True,
    seed: int = 42,
) -> List[int]:
    per_label = max(1, total_images // len(label_list))
    remainder = total_images - per_label * len(label_list)

    buckets = defaultdict(list)
    for idx in range(len(dataset)):
        _, _, csv_lbl = dataset[idx]
        mapped = _canonical_label(csv_lbl, classification_type)
        if mapped in label_list:
            buckets[mapped].append(idx)

    if randomize:
        random.seed(seed)
        for lst in buckets.values():
            random.shuffle(lst)

    chosen = []
    for lbl in label_list:
        chosen.extend(buckets[lbl][:per_label])

    extra_iter = (lbl for lbl in label_list for _ in range(remainder))
    for lbl in extra_iter:
        if buckets[lbl][per_label:]:
            chosen.append(buckets[lbl][per_label])

    return chosen[:total_images]


# --------------------------------------------------------------------------- #
# helper: resize → WebP *part*                                                #
# --------------------------------------------------------------------------- #
def _to_webp_part(img: Image.Image) -> Part:
    """Return a Gemini-ready dict: {'mime_type':'image/webp','data': <bytes>}"""
    if max(img.size) > 512:
        w, h = img.size
        scale = 512 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=90, method=6)
    buf.seek(0)
    return {"mime_type": "image/webp", "data": buf.getvalue()}


# --------------------------------------------------------------------------- #
# Few-shot factory                                                            #
# --------------------------------------------------------------------------- #
def build_few_shot_samples(
    config: dict,
    transform,
    classification_type: str,
    label_list: List[str],
) -> Dict[str, List[Tuple[Part, str]]]:
    strategy = config.get("sampling", {}).get("strategy", "random").lower()

    if strategy == "random":
        return _random_samples(
            train_csv=config["data"]["train_csv"],
            transform=transform,
            classification_type=classification_type,
            label_list=label_list,
            num_shots=config["data"]["num_shots"],
            randomize=config["data"].get("randomize_few_shot", True),
            seed=config["data"].get("seed", 42),
        )

    if strategy == "knn":
        knn_cfg = config["sampling"]
        return _knn_samples(
            transform=transform,
            classification_type=classification_type,
            label_list=label_list,
            num_shots=config["data"]["num_shots"],
            knn_csv_path=knn_cfg["knn_csv"],
            anchors=knn_cfg.get("anchors", {}),
            train_csv=config["data"]["train_csv"],   # for random anchors
            seed=config["data"].get("seed", 42),
            num_neighbors=knn_cfg.get("num_neighbors", 2),
        )

    raise ValueError(f"Unknown sampling strategy: {strategy}")


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
_LABEL_MAP = {
    "ADI":  "Adipose",
    "DEB":  "Debris",
    "LYM":  "Lymphocytes",
    "MUC":  "Mucus",
    "MUS":  "Smooth Muscle",
    "NORM": "Normal Colon Mucosa",
    "STR":  "Cancer-Associated Stroma",
    "TUM":  "Colorectal Adenocarcinoma Epithelium",
}


def _canonical_label(csv_label: str, classification_type: str) -> str:
    csv_label = csv_label.strip()
    return csv_label if classification_type == "binary" else _LABEL_MAP.get(csv_label, csv_label)


# ---------- RANDOM ---------------------------------------------------------- #
def _random_samples(
    train_csv: str,
    transform,
    classification_type: str,
    label_list: List[str],
    num_shots: int,
    randomize: bool,
    seed: int,
) -> Dict[str, List[Tuple[Part, str]]]:
    ds = CSVDataset(train_csv, transform=transform)
    lbl_to_idx = defaultdict(list)
    for i in range(len(ds)):
        _, _, raw = ds[i]
        mapped = _canonical_label(raw, classification_type)
        if mapped in label_list:
            lbl_to_idx[mapped].append(i)

    if randomize:
        random.seed(seed)
        for lst in lbl_to_idx.values():
            random.shuffle(lst)

    few = {}
    for lbl in label_list:
        items = []
        for ds_idx in lbl_to_idx.get(lbl, [])[:num_shots]:
            img_tensor, img_path, _ = ds[ds_idx]
            pil = T.ToPILImage()(img_tensor)       # already 512 px from transform
            items.append((_to_webp_part(pil), img_path))
        few[lbl] = items
    return few


# ---------- KNN ------------------------------------------------------------- #
def _load_knn_table(path: str) -> Dict[str, List[str]]:
    table, remove = {}, str.maketrans("", "", "[]'\"")

    def _clean(cell: str) -> str:
        cell = cell.strip().translate(remove)
        return cell.split()[0].split(",")[0]

    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                anchor = _clean(row[0])
                neighs = [_clean(c) for c in row[1:] if c.strip()]
                table[anchor] = neighs
    return table


def _knn_samples(
    transform,
    classification_type: str,
    label_list: List[str],
    num_shots: int,
    knn_csv_path: str,
    anchors: Dict[str, str],
    train_csv: str,        # kept for signature
    seed: int,
    num_neighbors: int,
) -> Dict[str, List[Tuple[Part, str]]]:
    if num_neighbors >= num_shots:
        raise ValueError("`num_neighbors` must be smaller than `data.num_shots`")

    _norm = lambda p: os.path.abspath(os.path.normpath(p))
    knn_table = {_norm(k): v for k, v in _load_knn_table(knn_csv_path).items()}

    def _label_from_path(p: str) -> str:
        prefix = os.path.basename(p).split("-")[0]
        return _canonical_label(prefix, classification_type)

    pools = defaultdict(list)
    for p in knn_table:
        pools[_label_from_path(p)].append(p)

    rand = random.Random(seed)
    few_shot = {}
    for lbl in label_list:
        anchor = anchors.get(lbl)
        if anchor in (None, "random"):
            if not pools[lbl]:
                raise ValueError(f"No anchors of label '{lbl}' in {knn_csv_path}.")
            anchor = rand.choice(pools[lbl])

        anchor = _norm(anchor)
        if anchor not in knn_table:
            raise ValueError(f"Anchor '{anchor}' not found in {knn_csv_path}.")

        neigh = knn_table[anchor][:num_neighbors]
        sample_paths = ([anchor] + neigh)[:num_shots]

        items = []
        for p in sample_paths:
            img = Image.open(p).convert("RGB")
            items.append((_to_webp_part(img), p))
        few_shot[lbl] = items

    return few_shot