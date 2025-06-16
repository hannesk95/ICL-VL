# ----------------------------------------------------------------------
# sampler.py  – helpers for class‑balanced test selection
# ----------------------------------------------------------------------
from collections import defaultdict
import random

__all__ = ["build_balanced_indices"]

# Raw → human‑readable mapping for the CRC100K dataset
_RAW2HUMAN = {
    "ADI": "Adipose",
    "DEB": "Debris",
    "LYM": "Lymphocytes",
    "MUC": "Mucus",
    "MUS": "Smooth Muscle",
    "NORM": "Normal Colon Mucosa",
    "STR": "Cancer-Associated Stroma",
    "TUM": "Colorectal Adenocarcinoma Epithelium",
}

def _map_label(raw: str, task: str) -> str:
    """Map the raw CSV label to the label used during training/inference."""
    raw = raw.strip()
    if task == "binary":
        if raw.lower() in {"tumor", "no tumor"}:
            return raw.title()
        return "Tumor" if raw == "TUM" else "No Tumor"
    return _RAW2HUMAN.get(raw, "Unknown")


def build_balanced_indices(
    dataset,
    classification_type: str,
    label_list: list[str],
    total_images: int,
    *,
    randomize: bool = True,
    seed: int = 42,
) -> list[int]:
    """Return **exactly** ``total_images`` indices, split evenly across classes.

    ``total_images`` must be divisible by the number of classes.  Both the order
    *within* each class bucket and the final concatenated list can be shuffled
    (respecting ``seed``) when ``randomize=True``.
    """
    n_classes = len(label_list)
    if total_images % n_classes:
        raise ValueError(
            f"total_images ({total_images}) must be divisible by number of classes ({n_classes})."
        )
    per_class = total_images // n_classes

    # gather indices per class
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        _, _, raw_label = dataset[idx]
        mapped = _map_label(raw_label, classification_type)
        if mapped in label_list:
            buckets[mapped].append(idx)

    if randomize:
        random.seed(seed)
        for arr in buckets.values():
            random.shuffle(arr)

    balanced: list[int] = []
    for lbl in label_list:
        pool = buckets.get(lbl, [])
        if len(pool) < per_class:
            raise RuntimeError(
                f"Requested {per_class} '{lbl}' images but only {len(pool)} present."
            )
        balanced.extend(pool[:per_class])

    if randomize:
        random.shuffle(balanced)

    return balanced