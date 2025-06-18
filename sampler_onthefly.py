"""
sampler.py – random OR on-the-fly K-nearest-neighbour few-shot selector
======================================================================

• build_balanced_indices – balanced subset for evaluation
• build_few_shot_samples – dispatches to _random_samples / _knn_samples
• _random_samples        – purely random few-shot
• _knn_samples           – on-the-fly cosine-KNN (no CSV required)
   · anchor = "random"  → draw from label pool using data.seed
   · anchor = /path/…   → fixed file
"""

from __future__ import annotations

import csv, io, os, random
from collections import defaultdict
from typing  import Dict, List, Tuple

import torch, torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models     as models
from PIL import Image

from dataset import CSVDataset


# ╭──────────────────────────────────────────────────────────────────╮
# │ HELPERS                                                          │
# ╰──────────────────────────────────────────────────────────────────╯
Part   = Dict[str, bytes]
Tensor = torch.Tensor

_LABEL_MAP = {  # only used for your multiclass tasks; keep if needed
    "ADI":"Adipose","DEB":"Debris","LYM":"Lymphocytes","MUC":"Mucus",
    "MUS":"Smooth Muscle","NORM":"Normal Colon Mucosa",
    "STR":"Cancer-Associated Stroma","TUM":"Colorectal Adenocarcinoma Epithelium",
}
def _canonical(lbl:str, typ:str)->str:
    return lbl.strip() if typ=="binary" else _LABEL_MAP.get(lbl.strip(), lbl.strip())


def _to_part(img:Image.Image)->Part:
    buf = io.BytesIO()
    img.save(buf, "WEBP", quality=90, method=6)
    buf.seek(0)
    return {"mime_type":"image/webp", "data":buf.getvalue()}


# ╭──────────────────────────────────────────────────────────────────╮
# │ PUBLIC ENTRY: build_few_shot_samples                            │
# ╰──────────────────────────────────────────────────────────────────╯
def build_few_shot_samples(config:dict, transform,
                           classification_type:str,
                           label_list:List[str]) -> Dict[str, List[Tuple[Part,str]]]:

    s_cfg   = config.get("sampling", {})
    strat   = s_cfg.get("strategy", "random").lower()

    if strat == "random":
        return _random_samples(
            train_csv          = config["data"]["train_csv"],
            transform          = transform,
            classification_type= classification_type,
            label_list         = label_list,
            num_shots          = config["data"]["num_shots"],
            randomize          = config["data"].get("randomize_few_shot", True),
            seed               = config["data"].get("seed", 42),
        )

    if strat == "knn":
        return _knn_samples(
            classification_type= classification_type,
            label_list         = label_list,
            num_shots          = config["data"]["num_shots"],
            train_csv          = config["data"]["train_csv"],
            seed               = config["data"].get("seed", 42),
            num_neighbors      = s_cfg.get("num_neighbors", 2),
            embedder           = s_cfg.get("embedder", "resnet50"),
            device             = s_cfg.get("device",   "cpu"),
            anchors            = s_cfg.get("anchors", {}),
            knn_csv            = s_cfg.get("knn_csv"),          # may be None
        )

    raise ValueError(f"Unknown sampling.strategy: {strat}")


# ╭──────────────────────────────────────────────────────────────────╮
# │ BALANCED TEST INDICES (unchanged)                               │
# ╰──────────────────────────────────────────────────────────────────╯
def build_balanced_indices(ds:CSVDataset, classification_type:str, label_list:List[str],
                           total_images:int, randomize=True, seed=42)->List[int]:
    per = max(1, total_images // len(label_list))
    rem = total_images - per * len(label_list)

    buckets = defaultdict(list)
    for i in range(len(ds)):
        _, _, raw = ds[i]
        mapped = _canonical(raw, classification_type)
        if mapped in label_list:
            buckets[mapped].append(i)

    if randomize:
        random.seed(seed)
        for v in buckets.values():
            random.shuffle(v)

    chosen=[]
    for lbl in label_list:
        chosen += buckets[lbl][:per]

    for lbl in (lbl for lbl in label_list for _ in range(rem)):
        if buckets[lbl][per:]:
            chosen.append(buckets[lbl][per])

    return chosen[:total_images]


# ╭──────────────────────────────────────────────────────────────────╮
# │ RANDOM FEW-SHOT                                                │
# ╰──────────────────────────────────────────────────────────────────╯
def _random_samples(train_csv, transform, classification_type, label_list,
                    num_shots, randomize, seed):
    ds = CSVDataset(train_csv, transform=transform)
    buckets = defaultdict(list)
    for i in range(len(ds)):
        _, _, raw = ds[i]
        mapped = _canonical(raw, classification_type)
        if mapped in label_list:
            buckets[mapped].append(i)

    if randomize:
        random.seed(seed)
        for v in buckets.values():
            random.shuffle(v)

    few = {}
    for lbl in label_list:
        few[lbl] = []
        for idx in buckets[lbl][:num_shots]:
            tensor, path, _ = ds[idx]
            few[lbl].append((_to_part(T.ToPILImage()(tensor)), path))
    return few


# ╭──────────────────────────────────────────────────────────────────╮
# │ KNN FEW-SHOT – PRE-COMPUTED OR LIVE                             │
# ╰──────────────────────────────────────────────────────────────────╯
_TRANS = T.Compose([
    T.Resize(256), T.CenterCrop(224), T.ToTensor(),
    T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])
_EMB_CACHE:Dict[str,Tensor] = {}

def _embedder(name:str, device:str):
    m = getattr(models, name)(weights="IMAGENET1K_V2")
    if hasattr(m, "fc"):
        m.fc = torch.nn.Identity()
    elif hasattr(m, "classifier"):
        m.classifier = torch.nn.Identity()
    return m.eval().to(device)

def _feat(path:str, model, device):
    if path in _EMB_CACHE:
        return _EMB_CACHE[path]
    img = Image.open(path).convert("RGB")
    x   = _TRANS(img).unsqueeze(0).to(device)
    with torch.no_grad():
        v = model(x).squeeze(0)
    v = F.normalize(v, dim=0).cpu()
    _EMB_CACHE[path] = v
    return v

def _load_table(csv_path)->Dict[str,List[str]]:
    table, strip = {}, str.maketrans("","",'[]\'"')
    with open(csv_path,newline="") as f:
        r=csv.reader(f); next(r,None)
        for row in r:
            if not row: continue
            anc   = row[0].translate(strip).split(",")[0].strip()
            neigh = [c.translate(strip).split(",")[0].strip() for c in row[1:] if c.strip()]
            table[os.path.abspath(anc)] = neigh
    return table


def _knn_samples(classification_type, label_list, num_shots, train_csv, seed,
                 num_neighbors, embedder, device, anchors, knn_csv):

    if num_neighbors >= num_shots:
        raise ValueError("num_neighbors must be < num_shots")

    # A) ---------- pre-computed table branch --------------------------
    if knn_csv:
        table = _load_table(knn_csv)
        def _lbl(p): return _canonical(os.path.basename(p).split("-")[0], classification_type)
        pools = defaultdict(list)
        for p in table:
            pools[_lbl(p)].append(p)

        rand = random.Random(seed)
        few  = {}
        for lbl in label_list:
            anchor = anchors.get(lbl, "random")
            if anchor == "random":
                anchor = rand.choice(pools[lbl])

            neigh  = table[anchor][:num_neighbors]
            paths  = ([anchor] + neigh)[:num_shots]
            few[lbl] = [(_to_part(Image.open(p).convert("RGB")), p) for p in paths]
        return few

    # B) ---------- live-embedding branch ------------------------------
    ds = CSVDataset(train_csv, transform=None)
    pools = defaultdict(list)
    for _, p, csv_lbl in ds:
        lbl = _canonical(csv_lbl, classification_type)
        if lbl in label_list:
            pools[lbl].append(p)

    model = _embedder(embedder, device)
    rand  = random.Random(seed)
    few   = {}

    for lbl in label_list:
        if len(pools[lbl]) < num_shots:
            raise ValueError(f"Need ≥{num_shots} images for '{lbl}', found {len(pools[lbl])}")

        anchor = anchors.get(lbl, "random")
        if anchor == "random":
            anchor = rand.choice(pools[lbl])

        feat_anchor = _feat(anchor, model, device)
        others      = [p for p in pools[lbl] if p != anchor]
        feats       = torch.stack([_feat(p, model, device) for p in others])
        sims        = torch.mv(feats, feat_anchor)
        neigh       = [others[i] for i in sims.topk(num_neighbors).indices.tolist()]

        paths = ([anchor] + neigh)[:num_shots]
        few[lbl] = [(_to_part(Image.open(p).convert("RGB")), p) for p in paths]

    return few