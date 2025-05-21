# main.py  – fully self-contained

import os, json, datetime, random, subprocess
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from PIL import Image

from config import load_config
from dataset import CSVDataset
from model import configure_gemini, build_gemini_prompt, gemini_api_call
from sampler import build_balanced_indices

# ------------------------------------------------------------
# optional k-NN retriever (only used if YAML says enable: true)
try:
    from retrieval.retrieve_knn import DinoRetriever
except ImportError as e:
    print(f"[WARN] cannot import DinoRetriever → {e}")
    DinoRetriever = None
# ------------------------------------------------------------


def get_few_shot_samples(
    csv_path, transform, classification_type, label_list,
    num_shots, randomize=False, seed=42
):
    """Original random sampler (fallback when retrieval.disabled)."""
    dataset = CSVDataset(csv_path, transform=transform)

    label_map = {
        "ADI": "Adipose", "DEB": "Debris", "LYM": "Lymphocytes",
        "MUC": "Mucus",   "MUS": "Smooth Muscle", "NORM": "Normal Colon Mucosa",
        "STR": "Cancer-Associated Stroma", "TUM": "Colorectal Adenocarcinoma Epithelium",
    }
    label_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, _, csv_lbl = dataset[idx]
        csv_lbl = csv_lbl.strip()

        mapped = (
            "Tumor" if csv_lbl == "TUM" else "No Tumor"
            if classification_type == "binary"
            else label_map.get(csv_lbl, "Unknown")
        )
        if mapped in label_list:
            label_to_indices[mapped].append(idx)

    if randomize:
        random.seed(seed)
        for v in label_to_indices.values():
            random.shuffle(v)

    few = {lbl: [] for lbl in label_list}
    for lbl in label_list:
        for idx in label_to_indices.get(lbl, [])[:num_shots]:
            tensor, path, _ = dataset[idx]
            few[lbl].append((T.ToPILImage()(tensor), path))
    return few


# ------------------------------------------------------------
def main():
    load_dotenv()

    # path to YAML can be overridden via ENV   (optional)
    cfg_path = os.getenv("CONFIG_PATH", "configs/CRC100K/binary/three_shot_knn.yaml")
    cfg      = load_config(cfg_path)

    # one global verbosity flag from YAML
    verbose = cfg.get("user_args", {}).get("verbose", False)

    # make prompt template discoverable by model.py
    prompt_path = cfg["user_args"]["prompt_path"]
    os.environ["PROMPT_PATH"] = prompt_path
    print(f"[INFO] Using prompt file: {prompt_path}")

    classification_type = cfg["classification"]["type"]
    label_list          = cfg["classification"]["labels"]
    num_shots           = cfg["data"]["num_shots"]

    # ---------- retrieval switch ----------
    r_cfg   = cfg.get("retrieval", {})
    use_knn = bool(r_cfg.get("enable", False)) and DinoRetriever is not None
    if use_knn:
        retriever = DinoRetriever(
            index_path = r_cfg["index_path"],
            meta_path  = r_cfg["meta_path"],
            model_name = r_cfg.get("model_name", "vit_small_patch14_dinov2"),
            img_size   = r_cfg.get("img_size", 224),
        )
        print("[INFO] k-NN retrieval ENABLED")
    else:
        retriever = None
        print("[INFO] k-NN retrieval disabled → random few-shot")

    # ---------- model & transforms ----------
    configure_gemini()
    transform = T.Compose([T.ToTensor()])

    # ---------- balanced test subset ----------
    test_full = CSVDataset(csv_path=cfg["data"]["test_csv"], transform=transform)
    balanced_ids = build_balanced_indices(
        test_full,
        classification_type=classification_type,
        label_list=label_list,
        total_images=cfg["data"]["num_test_images"],
        randomize=cfg["data"].get("randomize_test_images", True),
        seed=cfg["data"].get("test_seed", 42),
    )
    test_set  = Subset(test_full, balanced_ids)
    test_load = DataLoader(test_set, batch_size=1, shuffle=False)

    # ---------- random few-shot pool (built once) ----------
    if not use_knn:
        fewshot_cache = get_few_shot_samples(
            cfg["data"]["train_csv"], transform,
            classification_type, label_list, num_shots,
            cfg["data"].get("randomize_few_shot"),
            cfg["data"].get("seed", 42),
        )
        print("[INFO] Static few-shot pool built.")
        if verbose:
            print("  • random few-shot images:")
            for lbl, items in fewshot_cache.items():
                for _, p in items:
                    print(f"      {lbl:<10} {p}")

    # ---------- inference ----------
    results = []
    for idx, (test_tensor, test_path, _) in enumerate(test_load, 1):
        print(f"\n[IMAGE {idx}/{len(test_load)}] {test_path[0]}")
        test_img = T.ToPILImage()(test_tensor.squeeze(0))

        # choose exemplars
                # choose exemplars
        if use_knn:
            # ────────────────────────────────────────────────────────
            # 1) pull top-k×overquery neighbours without filtering
            # ────────────────────────────────────────────────────────
            neigh = retriever.query(
                test_img,
                k=num_shots,
                label_list=None,                       # no label filter
                overquery_factor=r_cfg.get("overquery_factor", 10),
            )

            # ────────────────────────────────────────────────────────
            # 2) map raw labels (TUM / NORM / …) → binary Tumor / No Tumor
            #    and build a balanced pool (≈ k/2 per class)
            # ────────────────────────────────────────────────────────
            def to_binary(raw):
                return "Tumor" if raw == "TUM" else "No Tumor"

            few = {"Tumor": [], "No Tumor": []}
            per_class_goal = num_shots // 2                     # e.g. 3 → 1+1

            for p, raw_lbl in neigh:
                b_lbl = to_binary(raw_lbl)
                if len(few[b_lbl]) < per_class_goal:
                    few[b_lbl].append((Image.open(p).convert("RGB"), str(p)))
                if sum(len(v) for v in few.values()) == num_shots:
                    break

            # top-up if perfect balance impossible
            if sum(len(v) for v in few.values()) < num_shots:
                for p, raw_lbl in neigh:
                    b_lbl = to_binary(raw_lbl)
                    if len(few[b_lbl]) < num_shots - sum(len(v) for v in few.values()):
                        few[b_lbl].append((Image.open(p).convert("RGB"), str(p)))
                    if sum(len(v) for v in few.values()) == num_shots:
                        break

            # ────────────────────────────────────────────────────────
            # 3) verbose listing
            # ────────────────────────────────────────────────────────
            if verbose:
                print("  • few-shot neighbours:")
                rank = 1
                for lbl in ("Tumor", "No Tumor"):                # fixed order
                    for _, path in few[lbl]:
                        print(f"      [{rank}] {lbl:<8} {path}")
                        rank += 1
        else:
            few = fewshot_cache

        # build prompt & call Gemini
        prompt = build_gemini_prompt(few, test_img, classification_type)
        pred   = gemini_api_call(prompt, classification_type)

        entry = {
            "image_path": test_path[0],
            "thoughts":   pred.get("thoughts", ""),
            "answer":     pred.get("answer", "Unknown"),
        }
        if classification_type == "binary":
            entry["score_tumor"]    = pred.get("score_tumor", -1)
            entry["score_no_tumor"] = pred.get("score_no_tumor", -1)
            entry["location"]       = pred.get("location")
        else:
            entry["score"] = pred.get("score", -1)
        results.append(entry)

    # ---------- save ----------
    out_dir = Path(cfg["data"]["save_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    jpth = out_dir / f"results_{ts}.json"
    json.dump({"results": results}, open(jpth, "w"), indent=2)
    print(f"[INFO] Results saved → {jpth}")

    # ---------- optional evaluation ----------
    lbls = cfg["data"].get("labels_path")
    if lbls and Path(lbls).exists():
        print("[INFO] Running evaluation …")
        subprocess.run(["python", "evaluation.py", "--results", jpth, "--labels", lbls])
    else:
        print("[WARN] Ground-truth labels missing → evaluation skipped.")


if __name__ == "__main__":
    main()