#!/usr/bin/env python
"""
Main entry-point for CRC100K few-shot classification.

Changes compared with your previous version
-------------------------------------------
* All filename keys are lower-cased once → prevents `.PNG` / `.png` issues
* Loads neighbour lists into **sets** for O(1) look-ups.
* **Fatal upfront check**: every test image must appear in the KNN table.
  Program exits immediately if not, telling you exactly how many/missing names.
"""

import os, csv, json, datetime, random, subprocess, sys
from collections import defaultdict
from dotenv import load_dotenv
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from config  import load_config
from dataset import CSVDataset
from model   import configure_gemini, build_gemini_prompt, gemini_api_call
from sampler2 import build_balanced_indices


# ─────────────────────────────────────────── few-shot utilities ──
def get_few_shot_samples(
    csv_path: str,
    transform,
    classification_type: str,
    label_list: list[str],
    num_shots: int,
    randomize: bool = False,
    seed: int = 42,
    allowed_filenames: set[str] | None = None,
):
    """
    Collect `num_shots` images per label from `csv_path`.

    * If `allowed_filenames` is given, restrict pool to those basenames.
    """
    dataset = CSVDataset(csv_path, transform=transform)

    label_map = {
        "ADI": "Adipose",
        "DEB": "Debris",
        "LYM": "Lymphocytes",
        "MUC": "Mucus",
        "MUS": "Smooth Muscle",
        "NORM": "Normal Colon Mucosa",
        "STR": "Cancer-Associated Stroma",
        "TUM": "Colorectal Adenocarcinoma Epithelium",
    }

    label_to_indices: dict[str, list[int]] = defaultdict(list)

    for idx in range(len(dataset)):
        _, img_path, csv_label = dataset[idx]
        fname = os.path.basename(img_path).lower()

        if allowed_filenames is not None and fname not in allowed_filenames:
            continue

        csv_label = csv_label.strip()

        if classification_type == "binary":
            mapped_label = "Tumor" if csv_label == "TUM" else "No Tumor"
        else:
            mapped_label = label_map.get(csv_label, "Unknown")

        if mapped_label in label_list:
            label_to_indices[mapped_label].append(idx)

    if randomize:
        random.seed(seed)
        for lbl in label_to_indices:
            random.shuffle(label_to_indices[lbl])

    few_shot_dict: dict[str, list[tuple[Image.Image, str]]] = {}
    for lbl in label_list:
        indices = label_to_indices.get(lbl, [])[:num_shots]
        if len(indices) < num_shots:
            print(f"[WARN] only {len(indices)}/{num_shots} shots available for {lbl}")
        items = []
        for cidx in indices:
            img_tensor, img_path, _ = dataset[cidx]
            img = T.ToPILImage()(img_tensor)
            items.append((img, img_path))
        few_shot_dict[lbl] = items

    return few_shot_dict


# ─────────────────────────────────────────────────── main ──
def main():
    load_dotenv()
    config = load_config("configs/CRC100K/binary/three_shot_knn2.yaml")

    train_csv           = config["data"]["train_csv"]
    test_csv            = config["data"]["test_csv"]
    save_path           = config["data"]["save_path"]
    prompt_path         = config["user_args"]["prompt_path"]
    classification_type = config["classification"]["type"]
    label_list          = config["classification"]["labels"]
    sampling_strategy   = config["data"].get("sampling_strategy", "random").lower()

    os.environ["PROMPT_PATH"] = prompt_path
    print(f"[INFO] Using prompt file: {prompt_path}")
    print(f"[INFO] Classification type: {classification_type}")
    print(f"[INFO] Label list:          {label_list}")
    print(f"[INFO] Sampling strategy:   {sampling_strategy}")

    # ── setup Gemini & transforms ──────────────────────────────────────────
    configure_gemini()
    transform = T.Compose([T.ToTensor()])

    # ── balanced test set --------------------------------------------------
    test_dataset_full = CSVDataset(csv_path=test_csv, transform=transform)

    balanced_indices = build_balanced_indices(
        test_dataset_full,
        classification_type=classification_type,
        label_list=label_list,
        total_images=config["data"]["num_test_images"],
        randomize=config["data"].get("randomize_test_images", True),
        seed=config["data"].get("test_seed", 42),
    )

    test_dataset = Subset(test_dataset_full, balanced_indices)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)
    num_tests    = len(test_loader.dataset)

    # ── preload random pool (used for random *or* fallback) ────────────────
    few_shot_samples_random = get_few_shot_samples(
        train_csv,
        transform,
        classification_type,
        label_list,
        config["data"]["num_shots"],
        config["data"].get("randomize_few_shot"),
        config["data"].get("seed", 42),
    )

    if sampling_strategy == "random":
        print("\n[INFO] Few-shot pool (random pre-selection):")
        for label in label_list:
            for _, path in few_shot_samples_random[label]:
                print(f"  [{label}] {path}")

    # ── if KNN: load neighbour lookup table once ---------------------------
    knn_dict: dict[str, set[str]] = {}
    knn_top_k = None
    if sampling_strategy == "knn":
        knn_csv_path = config["data"].get("knn_csv_path")
        if not knn_csv_path:
            raise ValueError("sampling_strategy='knn' but 'knn_csv_path' not set.")
        knn_top_k = config["data"].get("knn_top_k") or config["data"]["num_shots"]
        print(f"[INFO] Loading KNN neighbour file: {knn_csv_path}")
        with open(knn_csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            if header[0].lower() != "query":
                raise RuntimeError("KNN CSV header malformed; first column must be 'query'")
            for row in reader:
                query      = row[0].strip().lower()
                neighbours = {p.strip().lower() for p in row[1:] if p.strip()}
                knn_dict[query] = neighbours

        # ---- fatal upfront coverage check ---------------------------------
        if sampling_strategy == "knn":
            # collect *all* filenames that will actually be queried
            # (use the BALANCED subset, not the full test CSV)
            test_fnames = {
                os.path.basename(test_dataset.dataset.samples[i][0]).lower()
                for i in test_dataset.indices           # indices selected by build_balanced_indices
            }

            missing_queries = test_fnames - knn_dict.keys()
            if missing_queries:
                msg = (
                    f"KNN CSV '{knn_csv_path}' is missing {len(missing_queries)} "
                    f"test queries; e.g. {sorted(list(missing_queries))[:5]} …\n"
                    "Regenerate the neighbour table with all possible queries!"
                )
                sys.exit("[FATAL] " + msg)
                
    # ── inference loop -----------------------------------------------------
    results = []
    for idx, (test_tensors, test_paths, _) in enumerate(test_loader, 1):
        path_str     = test_paths[0]
        query_fname  = os.path.basename(path_str).lower()
        test_img_pil = T.ToPILImage()(test_tensors.squeeze(0))
        print(f"\n[IMAGE {idx}/{num_tests}] {path_str}")

        # ---- pick few-shot examples --------------------------------------
        if sampling_strategy == "random":
            few_shot_use = few_shot_samples_random

        else:  # KNN pathway
            neighbours = sorted(knn_dict[query_fname])[:knn_top_k]
            few_shot_use = get_few_shot_samples(
                train_csv,
                transform,
                classification_type,
                label_list,
                config["data"]["num_shots"],
                randomize=False,
                seed=42,
                allowed_filenames=set(neighbours),
            )

        # ---- Gemini prompt & call ----------------------------------------
        contents    = build_gemini_prompt(few_shot_use, test_img_pil, classification_type)
        predictions = gemini_api_call(contents, classification_type)

        entry = {
            "image_path": path_str,
            "thoughts":   predictions.get("thoughts", ""),
            "answer":     predictions.get("answer", "Unknown"),
            "score":      predictions.get("score", -1)
                         if classification_type != "binary"
                         else {
                             "score_tumor":    predictions.get("score_tumor", -1),
                             "score_no_tumor": predictions.get("score_no_tumor", -1),
                         },
        }
        if classification_type == "binary":
            entry["location"] = predictions.get("location")

        results.append(entry)

    # ── save results -------------------------------------------------------
    os.makedirs(save_path, exist_ok=True)
    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_path, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"[INFO] Results saved to {results_path}")

    # ── optional evaluation ------------------------------------------------
    labels_path = config["data"].get("labels_path")
    if labels_path and os.path.exists(labels_path):
        print("[INFO] Running evaluation…")
        subprocess.run(
            ["python", "evaluation.py", "--results", results_path, "--labels", labels_path],
            check=True,
        )
    else:
        print("[WARN] Labels file missing; skipping evaluation.")


if __name__ == "__main__":
    main()