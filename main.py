# main.py – unified Gemini / LLaVA-HF / Med-LLaVA pipeline for binary glioma grading
# ---------------------------------------------------------------------
# 2025-07-24: med_llava support
#   • Recognises backend "med_llava" with zero impact on the existing
#     Gemini or LLaVA-HF logic.
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import json
import datetime as dt
import subprocess
from functools import partial

from dotenv import load_dotenv
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from config import load_config
from dataset import CSVDataset
from model import (
    configure_vlm,
    build_gemini_prompt,
    build_llava_prompt,
    vlm_api_call,
)
from sampler import (
    build_few_shot_samples,
    build_balanced_indices,
)
from query_knn import build_query_knn_samples


def main() -> None:
    """
    Adds two optional CLI flags so an external wrapper can control
    randomness without touching the YAML:

        --config PATH   (YAML file; keeps old default if omitted)
        --seed  INT     (overrides data.seed & data.test_seed)

    If you run `python main.py` exactly like before, behaviour is *identical*.
    """
    import argparse, copy
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Unified VLM pipeline")
    parser.add_argument(
        "--config",
        default="configs/glioma/binary/t2/three_shot.yaml",
        help="YAML config to use (unchanged default)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional global seed override (used by bootstrap.py)",
    )
    args = parser.parse_args()

    # 1. Load YAML config & env vars
    # ------------------------------------------------------------------
    load_dotenv()
    config = load_config(args.config)

    # Optional seed override – lets bootstrap replicate runs
    if args.seed is not None:
        cfg = copy.deepcopy(config)
        cfg["data"]["seed"] = args.seed
        cfg["data"]["test_seed"] = args.seed
        config = cfg

    # ↓↓↓ everything from here on is **unchanged original code** ↓↓↓
    # ------------------------------------------------------------------
    verbose = config["user_args"].get("verbose", False)
    test_csv  = config["data"]["test_csv"]
    save_path = config["data"]["save_path"]

    prompt_path         = config["user_args"]["prompt_path"]
    classification_type = config["classification"]["type"]
    label_list          = config["classification"]["labels"]

    os.environ["PROMPT_PATH"] = prompt_path

    print(f"[INFO] Using prompt file:            {prompt_path}")
    print(f"[INFO] Classification type:          {classification_type}")
    print(f"[INFO] Label list:                   {label_list}")
    print(f"[INFO] Few-shot sampling strategy:   {config['sampling']['strategy']}")

    # 2. Configure the chosen VLM backend
    # -------------------------------------------------------------- #
    configure_vlm(config.get("model", {}))
    backend_name = config.get("model", {}).get("backend", "").lower()
    use_llava = backend_name in ("llava_hf", "med_llava")   # ← changed
    prompt_builder = build_llava_prompt if use_llava else build_gemini_prompt

    transform = T.Compose([T.ToTensor()])   # full-resolution images

    # 3. Balanced test subset
    # -------------------------------------------------------------- #
    full_test_ds = CSVDataset(csv_path=test_csv, transform=transform)
    balanced_indices = build_balanced_indices(
        full_test_ds,
        classification_type=classification_type,
        label_list=label_list,
        total_images=config["data"]["num_test_images"],
        randomize=config["data"].get("randomize_test_images", True),
        seed=config["data"].get("test_seed", 42),
    )
    test_loader = DataLoader(
        Subset(full_test_ds, balanced_indices),
        batch_size=1,
        shuffle=False,
    )

    # 4. Few-shot provider (static, random, or query-aware K-NN)
    # -------------------------------------------------------------- #
    strategy = config["sampling"]["strategy"].lower()

    # We support three strategies:
    #   • "random"       → prebuilt static pool (unchanged)
    #   • "knn"|"knn_dino"  → query-aware KNN using CNN/ViT (DINOv2)
    #   • "knn_radiomics"   → query-aware KNN using PyRadiomics + masks via suffix
    if strategy in ("knn", "knn_dino", "knn_radiomics"):
        # Choose feature backend by strategy
        if strategy == "knn_radiomics":
            embedder_name = "radiomics"  # signals the radiomics path
        else:
            embedder_name = config["sampling"].get("embedder", "resnet50")

        few_shot_provider = partial(
            build_query_knn_samples,
            train_csv=config["data"]["train_csv"],
            classification_type=classification_type,
            label_list=label_list,
            num_shots=config["data"]["num_shots"],
            embedder_name=embedder_name,
            device=config["sampling"].get("device", "cpu"),
            radiomics_cfg=config["sampling"].get("radiomics", {}),  # <- NEW
        )

    else:
        static_few_shot = build_few_shot_samples(
            config=config,
            transform=T.Compose([T.ToTensor()]),
            classification_type=classification_type,
            label_list=label_list,
        )
        if verbose:
            print("\n[INFO] Few-shot pool (static):")
            for lbl in label_list:
                print(f"  [{lbl}]")
                for _, p in static_few_shot[lbl]:
                    print(f"    – {p}")
        few_shot_provider = lambda *_, **__: static_few_shot   # noqa: E731

    # 5. Inference loop
    # -------------------------------------------------------------- #
    results = []
    for i, (img_tensor, img_paths, _) in enumerate(test_loader, start=1):
        img_path = img_paths[0]
        print(f"\n[IMAGE {i}/{len(test_loader.dataset)}] {img_path}")

        pil_img = T.ToPILImage()(img_tensor.squeeze(0))
        few_shot_samples = few_shot_provider(query_img=pil_img, query_path=img_path)

        if verbose and strategy == "knn":
            print("  Few-shot selection:")
            for lbl in label_list:
                print(f"    [{lbl}]")
                for _, p in few_shot_samples[lbl]:
                    print(f"      – {p}")

        contents = prompt_builder(
            few_shot_samples,
            pil_img,
            classification_type,
            label_list=label_list,
        )

        preds = vlm_api_call(contents, classification_type, label_list=label_list)

        entry = {
            "image_path": img_path,
            "thoughts":   preds.get("thoughts", ""),
            "answer":     preds.get("answer", "Unknown"),
            "location":   preds.get("location"),
        }
        if classification_type == "binary" and label_list and len(label_list) == 2:
            entry["score"] = {
                f"score_{lbl.lower().replace(' ', '_')}":
                preds.get(f"score_{lbl.lower().replace(' ', '_')}", -1)
                for lbl in label_list
            }
        else:
            entry["score"] = preds.get("score", -1)

        results.append(entry)

    # 6. Save results + optional evaluation
    # -------------------------------------------------------------- #
    os.makedirs(save_path, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(save_path, f"results_{ts}.json")

    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump({"results": results}, fp, indent=2)

    print(f"[INFO] Results saved to {out_json}")

    labels_path = config["data"].get("labels_path")
    if labels_path and os.path.exists(labels_path):
        print("[INFO] Running evaluation …")
        subprocess.run(
            ["python", "evaluation.py", "--results", out_json, "--labels", labels_path],
            check=True,
        )
    else:
        print("[WARN] Labels file missing; skipping evaluation.")


if __name__ == "__main__":
    main()