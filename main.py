# main.py – flexible binary labels **without legacy tumor/no_tumor keys**
#           + high-res few-shot transform & model-kwargs support

import os
import json
import datetime
import subprocess
from dotenv import load_dotenv

import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from config import load_config
from dataset import CSVDataset
from model import configure_gemini, build_gemini_prompt, gemini_api_call
from sampler import build_balanced_indices, build_few_shot_samples


def main() -> None:
    # ------------------------------------------------------------------ #
    # Load YAML config & env                                             #
    # ------------------------------------------------------------------ #
    load_dotenv()
    config = load_config("configs/glioma/binary/t2/three_shot.yaml")

    test_csv   = config["data"]["test_csv"]
    save_path  = config["data"]["save_path"]

    prompt_path         = config["user_args"]["prompt_path"]
    classification_type = config["classification"]["type"]
    label_list          = config["classification"]["labels"]

    os.environ["PROMPT_PATH"] = prompt_path

    print(f"[INFO] Using prompt file:            {prompt_path}")
    print(f"[INFO] Classification type:          {classification_type}")
    print(f"[INFO] Label list:                   {label_list}")
    print(f"[INFO] Few-shot sampling strategy:   {config['sampling']['strategy']}")

    # ------------------------------------------------------------------ #
    # Gemini & transforms                                                #
    # ------------------------------------------------------------------ #
    # Pass the model block so that temperature / top-p / max tokens
    # actually reach the GenerativeModel constructor.
    configure_gemini(config.get("model", {}))

    transform = T.Compose([T.ToTensor()])   # full-resolution test images

    # HIGH-RES DEMO IMAGES – no forced resize any more
    few_shot_transform = T.Compose([
        T.ToTensor(),                       # keep original resolution
    ])

    # ------------------------------------------------------------------ #
    # Balanced test subset                                               #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Few-shot pool (uses high-res transform)                            #
    # ------------------------------------------------------------------ #
    few_shot_samples = build_few_shot_samples(
        config=config,
        transform=few_shot_transform,
        classification_type=classification_type,
        label_list=label_list,
    )

    print("\n[INFO] Few-shot image selection:")
    for lbl in label_list:
        print(f"  [{lbl}]")
        for _, p in few_shot_samples[lbl]:
            print(f"    – {p}")

    # ------------------------------------------------------------------ #
    # Inference loop                                                     #
    # ------------------------------------------------------------------ #
    results = []
    for i, (img_tensor, img_paths, _) in enumerate(test_loader, start=1):
        img_path = img_paths[0]
        print(f"\n[IMAGE {i}/{len(test_loader.dataset)}] {img_path}")

        pil_img = T.ToPILImage()(img_tensor.squeeze(0))
        contents = build_gemini_prompt(
            few_shot_samples,
            pil_img,
            classification_type,
            label_list=label_list,
        )
        preds = gemini_api_call(contents, classification_type, label_list=label_list)

        entry = {
            "image_path": img_path,
            "thoughts":   preds.get("thoughts", ""),
            "answer":     preds.get("answer", "Unknown"),
            "location":   preds.get("location"),
        }
        if classification_type == "binary":
            entry["score"] = {
                f"score_{lbl.lower().replace(' ', '_')}":
                preds.get(f"score_{lbl.lower().replace(' ', '_')}", -1)
                for lbl in label_list
            }
        else:
            entry["score"] = preds.get("score", -1)

        results.append(entry)

    # ------------------------------------------------------------------ #
    # Save + optional evaluation                                         #
    # ------------------------------------------------------------------ #
    os.makedirs(save_path, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(save_path, f"results_{ts}.json")

    with open(out_json, "w") as fp:
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