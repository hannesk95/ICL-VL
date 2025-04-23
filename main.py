import os
import json
import datetime
import random
from dotenv import load_dotenv
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from config import load_config
from dataset import CSVDataset
from model import configure_gemini, build_gemini_prompt, gemini_api_call
import subprocess

def get_few_shot_samples(csv_path, transform, classification_type, label_list, num_shots,
                         randomize=False, seed=42):
    dataset = CSVDataset(csv_path, transform=transform)
    from collections import defaultdict
    label_to_indices = defaultdict(list)

    label_map = {
        "ADI": "Adipose",
        "DEB": "Debris",
        "LYM": "Lymphocytes",
        "MUC": "Mucus",
        "MUS": "Smooth Muscle",
        "NORM": "Normal Colon Mucosa",
        "STR": "Cancer-Associated Stroma",
        "TUM": "Colorectal Adenocarcinoma Epithelium"
    }

    for idx in range(len(dataset)):
        _, _, csv_label = dataset[idx]
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

    few_shot_dict = {}
    for lbl in label_list:
        indices = label_to_indices.get(lbl, [])[:num_shots]
        images = [T.ToPILImage()(dataset[cidx][0]) for cidx in indices]
        few_shot_dict[lbl] = images

    return few_shot_dict

def main():
    load_dotenv()
    config = load_config("configs/CRC100K/binary/one_shot.yaml")

    train_csv, test_csv, save_path = config["data"]["train_csv"], config["data"]["test_csv"], config["data"]["save_path"]
    prompt_path = config["user_args"]["prompt_path"]
    classification_type, label_list = config["classification"]["type"], config["classification"]["labels"]

    os.environ["PROMPT_PATH"] = prompt_path
    print(f"[INFO] Using prompt file: {prompt_path}")
    print(f"[INFO] Classification type: {classification_type}")
    print(f"[INFO] Label list: {label_list}")
    
    configure_gemini()
    transform = T.Compose([T.ToTensor()])

    test_dataset = CSVDataset(csv_path=test_csv, transform=transform)
    test_indices = list(range(len(test_dataset)))
    if config["data"].get("randomize_test_images"):
        random.seed(config["data"].get("test_seed", 42))
        random.shuffle(test_indices)
    test_dataset = Subset(test_dataset, test_indices[:config["data"]["num_test_images"]])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    few_shot_samples = get_few_shot_samples(
    train_csv, transform, classification_type, label_list,
    config["data"]["num_shots"], config["data"].get("randomize_few_shot"), config["data"].get("seed", 42)
    )

    print("\n[INFO] Few-shot image selection:")
    for label in label_list:
        print(f"  [{label}]")
        for img in few_shot_samples[label]:
            print(f"    - Image loaded (PIL object, source hidden for now)")  # Optional: add debug logic to show file path

    results = []
    for test_tensors, test_paths, _ in test_loader:
        test_img_pil = T.ToPILImage()(test_tensors.squeeze(0))
        path_str = test_paths[0]
        print(f"\n[LOADED TEST IMAGE] {path_str}")
        contents = build_gemini_prompt(few_shot_samples, test_img_pil, classification_type)
        predictions = gemini_api_call(contents, classification_type)

        entry = {
            "image_path": test_paths[0],
            "thoughts": predictions.get("thoughts", ""),
            "answer": predictions.get("answer", "Unknown"),
            "score": predictions.get("score", -1) if classification_type != "binary" else {
                "score_tumor": predictions.get("score_tumor", -1),
                "score_no_tumor": predictions.get("score_no_tumor", -1)
            }
        }
        if classification_type == "binary":
            entry["location"] = predictions.get("location")
        results.append(entry)

    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_path, f"results_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    print(f"[INFO] Results saved to {results_path}")

    labels_path = config["data"].get("labels_path")
    if labels_path and os.path.exists(labels_path):
        print("[INFO] Running evaluation...")
        subprocess.run([
            "python", "evaluation.py",
            "--results", results_path,
            "--labels", labels_path
        ], check=True)
    else:
        print("[WARN] Labels file missing; skipping evaluation.")

if __name__ == "__main__":
    main()