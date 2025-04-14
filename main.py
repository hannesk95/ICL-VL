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
from generate_labels import generate_labels_from_prefix

def main():
    # === 1. Load environment and config ===
    load_dotenv()
    config_path = "configs/CRC100K/one_shot.yaml"
    config = load_config(config_path)

    # === 2. Extract config parameters ===
    train_csv = config["data"]["train_csv"]
    test_csv = config["data"]["test_csv"]
    prompt_path = config["user_args"]["prompt_path"]
    save_path = config["data"]["save_path"]
    batch_size = config["data"]["batch_size"]
    num_pos_examples = config["data"]["num_shots"]
    num_neg_examples = config["data"]["num_shots"]
    num_test_images = config["data"].get("num_test_images", 10)

    randomize_few_shot = config["data"].get("randomize_few_shot", False)
    seed = config["data"].get("seed", 42)

    randomize_test = config["data"].get("randomize_test_images", False)
    test_seed = config["data"].get("test_seed", 123)

    os.environ["PROMPT_PATH"] = prompt_path
    print(f"[INFO] Using prompt file: {prompt_path}")

    configure_gemini()
    transform = T.Compose([T.ToTensor()])

    # === 3. Load datasets from CSVs ===
    pos_dataset = CSVDataset(csv_path=train_csv, transform=transform, label_filter="positive")
    neg_dataset = CSVDataset(csv_path=train_csv, transform=transform, label_filter="negative")
    test_dataset = CSVDataset(csv_path=test_csv, transform=transform)

    # === 4. Apply few-shot randomization ===
    if randomize_few_shot:
        print(f"[INFO] Randomizing few-shot examples with seed={seed}")
        random.seed(seed)
        pos_dataset = Subset(pos_dataset, random.sample(range(len(pos_dataset)), len(pos_dataset)))
        neg_dataset = Subset(neg_dataset, random.sample(range(len(neg_dataset)), len(neg_dataset)))

    # === 5. Apply test image randomization or slicing ===
    if randomize_test:
        print(f"[INFO] Randomizing test images with seed={test_seed}")
        random.seed(test_seed)
        test_indices = list(range(len(test_dataset)))
        random.shuffle(test_indices)
        test_dataset = Subset(test_dataset, test_indices[:num_test_images])
    else:
        test_dataset = Subset(test_dataset, list(range(min(len(test_dataset), num_test_images))))

    # === 6. Load data loaders ===
    pos_loader = DataLoader(pos_dataset, batch_size=1, shuffle=False)
    neg_loader = DataLoader(neg_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # === 7. Load few-shot examples ===
    pos_examples, neg_examples = [], []

    print("[INFO] Loading positive examples...")
    for i, (img_tensor, path, label) in enumerate(pos_loader):
        if i >= num_pos_examples: break
        pil_img = T.ToPILImage()(img_tensor.squeeze(0))
        print(f"[LOADED] Positive example {i}: {path[0]}")
        pos_examples.append(pil_img)

    print("[INFO] Loading negative examples...")
    for i, (img_tensor, path, label) in enumerate(neg_loader):
        if i >= num_neg_examples: break
        pil_img = T.ToPILImage()(img_tensor.squeeze(0))
        print(f"[LOADED] Negative example {i}: {path[0]}")
        neg_examples.append(pil_img)

    print(f"\n[SUMMARY] Using {len(pos_examples)} Positive and {len(neg_examples)} Negative examples.\n")

    # === 8. Run inference on test images ===
    all_results = []
    for test_tensors, test_paths, test_labels in test_loader:
        pil_img = T.ToPILImage()(test_tensors.squeeze(0))
        path_str = test_paths[0]
        print(f"[LOADED] Test image: {path_str}")

        gemini_contents = build_gemini_prompt(pos_examples, neg_examples, [pil_img])
        predictions = gemini_api_call(gemini_contents)

        all_results.append({
            "image_path": path_str,
            "thoughts": predictions.get("thoughts", ""),
            "answer": predictions.get("answer", "Unknown"),
            "score_tumor": predictions.get("score_tumor", -1),
            "score_no_tumor": predictions.get("score_no_tumor", -1),
            "location": predictions.get("location", None)
        })

    # === 9. Save results ===
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_path, f"results_{timestamp}.json")
    latest_results_path = os.path.join(save_path, "results_latest.json")

    results_to_save = {
        "summary": f"Used {len(pos_examples)} positive examples and {len(neg_examples)} negative examples.",
        "results": all_results
    }

    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    with open(latest_results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\n[INFO] Saved results to {results_path}")

    # === 10. Generate filtered labels.json based on predictions ===
    original_labels_path = os.path.join("data/tumor/test", "labels.json")
    filtered_labels_path = os.path.join(save_path, "labels_filtered.json")

    if os.path.exists(original_labels_path):
        with open(original_labels_path, "r") as f:
            all_labels = json.load(f)

        predicted_filenames = {os.path.basename(r["image_path"]) for r in all_results}
        filtered_labels = {
            fname: label for fname, label in all_labels.items()
            if fname in predicted_filenames
        }

        with open(filtered_labels_path, "w") as f:
            json.dump(filtered_labels, f, indent=2)

        print(f"[INFO] Saved filtered labels for evaluation to {filtered_labels_path}")
    else:
        print(f"[WARN] Original labels.json not found at {original_labels_path}")
        filtered_labels_path = original_labels_path  # fallback to original

    # === 11. Evaluate using filtered labels ===
    try:
        import subprocess
        subprocess.run([
            "python", "evaluation.py",
            "--results", latest_results_path,
            "--labels", filtered_labels_path
        ], check=True)
    except Exception as e:
        print(f"[WARN] Evaluation failed: {e}")

if __name__ == "__main__":
    main()