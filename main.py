import os
import json
import datetime
import argparse
from dotenv import load_dotenv
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from config import load_config  
from dataset import CustomImageDataset
from model import configure_gemini, build_gemini_prompt, gemini_api_call
from generate_labels import generate_labels_from_prefix

def main():
    # === Load CLI arguments ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/tumor/three_shot.yaml", help="Path to config YAML")
    args = parser.parse_args()

    # === Load environment and config ===
    load_dotenv()
    config = load_config(args.config)

    # === Extract config values ===
    image_root = config["data"]["image_root"]
    pos_dir = os.path.join(image_root, "positive")
    neg_dir = os.path.join(image_root, "negative")
    test_dir = os.path.join(image_root, "test")
    prompt_path = config["user_args"]["prompt_path"]
    save_path = config["data"]["save_path"]
    batch_size = config["data"]["batch_size"]
    num_pos_examples = config["data"]["num_shots"]
    num_neg_examples = config["data"]["num_shots"]

    os.environ["PROMPT_PATH"] = prompt_path
    print(f"[INFO] Using prompt file: {prompt_path}")
    
    # Configure Gemini API
    configure_gemini()

    # Define image transformation
    transform = T.Compose([
        T.ToTensor(),
    ])

    # Load datasets
    pos_dataset = CustomImageDataset(root_dir=pos_dir, transform=transform)
    neg_dataset = CustomImageDataset(root_dir=neg_dir, transform=transform)
    test_dataset = CustomImageDataset(root_dir=test_dir, transform=transform)

    # DataLoaders
    pos_loader = DataLoader(pos_dataset, batch_size=1, shuffle=True)
    neg_loader = DataLoader(neg_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Select examples
    pos_examples, neg_examples = [], []
    all_results = []

    print("[INFO] Loading positive examples...")
    for i, (img_tensor, path) in enumerate(pos_loader):
        if i >= num_pos_examples:
            break
        pil_img = T.ToPILImage()(img_tensor.squeeze(0))
        print(f"[LOADED] Positive example {i}: {path[0]}")
        pos_examples.append(pil_img)

    print("[INFO] Loading negative examples...")
    for i, (img_tensor, path) in enumerate(neg_loader):
        if i >= num_neg_examples:
            break
        pil_img = T.ToPILImage()(img_tensor.squeeze(0))
        print(f"[LOADED] Negative example {i}: {path[0]}")
        neg_examples.append(pil_img)

    print(f"\n[SUMMARY] Using {len(pos_examples)} Positive and {len(neg_examples)} Negative examples.\n")

    # Classify test images
    for test_tensors, test_paths in test_loader:
        for img_tensor, path in zip(test_tensors, test_paths):
            pil_img = T.ToPILImage()(img_tensor)
            print(f"[LOADED] Test image: {path}")

            gemini_contents = build_gemini_prompt(pos_examples, neg_examples, [pil_img])
            predictions = gemini_api_call(gemini_contents)

            all_results.append({
                "image_path": path,
                "thoughts": predictions.get("thoughts", ""),
                "answer": predictions.get("answer", "Unknown"),
                "score_tumor": predictions.get("score_tumor", -1),
                "score_no_tumor": predictions.get("score_no_tumor", -1),
                "location": predictions.get("location", None)
            })

    # Save results
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

    # Generate labels if not already present
    label_path = os.path.join(test_dir, "labels.json")
    if not os.path.exists(label_path):
        generate_labels_from_prefix(test_dir, label_path)

    # Run evaluation
    try:
        import subprocess
        subprocess.run([
            "python", "evaluation.py",
            "--results", latest_results_path,
            "--labels", os.path.join(test_dir, "labels.json")
        ], check=True)
    except Exception as e:
        print(f"[WARN] Evaluation failed: {e}")


if __name__ == "__main__":
    main()