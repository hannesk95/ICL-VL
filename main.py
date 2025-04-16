# main.py
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

def get_few_shot_samples(csv_path, transform, classification_type, label_list, num_shots,
                         randomize=False, seed=42):
    """
    Gather few-shot examples from the training CSV.
    Returns a dictionary mapping each label to a list of PIL images.
    """
    dataset = CSVDataset(csv_path, transform=transform)
    from collections import defaultdict
    label_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, _, lbl = dataset[idx]
        label_to_indices[lbl].append(idx)
    if randomize:
        random.seed(seed)
        for lbl in label_to_indices:
            random.shuffle(label_to_indices[lbl])
    few_shot_dict = {}
    for lbl in label_list:
        indices = label_to_indices.get(lbl, [])
        chosen_indices = indices[:num_shots]
        images = []
        for cidx in chosen_indices:
            img_tensor, _, _ = dataset[cidx]
            images.append(T.ToPILImage()(img_tensor))
        few_shot_dict[lbl] = images
    return few_shot_dict

def auto_generate_labels_if_needed(config):
    """
    Checks the labels file path defined in config.
    If it is set but the file does not exist, prints a warning.
    (Here you could also call your label-generation function.)
    """
    labels_path = config["data"].get("labels_path")
    if labels_path and not os.path.exists(labels_path):
        print(f"[WARN] Ground truth labels not found at {labels_path}.")
        print("[INFO] Please generate the ground truth labels using your preferred method.")
    else:
        print(f"[INFO] Using ground truth labels from {labels_path}")
    return labels_path

def main():
    # 1. Load environment and configuration
    load_dotenv()
    config_path = "configs/tumor/one_shot.yaml"  # update as needed
    config = load_config(config_path)

    train_csv = config["data"]["train_csv"]
    test_csv = config["data"]["test_csv"]
    save_path = config["data"]["save_path"]
    prompt_path = config["user_args"]["prompt_path"]

    batch_size = config["data"]["batch_size"]
    num_shots = config["data"]["num_shots"]
    num_test_images = config["data"].get("num_test_images", 10)
    randomize_few_shot = config["data"].get("randomize_few_shot", False)
    seed = config["data"].get("seed", 42)
    randomize_test = config["data"].get("randomize_test_images", False)
    test_seed = config["data"].get("test_seed", 123)
    classification_type = config["classification"]["type"]
    label_list = config["classification"]["labels"]

    os.environ["PROMPT_PATH"] = prompt_path
    print(f"[INFO] Using prompt file: {prompt_path}")
    print(f"[INFO] Classification type: {classification_type}")
    print(f"[INFO] Label list: {label_list}")

    # 2. Check for ground truth labels file
    labels_path = auto_generate_labels_if_needed(config)

    # 3. Configure Gemini and prepare image transform
    configure_gemini()
    transform = T.Compose([T.ToTensor()])

    # 4. Load test dataset and slice/shuffle it
    test_dataset = CSVDataset(csv_path=test_csv, transform=transform)
    if randomize_test:
        print(f"[INFO] Randomizing test images with seed={test_seed}")
        random.seed(test_seed)
        all_indices = list(range(len(test_dataset)))
        random.shuffle(all_indices)
        test_dataset = Subset(test_dataset, all_indices[:num_test_images])
    else:
        test_dataset = Subset(test_dataset, list(range(min(len(test_dataset), num_test_images))))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 5. Gather few-shot examples from the training set
    few_shot_samples = get_few_shot_samples(
        csv_path=train_csv,
        transform=transform,
        classification_type=classification_type,
        label_list=label_list,
        num_shots=num_shots,
        randomize=randomize_few_shot,
        seed=seed
    )
    print("\n[INFO] Few-shot summary:")
    for lbl, imgs in few_shot_samples.items():
        print(f"  Label '{lbl}': {len(imgs)} examples")

    # 6. Run inference on each test image using Gemini
    all_results = []
    for test_tensors, test_paths, test_labels in test_loader:
        test_img_pil = T.ToPILImage()(test_tensors.squeeze(0))
        path_str = test_paths[0]
        print(f"\n[LOADED] Test image: {path_str}")

        gemini_contents = build_gemini_prompt(
            few_shot_samples=few_shot_samples,
            test_image=test_img_pil,
            classification_type=classification_type
        )
        predictions = gemini_api_call(gemini_contents, classification_type=classification_type)

        if classification_type == "binary":
            entry = {
                "image_path": path_str,
                "thoughts": predictions.get("thoughts", ""),
                "answer": predictions.get("answer", "Unknown"),
                "score_tumor": predictions.get("score_tumor", -1),
                "score_no_tumor": predictions.get("score_no_tumor", -1),
                "location": predictions.get("location", None)
            }
        else:
            entry = {
                "image_path": path_str,
                "thoughts": predictions.get("thoughts", ""),
                "answer": predictions.get("answer", "Unknown"),
                "score": predictions.get("score", -1.0)
            }
        all_results.append(entry)

    # 7. Save the inference results
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(save_path, f"results_{timestamp}.json")
    latest_results_path = os.path.join(save_path, "results_latest.json")
    results_to_save = {
        "summary": f"Used {num_shots} shots per label in {classification_type} mode.",
        "results": all_results
    }
    with open(results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    with open(latest_results_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\n[INFO] Saved results to {results_path}")

    # 8. Filter ground truth labels to only include images that were predicted
    if labels_path and os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            all_labels = json.load(f)
        predicted_filenames = {os.path.basename(r["image_path"]) for r in all_results}
        filtered_labels = {fname: label for fname, label in all_labels.items() if fname in predicted_filenames}
        filtered_labels_path = os.path.join(save_path, "labels_filtered.json")
        with open(filtered_labels_path, "w") as f:
            json.dump(filtered_labels, f, indent=2)
        print(f"[INFO] Saved filtered labels for evaluation to {filtered_labels_path}")
    else:
        filtered_labels_path = labels_path
        print(f"[WARN] Labels file not found; evaluation may fail.")

    # 9. Call the evaluation script using the filtered ground truth
    try:
        import subprocess
        subprocess.run([
            "python", "evaluation.py",
            "--results", latest_results_path,
            "--labels", filtered_labels_path,
            "--classification_type", classification_type
        ], check=True)
    except Exception as e:
        print(f"[WARN] Evaluation failed: {e}")

if __name__ == "__main__":
    main()