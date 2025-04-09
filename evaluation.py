import json
import os
from sklearn.metrics import classification_report, accuracy_score

def load_ground_truth(label_path):
    with open(label_path, "r") as f:
        return json.load(f)

def load_predictions(results_path):
    with open(results_path, "r") as f:
        data = json.load(f)
    results = data["results"]
    predictions = {}
    for entry in results:
        file_name = os.path.basename(entry["image_path"])
        predictions[file_name] = entry["answer"]
    return predictions

def evaluate(results_path, ground_truth_path):
    ground_truth = load_ground_truth(ground_truth_path)
    predictions = load_predictions(results_path)

    y_true, y_pred = [], []

    for fname, true_label in ground_truth.items():
        pred_label = predictions.get(fname, "Unknown")
        y_true.append(true_label)
        y_pred.append(pred_label)

    print("\n=== Evaluation Report ===")
    report = classification_report(y_true, y_pred, labels=["Tumor", "No Tumor"])
    print(report)
    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", acc)

    # Save to file
    eval_dir = os.path.dirname(results_path)
    eval_report_path = os.path.join(eval_dir, "eval_report.txt")
    with open(eval_report_path, "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.2f}\n")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/results_latest.json", help="Path to predictions file")
    parser.add_argument("--labels", type=str, default="data/tumor/test/labels.json", help="Path to ground-truth labels")
    args = parser.parse_args()

    print(f"[INFO] Evaluating using results: {args.results}")
    print(f"[INFO] Ground truth labels: {args.labels}")

    evaluate(args.results, args.labels)