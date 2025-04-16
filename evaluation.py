# evaluation.py
import json
import os
import argparse
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

def load_ground_truth(label_path):
    with open(label_path, "r") as f:
        return json.load(f)

def load_predictions(results_path, classification_type):
    with open(results_path, "r") as f:
        data = json.load(f)
    results = data.get("results", [])
    predictions = {}
    for entry in results:
        file_name = os.path.basename(entry["image_path"])
        if classification_type == "binary":
            predictions[file_name] = {
                "answer": entry.get("answer", "Unknown"),
                "score_tumor": entry.get("score_tumor", 0.5)
            }
        else:
            predictions[file_name] = {
                "answer": entry.get("answer", "Unknown"),
                "score": entry.get("score", 0.5)
            }
    return predictions

def evaluate(results_path, ground_truth_path, classification_type):
    ground_truth = load_ground_truth(ground_truth_path)
    predictions = load_predictions(results_path, classification_type)

    y_true_str = []
    y_pred_str = []

    if classification_type == "binary":
        for fname, true_label in ground_truth.items():
            pred_data = predictions.get(fname, {"answer": "Unknown", "score_tumor": 0.5})
            y_true_str.append(true_label)
            y_pred_str.append(pred_data["answer"])
        report = classification_report(y_true_str, y_pred_str, labels=["Tumor", "No Tumor"])
        print("\n=== Binary Classification Report ===")
        print(report)
        acc = accuracy_score(y_true_str, y_pred_str)
        print("Accuracy:", acc)
    else:
        possible_labels = list(set(ground_truth.values()))
        for fname, true_label in ground_truth.items():
            pred_data = predictions.get(fname, {"answer": "Unknown", "score": 0.5})
            y_true_str.append(true_label)
            y_pred_str.append(pred_data["answer"])
        report = classification_report(y_true_str, y_pred_str, labels=possible_labels)
        print("\n=== Multi-class Classification Report ===")
        print(report)
        acc = accuracy_score(y_true_str, y_pred_str)
        print("Accuracy:", acc)
    eval_dir = os.path.dirname(results_path)
    eval_report_path = os.path.join(eval_dir, "eval_report.txt")
    with open(eval_report_path, "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\n")
    print(f"[INFO] Saved evaluation report to {eval_report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/results_latest.json",
                        help="Path to predictions JSON file")
    parser.add_argument("--labels", type=str, default="data/tumor/test/labels.json",
                        help="Path to ground truth JSON file")
    parser.add_argument("--classification_type", type=str, default="binary")
    args = parser.parse_args()
    print(f"[INFO] Evaluating with results: {args.results}")
    print(f"[INFO] Ground truth labels: {args.labels}")
    print(f"[INFO] Classification type: {args.classification_type}")
    evaluate(args.results, args.labels, classification_type=args.classification_type)