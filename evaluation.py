import json
import argparse
from collections import defaultdict
from sklearn.metrics import classification_report, accuracy_score

def evaluate(results_path, labels_path):
    with open(results_path, "r") as f:
        results_data = json.load(f)
        results = results_data["results"]

    with open(labels_path, "r") as f:
        labels = json.load(f)

    y_true = []
    y_pred = []

    for res in results:
        fname = res["image_path"].split("/")[-1]
        true_label = labels.get(fname, "Unknown")
        pred_label = res["answer"]
        y_true.append(true_label)
        y_pred.append(pred_label)

    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    # ✅ Overall Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

    # ✅ Per-Class Accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for true, pred in zip(y_true, y_pred):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    print("\nPer-Class Accuracy:")
    for cls in sorted(class_total.keys()):
        acc = class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0.0
        print(f"  {cls}: {acc * 100:.2f}% ({class_correct[cls]}/{class_total[cls]})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification results")
    parser.add_argument("--results", required=True, help="Path to the JSON results file")
    parser.add_argument("--labels", required=True, help="Path to the ground truth labels JSON file")

    args = parser.parse_args()
    evaluate(args.results, args.labels)