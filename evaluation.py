import json
import os
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def load_ground_truth(label_path):
    """
    Load ground-truth labels from a JSON file, returning a dict like:
    {
      "image1.jpg": "Tumor",
      "image2.jpg": "No Tumor",
      ...
    }
    """
    with open(label_path, "r") as f:
        return json.load(f)

def load_predictions(results_path):
    """
    Load the model's predictions (and scores) from the 'results' JSON file,
    returning a dict keyed by filename, e.g.:
    {
      "image1.jpg": {
         "answer": "Tumor",
         "score_tumor": 0.92
      },
      ...
    }
    If 'score_tumor' is missing, default to None (or 0.5, etc.).
    """
    with open(results_path, "r") as f:
        data = json.load(f)
    results = data["results"]
    predictions = {}
    for entry in results:
        file_name = os.path.basename(entry["image_path"])
        predictions[file_name] = {
            "answer": entry.get("answer", "Unknown"),
            "score_tumor": entry.get("score_tumor", None)
        }
    return predictions

def evaluate(results_path, ground_truth_path):
    """
    1. Reads ground truth and predictions.
    2. Prints a classification report for 'Tumor' vs. 'No Tumor'.
    3. Computes and prints accuracy.
    4. Attempts to compute ROC-AUC using 'score_tumor' if possible.
    5. Saves the classification report + accuracy (and optionally AUC) to a text file.
    6. If AUC is computed successfully, it plots and saves the ROC curve to 'plots/roc_curve.png'.
    """
    # Load the ground truth labels (string: "Tumor" or "No Tumor")
    ground_truth = load_ground_truth(ground_truth_path)
    
    # Load predictions (both string label + numeric score, if present)
    predictions = load_predictions(results_path)

    y_true_str, y_pred_str = [], []
    y_true_bin, y_score_tumor = [], []

    # Build lists for discrete evaluation (classification_report) and for ROC/AUC
    for fname, true_label in ground_truth.items():
        # Model's predicted data
        pred_data = predictions.get(fname, {"answer": "Unknown", "score_tumor": 0.5})
        pred_label = pred_data["answer"]
        score_tumor = pred_data["score_tumor"]

        # Collect for discrete metrics
        y_true_str.append(true_label)
        y_pred_str.append(pred_label)

        # Convert ground truth to 0 or 1
        if true_label == "Tumor":
            y_true_bin.append(1)
        else:
            y_true_bin.append(0)

        # If the model provided a numeric score, use it; otherwise fallback
        if score_tumor is None:
            score_tumor = 0.5  # or any neutral default
        y_score_tumor.append(score_tumor)

    # === 1) Classification Report (Discrete) ===
    print("\n=== Evaluation Report ===")
    # Focus on 'Tumor' and 'No Tumor' as your main labels
    report = classification_report(y_true_str, y_pred_str, labels=["Tumor", "No Tumor"])
    print(report)

    # === 2) Accuracy ===
    acc = accuracy_score(y_true_str, y_pred_str)
    print("Accuracy:", acc)

    # === 3) ROC & AUC (Continuous) ===
    # We need numeric ground truth (0/1) and predicted probabilities (score_tumor)
    # We'll catch if there's only one class or invalid data:
    eval_dir = os.path.dirname(results_path)
    eval_report_path = os.path.join(eval_dir, "eval_report.txt")

    try:
        auc_value = roc_auc_score(y_true_bin, y_score_tumor)
        print(f"AUC: {auc_value:.4f}")

        # === 4) Optionally Plot ROC Curve ===
        fpr, tpr, thresholds = roc_curve(y_true_bin, y_score_tumor)

        # We'll save the plot in a subfolder "plots/roc_curve.png"
        plot_dir = os.path.join(eval_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "roc_curve.png")

        # Generate and save the ROC curve
        plt.figure()  # single figure
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.savefig(plot_path)
        plt.close()
        print(f"[INFO] Saved ROC curve to {plot_path}")

    except ValueError as e:
        # Typically occurs if there's only one class present in y_true_bin
        print(f"Could not compute AUC: {e}")
        auc_value = None

    # === 5) Save everything to a text file ===
    with open(eval_report_path, "w") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.2f}\n")
        if auc_value is not None:
            f.write(f"AUC: {auc_value:.4f}\n")

    print(f"[INFO] Saved evaluation report to {eval_report_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="results/results_latest.json",
                        help="Path to predictions file (JSON with 'results' array)")
    parser.add_argument("--labels", type=str, default="data/tumor/test/labels.json",
                        help="Path to ground-truth labels (JSON dictionary)")
    args = parser.parse_args()

    print(f"[INFO] Evaluating using results: {args.results}")
    print(f"[INFO] Ground truth labels: {args.labels}")

    evaluate(args.results, args.labels)