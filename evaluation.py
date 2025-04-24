# evaluation.py
import os
import json
import argparse
from collections import defaultdict
from datetime import datetime

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
)

import numpy as np
import matplotlib

matplotlib.use("Agg")                        
import matplotlib.pyplot as plt              


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def save_confusion_matrix(cm, labels, outfile_png, title="Confusion Matrix"):
    n = len(labels)
    fig, ax = plt.subplots(
        figsize=(n * 0.7 + 2, n * 0.7 + 2),
        dpi=300,
        constrained_layout=True,
    )
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=range(n),
        yticks=range(n),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.savefig(outfile_png, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(fpr, tpr, auc_value, outfile_png, pos_label):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300, constrained_layout=True)
    ax.plot(fpr, tpr, linewidth=1.5, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set(
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"ROC curve (positive = '{pos_label}')",
    )
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(outfile_png, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def evaluate(results_path, labels_path):
    # ---------- load ----------
    with open(results_path, "r") as f:
        results = json.load(f)["results"]
    with open(labels_path, "r") as f:
        ground_truth = json.load(f)

    y_true, y_pred, y_score = [], [], []

    for res in results:
        fname = os.path.basename(res["image_path"])
        y_true.append(ground_truth.get(fname, "Unknown"))
        y_pred.append(res["answer"])

        # continuous score if present (binary)
        if "score_tumor" in res:  # legacy binary dict-style entry
            y_score.append(res["score_tumor"])
        elif isinstance(res.get("score"), dict):  # new dict wrapper
            y_score.append(res["score"].get("score_tumor"))
        else:
            y_score.append(None)

    # ---------- basic metrics ----------
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc * 100:.2f}%")

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1

    print("\nPer-Class Accuracy:")
    for cls in sorted(class_total):
        tot = class_total[cls]
        hits = class_correct[cls]
        pct = 100.0 * hits / tot if tot else 0.0
        print(f"  {cls}: {pct:.2f}% ({hits}/{tot})")

    # ---------- MCC ----------
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")

    # ---------- AUC & ROC ----------
    auc_value = None
    roc_saved = False
    labels_set = sorted({*y_true, *y_pred})
    if len(labels_set) == 2 and all(s is not None for s in y_score):
        pos_label = "Tumor" if "Tumor" in labels_set else labels_set[1]
        y_true_bin = [1 if t == pos_label else 0 for t in y_true]
        auc_value = roc_auc_score(y_true_bin, y_score)
        print(f"ROC-AUC (positive = '{pos_label}'): {auc_value:.4f}")
    else:
        print("ROC-AUC: n/a (need binary task **and** continuous scores)")

    # ---------- files ----------
    cm_labels = labels_set
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    base = os.path.splitext(os.path.basename(results_path))[0] 
    out_dir = os.path.dirname(results_path)

    cm_png = os.path.join(out_dir, f"{base}_cm.png")
    cm_json = os.path.join(out_dir, f"{base}_cm.json")
    roc_png = os.path.join(out_dir, f"{base}_roc.png")
    metrics_json = os.path.join(out_dir, f"{base}_metrics.json")

    save_confusion_matrix(cm, cm_labels, cm_png)
    with open(cm_json, "w") as f:
        json.dump({"labels": cm_labels, "matrix": cm.tolist()}, f, indent=2)

    if auc_value is not None:
        fpr, tpr, _ = roc_curve([1 if t == pos_label else 0 for t in y_true], y_score)
        save_roc_curve(fpr, tpr, auc_value, roc_png, pos_label)
        roc_saved = True

    # ---------- scalar metrics to JSON ----------
    with open(metrics_json, "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "mcc": mcc,
                "auc": auc_value,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            },
            f,
            indent=2,
        )

    # ---------- summary ----------
    print("\Results saved to:")
    print(f"  • {cm_png}")
    print(f"  • {cm_json}")
    if roc_saved:
        print(f"  • {roc_png}")
    print(f"  • {metrics_json}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification results")
    parser.add_argument("--results", required=True, help="Path to the JSON results file")
    parser.add_argument("--labels", required=True, help="Path to the ground truth labels JSON file")
    args = parser.parse_args()

    evaluate(args.results, args.labels)