# evaluation.py – flexible two-label evaluator

"""Evaluate classification results saved by main.py.

Highlights vs. original:
• Understands *nested* score dictionaries produced by the new main.py.
• Works with any two-label list (class1/class2, Tumor/No Tumor, etc.).
• Fixes the UTC deprecation warning.
"""

from __future__ import annotations

import os
import json
import argparse
from collections import defaultdict
from datetime import datetime, UTC
from typing import List, Dict, Any

import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    matthews_corrcoef,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------
# plotting helpers
# ----------------------------------------------------------------------

def save_confusion_matrix(cm: np.ndarray, labels: List[str], outfile_png: str, title: str = "Confusion Matrix") -> None:
    n = len(labels)
    fig, ax = plt.subplots(figsize=(n * 0.7 + 2, n * 0.7 + 2), dpi=300, constrained_layout=True)
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
            ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)
    fig.savefig(outfile_png, bbox_inches="tight")
    plt.close(fig)


def save_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_value: float, outfile_png: str, pos_label: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300, constrained_layout=True)
    ax.plot(fpr, tpr, linewidth=1.5, label=f"AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], "--", linewidth=1)
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel="False Positive Rate", ylabel="True Positive Rate", title=f"ROC curve (positive = '{pos_label}')")
    ax.legend(loc="lower right", fontsize=7)
    fig.savefig(outfile_png, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# evaluation core
# ----------------------------------------------------------------------

def _snake(label: str) -> str:
    """Lower-case and replace whitespace with underscores."""
    return "_".join(label.lower().split())


def evaluate(results_path: str, labels_path: str) -> None:
    # ---------- load ----------
    with open(results_path, "r") as f:
        results: List[Dict[str, Any]] = json.load(f)["results"]
    with open(labels_path, "r") as f:
        ground_truth: Dict[str, str] = json.load(f)

    y_true, y_pred, score_dicts = [], [], []
    for res in results:
        fname = os.path.basename(res["image_path"])
        y_true.append(ground_truth.get(fname, "Unknown"))
        y_pred.append(res.get("answer", "Unknown"))

        # collect any scores (nested or legacy flat)
        if isinstance(res.get("score"), dict):
            score_dicts.append(res["score"])
        else:
            score_dicts.append({k: res.get(k) for k in ("score_tumor", "score_no_tumor") if k in res})

    # ---------- determine positive label and build y_score ----------
    labels_set = sorted({*y_true, *y_pred})
    y_score: List[float | None] = [None] * len(y_true)
    pos_label = None
    if len(labels_set) == 2:
        # choose lexicographically first label as positive (historic behaviour)
        pos_label = labels_set[0]
        pos_key = f"score_{_snake(pos_label)}"
        for i, sdict in enumerate(score_dicts):
            if sdict and pos_key in sdict:
                y_score[i] = sdict[pos_key]

    # ---------- basic metrics ----------
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))

    acc = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc * 100:.2f}%")

    # per-class accuracy
    class_correct, class_total = defaultdict(int), defaultdict(int)
    for t, p in zip(y_true, y_pred):
        class_total[t] += 1
        if t == p:
            class_correct[t] += 1
    print("\nPer-Class Accuracy:")
    for cls in sorted(class_total):
        pct = 100.0 * class_correct[cls] / class_total[cls] if class_total[cls] else 0.0
        print(f"  {cls}: {pct:.2f}% ({class_correct[cls]}/{class_total[cls]})")

    # MCC
    mcc = matthews_corrcoef(y_true, y_pred)
    print(f"\nMatthews Correlation Coefficient (MCC): {mcc:.4f}")

    # AUC & ROC
    auc_value, roc_saved = None, False
    if pos_label and all(s is not None for s in y_score):
        y_true_bin = [1 if t == pos_label else 0 for t in y_true]
        auc_value = roc_auc_score(y_true_bin, y_score)
        print(f"ROC-AUC (positive = '{pos_label}'): {auc_value:.4f}")
    else:
        print("ROC-AUC: n/a (need binary task and continuous scores)")

    # ---------- confusion matrix & files ----------
    cm = confusion_matrix(y_true, y_pred, labels=labels_set)
    base = os.path.splitext(os.path.basename(results_path))[0]
    out_dir = os.path.dirname(results_path)

    cm_png  = os.path.join(out_dir, f"{base}_cm.png")
    cm_json = os.path.join(out_dir, f"{base}_cm.json")
    roc_png = os.path.join(out_dir, f"{base}_roc.png")
    metrics_json = os.path.join(out_dir, f"{base}_metrics.json")

    save_confusion_matrix(cm, labels_set, cm_png)
    with open(cm_json, "w") as f:
        json.dump({"labels": labels_set, "matrix": cm.tolist()}, f, indent=2)

    if auc_value is not None:
        fpr, tpr, _ = roc_curve([1 if t == pos_label else 0 for t in y_true], y_score)
        save_roc_curve(fpr, tpr, auc_value, roc_png, pos_label)
        roc_saved = True

    # ---------- scalar metrics ----------
    with open(metrics_json, "w") as f:
        json.dump(
            {
                "accuracy": acc,
                "mcc":       mcc,
                "auc":       auc_value,
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )

    # ---------- summary ----------
    print("\nResults saved to:")
    print(f"  • {cm_png}")
    print(f"  • {cm_json}")
    if roc_saved:
        print(f"  • {roc_png}")
    print(f"  • {metrics_json}")


# ----------------------------------------------------------------------
# CLI wrapper
# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate classification results")
    parser.add_argument("--results", required=True, help="Path to the JSON results file")
    parser.add_argument("--labels",  required=True, help="Path to the ground truth labels JSON file")
    args = parser.parse_args()

    evaluate(args.results, args.labels)
