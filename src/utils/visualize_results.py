import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)


def plot_roc(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig("results/figures/roc_curve.png", dpi=300)
    plt.close()

    return roc_auc


def plot_pr(y_true, y_scores, title):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()

    plt.savefig("results/figures/pr_curve.png", dpi=300)
    plt.close()

    return ap


def plot_compression_robustness(results_dict):
    qualities = list(results_dict.keys())
    accuracies = list(results_dict.values())

    plt.figure(figsize=(6, 5))
    plt.plot(qualities, accuracies, marker="o", linewidth=2)
    plt.xlabel("Compression Level")
    plt.ylabel("Accuracy")
    plt.title("Compression Robustness")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("results/figures/compression_robustness.png", dpi=300)
    plt.close()
