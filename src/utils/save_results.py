import os
import json
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def ensure_dirs():
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)


def save_metrics(accuracy, dataset_name):
    metrics = {
        "dataset": dataset_name,
        "accuracy": round(float(accuracy), 4),
        "timestamp": datetime.datetime.now().isoformat()
    }

    with open("results/metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    with open("results/metrics/metrics.txt", "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")

    return metrics


def save_log(metrics):
    with open("results/logs/run.log", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")


def save_confusion_matrix(y_true, y_pred, dataset_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Real", "Fake"],
        yticklabels=["Real", "Fake"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(f"Confusion Matrix – {dataset_name}")

    plt.tight_layout()
    plt.savefig("results/figures/confusion_matrix.png")
    plt.close()
