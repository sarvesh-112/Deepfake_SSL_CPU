import os
import torch
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from src.data.dataset import FramePairDataset
from src.models.backbone import MobileNetBackbone
from src.models.classifier import BinaryClassifier
from src.utils.save_results import (
    ensure_dirs,
    save_metrics,
    save_log,
    save_confusion_matrix
)
from src.utils.visualize_results import (
    plot_roc,
    plot_pr,
    plot_compression_robustness
)

# =========================
# CONFIG
# =========================
DEVICE = "cpu"
BATCH_SIZE = 16

DATASETS = {
    "Original": "data/compressed/original/sdfvd_frames",
    "JPEG Q60": "data/compressed/jpeg_q60/sdfvd_frames",
    "JPEG Q30": "data/compressed/jpeg_q30/sdfvd_frames"
}

MODEL_PATH = "results/models/classifier.pth"

# =========================
# SETUP
# =========================
ensure_dirs()

# Load models
backbone = MobileNetBackbone().to(DEVICE)
classifier = BinaryClassifier(1280).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classifier.load_state_dict(checkpoint["classifier_state_dict"])

backbone.eval()
classifier.eval()

compression_results = {}

# =========================
# EVALUATION LOOP
# =========================
for dataset_name, data_root in DATASETS.items():
    print(f"\nEvaluating on: {dataset_name}")

    dataset = FramePairDataset(data_root)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for img1, img2, labels in loader:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)

            f1 = backbone(img1)
            f2 = backbone(img2)
            features = torch.abs(f1 - f2)

            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            y_scores.extend(probs.tolist())

    accuracy = accuracy_score(y_true, y_pred)
    compression_results[dataset_name] = round(float(accuracy), 4)

    print(f"{dataset_name} Accuracy: {accuracy:.4f}")

    # =========================
    # SAVE DETAILED RESULTS ONLY FOR ORIGINAL
    # =========================
    if dataset_name == "Original":
        roc_auc = plot_roc(
            y_true,
            y_scores,
            title="ROC Curve (SDFVD)"
        )

        pr_ap = plot_pr(
            y_true,
            y_scores,
            title="Precision-Recall Curve (SDFVD)"
        )

        metrics = save_metrics(accuracy, "SDFVD (Original)")
        metrics["roc_auc"] = round(float(roc_auc), 4)
        metrics["pr_ap"] = round(float(pr_ap), 4)

        save_log(metrics)
        save_confusion_matrix(
            y_true,
            y_pred,
            "SDFVD (Original)"
        )

# =========================
# COMPRESSION ROBUSTNESS PLOT
# =========================
plot_compression_robustness(compression_results)

# Save compression robustness table
with open("results/metrics/compression_results.txt", "w") as f:
    for k, v in compression_results.items():
        f.write(f"{k}: {v}\n")

print("\n✅ Evaluation completed successfully.")
print("📊 Metrics saved to results/metrics/")
print("🖼️ Figures saved to results/figures/")
print("🧾 Logs saved to results/logs/")
