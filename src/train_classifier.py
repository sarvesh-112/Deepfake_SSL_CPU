import os
import torch
from torch.utils.data import DataLoader

from src.data.dataset import FramePairDataset
from src.models.backbone import MobileNetBackbone
from src.models.classifier import BinaryClassifier

# =====================
# CONFIG
# =====================
DEVICE = "cpu"
DATA_ROOT = "data/compressed/original/sdfvd_frames"
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
MODEL_DIR = "results/models"

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================
# DATA
# =====================
dataset = FramePairDataset(DATA_ROOT)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# =====================
# MODELS
# =====================
backbone = MobileNetBackbone().to(DEVICE)
classifier = BinaryClassifier(1280).to(DEVICE)

# 🔒 FREEZE BACKBONE
for p in backbone.parameters():
    p.requires_grad = False

# =====================
# TRAINING SETUP
# =====================
optimizer = torch.optim.Adam(classifier.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()

# =====================
# TRAIN LOOP
# =====================
classifier.train()
backbone.eval()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for img1, img2, labels in loader:
        img1 = img1.to(DEVICE)
        img2 = img2.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():
            f1 = backbone(img1)
            f2 = backbone(img2)
            features = torch.abs(f1 - f2)

        preds = classifier(features)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# =====================
# SAVE MODEL
# =====================
torch.save(
    {"classifier_state_dict": classifier.state_dict()},
    os.path.join(MODEL_DIR, "classifier.pth")
)

print("✅ Classifier training completed and saved.")
