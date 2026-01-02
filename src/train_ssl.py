import torch
from torch.utils.data import DataLoader
from src.data.dataset import VideoFrameDataset
from src.models.backbone import MobileNetBackbone
from src.models.temporal_module import TemporalDifference
from src.ssl.temporal_shuffle import temporal_shuffle
from src.ssl.losses import ssl_loss

DEVICE = "cpu"
DATA_ROOT = "data/compressed/jpeg_q60/sdfvd_frames"

dataset = VideoFrameDataset(DATA_ROOT)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

backbone = MobileNetBackbone().to(DEVICE)
temporal = TemporalDifference().to(DEVICE)

optimizer = torch.optim.Adam(
    list(backbone.parameters()) + list(temporal.parameters()),
    lr=1e-4
)

for epoch in range(5):
    for frames, _ in loader:
        frames, label = temporal_shuffle(frames)
        frames = frames.to(DEVICE)

        b, t, c, h, w = frames.shape
        feats = []

        for i in range(t):
            feats.append(backbone(frames[:, i]))

        feats = torch.stack(feats, dim=1)
        out = temporal(feats)

        loss = ssl_loss(out, label.repeat(out.size(0)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} SSL loss: {loss.item():.4f}")

print("SSL training completed.")
