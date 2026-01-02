import torch.nn as nn
from torchvision.models import mobilenet_v2


class MobileNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v2(pretrained=True)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.flatten(1)
