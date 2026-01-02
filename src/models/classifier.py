import torch.nn as nn


class BinaryClassifier(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.fc(x)
