import torch
import torch.nn as nn


class TemporalDifference(nn.Module):
    def forward(self, features):
        diffs = []
        for i in range(features.size(1) - 1):
            diffs.append(
                torch.abs(features[:, i] - features[:, i + 1])
            )
        return torch.stack(diffs, dim=1).mean(dim=1)
