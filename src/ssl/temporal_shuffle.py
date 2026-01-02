import random
import torch


def temporal_shuffle(frames):
    if random.random() < 0.5:
        return frames, torch.tensor(0)
    idx = torch.randperm(frames.size(1))
    return frames[:, idx], torch.tensor(1)
