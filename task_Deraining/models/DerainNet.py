"""Task-defined three-scale network; not a claim to reproduce a named paper."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .baseline_net import ResidualBlock


class DerainNet(nn.Module):
    """Independent full/half/quarter branches, each with an eight-convolution block."""
    def __init__(self, channels=64):
        super().__init__()
        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.branches = nn.ModuleList([ResidualBlock(channels, 8) for _ in range(3)])
        self.fuse = nn.Sequential(nn.Conv2d(channels * 3, channels, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(channels, 3, 3, padding=1))

    def forward(self, x):
        feature = self.stem(x)
        outputs = []
        for scale, branch in zip((1, 2, 4), self.branches):
            pooled = feature if scale == 1 else F.max_pool2d(feature, scale)
            output = branch(pooled)
            outputs.append(F.interpolate(output, size=feature.shape[-2:], mode='bilinear', align_corners=False))
        return x - self.fuse(torch.cat(outputs, dim=1))
