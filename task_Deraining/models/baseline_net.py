"""Residual rain prediction baseline; an eight-convolution, 64-channel residual block."""
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Even-depth 64-channel residual branch, retaining spatial resolution."""
    def __init__(self, channels=64, depth=8):
        super().__init__()
        layers = []
        for i in range(depth):
            layers.append(nn.Conv2d(channels, channels, 3, padding=1))
            if i < depth - 1:
                layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.body(x)


class BaselineNet(nn.Module):
    """1 input + 8 hidden + 1 output convolutions; output = rainy - rain."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.body = ResidualBlock(64, depth=8)
        self.head = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        return x - self.head(self.body(self.stem(x)))
