"""Compact multi-scale/channel-attention network inspired by restoration literature."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Global pooling gates channels; inspired by channel-attention restoration modules (MPRNet)."""
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, max(4, channels // 8), 1),
                                  nn.ReLU(inplace=True), nn.Conv2d(max(4, channels // 8), channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x)


class DilatedBlock(nn.Module):
    """Dilated convolution increases context without downsampling; residual path eases optimization."""
    def __init__(self, channels, dilation):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation),
                                  nn.ReLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1),
                                  ChannelAttention(channels))

    def forward(self, x):
        return x + .1 * self.body(x)


class MyModel(nn.Module):
    """Two shared-weight refinement stages; multi-scale context inspired by task DeRainNet.

    Iterative refinement draws inspiration from PReNet, but this model has no recurrent LSTM
    and is not its reproduction. Fixed 32 channels keep parameter count well below 4 million.
    """
    def __init__(self, channels=32, stages=2):
        super().__init__()
        self.stages = stages
        self.stem = nn.Conv2d(6, channels, 3, padding=1)
        self.full = nn.Sequential(DilatedBlock(channels, 1), DilatedBlock(channels, 2))
        # Do not shadow nn.Module.half(), the dtype-conversion method.
        self.half_scale = nn.Sequential(DilatedBlock(channels, 1), DilatedBlock(channels, 3))
        self.head = nn.Sequential(nn.Conv2d(channels * 2, channels, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(channels, 3, 3, padding=1))

    def forward(self, x):
        estimate = x
        for _ in range(self.stages):
            feature = self.stem(torch.cat([x, estimate], dim=1))
            low = self.half_scale(F.avg_pool2d(feature, 2))
            low = F.interpolate(low, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            estimate = x - self.head(torch.cat([self.full(feature), low], dim=1))
        return estimate
