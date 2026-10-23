"""Frozen VGG19 feature losses. Predictions retain gradients through the encoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGFeatures(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights
        self.features = vgg19(weights=VGG19_Weights.DEFAULT if pretrained else None).features[:35].eval()
        self.features.requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(False)
        return self

    def forward(self, x):
        return self.features((x.float() - self.mean) / self.std)


class PerceptualLoss(nn.Module):
    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder if encoder is not None else VGGFeatures()

    def forward(self, generated, ground_truth):
        predicted = self.encoder(generated)
        with torch.no_grad():
            target = self.encoder(ground_truth)
        return F.l1_loss(predicted, target)
