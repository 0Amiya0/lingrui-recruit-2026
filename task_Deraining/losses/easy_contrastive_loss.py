"""Task's cosine objective plus optional two-candidate InfoNCE."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .perceptual_loss import VGGFeatures


class EasyContrastiveLoss(nn.Module):
    """Anchor=prediction, positive=clean, negative=rainy. Shared background can be penalized too."""
    def __init__(self, encoder=None, mode='cosine', temperature=.1):
        super().__init__()
        if mode not in ('cosine', 'infonce') or temperature <= 0:
            raise ValueError('invalid contrastive mode/temperature')
        self.encoder = encoder if encoder is not None else VGGFeatures()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mode, self.temperature = mode, temperature

    def encode(self, image):
        return F.normalize(self.pool(self.encoder(image)).flatten(1), dim=1, eps=1e-8)

    def forward(self, rainy, generated, clean):
        anchor = self.encode(generated)
        with torch.no_grad():
            positive, negative = self.encode(clean), self.encode(rainy)
        positive_sim = (anchor * positive).sum(1)
        negative_sim = (anchor * negative).sum(1)
        if self.mode == 'cosine':
            return (1 - positive_sim + negative_sim).mean()
        logits = torch.stack([positive_sim, negative_sim], dim=1) / self.temperature
        return F.cross_entropy(logits, torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device))
