"""Per-image metrics (RGB, [0,1], full resolution), checkpoint and output helpers."""
from pathlib import Path
import math
import numpy as np
import torch
from skimage.metrics import structural_similarity
from torchvision.utils import save_image


def psnr(prediction, target, data_range=1.0):
    if prediction.shape != target.shape or data_range <= 0:
        raise ValueError('PSNR shape/range mismatch')
    difference = prediction.double() - target.double()
    mse = difference.square().mean().item()
    return float('inf') if mse == 0 else 10 * math.log10(data_range ** 2 / mse)


def calculate_metrics(img1, img2, lpips_fn=None):
    prediction, target = img1.detach().float().clamp(0,1), img2.detach().float().clamp(0,1)
    p = prediction.cpu().numpy().transpose(0,2,3,1)
    t = target.cpu().numpy().transpose(0,2,3,1)
    ssim_values = []
    for a,b in zip(p,t):
        window = min(7, a.shape[0], a.shape[1])
        window -= (window + 1) % 2
        if window < 3:
            raise ValueError('SSIM needs spatial dimensions >= 3')
        ssim_values.append(structural_similarity(a,b,data_range=1.,channel_axis=2,win_size=window))
    p_value = np.mean([psnr(a,b) for a,b in zip(prediction,target)])
    lp_value = None
    if lpips_fn is not None:
        with torch.no_grad():
            lp_value = float(lpips_fn(prediction*2-1, target*2-1).mean().item())
    return float(p_value), float(np.mean(ssim_values)), lp_value


def save_checkpoint(state, folder='checkpoints', filename='last.pt'):
    path = Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / (filename + '.tmp')
    torch.save(state, temporary)
    temporary.replace(path / filename)


def save_comparison(rainy, restored, clean, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat([rainy, restored.clamp(0,1), clean], dim=0), path, nrow=1)
