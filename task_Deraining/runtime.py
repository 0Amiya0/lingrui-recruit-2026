"""Shared model loading/device handling for training, inference and Flask."""
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from models import build_model


def select_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(name)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('only CPU/CUDA supported')
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable: install a CUDA-enabled PyTorch build')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError('CUDA device index out of range')
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(path, device='cpu'):
    # Our checkpoints use tensors/primitives only; no unrestricted pickle execution.
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    name = checkpoint['model']
    model = build_model(name)
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint


def restore_image(model, x):
    """Preserve full resolution; pad for upstream U-Net's 3 pooling operations."""
    h, w = x.shape[-2:]
    target_h, target_w = max(32, (h + 7) // 8 * 8), max(32, (w + 7) // 8 * 8)
    x = F.pad(x, (0, target_w - w, 0, target_h - h), mode='replicate')
    return model(x)[..., :h, :w]
