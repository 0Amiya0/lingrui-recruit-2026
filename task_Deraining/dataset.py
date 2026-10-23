"""Paired Rain200L loading, train-only validation and synchronized augmentation."""
from pathlib import Path
import random
import hashlib
import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp'}


def list_pairs(root, folder):
    base = Path(root) / folder
    inputs = sorted(p for p in (base / 'input').iterdir() if p.suffix.lower() in EXTENSIONS)
    pairs = [(p, base / 'target' / p.name) for p in inputs]
    if not pairs:
        raise ValueError(f'no images in {base / "input"}')
    missing = [str(b) for _, b in pairs if not b.is_file()]
    if missing:
        raise FileNotFoundError(f'paired target names must match: {missing[:5]}')
    return pairs


def split_pairs(root, seed=42, val_fraction=.1):
    """Group variants such as 1_1.png, 1_2.png by background prefix before underscore."""
    pairs = list_pairs(root, 'train')
    if not 0 < val_fraction < 1:
        raise ValueError('validation fraction must be in (0,1)')
    groups = sorted({a.stem.split('_')[0] for a, _ in pairs})
    if len(groups) < 2:
        raise ValueError('need at least two background groups for a train/val split')
    random.Random(seed).shuffle(groups)
    n_val = min(len(groups)-1, max(1, round(len(groups)*val_fraction)))
    validation = set(groups[:n_val])
    train = [(a,b) for a,b in pairs if a.stem.split('_')[0] not in validation]
    val = [(a,b) for a,b in pairs if a.stem.split('_')[0] in validation]
    return train, val


def split_manifest(train, val):
    result = {'train': [a.name for a,_ in train], 'val': [a.name for a,_ in val]}
    result['sha256'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


class DerainDataset(Dataset):
    def __init__(self, root_dir=None, is_train=True, image_size=128, pairs=None):
        self.pairs = pairs if pairs is not None else list_pairs(root_dir, 'train' if is_train else 'test')
        self.training = is_train
        self.patch_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        a, b = self.pairs[index]
        with Image.open(a) as image:
            rainy = image.convert('RGB')
        with Image.open(b) as image:
            clean = image.convert('RGB')
        if rainy.size != clean.size:
            raise ValueError(f'pair dimensions differ: {a.name}')
        if self.training:
            size = self.patch_size
            w, h = rainy.size
            if min(w, h) < size:
                padding = [0, 0, max(0, size-w), max(0, size-h)]
                rainy, clean = TF.pad(rainy, padding, padding_mode='edge'), TF.pad(clean, padding, padding_mode='edge')
            w, h = rainy.size
            left, top = random.randint(0, w-size), random.randint(0, h-size)
            rainy, clean = TF.crop(rainy, top, left, size, size), TF.crop(clean, top, left, size, size)
            if random.random() < .5:
                rainy, clean = TF.hflip(rainy), TF.hflip(clean)
            if random.random() < .5:
                rainy, clean = TF.vflip(rainy), TF.vflip(clean)
        return TF.to_tensor(rainy), TF.to_tensor(clean), a.name
