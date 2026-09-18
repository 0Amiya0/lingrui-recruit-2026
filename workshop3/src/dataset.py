"""CIFAR-10 加载、few-shot support 划分、标签噪声注入、特征提取。"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 冻结的 ImageNet 预训练 backbone 必须用 ImageNet 归一化统计量
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transform(image_size: int = 224):
    # 冻结 backbone 只做确定性预处理，不做数据增强
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_cifar10(root: str, image_size: int = 224):
    transform = get_transform(image_size)
    train_set = datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
    return train_set, test_set


def sample_support(train_targets, n_way: int, k_shot: int, seed: int):
    """每类随机抽 k_shot 张，返回 support 样本在训练集中的下标。"""
    rng = np.random.RandomState(seed)
    train_targets = np.asarray(train_targets)
    indices = []
    for c in range(n_way):
        cls_idx = np.where(train_targets == c)[0]
        chosen = rng.permutation(cls_idx)[:k_shot]
        indices.append(chosen)
    return np.concatenate(indices).astype(np.int64)


def inject_symmetric_noise(labels, n_way: int, noise_rate: float, seed: int):
    """对称噪声：把 noise_rate 比例的样本标签翻成其它随机类别（等概率）。"""
    labels = np.asarray(labels).copy()
    rng = np.random.RandomState(seed)
    n = len(labels)
    n_noise = int(round(n * noise_rate))
    flip_idx = rng.permutation(n)[:n_noise]
    mask = np.zeros(n, dtype=bool)
    mask[flip_idx] = True
    for i in flip_idx:
        new_label = rng.randint(0, n_way - 1)
        if new_label >= labels[i]:
            new_label += 1
        labels[i] = new_label
    return labels, mask


@torch.no_grad()
def extract_features(backbone, dataset, indices, device, batch_size: int):
    """对给定样本下标提取特征，返回 (feats[N, dim], labels[N])。"""
    backbone = backbone.to(device)
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0)
    feats, labels = [], []
    for x, y in loader:
        f = backbone(x.to(device))
        feats.append(f.cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labels, axis=0)
