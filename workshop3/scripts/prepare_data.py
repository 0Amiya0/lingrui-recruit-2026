"""Step 1：准备数据。

下载 CIFAR-10 → 划分 few-shot support（每类 k_shot）→ 注入对称标签噪声
→ 用冻结 backbone 提取特征 → 落盘特征/标签/噪声 mask → 自检。
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 修复 anaconda(MKL) 与 torch 的 OpenMP 冲突

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import Config
from src.utils import seed_everything, get_device, save_json, save_npz
from src.backbone import Backbone
from src.dataset import load_cifar10, sample_support, inject_symmetric_noise, extract_features


def main(cfg: Config):
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    print(f"设备: {device}")

    data_dir = Path(cfg.data_dir)
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # 1) 下载/加载 CIFAR-10
    train_set, test_set = load_cifar10(str(data_dir), image_size=cfg.image_size)
    train_targets = np.asarray(train_set.targets)
    print(f"训练集 {len(train_set)} 张，测试集 {len(test_set)} 张")

    # 2) 划分 few-shot support（每类 k_shot）
    support_idx = sample_support(train_targets, cfg.n_way, cfg.k_shot, cfg.seed)
    support_true = train_targets[support_idx]

    # 3) 注入标签噪声
    support_noisy, noise_mask = inject_symmetric_noise(support_true, cfg.n_way, cfg.noise_rate, cfg.seed)
    actual_rate = float(noise_mask.mean())
    print(f"support: {len(support_idx)} 张 | 请求噪声率 {cfg.noise_rate} | 实际噪声率 {actual_rate:.4f}")

    # 4) 提取特征
    backbone = Backbone(cfg.backbone, pretrained=cfg.pretrained, normalize=cfg.normalize)
    support_feats, _ = extract_features(backbone, train_set, support_idx, device, cfg.batch_size)

    # 测试特征只算一次（共享），之后复用缓存
    test_cache = processed_dir / "cifar10_test_features.npz"
    if test_cache.exists():
        data = np.load(test_cache)
        test_feats, test_labels = data["test_feats"], data["test_labels"]
        print("复用缓存的测试特征")
    else:
        print("首次提取测试特征（10k 张，CPU 上较慢）……")
        test_idx = np.arange(len(test_set))
        test_feats, test_labels = extract_features(backbone, test_set, test_idx, device, cfg.batch_size)
        save_npz(test_cache, test_feats=test_feats.astype(np.float32), test_labels=test_labels.astype(np.int64))

    # 5) 落盘 support 特征 + 标签 + 噪声 mask
    tag = f"cifar10_K{cfg.k_shot}_r{cfg.noise_rate}_seed{cfg.seed}"
    out_path = processed_dir / f"support_{tag}.npz"
    save_npz(out_path,
              support_feats=support_feats.astype(np.float32),
              support_true=support_true.astype(np.int64),
              support_noisy=support_noisy.astype(np.int64),
              noise_mask=noise_mask,
              support_indices=support_idx)

    meta = {
        "tag": tag,
        "config": cfg.to_dict(),
        "n_support": int(len(support_idx)),
        "requested_noise_rate": cfg.noise_rate,
        "actual_noise_rate": float(actual_rate),
        "n_noisy": int(noise_mask.sum()),
        "feature_dim": int(support_feats.shape[1]),
        "test_n": int(len(test_feats)),
    }
    save_json(meta, processed_dir / f"support_{tag}.json")

    # 6) 自检：用干净真实标签做最近原型分类，确认特征可用
    proto = np.stack([support_feats[support_true == c].mean(0) for c in range(cfg.n_way)])
    proto = proto / (np.linalg.norm(proto, axis=1, keepdims=True) + 1e-12)
    sim = test_feats @ proto.T
    pred = sim.argmax(1)
    acc = float((pred == test_labels).mean())
    print(f"[自检] 干净原型 -> 测试集 最近邻准确率: {acc * 100:.2f}%")
    print(f"[完成] 输出: {out_path}")

    return out_path


if __name__ == "__main__":
    main(Config.from_args())
