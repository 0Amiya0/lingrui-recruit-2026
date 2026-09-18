"""Step 6：错误分析——混淆矩阵、噪声识别失败、t-SNE、原型漂移、距离分布。

分析两类失败：模型误判（干净测试样本为何分错）与噪声识别失败（噪声样本为何没被筛掉）。
用法：python scripts/error_analysis.py --k-shot 10 --noise-rate 0.4 --seed 0
"""
import json
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 修复 OpenMP 冲突

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from src.config import Config
from src.utils import seed_everything, get_device, save_json
from src.method import distance_based_selection, train_linear_head, head_proba

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def load_data(cfg: Config):
    data_dir = Path(cfg.data_dir) / "processed"
    tag = f"support_cifar10_K{cfg.k_shot}_r{cfg.noise_rate}_seed{cfg.seed}"
    support = np.load(data_dir / f"{tag}.npz")
    test = np.load(data_dir / "cifar10_test_features.npz")
    return (support["support_feats"], support["support_noisy"], support["support_true"],
            support["noise_mask"], test["test_feats"], test["test_labels"])


def main():
    cfg = Config.from_args()
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    support_feats, support_noisy, support_true, noise_mask, test_feats, test_labels = load_data(cfg)
    feat_dim = support_feats.shape[1]
    out_dir = Path(cfg.out_dir)
    figs_dir = out_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # ---- full 方法 ----
    pred_noise, keep = distance_based_selection(
        support_feats, support_noisy, cfg.n_way, n_iters=cfg.sel_iters, beta=cfg.sel_beta)
    head = train_linear_head(support_feats[keep], support_noisy[keep], feat_dim, cfg.n_way,
                             loss_name=cfg.loss, loss_kw={"q": cfg.gce_q}, device=str(device), seed=cfg.seed)
    probs = head_proba(head, test_feats, device)
    pred = probs.argmax(1)

    report = {"config": {"k_shot": cfg.k_shot, "noise_rate": cfg.noise_rate, "seed": cfg.seed}}

    # ---- 1) 混淆矩阵 + 误判分析 ----
    cm = confusion_matrix(test_labels, pred, labels=list(range(cfg.n_way)))
    report["confusion_matrix"] = cm.tolist()
    # 最易混淆的类对（取非对角最大 5 个）
    offdiag = []
    for i in range(cfg.n_way):
        for j in range(cfg.n_way):
            if i != j:
                offdiag.append((cm[i, j], i, j))
    offdiag.sort(reverse=True)
    report["top_confused_pairs"] = [
        {"from": CIFAR10_CLASSES[i], "to": CIFAR10_CLASSES[j], "count": int(n)}
        for n, i, j in offdiag[:5]
    ]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(cfg.n_way), CIFAR10_CLASSES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(cfg.n_way), CIFAR10_CLASSES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix (full method, {cfg.noise_rate} noise)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figs_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ---- 2) 噪声识别失败分析 ----
    fn_mask = (~pred_noise) & noise_mask      # 漏筛的噪声样本
    fp_mask = pred_noise & (~noise_mask)      # 误伤的干净样本
    report["noise_detection"] = {
        "n_noisy": int(noise_mask.sum()),
        "n_flagged": int(pred_noise.sum()),
        "missed_noisy": int(fn_mask.sum()),
        "false_flagged_clean": int(fp_mask.sum()),
        "missed_noisy_indices": np.where(fn_mask)[0].tolist(),
        "false_flagged_indices": np.where(fp_mask)[0].tolist(),
    }

    # ---- 3) 原型漂移：干净/噪声/鲁棒 原型与干净原型的余弦相似度 ----
    def proto(feats, labels):
        ps = np.stack([feats[labels == c].mean(0) for c in range(cfg.n_way)])
        return ps / (np.linalg.norm(ps, axis=1, keepdims=True) + 1e-12)

    clean_p = proto(support_feats, support_true)
    noisy_p = proto(support_feats, support_noisy)
    robust_p = proto(support_feats[keep], support_noisy[keep])
    sim_noisy = (clean_p * noisy_p).sum(1)   # 每类 干净 vs 噪声 原型相似度
    sim_robust = (clean_p * robust_p).sum(1)
    report["prototype_drift"] = {
        "cos_sim_clean_vs_noisy": sim_noisy.tolist(),
        "cos_sim_clean_vs_robust": sim_robust.tolist(),
        "mean_clean_vs_noisy": float(sim_noisy.mean()),
        "mean_clean_vs_robust": float(sim_robust.mean()),
    }

    x = np.arange(cfg.n_way)
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, sim_noisy, w, label="clean vs noisy prototype", color="#ff7f0e")
    ax.bar(x + w / 2, sim_robust, w, label="clean vs robust prototype", color="#2ca02c")
    ax.set_xticks(x, CIFAR10_CLASSES, fontsize=8)
    ax.set_ylabel("cosine similarity to clean prototype")
    ax.set_title("Prototype drift by class (higher = less drift)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figs_dir / "prototype_drift.png", dpi=150)
    plt.close(fig)

    # ---- 4) 距离分布：噪声 vs 干净 support 样本到其干净类原型的距离 ----
    d_all = 1.0 - (support_feats * clean_p[support_true]).sum(1)
    d_noisy = d_all[noise_mask]
    d_clean = d_all[~noise_mask]
    report["distance_distribution"] = {
        "clean_mean": float(d_clean.mean()), "clean_std": float(d_clean.std()),
        "noisy_mean": float(d_noisy.mean()), "noisy_std": float(d_noisy.std()),
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d_clean, bins=30, alpha=0.6, label="clean support", color="#2ca02c", density=True)
    ax.hist(d_noisy, bins=30, alpha=0.6, label="noisy support", color="#d62728", density=True)
    ax.set_xlabel("cosine distance to clean prototype")
    ax.set_ylabel("density")
    ax.set_title("Distance distribution: clean vs noisy support samples")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs_dir / "distance_distribution.png", dpi=150)
    plt.close(fig)

    # ---- 5) t-SNE（support 全部 + test 子集），标出噪声/筛除样本 ----
    rng = np.random.RandomState(cfg.seed)
    test_sub = rng.choice(len(test_feats), 300, replace=False)
    all_feats = np.concatenate([support_feats, test_feats[test_sub]], axis=0)
    emb = TSNE(n_components=2, perplexity=30, random_state=cfg.seed, init="pca").fit_transform(all_feats)
    sup_emb, test_emb = emb[:len(support_feats)], emb[len(support_feats):]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(test_emb[:, 0], test_emb[:, 1], c=test_labels[test_sub], cmap="tab10",
               alpha=0.25, s=8, label="test (clean)")
    # 干净 support
    ax.scatter(sup_emb[~noise_mask, 0], sup_emb[~noise_mask, 1], c=support_true[~noise_mask],
               cmap="tab10", edgecolors="black", s=45, marker="o")
    # 噪声 support（漏筛的用红叉，被筛掉用黑叉）
    ax.scatter(sup_emb[fn_mask, 0], sup_emb[fn_mask, 1], marker="x", color="red", s=90,
               label="missed noisy")
    ax.scatter(sup_emb[fp_mask, 0], sup_emb[fp_mask, 1], marker="+", color="black", s=90,
               label="false-flagged clean")
    ax.set_title("t-SNE of features (support + test subset)")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(figs_dir / "tsne.png", dpi=150)
    plt.close(fig)

    save_json(report, out_dir / "error_analysis.json")
    print(f"[完成] 错误分析报告 -> {out_dir / 'error_analysis.json'}")
    print(f"  最易混淆类对: {report['top_confused_pairs'][:3]}")
    print(f"  噪声识别: 漏筛 {int(fn_mask.sum())}/{int(noise_mask.sum())}，误伤 {int(fp_mask.sum())}")
    print(f"  原型漂移: 噪声均值 {sim_noisy.mean():.4f} -> 鲁棒均值 {sim_robust.mean():.4f}")


if __name__ == "__main__":
    main()
