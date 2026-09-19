"""Step 3：跑核心方法（样本筛选 + 鲁棒损失），含三种变体：
selection-only / robust-loss-only / full（筛选 + 鲁棒损失）。"""
import csv
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 修复 anaconda(MKL) 与 torch 的 OpenMP 冲突

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import Config
from src.utils import seed_everything, get_device, save_json
from src.method import distance_based_selection, train_linear_head, head_proba
from src.metrics import (accuracy, macro_f1, expected_calibration_error,
                         noise_detection_metrics, noise_detection_auroc)
from src.dataset import inject_symmetric_noise


FIELDS = ["k_shot", "noise_rate", "seed", "method", "accuracy", "macro_f1", "ece",
          "noisy_test_acc", "noise_precision", "noise_recall", "noise_f1", "noise_auroc"]


def load_data(cfg: Config):
    data_dir = Path(cfg.data_dir) / "processed"
    tag = f"support_cifar10_K{cfg.k_shot}_r{cfg.noise_rate}_seed{cfg.seed}"
    support = np.load(data_dir / f"{tag}.npz")
    test = np.load(data_dir / "cifar10_test_features.npz")
    return (support["support_feats"], support["support_noisy"], support["support_true"],
            support["noise_mask"], test["test_feats"], test["test_labels"])


def main(cfg: Config):
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    print(f"设备: {device} | K={cfg.k_shot} | 噪声率={cfg.noise_rate} | loss={cfg.loss} | "
          f"sel_beta={cfg.sel_beta} | sel_iters={cfg.sel_iters}")

    support_feats, support_noisy, support_true, noise_mask, test_feats, test_labels = load_data(cfg)
    feat_dim = support_feats.shape[1]
    print(f"support {support_feats.shape} | test {test_feats.shape} | 噪声样本 {int(noise_mask.sum())} 个")

    # 样本筛选（selection-only 与 full 共用）；dist 为噪声似然分数（用于 AUROC）
    pred_noise, keep, dist = distance_based_selection(
        support_feats, support_noisy, cfg.n_way, n_iters=cfg.sel_iters, beta=cfg.sel_beta)
    noise_metrics = noise_detection_metrics(pred_noise, noise_mask)
    noise_auroc = noise_detection_auroc(dist, noise_mask)
    print(f"[筛选] 判为噪声 {int(pred_noise.sum())} 个 | "
          f"precision={noise_metrics['precision']:.3f} recall={noise_metrics['recall']:.3f} "
          f"f1={noise_metrics['f1']:.3f} auroc={noise_auroc:.3f}")

    # 三种变体：(名称, 是否用筛选后的子集, 损失)
    variants = [
        ("selection-only", keep, "ce"),
        ("robust-loss-only", None, cfg.loss),
        ("full", keep, cfg.loss),
    ]

    # 含噪测试集（独立 seed 注入）
    noisy_test_labels, _ = inject_symmetric_noise(
        test_labels, cfg.n_way, cfg.noise_rate, seed=cfg.seed + 10000)

    rows = []
    for name, sel_mask, loss_name in variants:
        X = support_feats if sel_mask is None else support_feats[sel_mask]
        y = support_noisy if sel_mask is None else support_noisy[sel_mask]
        head = train_linear_head(X, y, feat_dim, cfg.n_way, loss_name=loss_name,
                                 loss_kw={"q": cfg.gce_q}, device=str(device), seed=cfg.seed)
        probs = head_proba(head, test_feats, device)
        pred = probs.argmax(1)
        row = {
            "k_shot": cfg.k_shot, "noise_rate": cfg.noise_rate, "seed": cfg.seed,
            "method": name,
            "accuracy": accuracy(test_labels, pred),
            "macro_f1": macro_f1(test_labels, pred),
            "ece": expected_calibration_error(probs, test_labels),
            "noisy_test_acc": accuracy(noisy_test_labels, pred),
        }
        if sel_mask is not None:
            row.update({
                "noise_precision": noise_metrics["precision"],
                "noise_recall": noise_metrics["recall"],
                "noise_f1": noise_metrics["f1"],
                "noise_auroc": noise_auroc,
            })
        rows.append(row)

    # 落盘
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(rows, out_dir / f"method_K{cfg.k_shot}_r{cfg.noise_rate}_seed{cfg.seed}.json")

    csv_path = out_dir / "method_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in FIELDS})

    for r in rows:
        extra = ""
        if r.get("noise_f1") is not None:
            extra = f"  noiseF1={r['noise_f1']:.3f}"
        print(f"[{r['method']}] acc={r['accuracy']*100:.2f}%  macroF1={r['macro_f1']:.4f}  ece={r['ece']:.4f}{extra}")
    print(f"[完成] 结果写入 {csv_path}")


if __name__ == "__main__":
    main(Config.from_args())
