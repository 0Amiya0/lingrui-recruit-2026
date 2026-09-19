"""Step 4：消融实验——超参扫描（sel_beta / gce_q / sel_iters / top_tau）。

固定在某噪声率（默认 40%）下，逐一扫描关键超参，输出 accuracy / macro-f1 / 噪声识别指标。
用法：python scripts/run_ablation.py --k-shot 10 --noise-rate 0.4 --seed 0
"""
import csv
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 修复 OpenMP 冲突

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import Config
from src.utils import seed_everything, get_device
from src.method import distance_based_selection, train_linear_head, head_proba
from src.metrics import accuracy, macro_f1, noise_detection_metrics

FIELDS = ["config", "sel_beta", "gce_q", "sel_iters", "top_tau",
          "accuracy", "macro_f1", "noise_precision", "noise_recall", "noise_f1"]


def load_data(cfg: Config):
    data_dir = Path(cfg.data_dir) / "processed"
    tag = f"support_cifar10_K{cfg.k_shot}_r{cfg.noise_rate}_seed{cfg.seed}"
    support = np.load(data_dir / f"{tag}.npz")
    test = np.load(data_dir / "cifar10_test_features.npz")
    return (support["support_feats"], support["support_noisy"], support["support_true"],
            support["noise_mask"], test["test_feats"], test["test_labels"])


def run_one(cfg, support_feats, support_noisy, noise_mask, test_feats, test_labels,
            device, sel_beta, gce_q, sel_iters, top_tau, label):
    feat_dim = support_feats.shape[1]
    pred_noise, keep, _ = distance_based_selection(
        support_feats, support_noisy, cfg.n_way,
        n_iters=sel_iters, beta=sel_beta, top_tau=top_tau)
    nd = noise_detection_metrics(pred_noise, noise_mask)
    head = train_linear_head(support_feats[keep], support_noisy[keep], feat_dim, cfg.n_way,
                             loss_name="gce", loss_kw={"q": gce_q}, device=str(device), seed=cfg.seed)
    probs = head_proba(head, test_feats, device)
    pred = probs.argmax(1)
    return {
        "config": label, "sel_beta": sel_beta, "gce_q": gce_q, "sel_iters": sel_iters,
        "top_tau": ("" if top_tau is None else top_tau),
        "accuracy": accuracy(test_labels, pred),
        "macro_f1": macro_f1(test_labels, pred),
        "noise_precision": nd["precision"], "noise_recall": nd["recall"], "noise_f1": nd["f1"],
    }


def main():
    cfg = Config.from_args()
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    support_feats, support_noisy, support_true, noise_mask, test_feats, test_labels = load_data(cfg)
    print(f"消融实验 | K={cfg.k_shot} | 噪声率={cfg.noise_rate} | 设备={device}")

    sweeps = []
    # 1) sel_beta 扫描（固定 gce_q=0.7, sel_iters=5）
    sweeps += [dict(sel_beta=b, gce_q=0.7, sel_iters=5, top_tau=None, label=f"beta={b}")
               for b in (0.5, 1.0, 1.5, 2.0)]
    # 2) gce_q 扫描（固定 sel_beta=1.0, sel_iters=5）
    sweeps += [dict(sel_beta=1.0, gce_q=q, sel_iters=5, top_tau=None, label=f"q={q}")
               for q in (0.3, 0.5, 0.7, 0.9)]
    # 3) sel_iters 扫描
    sweeps += [dict(sel_beta=1.0, gce_q=0.7, sel_iters=it, top_tau=None, label=f"iters={it}")
               for it in (1, 3, 5, 10)]
    # 4) top_tau 变体（固定比例保留，替代 beta 阈值）
    sweeps += [dict(sel_beta=1.0, gce_q=0.7, sel_iters=5, top_tau=tau, label=f"tau={tau}")
               for tau in (0.5, 0.6, 0.7, 0.8)]

    rows = []
    for kw in sweeps:
        label = kw.pop("label")
        r = run_one(cfg, support_feats, support_noisy, noise_mask, test_feats, test_labels,
                    device, label=label, **kw)
        rows.append(r)
        print(f"[{label}] acc={r['accuracy']*100:.2f}%  noiseF1={r['noise_f1']:.3f}")

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "ablation_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[完成] 消融结果写入 {csv_path}")


if __name__ == "__main__":
    main()
