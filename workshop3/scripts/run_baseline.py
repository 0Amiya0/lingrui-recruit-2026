"""Step 2：跑 baseline（ProtoNet + CE linear probe），在干净测试集上评估并存指标。"""
import csv
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 修复 anaconda(MKL) 与 torch 的 OpenMP 冲突

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import Config
from src.utils import seed_everything, get_device, save_json
from src.baseline import ProtoNet, CELinearProbe
from src.metrics import accuracy, macro_f1, expected_calibration_error
from src.dataset import inject_symmetric_noise


FIELDS = ["k_shot", "noise_rate", "seed", "method", "accuracy", "macro_f1", "ece", "noisy_test_acc"]


def load_data(cfg: Config):
    data_dir = Path(cfg.data_dir) / "processed"
    tag = f"support_cifar10_K{cfg.k_shot}_r{cfg.noise_rate}_seed{cfg.seed}"
    support = np.load(data_dir / f"{tag}.npz")
    test = np.load(data_dir / "cifar10_test_features.npz")
    return (
        support["support_feats"], support["support_noisy"], support["support_true"],
        support["noise_mask"],
        test["test_feats"], test["test_labels"],
    )


def evaluate(method_name, probs, test_labels, noisy_test_labels=None):
    pred = probs.argmax(1)
    row = {
        "method": method_name,
        "accuracy": accuracy(test_labels, pred),
        "macro_f1": macro_f1(test_labels, pred),
        "ece": expected_calibration_error(probs, test_labels),
    }
    if noisy_test_labels is not None:
        row["noisy_test_acc"] = accuracy(noisy_test_labels, pred)
    return row


def main(cfg: Config):
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    print(f"设备: {device} | K={cfg.k_shot} | 噪声率={cfg.noise_rate} | seed={cfg.seed}")

    support_feats, support_noisy, support_true, noise_mask, test_feats, test_labels = load_data(cfg)
    feat_dim = support_feats.shape[1]
    print(f"support {support_feats.shape} | test {test_feats.shape} | 噪声样本 {int(noise_mask.sum())} 个")

    rows = []

    # 含噪测试集：给干净测试标签注入同比例对称噪声（独立 seed）。
    # 注意：含噪参考下，即使完美分类器，一致性上限也≈1-noise_rate（错标签本身不可预测）。
    noisy_test_labels, _ = inject_symmetric_noise(
        test_labels, cfg.n_way, cfg.noise_rate, seed=cfg.seed + 10000)

    # Baseline A：ProtoNet（用噪声标签建原型）
    proto = ProtoNet(n_way=cfg.n_way).fit(support_feats, support_noisy)
    rows.append(evaluate("ProtoNet", proto.predict_proba(test_feats), test_labels, noisy_test_labels))

    # Baseline B：CE 线性分类头（强基线：label smoothing + weight decay）
    probe = CELinearProbe(feat_dim, cfg.n_way, device=str(device), seed=cfg.seed).fit(support_feats, support_noisy)
    rows.append(evaluate("CE-linear-probe", probe.predict_proba(test_feats), test_labels, noisy_test_labels))

    for r in rows:
        r.update({"k_shot": cfg.k_shot, "noise_rate": cfg.noise_rate, "seed": cfg.seed})

    # 落盘：JSON（完整）+ CSV（汇总追加）
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(rows, out_dir / f"baseline_K{cfg.k_shot}_r{cfg.noise_rate}_seed{cfg.seed}.json")

    csv_path = out_dir / "baseline_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in FIELDS})

    for r in rows:
        print(f"[{r['method']}] acc={r['accuracy']*100:.2f}%  macroF1={r['macro_f1']:.4f}  "
              f"ece={r['ece']:.4f}  noisyAcc={r['noisy_test_acc']*100:.2f}%")
    print(f"[完成] 结果写入 {csv_path}")


if __name__ == "__main__":
    main(Config.from_args())
