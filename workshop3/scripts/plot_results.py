"""Step 4：画主实验结果图（Acc / Macro-F1 / ECE vs 噪声率）+ 噪声识别曲线。

优先读 results/summary_meanstd.csv（多种子 mean±std，由 aggregate_seeds.py 产出），
画均值并带 ±std 误差棒；若无则回退到单种子原始 CSV。
用法：python scripts/plot_results.py
"""
import csv
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 修复 OpenMP 冲突

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 方法显示顺序与配色（colorblind-friendly）
METHODS = ["ProtoNet", "CE-linear-probe", "selection-only", "robust-loss-only", "full"]
STYLE = {
    "ProtoNet": dict(color="#1f77b4", marker="o"),
    "CE-linear-probe": dict(color="#ff7f0e", marker="s"),
    "selection-only": dict(color="#2ca02c", marker="^"),
    "robust-loss-only": dict(color="#9467bd", marker="v"),
    "full": dict(color="#d62728", marker="D"),
}


def read_csv(path):
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data = {}
    for r in rows:
        key = (r["method"], float(r["noise_rate"]))
        data[key] = r
    return data


def series(data, method, field):
    """按噪声率升序返回 (rates, mean_vals, std_vals)，缺失则跳过。"""
    rates, vals, errs = [], [], []
    for rate in sorted({k[1] for k in data if k[0] == method}):
        r = data.get((method, rate))
        if r is None or r.get(field) in ("", None):
            continue
        rates.append(rate)
        vals.append(float(r[field]))
        std = r.get(field + "_std")
        errs.append(float(std) if std not in ("", None) else 0.0)
    return rates, vals, errs


def plot_metric(ax, data, field, ylabel, title):
    for m in METHODS:
        rates, vals, errs = series(data, m, field)
        if vals:
            ax.errorbar(rates, vals, yerr=errs, label=m, capsize=3, **STYLE[m])
    ax.set_xlabel("label noise rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main():
    # 优先用多种子 mean±std；缺失则回退单种子原始 CSV
    data = read_csv("results/summary_meanstd.csv")
    if not data:
        data = {**read_csv("results/baseline_summary.csv"),
                **read_csv("results/method_summary.csv")}
    if not data:
        print("[警告] 未找到结果 CSV，请先跑 run_all.ps1 生成。")
        return

    out_dir = Path("results") / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_metric(axes[0, 0], data, "accuracy", "accuracy", "Accuracy vs. label noise")
    plot_metric(axes[0, 1], data, "macro_f1", "macro-F1", "Macro-F1 vs. label noise")
    plot_metric(axes[1, 0], data, "ece", "ECE", "ECE vs. label noise")

    # 噪声识别指标（仅 selection 类方法有）
    ax = axes[1, 1]
    for field in ("noise_precision", "noise_recall", "noise_f1", "noise_auroc"):
        rates, vals, errs = series(data, "full", field)
        if vals:
            ax.errorbar(rates, vals, yerr=errs, marker="o", capsize=3, label=field)
    ax.set_xlabel("label noise rate")
    ax.set_ylabel("noise detection score")
    ax.set_title("Noise detection (full method)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Few-shot classification under label noise (CIFAR-10)")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_dir / "main_results.png", dpi=150)
    print(f"[完成] 主结果图 -> {out_dir / 'main_results.png'}")


if __name__ == "__main__":
    main()
