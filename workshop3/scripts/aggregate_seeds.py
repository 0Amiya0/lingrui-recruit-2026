"""Step 4：多种子聚合 —— 把 baseline/method 的逐种子 CSV 汇总成 mean±std。

读 results/baseline_summary.csv 与 results/method_summary.csv（每行一个 seed），
按 (method, noise_rate) 分组，输出 results/summary_meanstd.csv：
  - 均值沿用原字段名（accuracy / macro_f1 / ece / noisy_test_acc / noise_*）
  - 标准差用 <field>_std 列
用法：python scripts/aggregate_seeds.py
"""
import csv
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 修复 OpenMP 冲突

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

METRIC_FIELDS = ["accuracy", "macro_f1", "ece", "noisy_test_acc",
                 "noise_precision", "noise_recall", "noise_f1", "noise_auroc"]


def load(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dedupe(rows):
    """按 (method, noise_rate, seed) 去重，避免重复跑 run_all.ps1 时重复计数。"""
    seen, out = set(), []
    for r in rows:
        key = (r.get("method"), r.get("noise_rate"), r.get("seed"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def aggregate(rows):
    groups = {}
    for r in rows:
        key = (r["method"], float(r["noise_rate"]))
        groups.setdefault(key, []).append(r)
    out = []
    for (method, rate), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        row = {"method": method, "noise_rate": f"{rate:g}", "n_seeds": len(rs)}
        for f in METRIC_FIELDS:
            vals = [float(r[f]) for r in rs if r.get(f) not in ("", None)]
            if vals:
                row[f] = f"{np.mean(vals):.6f}"
                row[f"{f}_std"] = f"{np.std(vals):.6f}" if len(vals) > 1 else ""
            else:
                row[f] = ""
                row[f"{f}_std"] = ""
        out.append(row)
    return out


def main():
    rows = dedupe(load("results/baseline_summary.csv") + load("results/method_summary.csv"))
    out_rows = aggregate(rows)
    if not out_rows:
        print("[警告] 未找到结果 CSV，请先跑 run_all.ps1 生成。")
        return

    fields = ["method", "noise_rate", "n_seeds"]
    for f in METRIC_FIELDS:
        fields += [f, f"{f}_std"]

    out = Path("results") / "summary_meanstd.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"[完成] mean±std 汇总 -> {out}（{len(out_rows)} 行，来自 {len(rows)} 个 seed 记录）")


if __name__ == "__main__":
    main()
