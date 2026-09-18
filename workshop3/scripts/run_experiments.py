"""Step 4：批量跑主实验矩阵（K 固定，扫噪声率 0 / 10 / 20 / 40%）。

对每个噪声率依次：prepare_data -> run_baseline -> run_method。
用法：python scripts/run_experiments.py --k-shot 10 --seed 0 --loss gce
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 修复 anaconda(MKL) 与 torch 的 OpenMP 冲突

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # workshop3/ 根
sys.path.insert(0, str(Path(__file__).resolve().parent))       # scripts/

from dataclasses import replace

from src.config import Config
import prepare_data
import run_baseline
import run_method

RATES = [0.0, 0.1, 0.2, 0.4]


def main():
    base = Config.from_args()
    for r in RATES:
        cfg = replace(base, noise_rate=r)
        print(f"\n{'=' * 60}\n噪声率 {r}  (K={cfg.k_shot}, seed={cfg.seed})\n{'=' * 60}")
        prepare_data.main(cfg)
        run_baseline.main(cfg)
        run_method.main(cfg)


if __name__ == "__main__":
    main()
