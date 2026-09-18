"""Step 5：metrics 的单元级 sanity check（不依赖 GPU/数据）。

用法：python scripts/test_metrics.py
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 修复 OpenMP 冲突

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.metrics import (accuracy, macro_f1, expected_calibration_error,
                         noise_detection_metrics, softmax)


def check(name, got, expected, tol=1e-4):
    ok = abs(got - expected) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: got={got:.6f} expected={expected:.6f}")
    assert ok, f"{name} 不通过"


def main():
    # accuracy
    check("accuracy perfect", accuracy([0, 1, 2, 3], [0, 1, 2, 3]), 1.0)
    check("accuracy 25%", accuracy([0, 0, 0, 0], [0, 1, 2, 3]), 0.25)

    # macro_f1
    check("macro_f1 perfect", macro_f1([0, 1, 2], [0, 1, 2]), 1.0)
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 0, 0, 1, 0]
    # 类0 f1=0.6667, 类1 f1=1.0, 类2 f1=0.0 -> macro=0.5556
    check("macro_f1 mixed", macro_f1(y_true, y_pred), 0.5556, tol=1e-3)

    # softmax
    p = softmax(np.array([[0.0, 0.0]]))
    check("softmax uniform", p[0, 0], 0.5)
    check("softmax sums to 1", float(p.sum()), 1.0)

    # ECE：完全校准 -> 0
    probs = np.array([[0.5, 0.5]] * 10)
    labels = np.array([0] * 5 + [1] * 5)
    check("ECE perfectly calibrated", expected_calibration_error(probs, labels), 0.0)

    # ECE：过度自信且一半错 -> |0.5 - 0.99| = 0.49
    probs2 = np.array([[0.99, 0.01]] * 10)
    labels2 = np.array([0] * 5 + [1] * 5)
    check("ECE overconfident", expected_calibration_error(probs2, labels2), 0.49, tol=0.02)

    # 噪声识别：tp=2 fp=1 fn=1 -> prec/recall/f1 = 2/3
    true_noise = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    pred_noise = np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    m = noise_detection_metrics(pred_noise, true_noise)
    check("noise precision", m["precision"], 2 / 3, tol=1e-3)
    check("noise recall", m["recall"], 2 / 3, tol=1e-3)
    check("noise f1", m["f1"], 2 / 3, tol=1e-3)

    print("\nALL METRICS SANITY CHECKS PASSED")


if __name__ == "__main__":
    main()
