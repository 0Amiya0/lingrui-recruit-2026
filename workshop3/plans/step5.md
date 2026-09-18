# Step 5 · 指标实现（建议指标）

**目标**：统一实现 4 类指标，供所有方法复用。

## 任务清单

- [ ] `src/metrics.py`：
  - `accuracy(y_true, y_pred)`
  - `macro_f1(y_true, y_pred)`
  - `expected_calibration_error(probs, y_true, n_bins=15)`（含 temperature scaling 辅助）
  - `noise_detection_metrics(pred_mask, noise_mask)` → Precision / Recall / F1

## 关键细节

- **ECE**：把预测概率按置信度分 15 个 bin，`sum(|bin_acc - bin_conf| * bin_size / N)`。
- 线性头：`probs = softmax(logits)`；ProtoNet：`probs = softmax(-d / T)`，T 用验证集/支持集做 temperature scaling。
- **噪声识别准确率**：`pred_mask` 来自 Step 3 筛选结果，`noise_mask` 来自 Step 1，直接算 P/R/F1。

## 产出

- `src/metrics.py` + 所有方法统一调用同一套指标函数。

## 验收标准（DoD）

1. 每个指标都有单元级 sanity check（构造已知答案的输入验证数值）。
2. 各方法的 CSV 里 4 项指标齐全、口径一致。
