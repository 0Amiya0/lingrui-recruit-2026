# Step 3 · 核心方法：样本筛选 + 鲁棒损失（要求④）

**目标**：实现可插拔的"距离驱动样本筛选 + 鲁棒损失"方法，并支持 `selection_only / robust_loss_only / full` 三种开关。

## 任务清单

- [ ] `src/method.py`：
  - `distance_based_selection`：迭代 T 轮 trimmed-mean 原型 + 大距离剔除
  - `GCELoss` / `SCELoss`：广义/对称交叉熵
  - `RobustClassifier`：筛选后的特征上训练线性头（可换 CE / GCE / SCE）
- [ ] `scripts/run_method.py`：加载 npz → 跑方法 → 存指标 CSV

## 关键细节

- **筛选伪代码**：
  ```
  keep = all(True)
  for t in 1..T:
      proto[c] = mean(support_feats[keep & noisy==c], axis=0)   # 用当前保留样本
      d[i] = ||feat[i] - proto[noisy[i]]||                       # 到其标注类原型的距离
      threshold = mu + beta * sigma  或  top-tau 比例
      keep = d <= threshold
  ```
- **鲁棒损失**：
  - GCE：`L = (1 - p_y^q) / q`（q 越小越鲁棒，q→0 趋近 MAE）
  - SCE：`CE + beta_ce * 反向交叉熵`（对噪声更对称）
- 筛选用**标注类**（noisy label）算距离，而不是真实类——这样"噪声识别准确率"指标才可计算（与 noise mask 对齐）。

## 产出

- `results/method_*.csv` + 筛选出的 `predicted_clean_mask`（用于噪声识别准确率）

## 验收标准（DoD）

1. 40% 噪声下，full 方法的干净测试 Acc 显著高于 CE baseline。
2. `selection_only` 在低噪声（0/10%）下不劣于 baseline（不应误伤干净样本太多）。
3. 筛选输出的 mask 与真实 noise mask 的 Precision/Recall 可计算（Step 5）。
