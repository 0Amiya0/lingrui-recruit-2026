# Step 2 · Baseline：普通交叉熵（要求③）

**目标**：实现两个 baseline，得到不同噪声率下的基准曲线。

## 任务清单

- [ ] `src/baseline.py`：
  - `ProtoNet`：类均值原型 + 最近邻（cosine）分类，无训练
  - `CELinearProbe`：support 特征上训练线性分类头（CE loss，含可选 temperature）
- [ ] `scripts/run_baseline.py`：加载 Step 1 的 npz → 训练/推理 → 存指标 CSV
- [ ] 跑 4 个噪声率 × 2 baseline

## 关键细节

- 线性头输入 512-d、输出 10 类；用 `torch.optim.Adam` + CE；support 样本少（≤100），训练到收敛即可。
- 因为 support 可能含噪声，CE baseline 会**过拟合噪声样本**，这正是要展示的退化现象。
- 输出概率：线性头 softmax 直接给；ProtoNet 用 `softmax(-d / T)`（先固定 T=1，Step 5 再做 temperature scaling 算 ECE）。

## 产出

- `results/baseline_*.csv`：每噪声率一行的 Acc / Macro-F1 / ECE（ECE 在 Step 5 补全）

## 验收标准（DoD）

1. 0% 噪声下 CE-linear-probe 与 ProtoNet 的 Acc 应接近（都在自检水平附近）。
2. 随噪声率升高，两者 Acc 单调下降——这是后续方法要修复的目标。
3. 复现：同 seed 跑两次结果一致。
