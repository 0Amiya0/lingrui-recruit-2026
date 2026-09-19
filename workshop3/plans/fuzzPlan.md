# fuzzPlan · 标签噪声下的小样本图像分类（研究总纲）

> Workshop 题目三。目标：按 Workshop short paper 标准完成一次完整研究流程，产出 3–4 页英文 short paper + 代码 + 实验配置 + 图表日志 + AI 工具使用说明。

## 1. 研究问题

在 **few-shot（每类仅 K 张标注样本）** 且这些标注 **被对称标签噪声污染（10% / 20% / 40%）** 的条件下，如何训练一个分类器，使其在**干净测试集**上保持尽可能好的性能？

核心张力：few-shot 本就让每个样本都很珍贵，标签噪声会直接污染原型/决策边界，且样本越少噪声破坏力越强。

## 2. 关键设定（已与用户确认）

| 决策点 | 选择 |
|---|---|
| 数据集 | CIFAR-10（10 类） |
| 小样本含义 | 两者结合：既限制每类标注量 K，又叠加标签噪声 |
| 特征提取 | ImageNet 预训练 ResNet-18（**冻结**，512-d）——backbone 从未见过 CIFAR 标签 |
| 噪声类型 | 对称噪声（uniform，翻成其它随机类别） |
| 干净测试集 | CIFAR-10 官方 test（10k 张，永不污染） |
| 核心方法 | 样本筛选（距离驱动）+ 鲁棒损失（GCE/SCE）组合 |

## 3. 方法概述

**核心思路**：把经典"小损失样本 = 干净样本"的样本筛选范式，迁移到 few-shot 原型空间，变成"**大距离样本 = 疑似噪声**"。

1. **样本筛选（组件一）**：迭代 T 轮——计算各类原型 → 每个 support 样本到其标注类原型的距离 → 按阈值（μ+βσ 或固定 top-τ 比例）剔除疑似噪声 → 用剩余样本重算原型（trimmed mean）。
2. **鲁棒损失（组件二）**：对筛选后的样本，用 GCE（广义交叉熵）或 SCE（对称交叉熵）训练分类头，抵御筛选漏掉的残余噪声。

**Baseline**：
- A（必做，对应"普通交叉熵"）：support 特征上训练线性分类头 + CE。
- B（few-shot 标准参考）：Prototypical Network（类均值原型 + 最近邻）。

## 4. 实验矩阵

主实验固定 **K=10**，横扫噪声率；K 作为第二轴只在"few-shot 规模敏感性"单独做。

| 方法 | 0% | 10% | 20% | 40% |
|---|---|---|---|---|
| CE-linear-probe（baseline） | ✓ | ✓ | ✓ | ✓ |
| ProtoNet（参考） | ✓ | ✓ | ✓ | ✓ |
| + 样本筛选 | ✓ | ✓ | ✓ | ✓ |
| + 筛选 + 鲁棒损失（full） | ✓ | ✓ | ✓ | ✓ |

**消融**：`CE → +selection → +robust_loss → +both`；再消融迭代轮数 T、阈值 β/τ。
**规模敏感性**：K ∈ {5, 10, 20} 在最高噪声率 40% 下的表现。

## 5. 评估指标

- **Accuracy**、**Macro-F1**（干净测试集，sklearn）
- **ECE**（Expected Calibration Error）：线性头 softmax 概率 / ProtoNet `softmax(-d/T)` + temperature scaling，15-bin
- **噪声样本识别准确率**（核心自定义指标）：筛选结果 vs 真实 noise mask 的 Precision / Recall / F1
- **性能-噪声率曲线**：不同噪声比例下的性能变化

## 6. 错误分析（要求⑥）

两类失败：
1. **模型误判**：干净样本为何被分错（类间混淆对、原型漂移）。
2. **噪声识别失败**：被污染样本为何没被距离法筛掉（高噪声率下原型本身被污染、少数类整体漂移）。

可视化：t-SNE 特征图（标出噪声样本）、每类原型漂移方向、筛选阈值下的距离分布。

## 7. 论文结构（对应 task.md 第 9 节）

Abstract / Introduction / Related Work / Method / Experiments / Error Analysis / Limitations and Conclusion。

## 8. 风险与应对

1. **ImageNet→CIFAR 迁移精度一般**：作为相对比较足够（噪声方法 vs baseline 的增益是主要结论）。可选升级：自监督预训练 backbone，论文加"标签完全未被使用"的更强声明。
2. **高噪声 + 低 K 时原型极易漂移**：40% × K=5 平均每类只剩 3 张干净样本——这是论文的"困难区"卖点，重点分析而非回避。
3. **ECE 需要概率输出**：统一口径（线性头 softmax / 带 temperature 的 ProtoNet）。
4. **可复现**：所有随机（support 抽样、噪声注入、训练）seed 固定，附 noise mask 与 config。

## 9. 分步路线

Step 0 工程骨架 → Step 1 数据+噪声构造 → Step 2 CE baseline → Step 3 核心方法 → Step 4 对比+消融 → Step 5 指标 → Step 6 错误分析+论文。详见 [PLAN.md](../PLAN.md) 与本目录 step0–step6.md。
