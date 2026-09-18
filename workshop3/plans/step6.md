# Step 6 · 错误分析 + 论文（要求⑥）

**目标**：分析模型误判与噪声识别失败的原因，并产出 short paper 草稿。

## 任务清单

- [ ] 错误案例分析：统计误判样本的类间混淆矩阵；统计"漏筛"噪声样本特征
- [ ] 可视化：t-SNE 特征图（标出噪声样本 + 筛选结果）；每类原型漂移方向；筛选阈值下的距离分布
- [ ] 论文草稿 `paper/`：按 Abstract / Introduction / Related Work / Method / Experiments / Error Analysis / Limitations and Conclusion

## 两类失败的分析要点

1. **模型误判**（干净样本被分错）：
   - 类间混淆对（如 cat/dog、truck/automobile）
   - 噪声导致原型漂移，使相邻类决策边界偏移
2. **噪声识别失败**（被污染样本没被筛掉）：
   - 高噪声率下原型本身被污染，阈值失准
   - 少数类 / 类内变异大的样本距离天然大，被误判为噪声或漏判

## 产出

- `results/figs/tsne.png`、`results/figs/error_analysis.png`
- `paper/short_paper.md`（英文草稿）

## 验收标准（DoD）

1. 每个结论都有对应的图表或数字支撑。
2. 论文能回答"方法在哪些条件下有效 / 失效"。
