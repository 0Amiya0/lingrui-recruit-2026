# Step 1 · 数据集 + 标签噪声构造（要求①②）

**目标**：产出可复现的 few-shot support 特征、噪声标签、噪声 mask，以及干净测试特征。

## 任务清单

- [x] `load_cifar10`：torchvision 下载 CIFAR-10，resize 到 224，ImageNet 归一化
- [x] `sample_support`：每类抽 K 张，记录真实标签 `support_true`
- [x] `inject_symmetric_noise`：按 r ∈ {0,10,20,40}% 对称翻转，保留 `noise_mask`
- [x] `extract_features`：冻结 backbone 提取 support + test 特征
- [x] `scripts/prepare_data.py`：串联上述流程，落盘 npz + json，含自检
- [x] 测试特征缓存（`cifar10_test_features.npz`，只算一次）

## 关键细节

- **对称噪声翻转**：`new_label = randint(0, n_way-1)`，若 `>= 原标签` 则 `+1`，保证等概率翻到"其它"类别。
- **自检**：用干净标签算类均值原型，在测试集上做最近邻，期望 ~50–70%（ResNet-18 ImageNet 迁移到 CIFAR 的正常水平）。若明显低于此区间，说明特征提取或归一化有问题。
- **CPU 耗时**：首次提取 10k 测试特征需数分钟到十几分钟，之后缓存复用。

## 产出

- `data/processed/cifar10_test_features.npz`（共享）
- `data/processed/support_cifar10_K{K}_r{r}_seed{s}.npz` + 同名 `.json`

## 验收标准（DoD）

1. 实际噪声率 ≈ 请求值（`round(n*r)` 取整误差 ≤ 1 个样本）。
2. 自检干净原型准确率落在合理区间。
3. npz 中 `noise_mask.sum() == n_noisy`，且 `support_true[noise_mask] != support_noisy[noise_mask]`（被污染处标签确实变了）。
