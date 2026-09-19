# workshop3 · 标签噪声下的小样本图像分类

> Robust Few-shot Image Classification under Label Noise

Workshop 题目三（见 [task.md](../task.md) 第 5 节）：在 CIFAR-10 上构造 few-shot（每类 K 张标注）支持集，
注入 10% / 20% / 40% 对称标签噪声，研究其对训练的影响，并用「样本筛选 + 鲁棒损失」方法提升干净测试集表现。

**跨平台**（Windows / macOS / Linux）· **设备自适应**（有 CUDA 用 GPU，否则 CPU）· **模块化 pipeline**。

- 研究总纲：见 [fuzzPlan.md](plans/fuzzPlan.md)
- 工程实施计划：见 [PLAN.md](PLAN.md)
- 每步详细计划：见 [plans/](plans/)
- 论文草稿：见 [paper/short_paper.md](paper/short_paper.md)（LaTeX 版 [short_paper.tex](paper/short_paper.tex)）
- AI 使用报告：见 [AI_USAGE_REPORT.md](AI_USAGE_REPORT.md)

## 目录结构

```
workshop3/
├── PLAN.md                  # 工程实施计划（研究总纲见 plans/fuzzPlan.md）
├── plans/                   # 每步详细计划 step0 ~ step6
├── paper/
│   ├── short_paper.md       # 3-4 页英文 short paper 草稿
│   └── short_paper.tex      # ICLR 2023 模板 LaTeX 版
├── AI_USAGE_REPORT.md       # AI 使用报告（课程要求）
├── run_all.ps1              # [Windows] PowerShell 全流程编排
├── configs/*.json           # 实验配置
├── src/
│   ├── config.py            # 配置 dataclass + CLI 覆盖
│   ├── utils.py             # seed / 设备(auto) / 文件 IO
│   ├── backbone.py          # 冻结 ImageNet 预训练 ResNet
│   ├── dataset.py           # CIFAR 加载 + support 划分 + 噪声注入 + 特征提取
│   ├── baseline.py          # ProtoNet / CE 线性头
│   ├── method.py            # 距离筛选 + 鲁棒损失(GCE/SCE)
│   └── metrics.py           # Acc / Macro-F1 / ECE / 噪声识别 / AUROC / risk-coverage
├── scripts/
│   ├── prepare_data.py      # Step1 数据准备
│   ├── run_baseline.py      # Step2 baseline
│   ├── run_method.py        # Step3 方法（三变体）
│   ├── run_experiments.py   # [跨平台] Python 批量编排（单种子扫噪声率）
│   ├── run_ablation.py      # Step4 超参消融
│   ├── aggregate_seeds.py   # Step4 多种子聚合 mean±std
│   ├── plot_results.py      # Step4 画主结果图
│   ├── test_metrics.py      # Step5 指标 sanity check
│   └── error_analysis.py    # Step6 错误分析
├── data/                    # 原始数据 + 处理后的特征（.gitignore）
└── results/                 # 指标 CSV / 图表 / 错误分析报告
```

## 环境安装（跨平台）

依赖 Python 3.9+（推荐 3.10–3.13）与 PyTorch。代码通过 `--device auto` 自动选择设备。

**A. 有 NVIDIA 显卡（推荐，跑得快）**

先确认驱动支持的 CUDA 版本（`nvidia-smi` 右上角 `CUDA Version`），再装对应 CUDA 版 PyTorch
（无需单独装 CUDA Toolkit，只要版本号 ≤ 驱动支持的上限）：

```bash
# CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# CUDA 11.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**B. 无 NVIDIA 显卡（CPU）**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

装其余依赖：

```bash
pip install -r requirements.txt
```

验证（GPU 环境应输出 `True` 与显卡名）：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> ⚠️ **OpenMP 冲突**：部分 Anaconda 环境的 MKL 与 torch 自带的 OpenMP 会冲突报
> `OMP: Error #15`。所有入口脚本顶部已设置 `KMP_DUPLICATE_LIB_OK=TRUE` 修复；
> 若手动 import 报错，请自行设置该环境变量（bash：`export KMP_DUPLICATE_LIB_OK=TRUE`）。

## 运行实验

### 方式一：Windows · PowerShell 全流程（一键）

```powershell
cd workshop3
Set-ExecutionPolicy -Scope Process Bypass   # 首次若被禁止运行脚本时执行
.\run_all.ps1                               # 全流程：数据→baseline→方法→消融→聚合→画图→错误分析
.\run_all.ps1 -K 10 -Seeds "0,1,2" -Loss gce -Rates "0.0,0.1,0.2,0.4"
.\run_all.ps1 -Seeds "0"                    # 单种子快速验证
.\run_all.ps1 -Device cuda:0                # 多卡机器显式指定某块 GPU
.\run_all.ps1 -Batch 64                     # 显存小时降低特征提取 batch
```

该脚本依次完成：数据准备（4 噪声率 × 多 seed）→ 主实验矩阵（baseline + 方法，多 seed）
→ 消融（40%）→ few-shot 规模敏感性（K=1/5/20）→ 指标自检 → 多种子聚合（mean±std）→ 画图 → 错误分析。

### 方式二：跨平台 · Python（Windows / macOS / Linux 通用）

用 `scripts/run_experiments.py` 跑主实验矩阵（单种子，扫 0 / 10 / 20 / 40% 噪声率）：

```bash
cd workshop3
python scripts/run_experiments.py --k-shot 10 --seed 0 --loss gce --device auto
```

其余步骤（消融 / 多种子聚合 / 画图 / 自检 / 错误分析）用下面的单步命令补齐；多种子可在 shell 里对
`--seed 0/1/2` 循环调用。

> 首次运行会下载 CIFAR-10（约 170 MB）+ ResNet-18 权重（约 44 MB），并在所选设备上提取
> 10k 张测试特征（GPU 上几秒到十几秒，CPU 上数分钟；之后缓存复用）。

## GPU / 显存说明

- **真正被训练的东西很小**：只有一个 `512×10` 的线性分类头；ResNet-18 只做**冻结的前向特征提取**
  （`@torch.no_grad()`，不存梯度、不反向），所以显存和算力需求都很低。
- **显存峰值**出现在 `prepare_data.py` 首次提取 10k 张测试特征时（224×224，batch=256）：
  - **4 GB 及以上显存**：默认参数直接跑，峰值约 2–3 GB；
  - **2 GB 显存**：把特征提取 batch 降到 64/128（PS 版 `-Batch 64`，Python 版 `--batch-size 64`），
    峰值降到约 1 GB；
  - 结论：**任何支持 CUDA 的独立显卡都能跑**，瓶颈在数据加载而非算力。
- **显式控制**：所有脚本都支持 `--device auto|cuda|cuda:0|cpu`；`run_all.ps1` 对应 `-Device`。
- 测试特征算一次即缓存（`data/processed/cifar10_test_features.npz`），后续步骤只做矩阵运算，几乎不占显存。

## 单步运行（Python，跨平台，便于调试）

```bash
# Step 1：数据准备
python scripts/prepare_data.py --k-shot 10 --noise-rate 0.4 --seed 0

# Step 2：baseline
python scripts/run_baseline.py --k-shot 10 --noise-rate 0.4 --seed 0

# Step 3：方法（selection-only / robust-loss-only / full）
python scripts/run_method.py --k-shot 10 --noise-rate 0.4 --seed 0 --loss gce

# Step 4：消融 + 多种子聚合 + 画图
python scripts/run_ablation.py --k-shot 10 --noise-rate 0.4 --seed 0
python scripts/aggregate_seeds.py
python scripts/plot_results.py

# Step 5：指标自检（不依赖数据/GPU）
python scripts/test_metrics.py

# Step 6：错误分析
python scripts/error_analysis.py --k-shot 10 --noise-rate 0.4 --seed 0
```

所有脚本都支持 `--help` 查看参数，并共享 `--device`、`--seed`、`--k-shot`、`--noise-rate`、
`--backbone`、`--batch-size` 等覆盖项（见 `src/config.py`）。

## 产出

- `data/processed/`：support 特征 + 干净/噪声标签 + 噪声 mask；干净测试特征（缓存复用）
- `results/baseline_summary.csv`、`method_summary.csv`（逐种子）、`summary_meanstd.csv`（mean±std）、`ablation_summary.csv`
- `results/error_analysis.json`（混淆矩阵、噪声识别失败、AUROC、原型漂移、选择性预测）
- `results/figs/*.png`（主结果曲线、混淆矩阵、t-SNE、原型漂移、距离分布、风险-覆盖）
