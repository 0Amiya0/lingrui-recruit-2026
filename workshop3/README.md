# workshop3 · 标签噪声下的小样本图像分类

Workshop 题目三（见 [task.md](../task.md) 第 5 节）：在 CIFAR-10 上构造 few-shot（每类 K 张标注）支持集，
注入 10% / 20% / 40% 对称标签噪声，研究其对训练的影响，并用「样本筛选 + 鲁棒损失」方法提升干净测试集表现。

- 研究总纲：见 [fuzzPlan.md](plans/fuzzPlan.md)
- 工程实施计划：见 [PLAN.md](PLAN.md)
- 每步详细计划：见 [plans/](plans/)
- 论文草稿：见 [paper/short_paper.md](paper/short_paper.md)

## 目录结构

```
workshop3/
├── PLAN.md                # 工程实施计划（研究总纲见 plans/fuzzPlan.md）
├── plans/                   # 每步详细计划 step0 ~ step6
├── paper/short_paper.md     # 3-4 页英文 short paper 草稿
├── run_all.ps1              # PowerShell 全流程实验脚本（本机入口）
├── configs/*.json           # 实验配置
├── src/
│   ├── config.py            # 配置 dataclass + CLI 覆盖
│   ├── utils.py             # seed / 设备 / 文件 IO
│   ├── backbone.py          # 冻结 ImageNet 预训练 ResNet
│   ├── dataset.py           # CIFAR 加载 + support 划分 + 噪声注入 + 特征提取
│   ├── baseline.py          # [Step2] ProtoNet / CE 线性头
│   ├── method.py            # [Step3] 距离筛选 + 鲁棒损失(GCE/SCE)
│   └── metrics.py           # [Step5] Acc / Macro-F1 / ECE / 噪声识别
├── scripts/
│   ├── prepare_data.py      # [Step1] 数据准备流水线
│   ├── run_baseline.py      # [Step2] 跑 baseline
│   ├── run_method.py        # [Step3] 跑方法（三变体）
│   ├── run_experiments.py   # [Step4] Python 版批量编排（替代方案）
│   ├── run_ablation.py      # [Step4] 超参消融
│   ├── plot_results.py      # [Step4] 画主结果图
│   ├── test_metrics.py      # [Step5] 指标 sanity check
│   └── error_analysis.py    # [Step6] 错误分析
├── data/                    # 原始数据 + 处理后的特征（.gitignore）
└── results/                 # 指标 CSV / 图表 / 错误分析报告
```

## 环境安装（Windows · Python 3.13 · 无 GPU）

PyTorch 请**先用 CPU 源**安装，再装其余依赖：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> 已确认本机 numpy / scikit-learn / matplotlib / pyyaml 已装，其余只需 torch + torchvision。

> ⚠️ **OpenMP 冲突修复**：anaconda 的 MKL 与 torch 自带的 OpenMP 会冲突报
> `OMP: Error #15`。所有入口脚本顶部已设置 `KMP_DUPLICATE_LIB_OK=TRUE` 修复；
> 若在其它环境手动 import，需自行设置该环境变量。

## 全流程运行（推荐 · PowerShell）

本机无 GPU，所有实验统一由 [run_all.ps1](run_all.ps1) 编排，在 CPU 上执行：

```powershell
cd Lingrui\workshop3
Set-ExecutionPolicy -Scope Process Bypass   # 首次若被禁止运行脚本时执行
.\run_all.ps1                               # 全流程：数据→baseline→方法→消融→画图→错误分析
.\run_all.ps1 -K 10 -Seed 0 -Loss gce -Rates "0.0,0.1,0.2,0.4"
```

脚本会依次完成：数据准备（4 噪声率）→ 主实验矩阵（baseline + 方法）→ 消融（40%）
→ few-shot 规模敏感性（K=5/20）→ 指标自检 → 画图 → 错误分析。

> 首次运行会下载 CIFAR-10（约 170 MB）+ ResNet-18 权重（约 44 MB），
> 并在 CPU 上提取 10k 张测试特征（数分钟到十几分钟，之后缓存复用）。

## 单步运行（Python，便于调试）

```bash
# Step 1：数据准备
python scripts/prepare_data.py --k-shot 10 --noise-rate 0.4 --seed 0

# Step 2：baseline
python scripts/run_baseline.py --k-shot 10 --noise-rate 0.4 --seed 0

# Step 3：方法（selection-only / robust-loss-only / full）
python scripts/run_method.py --k-shot 10 --noise-rate 0.4 --seed 0 --loss gce

# Step 4：消融 + 画图
python scripts/run_ablation.py --k-shot 10 --noise-rate 0.4 --seed 0
python scripts/plot_results.py

# Step 5：指标自检（不依赖数据/GPU）
python scripts/test_metrics.py

# Step 6：错误分析
python scripts/error_analysis.py --k-shot 10 --noise-rate 0.4 --seed 0
```

## 产出

- `data/processed/`：support 特征 + 干净/噪声标签 + 噪声 mask；干净测试特征（缓存复用）
- `results/baseline_summary.csv`、`method_summary.csv`、`ablation_summary.csv`
- `results/error_analysis.json`（混淆矩阵、噪声识别失败、原型漂移）
- `results/figs/*.png`（主结果曲线、混淆矩阵、t-SNE、原型漂移、距离分布）
