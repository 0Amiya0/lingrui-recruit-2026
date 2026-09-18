# PLAN · 工程实施计划（足够具体的 how）

> 本文档面向"照着做"：环境、目录、模块接口、数据流、配置 schema、运行命令、每步验收标准。研究层面的 what/why 见 [fuzzPlan.md](plans/fuzzPlan.md)。

## 1. 环境（已实测）

- Python 3.13.5（anaconda，`C:\Users\NiZhe\anaconda3\python.exe`）
- 无 NVIDIA GPU（nvidia-smi 不存在）→ **CPU 版 PyTorch**
- 已装：numpy 2.1.3 / sklearn 1.6.1 / matplotlib 3.10.0 / pyyaml 6.0.2
- 待装：torch + torchvision（CPU 源）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. 目录结构

```
workshop3/
├── PLAN.md / plans/            # plans/ 含 fuzzPlan.md 与 step0~step6
├── configs/*.json          # 每个实验一份配置
├── src/
│   ├── config.py           # Config dataclass + from_json + from_args(CLI覆盖)
│   ├── utils.py            # seed_everything / get_device / save&load json/npz / now_str
│   ├── backbone.py         # Backbone(name, pretrained, normalize).forward -> (B,dim)
│   ├── dataset.py          # load_cifar10 / sample_support / inject_symmetric_noise / extract_features
│   ├── baseline.py         # [Step2] CE-linear-probe / ProtoNet
│   ├── method.py           # [Step3] 样本筛选 + 鲁棒损失
│   ├── metrics.py          # [Step5] accuracy / macro-F1 / ECE / 噪声识别准确率
│   └── train_eval.py       # [Step2+] 主流程
├── scripts/
│   ├── prepare_data.py     # [Step1] 数据准备流水线 ✅
│   ├── run_baseline.py     # [Step2]
│   ├── run_method.py       # [Step3]
│   ├── run_experiments.py  # [Step4] 批量跑矩阵
│   └── plot_results.py     # [Step4/6] 画图
├── data/processed/         # 特征/标签/噪声 mask（.gitignore）
└── results/                # CSV/图表/日志
```

## 3. 核心数据流

```
CIFAR-10(train 50k, test 10k)
   │ sample_support(n_way, k_shot, seed)          # 每类抽 K 张
   ▼
support_idx (N·K) + support_true 标签
   │ inject_symmetric_noise(noise_rate, seed)     # 对称翻转
   ▼
support_noisy 标签 + noise_mask (bool)
   │ Backbone.forward (冻结 ResNet-18, L2 归一化)
   ▼
support_feats (N·K, 512) / test_feats (10k, 512)   # test 特征缓存复用
   ▼
下游方法（baseline / 筛选 / 鲁棒损失）→ 干净测试集指标
```

## 4. 模块接口契约

| 模块 | 关键函数 | 输入 → 输出 |
|---|---|---|
| `dataset.sample_support` | `(train_targets, n_way, k_shot, seed)` | `int64[N·K]` 下标 |
| `dataset.inject_symmetric_noise` | `(labels, n_way, noise_rate, seed)` | `(noisy_labels, mask)` |
| `dataset.extract_features` | `(backbone, dataset, indices, device, bs)` | `(feats[N,dim], labels[N])` |
| `backbone.Backbone` | `(name, pretrained, normalize)` | `.forward(x)->(B,dim)` |
| `metrics.*` | `(pred, target, ...)` | 标量指标 |

**数据文件格式（npz）**：
- `data/processed/cifar10_test_features.npz`：`test_feats (10000,dim) f32`, `test_labels (10000,) i64`（共享，只算一次）
- `data/processed/support_cifar10_K{K}_r{r}_seed{s}.npz`：`support_feats`, `support_true`, `support_noisy`, `noise_mask`, `support_indices`
- 同名 `.json`：配置 + 实际噪声率 + 特征维度 + 自检准确率

## 5. 配置 schema（configs/*.json）

```json
{
  "dataset": "cifar10", "n_way": 10, "k_shot": 10,
  "noise_rate": 0.4, "noise_type": "symmetric", "seed": 0,
  "backbone": "resnet18", "pretrained": true, "image_size": 224,
  "normalize": true, "batch_size": 256, "device": "auto",
  "data_dir": "data", "out_dir": "results"
}
```

命令行覆盖：`--k-shot 5 --noise-rate 0.2 --seed 1`（`config.py` 已实现）。

## 6. 里程碑与验收标准

| Step | 交付物 | 验收标准（DoD） |
|---|---|---|
| 0 骨架 | 目录 + config/utils/backbone/dataset + README | `python scripts/prepare_data.py --help` 可跑，模块可 import |
| 1 数据 | prepare_data.py 跑通，产出 npz/json | 实际噪声率≈请求值；自检干净原型 acc 在合理区间（~50–70%） |
| 2 baseline | CE-linear-probe + ProtoNet | 4 噪声率 × 干净测试集的 Acc/F1/ECE 有结果 |
| 3 方法 | 筛选 + GCE/SCE | 40% 噪声下显著优于 CE baseline |
| 4 实验 | 矩阵 + 消融脚本/图 | 对比曲线 + 消融表齐备 |
| 5 指标 | metrics.py | 4 项指标全部复现一致 |
| 6 分析 | 错误案例 + t-SNE + 论文 | 3–4 页 short paper 草稿 |

## 7. 复现性原则

- 所有随机（support 抽样、噪声注入、训练、数据加载）都用 `seed_everything(seed)` 固定。
- 每个实验输出保存：配置 JSON + 指标 CSV + 日志，附 noise mask。
- 特征只依赖冻结 backbone + 固定 seed，可精确复现。
