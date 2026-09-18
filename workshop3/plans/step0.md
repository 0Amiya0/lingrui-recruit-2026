# Step 0 · 工程骨架

**目标**：搭好可扩展、可复现的项目骨架，让后续每一步只往里面填模块和脚本。

## 任务清单

- [x] 目录结构（src / scripts / configs / data / results / plans）
- [x] `requirements.txt`（torch/torchvision 走 CPU 源，README 说明）
- [x] `README.md`（安装、快速开始、产出说明）
- [x] `src/config.py`：`Config` dataclass + `from_json` + `from_args`（CLI 覆盖）
- [x] `src/utils.py`：`seed_everything` / `get_device` / `save_json` / `load_json` / `save_npz` / `now_str`
- [x] `src/backbone.py`：冻结 ResNet 特征提取器
- [x] `src/dataset.py`：数据加载 / support 划分 / 噪声注入 / 特征提取
- [ ] `.gitignore`（忽略 `data/`、`results/`、`__pycache__/`、`*.npz`）

## 设计决策

- 配置用 **dataclass + JSON + argparse 覆盖**：零额外依赖、可序列化、便于批量跑实验。
- 特征提取用 `num_workers=0`：Windows 上避免多进程加载的坑。
- 冻结 backbone 用 **ImageNet 归一化统计量**（不是 CIFAR 的），因为权重是 ImageNet 预训练的。

## 验收标准（DoD）

1. `python scripts/prepare_data.py --help` 正常打印参数。
2. `python -c "from src.config import Config; from src.backbone import Backbone; print('ok')"` 可 import。
3. 目录结构与本文一致。
