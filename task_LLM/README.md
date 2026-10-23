# task_LLM：Transformer 与 nanoGPT 学习笔记 + 实验文档

本目录是"从 Transformer 原理到 nanoGPT 部署/训练/生成"的完整产出，对应三个任务：

## 目录结构

```
task_LLM/
├── nanoGPT/                    # 克隆的 nanoGPT 源码（karpathy/nanoGPT）
│   ├── model.py                # 模型定义（GPT / Attention / MLP / Block）
│   ├── train.py                # 训练脚本
│   ├── sample.py               # 推理生成脚本
│   ├── configurator.py         # 配置器
│   ├── config/train_shakespeare_char_cpu.py  # 本实验的 CPU 训练配置
│   └── data/shakespeare_char/  # 莎士比亚字符级数据（prepare.py + 生成的数据文件）
├── notes/
│   ├── 01_Transformer笔记.md   # Task 1：论文《Attention Is All You Need》笔记
│   ├── 02_nanoGPT笔记.md       # Task 2：nanoGPT 源码阅读 + 架构对应 + Tensor shape
│   └── 03_实验报告.md          # Task 3：部署、训练、生成、实验分析
├── train_log.txt               # 完整训练日志
├── loss_curve.png              # loss 曲线图（plot_loss.py 生成）
├── generate_demo.py            # 生成脚本（训练模型 vs 未训练模型对比）
├── plot_loss.py                # 解析训练日志并绘制 loss 曲线
└── bench_cpu.py / bench_attn.py# CPU 速度基准测试脚本
```

## 三个任务对应关系

| 任务 | 产出 | 关键问题 |
|---|---|---|
| Task 1 | [notes/01_Transformer笔记.md](notes/01_Transformer笔记.md) | Transformer 是如何工作的？ |
| Task 2 | [notes/02_nanoGPT笔记.md](notes/02_nanoGPT笔记.md) | 论文结构在真实 GPT 代码里如何实现？ |
| Task 3 | [notes/03_实验报告.md](notes/03_实验报告.md) | 从数据、模型、训练到生成的完整流程？ |

## 复现实验

从 `task_LLM/` 运行以下 PowerShell 命令。新训练写入独立目录，后续生成与绘图明确读取同一轮产物：

```powershell
New-Item -ItemType Directory -Force artifacts/reproduction | Out-Null
cd nanoGPT
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char_cpu.py --out_dir=../artifacts/reproduction/model *> ../artifacts/reproduction/train.log
cd ..
python generate_demo.py --checkpoint artifacts/reproduction/model/ckpt.pt --output artifacts/reproduction/generation.json
python plot_loss.py --log artifacts/reproduction/train.log --csv artifacts/reproduction/loss.csv --output artifacts/reproduction/loss_curve.png
```

> 本机无 GPU，实验在 CPU（i5-13420H）上以 float32 完成，模型缩小为 0.80M 参数（4 层 / 4 头 / 128 维 / block 128）。

## 2026-10-15 完善后的验证入口

以下命令从 `task_LLM/` 目录运行；安装依赖可用 `python -m pip install -r requirements.txt`。
CPU/GPU PyTorch 的具体安装包按自己的设备选择。当前验证使用已有 CPU 环境，没有重新安装依赖。

```powershell
python -m unittest discover -s tests -v
python scripts/verify_environment.py
python generate_demo.py --seed 1337 --num-samples 2 --max-new-tokens 300 --output artifacts/verification/generation.json
python plot_loss.py --output artifacts/verification/loss_curve.png
```

`generate_demo.py` 可从任意工作目录执行，默认 checkpoint/meta 使用脚本所在目录定位。
`--checkpoint`、`--meta`、`--output` 的显式相对路径按当前工作目录解释。
可使用 `--prompt "ROMEO:" --temperature 0.8 --top-k 20` 调整提示与采样。随机对照从 checkpoint 读取同一架构，
每次采样单独固定种子。结果 JSON 记录 checkpoint、配置和两组文本；复现保证限于相同设备与软件环境。

`plot_loss.py --log 路径 --csv 路径 --output 路径` 可绘制其他日志。
绘图应在独立 Python 进程执行；本机同进程导入 torch 与 matplotlib 出现 OpenMP 运行库冲突，工具因此延迟加载 matplotlib。

短训练验证使用独立目录，避免覆盖历史训练模型：

```powershell
cd nanoGPT
python train.py config/train_shakespeare_char_cpu.py --out_dir=../artifacts/verification/smoke-model --batch_size=2 --block_size=32 --n_layer=1 --n_head=2 --n_embd=32 --max_iters=4 --eval_interval=2 --eval_iters=2 --log_interval=1 --warmup_iters=1 --lr_decay_iters=4 *> ../artifacts/verification/smoke_train.log
cd ..
python generate_demo.py --checkpoint artifacts/verification/smoke-model/ckpt.pt --max-new-tokens 40 --num-samples 1 --output artifacts/verification/smoke_generation.json
```

先运行环境采集或生成工具创建 `artifacts/verification/`，再执行上面的日志重定向。
4 步训练用于验证链路，不代表模型已收敛；原有 1200 步实验承担正式训练结果。

恢复历史训练（会写回原输出目录，应仅在确实希望继续训练时运行）：

```powershell
cd nanoGPT
python train.py config/train_shakespeare_char_cpu.py --init_from=resume --max_iters=1500 --lr_decay_iters=1500
```

上游 `sample.py` 导入 tiktoken；本地字符级生成工具不需要该依赖。默认使用可信本地 checkpoint 和 meta。
提交时应附带 `notes/`、运行脚本、训练配置、原训练日志、图表、生成结果、验证产物与获取数据的说明。
checkpoint 在 `nanoGPT/out-shakespeare-char-cpu/ckpt.pt`，数据下载脚本在 `nanoGPT/data/shakespeare_char/prepare.py`。
