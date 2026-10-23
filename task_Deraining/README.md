# Rain200L 去雨：代码与学习笔记

**本轮仅交付代码和必要笔记，按用户要求未安装、未运行、未训练、未测试、未导出 ONNX。**
所有性能指标、权重、训练截图与图像对比均需后续在游戏本或 AutoDL 实际生成。

基于出题人 [公开项目](https://github.com/Lingrui-Studio/Image-Deraining-Recruitment-Task) 的职责与接口完善。
`models/unet.py` 保留上游结构，许可证在 `upstream/LICENSE`；上游说明在 `upstream/README.md`。
本目录没有替你 Fork 或发布仓库；若正式提交，按上游说明自行 Fork 并提交源码和 Reports。

## 文件与任务

| 文件 | 内容 |
|---|---|
| models/baseline_net.py | 八层 3×3/64 通道残差块，首尾投影另计，预测雨层 |
| models/DerainNet.py | 题目规定的全/半/四分之一分辨率三分支，分支八层残差块 |
| models/MyModel.py | 两阶段共享权重、空洞卷积、多尺度和通道注意力，参数上限检查 |
| models/unet.py | 上游给定 U-Net，损失实验固定使用 |
| dataset.py | 按文件名配对、背景分组划分、同步增强，不缩放测试图 |
| losses/ | VGG19 前 35 层感知 L1、余弦对比及附加 InfoNCE |
| main.py | 训练、恢复、验证、最终测试、日志、best/last checkpoint |
| runtime.py / infer.py | 模型恢复、设备选择、全分辨率填充推理 |
| utils.py | 手写 PSNR、SSIM、可选官方 LPIPS、图像保存 |
| export_onnx.py | 参数统计与静态图导出，供 Netron 打开 |
| run_experiments.py | 顺序实验矩阵，默认只打印命令 |
| tests/test_core.py | 后续在目标机执行的行为测试 |
| Reports/ | 按小题整理的理论、实现与实验执行笔记 |

## 环境：游戏本与 AutoDL

推荐 Python 3.10–3.12，torch>=2.3 与匹配的 torchvision>=0.18。优先使用兼容的官方版本组合，
不要将 CPU 版 torch 复制到 CUDA 机器。在 [官方安装选择器](https://pytorch.org/get-started/locally/) 中选择平台与 CUDA wheel。
`nvidia-smi` 显示驱动支持上限，不是本机已安装的 Toolkit 版本；PyTorch wheel 自带所需 runtime。

Windows PowerShell：

```powershell
cd Lingrui/task_Deraining
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# 在此安装官方选择器给出的 torch / torchvision 组合
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__,torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

AutoDL：上传整个 `task_Deraining` 文件夹，在平台提供的兼容 PyTorch 环境中安装其他 requirements；
若自建 venv，使用 `python -m venv .venv` 和 `source .venv/bin/activate`。在持久数据盘保存数据和 runs，避免临时盘回收。
Windows 先用 `--num_workers 0`；Linux 可用 4。默认 auto 选择设备，建议正式 CUDA 实验显式指定 `--device cuda:0`。
单 GPU 已足够完成本代码；未提供 DDP。CUDA 环境可加 `--amp` 节省显存，VGG 特征损失仍在 float32 下计算。

## 指定数据集

下载上游指定 [Rain200L](https://drive.google.com/drive/folders/1wjEHV9D4UfL4VUS-IywKJDVX9iHcBR9J?usp=drive_link)，解压为：

```text
data/Rain200L/
  train/input/*.png
  train/target/*.png
  test/input/*.png
  test/target/*.png
```

input 与 target 文件名需一一对应且像素对齐，读取时检查尺寸。如果原始压缩包命名不同，先按数据说明整理映射，
不要仅通过排序配对。训练集按下划线前的背景编号分组，再按 seed 划分 90%/10% 训练/验证；检查文件名是否符合此规则。
验证集仅用于选择 best.pt，test 不用于调参。测试协议为完整分辨率 RGB、[0,1]、逐图平均、不裁边；与上游 resize128 结果不直接混比。
训练图像共享随机 crop/flip；尺寸不足先边缘填充。推理补到 8 的倍数，之后裁回原尺寸。

## 按题目训练

以下命令均在 `task_Deraining/` 目录执行。第一轮可去掉 `--amp` 建立 float32 参考，方法对比保持相同精度、patch、划分与种子。

```bash
python main.py --data_dir data/Rain200L --model baseline --epochs 100 --run_dir runs/baseline --device cuda:0 --amp
python main.py --data_dir data/Rain200L --model derainnet --epochs 50 --run_dir runs/derainnet --device cuda:0 --amp
python main.py --data_dir data/Rain200L --model mymodel --epochs 200 --run_dir runs/mymodel --device cuda:0 --amp
python main.py --data_dir data/Rain200L --model unet --epochs 100 --run_dir runs/unet_mse --device cuda:0 --amp
python main.py --data_dir data/Rain200L --model unet --epochs 100 --use_perceptual_loss --run_dir runs/unet_perceptual --device cuda:0 --amp
python main.py --data_dir data/Rain200L --model unet --epochs 100 --use_contrastive_loss --run_dir runs/unet_cosine --device cuda:0 --amp
python main.py --data_dir data/Rain200L --model unet --epochs 100 --use_contrastive_loss --contrastive_mode infonce --run_dir runs/unet_infonce --device cuda:0 --amp
```

VGG19 首次运行会下载较大的 ImageNet 权重，LPIPS 也会下载 AlexNet/感知权重；可提前在目标设备联网准备缓存，
或把缓存放在 `TORCH_HOME` 指定的持久目录。代码不会以随机 VGG 权重冒充预训练权重。
没有规定损失实验的固定轮数，这里统一使用 100，公平比较需要保持一致，不宣称这是题目硬性轮数。

统一打印/运行整套实验：

```bash
python run_experiments.py --data_dir data/Rain200L --device cuda:0 --amp
# 确认命令和算力预算后，在上条命令末尾加 --execute 才会真正训练。
```

脚本完成每种方法训练后使用 best.pt 在测试集计算 PSNR/SSIM/LPIPS。结构比较只使用 MSE；
损失比较不替换上游 U-Net。`--lambda_pixel`、`--lambda_perceptual`、`--lambda_contrastive` 可控制权重，默认 1/.1/.1。

## 恢复、测试、推理、导出

恢复必须带上与原运行相同的配置（含目标 epochs、损失开关等）；epochs 指完整目标轮数，epoch 表示已完成轮数。
为了调度一致，不支持在恢复时任意延长总轮数。恢复使用 checkpoint 所在的原 run_dir，保留 best.pt 与连续日志。
如需在其他机器继续，复制完整运行目录后再指定其中的 last.pt，不能只复制 last.pt 丢失已有 best.pt。

```bash
python main.py --data_dir data/Rain200L --model baseline --epochs 100 --run_dir runs/baseline --resume runs/baseline/last.pt --device cuda:0 --amp
python main.py --data_dir data/Rain200L --mode test --checkpoint runs/baseline/best.pt --run_dir runs/baseline --device cuda:0 --lpips
python infer.py --checkpoint runs/baseline/best.pt --input data/Rain200L/test/input/example.png --output outputs/example.png --device cuda:0
python export_onnx.py --model mymodel --checkpoint runs/mymodel/best.pt --output exports/mymodel.onnx
python -m unittest discover -s tests -v
```

example.png 替换为实际文件名。无 checkpoint 的 ONNX 导出只用于检查随机初始化架构，不能作为训练结果。
导出输入为固定 `(1,3,128,128)`；先在 [Netron](https://netron.app/) 打开图并保存截图。
不同尺寸推理由 PyTorch 的 runtime 填充处理，当前 ONNX 不承诺动态尺寸支持。

每次训练保存 config.json（参数量、划分、设备）、metrics.csv（原始损失项、验证指标、时间、峰值分配显存）、
best.pt、last.pt 和验证对比图。测试生成 test_metrics.json 与测试图。未开启 `--lpips` 时该字段为 null，不填 0。
对比图按从上到下为有雨、输出、干净。不要将测试指标当验证指标选择模型。

显存不足：先减小 batch_size，例如 2；再减 patch_size（至少 32 且为 8 的倍数），并记录配置变化。
完整图测试仍可能超过显存；本版未实现分块推理，可显式 `--device cpu` 完成测试，勿悄悄 resize 后与全图协议混比。
GPU 驱动与算子差异仍可能导致跨设备非逐位一致，固定种子不等于跨平台完全相同。

## 尚待实际完成

安装截图、完整训练、权重、指标、输出图片、ONNX 文件、Netron 截图、Apifox/云端运行截图均未生成。
代码和测试仅做静态检查。不要把本次代码交付写成已完成训练或已通过测试。
