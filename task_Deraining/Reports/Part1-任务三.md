# 环境、八层 baseline 与训练入口

目标环境与安装步骤见根 README；当前只写代码，尚无安装或训练截图。
训练命令：`python main.py --data_dir data/Rain200L --model baseline --mode train --epochs 100 --device cuda:0 --amp --run_dir runs/baseline`。

baseline 的中间残差块为 8 层 3×3、64→64 卷积；另加输入 3→64 和输出 64→3 投影各一层，总卷积数 10。
所有卷积 stride=1、padding=1。这里按题意把“8 层”计为残差块内部层数，首尾投影另计；
三尺度模型每个分支也使用深度为 8 的残差块，避免混淆两种计数口径。

核心实现摘录（完整 ResidualBlock 与 import 见 `models/baseline_net.py`）：

```python
class BaselineNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3,64,3,padding=1), nn.ReLU(inplace=True))
        self.body = ResidualBlock(64,depth=8)
        self.head = nn.Conv2d(64,3,3,padding=1)

    def forward(self,x):
        return x-self.head(self.body(self.stem(x)))
```

串联八层 3×3、stride1 的块本身理论感受野最大为 17×17，加首尾卷积后为 21×21；实际有效感受野通常更集中。
不改变分辨率的扩大方式包括空洞卷积、较大卷积核、窗口/全局注意力；多尺度编码后上采样也能扩大原图尺度的上下文。
残差连接有助于梯度传播，但不保证任意深度都能稳定训练；下采样会损失高频细节，需通过跳连和融合弥补。

代码学习要点：argparse 管理配置；Dataset 负责成对读取；DataLoader 负责采样与批次；
train/eval 与自动微分开关含义不同；zero_grad/backward/step 分别清梯度、求梯度、更新参数。
CUDA 张量与模型需在同一设备，AMP 下先 unscale 再裁梯度。测试集不能每轮拿来选模型。
本项目从 train 内留验证，best.pt 以验证 PSNR 选择，test 模式再读 test 文件夹。

训练输出：config.json、metrics.csv、last.pt、best.pt、val_images。PSNR 实现在 utils.py，可在目标机执行 tests/test_core.py。
需后续保存完整 100 轮日志、终端/环境截图、真实 checkpoint、最终测试指标与对比图；当前均未生成。
