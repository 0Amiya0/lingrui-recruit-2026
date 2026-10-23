# 像素损失与 VGG 感知损失

像素损失逐位置度量颜色误差，不能直接衡量边缘/纹理/对象结构；对齐误差可导致大惩罚。
但不应说像素损失“完全没有空间信息”：有空间上下文的网络仍可从像素监督学习结构，只是损失公式本身不显式建模邻域。
感知损失将两图经过固定的特征提取器，再比较特征。一个特征位置的感受野覆盖多个输入像素，
特征通道也响应不同模式，因此特征上的逐元素 L1 与原图逐像素 L1 的含义不同。

使用 torchvision VGG19 ImageNet 预训练权重的 `features[:35]`，即 Python 索引 0–34，
并将 RGB [0,1] 图按 ImageNet mean/std 归一化。它不是“图像变成更清晰的图片”，而是高维特征图，
不能直接当 RGB 图展示。可对单通道或通道范数做可视化，但本轮未生成这类图。

\[
L_{perc}=\frac{1}{CHW}\|\phi(\hat B)-\phi(B)\|_1,\qquad
L=L_{MSE}+\lambda_p L_{perc}.
\]

VGG 参数被冻结、保持 eval；预测图走 VGG 的路径仍需记录梯度，不能整体 no_grad。
干净目标分支可 no_grad，减少显存。不要将预测特征 detach，否则感知损失无法指导去雨网络。
本实现外层在 float32 计算特征损失，避免 AMP 下特征距离数值不稳；网络前向仍可使用 AMP。

训练必须固定上游 U-Net，用与 U-Net MSE 基线相同的划分、seed、patch、epochs 和精度比较。
记录 PSNR/SSIM/LPIPS 和原图对照；感知质量与 PSNR 不一定同向变化，不能先假定必然提升。
命令见根 README，原始 p_loss 和加权总损失分别写入 metrics.csv。

参考：[感知损失原论文](https://arxiv.org/abs/1603.08155)、
[torchvision VGG19](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg19.html)。
状态：实现已写，预训练权重尚未在本轮下载，未训练、未产出指标。
