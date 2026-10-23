# 损失与评价指标

误差向量 e：L1 范数为 sum|e_i|，表示到原点的曼哈顿距离；L2 范数为 sqrt(sum e_i²)，是欧氏距离。
e=(3,4) 时 L1=7、L2=5。图像训练通常使用 MAE=mean|e_i| 和 MSE=mean e_i²，MSE 是平均平方误差，
并非直接取 L2 范数。本项目结构实验使用 `F.mse_loss`，对 NCHW 所有元素取平均。

L1 的梯度对非零误差为符号项，L2 平方项对大误差惩罚更强；各自适合程度取决于数据、目标和噪声。
像素损失要求配对对齐；两张视觉相同但偏移一像素的图片，也可能产生很大的误差。

对 RGB [0,1] 图像，MSE 在全部像素和通道上计算：

\[
MSE=\frac{1}{3HW}\sum_{h,w,c}(\hat B_{hwc}-B_{hwc})^2,\qquad
PSNR=10\log_{10}\frac{MAX^2}{MSE}.
\]

MAX=1；uint8 未归一化图像才用 255。MSE=0 时 PSNR=+∞。PSNR 越高越好，
它是 MSE 的单调变换，无法单独说明纹理、语义或人类观感。实现位于 `utils.psnr`，采用 float64 计算误差。
最终先逐图计算再平均，不先跨图平均 MSE 后转换，二者不等价。

SSIM 在局部窗口比较亮度、对比度和结构，再汇总各窗口：

\[
SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}.
\]

标准形式理论范围为 [-1,1]，自然图像常见结果在 [0,1]，越高越好；不能把 0 说成理论下界。
本代码使用 RGB channel_axis=2、data_range=1、最大窗口 7，不使用过时 multichannel 参数。
评价单位是局部窗口而非单个孤立像素。PSNR/SSIM 不能替代视觉检查，需同时查看有雨、输出与真实图。

LPIPS 越低越好；使用官方 lpips 库的 AlexNet 实现，对 [0,1] 图像映射到 [-1,1]。
LPIPS 是归一化深层特征差异及学习权重的组合，不等于“VGG 特征简单 L1”。题目的 VGG19 L1 是感知训练损失，
与 LPIPS 有关联但不是同一个公式。

参考：[SSIM 原始作者页面](https://ece.uwaterloo.ca/~z70wang/research/ssim/)、
[LPIPS 论文](https://arxiv.org/abs/1801.03924)、[LPIPS 官方代码](https://github.com/richzhang/PerceptualSimilarity)。
本轮未计算任何真实图像指标。
