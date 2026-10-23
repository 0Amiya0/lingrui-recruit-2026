# 向量、池化与余弦对比损失

向量是一组有序数，可代表空间位移或学习得到的特征；维度是分量数，不是 batch 或图像张数。
内积 a·b=sum a_i b_i=||a||||b||cos(theta)，矩阵乘法由多次内积组成，说明它与线性变换密切相关。
余弦相似度 cos(a,b)=a·b/(||a||||b||)，范围 [-1,1]，衡量方向而非长度；零向量需 eps 防止除零。

VGG 输出 N×C×h×w，通过 AdaptiveAvgPool2d(1) 变为 N×C×1×1，再 flatten(1) 得 N×C，
每个通道对空间位置平均。它适配不同 h/w，但也丢失精确空间布局；局部雨纹的差异可能因此被稀释。
本项目用 F.normalize 得单位向量，点积即可计算余弦相似度。

定义 anchor 为去雨输出特征 a，positive 为干净目标特征 p，negative 为有雨输入特征 n：

\[
L_{cos}=1-\cos(a,p)+\cos(a,n),\qquad
L=L_{MSE}+\lambda_c L_{cos}.
\]

最小化时希望输出靠近干净、远离有雨。这个损失可能为负，没有非负损失的普遍要求；
关注梯度方向而不是只看数字大小。预测路径有梯度，输入和目标分支不记录梯度，编码器冻结。

关键局限：有雨和无雨图共享大量背景，特征可能非常接近，不能先假定两者余弦很小。
强行远离有雨图可能也损害真实背景；这种负样本定义需要实验验证，可以尝试局部雨纹特征或 margin。
即使 loss 数值降低也不保证图像更好，需检查 PSNR/SSIM/LPIPS 与可视化。

命令：`python main.py --data_dir data/Rain200L --model unet --use_contrastive_loss --run_dir runs/unet_cosine --device cuda:0 --amp`。
实现：losses/easy_contrastive_loss.py；状态：未运行。
