# 与标准对比学习的关系，以及三选一实现

本项目选择“把感知变体改为 InfoNCE 并提供训练入口”这一项。未观看题目 B 站视频，
以下基于论文定义与源码整理，不写成已看完视频或已训练验证。

对比学习通常通过相关视图/样本形成正对，其他视图/样本形成负对，学习能分开它们的表示。
也有无显式负样本的方法，所以“所有对比学习都必须有负样本”不成立。
本任务固定预训练编码器、只训练图像恢复网络，区别于通常直接训练表示编码器的对比学习。
原余弦差值有拉近/推远思想，但没有 InfoNCE 的归一化分类形式。

这里保留同一个 anchor/positive/negative，构造最小的两候选 InfoNCE：

\[
L_{NCE}=-\log\frac{\exp(s(a,p)/\tau)}{\exp(s(a,p)/\tau)+\exp(s(a,n)/\tau)}.
\]

实现用 `cross_entropy(stack([positive_sim,negative_sim])/temperature, zeros)`，避免手工 exp 溢出。
temperature>0，越小对相似度差异越敏感；不是学习率，也不是噪声比例。
该版本只有一个负候选；可扩展到 batch 负样本，但同背景或语义相近的图像可能是假负样本，需要检查。

命令：`python main.py --data_dir data/Rain200L --model unet --use_contrastive_loss --contrastive_mode infonce --run_dir runs/unet_infonce --device cuda:0 --amp`。
实现与原余弦项共用 `EasyContrastiveLoss` 的 mode 参数。固定 U-Net、MSE 权重和数据划分，与不带特征项的 U-Net 比较。

参考：[CPC / InfoNCE](https://arxiv.org/abs/1807.03748)。
状态：公式、代码与命令已提供；题目要求的实际运行与结果分析需要你在目标 CUDA 设备上完成。
