任务目标

本部分希望你从 Transformer 的基本原理出发，进一步理解一个真实 GPT 项目的代码实现，并最终完成模型的部署、训练和文本生成。

完成本部分后，你应当能够回答三个问题：

    Transformer 是如何工作的？
    论文中的 Transformer 结构，在实际 GPT 代码中是如何实现的？
    一个语言模型从数据处理、模型定义、训练到推理生成，完整流程是什么？

不要求在开始本任务前已经学习过 Transformer 或 PyTorch，但要求你能够通过论文、文档、源码和其他公开资料完成自主学习。
Task 1：阅读《Attention Is All You Need》，理解 Transformer

阅读论文 Attention Is All You Need，重点理解论文第 3 节 Model Architecture，包括：

    Encoder 与 Decoder 的整体结构；
    Self-Attention、Masked Self-Attention、Cross-Attention 的作用；
    Scaled Dot-Product Attention 的计算过程；
    Multi-Head Attention 的结构与作用；
    Position Encoding 的必要性；
    Feed-Forward Network；
    Residual Connection 与 Layer Normalization。

提交要求

请根据自己的理解，完善transformer的笔记内容。

可供参考的方面：

    Transformer Encoder–Decoder 架构；
    Encoder Block 和一个 Decoder Block 的主要组成部分；
    说明 (Q)、(K)、(V) 的含义以及 Attention 的基本计算过程；
    【进阶】解释为什么 Decoder 中需要 Causal / Masked Attention；
    【进阶】解释 Multi-Head Attention 与单头 Attention 的区别；
    【进阶】说明 Attention、MLP、Residual Connection、LayerNorm 各自承担什么功能。

Task 2：阅读 nanoGPT 源码，并与论文架构进行对应

https://github.com/karpathy/nanoGPT/ git clone nanoGPT项目，阅读项目源码。

请至少阅读并理解以下内容：
复制代码

nanoGPT/
├── model.py
├── train.py
├── sample.py
├── configurator.py
├── config/
└── data/

其中重点阅读：
复制代码

model.py
train.py
sample.py

阅读过程中，请尝试回答：

    “一个 token 从进入 nanoGPT，到最终产生 next-token prediction，中间经历了什么？”

你需要理解并能够追踪下面这条基本数据流：
复制代码

raw text
    ↓
tokenization
    ↓
token ids
    ↓
token embedding + position embedding
    ↓
Transformer Blocks
    ↓
Causal Self-Attention
    ↓
MLP
    ↓
LM Head
    ↓
logits
    ↓
Cross-Entropy Loss / Sampling

提交要求

完善一份关于 nanoGPT 项目结构与模型流程的笔记

可供参考的方面：

1. 项目结构

说明以下文件分别负责什么：
复制代码

model.py
train.py
sample.py
configurator.py
data/
config/

2. nanoGPT 模型结构

根据 model.py，整理出：
复制代码

GPT
 ├── Token Embedding
 ├── Position Embedding
 ├── Transformer Block × N
 │    ├── LayerNorm
 │    ├── Causal Self-Attention
 │    ├── Residual Connection
 │    ├── LayerNorm
 │    ├── MLP
 │    └── Residual Connection
 ├── Final LayerNorm
 └── LM Head

并说明其中各模块的作用。

3. 【进阶】Tensor Shape

任选一组合理的：
复制代码

B       batch size
T       sequence length
C       embedding dimension
n_head  attention heads

追踪下面几个关键 tensor 的 shape：
复制代码

input token ids
token embedding
Q / K / V
multi-head Q / K / V
attention matrix
attention output
MLP input/output
logits

要求能够解释每一次 view、transpose、split 等操作为什么成立，而不仅仅记录最终结果。
Task 3：部署并训练 nanoGPT

在本地计算机或服务器上完成 nanoGPT 的环境配置和运行。

建议首先使用项目提供的 Shakespeare character-level dataset 完成一次小规模训练。

你需要完成：
复制代码

数据准备
    ↓
训练
    ↓
保存 checkpoint
    ↓
加载 checkpoint
    ↓
文本生成

实验要求

训练至少一个可以正常收敛并生成文本的 nanoGPT 模型。

请记录：
复制代码

运行环境
Python / PyTorch 版本
CPU / GPU 型号
训练配置

复制代码

【进阶】
n_layer
n_head
n_embd
block_size
batch_size
learning_rate
max_iters

同时记录训练过程中：
复制代码

train loss
validation loss
训练时间

并使用训练得到的 checkpoint 完成文本生成。
提交结果

至少提交：

1. 环境部署结果

证明 nanoGPT 可以正常运行。

2. 数据准备结果

说明训练数据经过了怎样的处理，以及模型实际接收到的数据是什么。

3. 训练结果

提供至少一段能够体现 loss 变化的训练日志或可视化结果。

4. 文本生成结果

使用训练得到的模型生成一段文本。

【进阶】：实验分析

    训练开始和结束时 loss 有什么变化？
    生成文本相比未训练模型有什么变化？
    n_layer、n_head、n_embd 分别控制模型的什么属性？
    block_size 表示什么？
    train loss 和 validation loss 分别反映什么？

最终提交内容

将本部分整理成一份笔记+实验文档

允许参考论文、PyTorch 文档、技术博客、课程和 AI 工具，但最终提交内容应体现你自己的理解。