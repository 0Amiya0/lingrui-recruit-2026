# Task 1：《Attention Is All You Need》Transformer 笔记

> 本文是基于论文 *Attention Is All You Need*（Vaswani et al., 2017）第 3 节 Model Architecture 的阅读笔记，
> 重点回答三个问题：Transformer 是如何工作的？每个子模块承担什么功能？为什么这样设计？

---

## 1. 整体架构：Encoder–Decoder

Transformer 最初是为**机器翻译**这类 sequence-to-sequence 任务设计的，因此整体是
**Encoder–Decoder** 结构：

```
            Encoder (N × Encoder Block)
 输入序列 ────────────────────────────────┐
                                          │ (最后一层 Encoder 的输出 K、V)
 输出序列 ──> Decoder (N × Decoder Block) ─┴──> 线性层 + Softmax ──> 逐 token 概率分布
```

- **Encoder**：读入整个**源序列**（如英文句子），把每个位置编码成一个上下文相关的向量表示。
  它是**双向**的：每个位置都能看到前后所有位置的信息（因为用的是普通的 Self-Attention，不加因果掩码）。
- **Decoder**：根据**已经生成的前缀**，一个 token 一个 token 地**自回归**生成目标序列（如中文句子）。
  它既看自己已生成的前缀（Masked Self-Attention），又看 Encoder 输出的源序列（Cross-Attention）。

nanoGPT 只用到了其中的 **Decoder** 部分（GPT 是 decoder-only 模型），
所以后文和 Task 2 会反复出现 Decoder Block，但理解完整的 Encoder–Decoder 结构仍然重要——
它解释了 Cross-Attention 的来历，以及为什么 GPT 没有 Cross-Attention（GPT 没有 Encoder 可 attend）。

---

## 2. Encoder Block 与 Decoder Block 的组成

一个 **Encoder Block** 有两大块，每块都带一个残差连接 + LayerNorm：

```
x → Multi-Head Self-Attention → 残差相加 → LayerNorm → FFN → 残差相加 → LayerNorm → out
                         (无掩码，双向)                        (逐位置 MLP)
```

一个 **Decoder Block** 比 Encoder Block 多一层 **Cross-Attention**，并且第一层 Self-Attention 带**因果掩码**：

```
x → Masked Self-Attention → 残差相加 → LayerNorm
  → Cross-Attention（K、V 来自 Encoder）→ 残差相加 → LayerNorm
  → FFN → 残差相加 → LayerNorm → out
```

要点：

- 原论文每个子层使用 **`LayerNorm(x + Sublayer(x))`（Post-LN）**；nanoGPT 则使用
  **`x + Sublayer(LayerNorm(x))`（Pre-LN）**。上面的架构图表示原论文，不能混用两种排列。
- Decoder 的 **Masked Self-Attention**：每个位置只能 attend 到它自己以及它**左边**的位置，
  从而保证训练时不会"偷看未来"。
- Decoder 的 **Cross-Attention**：Query 来自 Decoder 自己，Key/Value 来自 Encoder 的输出，
  这是 Decoder 从源序列里"取信息"的通道。

---

## 3. Q、K、V 的含义与 Attention 计算过程

Self-Attention 的核心可以概括为一句话：

> **让序列中每个 token 的表示，去"查"其他 token，按相关程度加权汇总它们的 Value。**

具体到符号上，每个 token 的向量 `x` 会通过三个可学习的线性变换，分别得到：

- **Q（Query，查询）**：表示"我（这个 token）在找什么"。相当于数据库查询里的查询条件。
- **K（Key，键）**：表示"我能提供什么、用来被匹配的标签"。相当于数据库里每条记录的索引键。
- **V（Value，值）**：表示"我实际携带的信息内容"。匹配上之后真正被取走、加权求和的东西。

用检索来类比：拿 **Q** 去和所有位置的 **K** 计算相似度，得到一组"注意力权重"，
再用这组权重对 **V** 做加权平均。**Q、K 决定"该看谁"，V 决定"看到的是什么"。**

### Scaled Dot-Product Attention（第 3.2.1 节）

论文给出的公式：

```
Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V
```

逐步骤拆解（设 Q、K 维度为 `d_k`，V 维度为 `d_v`，序列长度为 `T`）：

1. **算分数**：`scores = Q·Kᵀ`，形状 `(T, T)`。第 `i` 行第 `j` 列表示"位置 i 对位置 j 的原始相关分数"。
2. **缩放**：`scores / √d_k`。这是论文里 "Scaled" 的由来。原因：当 `d_k` 很大时点积的方差会变大，
   导致 softmax 输入落在梯度很小的饱和区，除以 `√d_k` 可以把方差拉回 1 附近，训练更稳定。
3. **归一化**：对每一行做 `softmax`，得到 `(T, T)` 的**注意力权重矩阵**，每一行和为 1，
   可解释为"位置 i 把注意力分配给了哪些位置、各占多少比例"。
4. **加权求和**：`softmax(·) · V`，得到每个位置的输出，形状 `(T, d_v)`。

### 为什么用 Q/K/V 而不是更简单的方式

如果所有位置直接对 "原始向量 x" 两两算相似度再加权，那"用来匹配"和"实际输出的内容"就被绑死了。
拆成 Q/K/V 三套投影后，模型可以学出：匹配时关注某一类特征（K 的投影方向），
而汇总时输出另一类特征（V 的投影方向），表达能力更强。

---

## 4. Multi-Head Attention 与单头 Attention 的区别（进阶）

单头 Attention 只有一组 Q/K/V 投影，整条序列只能学到**一种**"相关关系"。
但语言里"相关"有多种含义：句法上的主谓关系、指代关系、词义上的近义/反义、位置上的相邻关系……

**Multi-Head Attention** 的做法：把 `d_model` 维切成 `h` 份（每份 `d_k = d_model / h`），
每一份独立地做一次 Attention（一个"头"），最后把所有头的输出**拼接**起来再过一个线性变换：

```
MultiHead(Q,K,V) = Concat(head_1, ..., head_h) · W_O
head_i = Attention(Q·W_i^Q, K·W_i^K, V·W_i^V)
```

好处：

1. **每个头可以关注不同的子空间 / 不同的关系**：一个头看语法、一个头看指代、一个头看局部词序……
2. **总计算量几乎不变**：虽然分了 h 个头，但每个头的维度变成 `1/h`，总参数量和单头几乎一样，
   却获得了 h 套独立的注意力模式。
3. 单头是 `h=1` 的特例；多头的关键差异在于**并行地、在多个投影子空间里**做 Attention。

---

## 5. 为什么 Decoder 需要 Causal / Masked Attention（进阶）

Decoder 是**自回归**生成模型的训练方式：训练时我们希望模型学会"根据前 t 个 token 预测第 t+1 个 token"。
如果训练时允许第 t 个位置 attend 到第 t+1、t+2…… 个位置（未来），就相当于**泄题**：
模型可以直接抄答案，而学不会真正的"预测下一个词"。

所以 Decoder 的 Self-Attention 需要加一个**因果掩码（Causal Mask）**：

```
Attention 分数矩阵（softmax 前）     加性因果掩码           掩码后分数
  1  2  3  4                       0  -∞ -∞ -∞          1 -∞ -∞ -∞
  5  6  7  8                       0  0  -∞ -∞          5  6 -∞ -∞
  9 10 11 12                       0  0   0 -∞          9 10 11 -∞
 13 14 15 16                       0  0   0  0         13 14 15 16
```

实现上是把未来位置的分数设为 `-∞`，这样经过 softmax 后那些位置的权重就是 0。
一个下三角（`tril`）矩阵即可表达这个掩码，nanoGPT 里正是用 `torch.tril` 生成的（见 Task 2）。

> 与之对应，Encoder 不需要这个掩码，因为 Encoder 本来就可以（也应该）看到整个源序列。

---

## 6. Position Encoding 的必要性

不含位置编码、掩码等位置信息的 Self-Attention 是排列等变的：重排输入，输出会对应重排，
它无法据此区分 "A 在 B 前面" 和 "B 在 A 前面"。因果掩码已引入部分顺序约束，但不等同于显式位置编码。

因此 Transformer 需要显式地注入位置信息，即 **Positional Encoding**。论文用的是**正弦/余弦编码**：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- 每个位置 `pos` 有一个唯一的位置向量，不同频率的正余弦函数让不同维度编码不同尺度的位置信息。
- 它与 token embedding **相加**后进入模型，让模型"知道"每个 token 在第几个位置。
- 对固定偏移，正余弦位置向量之间可以通过线性变换联系；论文认为它可能支持更长序列的外推，
  但这不是实际长序列泛化的保证。论文也比较了学习式位置嵌入，效果接近。

（GPT/nanoGPT 用的是**可学习**的 Position Embedding `wpe`，思路一致、实现不同，见 Task 2。）

---

## 7. Feed-Forward Network（FFN / MLP）

每个 Attention 之后接一个**逐位置（position-wise）的全连接前馈网络**：

```
FFN(x) = ReLU(x·W1 + b1)·W2 + b2      （论文原文用 ReLU）
```

特点：

- 它**对每个位置独立作用**（同一套权重在所有位置共享），不跨位置交互——跨位置的交互已经由 Attention 完成。
- 中间层维度通常膨胀 4 倍（`d_model → 4·d_model → d_model`），提供更大的容量去做非线性变换。

分工可以这样理解：**Attention 负责"在位置之间搬运和混合信息"，FFN 负责"对每个位置的信息做非线性加工/记忆"**。
如果把 Attention 类比成"对话/交流"，FFN 就是"消化吸收"。GPT 家族用 GELU 替代 ReLU，见 Task 2。

---

## 8. Residual Connection 与 Layer Normalization

Transformer 里每个子层（Attention / FFN）都包了一层 **残差连接 + LayerNorm**：

```
out = LayerNorm( x + Sublayer(x) )     （Post-LN 写法）
```

- **Residual Connection（残差连接，`x + Sublayer(x)`）**：
  相当于给梯度一条"高速公路"，缓解深层网络训练时的梯度消失/爆炸，让很深的网络（几十上百层）也能稳定训练。
  同时它保证"最坏情况下这一层什么都不做也能把 x 原样传下去"。
- **Layer Normalization（层归一化）**：
  对**单个样本的隐藏维**做归一化（均值归 0、方差归 1，再乘可学习的 `γ`、加 `β`）。
  它稳定了每层的激活分布，让训练对学习率和初始化不那么敏感，加速收敛。
  和 BatchNorm 的差别在于：LayerNorm 沿**特征维度**归一化，不依赖 batch 内其他样本，天然适合序列/变长输入。

### 三个模块的分工总结

| 模块 | 功能 |
|------|------|
| **Attention** | 跨位置的信息交互：决定"每个位置从哪些位置拿信息、拿多少" |
| **MLP / FFN** | 每个位置内部的非线性变换与信息加工，提供容量 |
| **Residual Connection** | 梯度与信息的直通通道，让深层网络可训练 |
| **LayerNorm** | 稳定每层激活分布，加速并稳定训练 |

---

## 9. 一句话回顾整个数据流

```
输入 token 序列
   → Embedding（token 语义 + 位置编码相加）
   → Encoder：双向 Self-Attention → Add & Norm → FFN → Add & Norm
   → Decoder：Masked Self-Attention → Add & Norm → Cross-Attention → Add & Norm → FFN → Add & Norm
   → 线性层（映射回词表大小）
   → Softmax → 每个位置对下一个 token 的概率分布
```

训练时用这个概率分布和真实下一个 token 计算交叉熵损失；推理时按这个分布采样生成下一个 token。
这条数据流在 Task 2 的 nanoGPT 代码里会被逐行对应实现。

原论文用于条件翻译：Decoder 根据源序列与偏移后的目标前缀预测目标 token；nanoGPT 没有 Encoder
或 Cross-Attention，是纯自回归语言模型。上面展示的是模块对应关系，不能把二者视为完全相同的模型。

参考：[Attention Is All You Need，第 3 节](https://arxiv.org/html/1706.03762v7#S3)。
