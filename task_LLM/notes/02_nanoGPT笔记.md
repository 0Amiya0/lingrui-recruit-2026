# Task 2：nanoGPT 源码阅读笔记

> 项目地址：https://github.com/karpathy/nanoGPT
> 本文把 nanoGPT 的源码和论文里的 Transformer 结构一一对应起来，并追踪一个 token 的完整数据流。
> 文中行号以本仓库 `nanoGPT/` 下的源码为准。

---

## 1. 项目结构：每个文件负责什么

```
nanoGPT/
├── model.py          # 模型定义：GPT、CausalSelfAttention、MLP、Block、LayerNorm、GPTConfig
├── train.py          # 训练脚本：数据加载、优化器、学习率调度、训练循环、checkpoint 保存
├── sample.py         # 推理脚本：加载 checkpoint，自回归生成文本
├── configurator.py   # "穷人版"配置器：从命令行 --key=value 或配置文件覆盖全局变量
├── bench.py          # 基准测试脚本（对比不同模型的单次迭代耗时）
├── config/           # 各种预设配置：训练 GPT-2、finetune shakespeare、train shakespeare_char 等
│   ├── train_gpt2.py              # 从头训练 GPT-2 124M（OpenWebText）
│   ├── train_shakespeare_char.py  # 字符级莎士比亚小模型（调试用）
│   ├── finetune_shakespeare.py    # 用 GPT-2 权重 finetune 莎士比亚
│   └── eval_gpt2*.py              # 加载 GPT-2 权重做评估的配置
└── data/             # 数据准备脚本：把原始文本处理成 train.bin / val.bin / meta.pkl
    ├── shakespeare_char/prepare.py  # 字符级莎士比亚（本项目使用）
    ├── shakespeare/prepare.py       # GPT-2 BPE 词表级的莎士比亚
    └── openwebtext/prepare.py       # OpenWebText 大语料
```

### 各文件职责详解

- **model.py**：整个模型的**唯一定义**。从底层的 `LayerNorm`、`CausalSelfAttention`、`MLP`、`Block`，
  到顶层的 `GPT` 和配置类 `GPTConfig` 全部在这里。还包含 `generate()`（生成）、
  `configure_optimizers()`（构建 AdamW）、`estimate_mfu()`（估算算力利用率）、
  `from_pretrained()`（加载 GPT-2 官方权重）等方法。
- **train.py**：负责**整个训练流程**。用 `np.memmap` 实现"穷人版数据加载器" `get_batch`，
  构建模型和优化器，实现 cosine + warmup 的学习率调度，跑训练循环，
  周期性用 `estimate_loss()` 评估 train/val loss 并保存 checkpoint。
- **sample.py**：**推理/生成**入口。从 checkpoint 恢复模型，`model.eval()` 后调用 `model.generate()`
  自回归采样，再用 `meta.pkl` 里的 `itos` 把 id 解码回字符。
- **configurator.py**：一段被 `exec` 进 `train.py`/`sample.py` 的代码，读取 `--key=value` 参数
  或一个配置文件，**覆盖脚本里的同名全局变量**。这就是为什么所有超参数在 `train.py` 里只是普通变量。
- **data/**：`prepare.py` 把原始文本 `input.txt` 编码成 `train.bin`/`val.bin`（token id 的二进制数组），
  并把词表映射 `stoi`/`itos` 存进 `meta.pkl`。
- **config/**：纯配置文件，没有新逻辑，只是给 `configurator.py` 提供一组"预设超参数"。

---

## 2. nanoGPT 的模型结构（对应论文）

`GPT` 类（`model.py`）是一个 **decoder-only** 的 Transformer，结构如下：

```
GPT
 ├── Token Embedding      wte  : Embedding(vocab_size, n_embd)   ← 论文的 token embedding
 ├── Position Embedding   wpe  : Embedding(block_size, n_embd)   ← 论文的 position encoding（可学习版）
 ├── Dropout              drop
 ├── Transformer Block × n_layer
 │    ├── LayerNorm       ln_1   （Pre-LN）
 │    ├── Causal Self-Attention   ← 论文 Decoder 的 Masked Multi-Head Self-Attention
 │    ├── Residual Connection
 │    ├── LayerNorm       ln_2
 │    ├── MLP                      ← 论文的 Feed-Forward Network（GELU 版）
 │    └── Residual Connection
 ├── Final LayerNorm      ln_f
 └── LM Head              lm_head: Linear(n_embd, vocab_size, bias=False)  ← 映射回词表
```

### 与论文结构的对应与差异

| 论文（Attention Is All You Need） | nanoGPT（decoder-only） |
|---|---|
| Encoder 部分 | **没有**（GPT 是 decoder-only） |
| Decoder 的 Masked Self-Attention | `CausalSelfAttention` |
| Decoder 的 Cross-Attention | **没有**（没有 Encoder 可 attend） |
| 正弦/余弦 Positional Encoding | **可学习**的 `wpe`（`Embedding`） |
| FFN 用 ReLU | FFN 用 **GELU**（GPT-2 沿用下来的选择） |
| Post-LN（Sublayer 后归一化） | **Pre-LN**（先 `ln_1/ln_2` 再进子层） |
| 子层输出及 embedding 与位置编码之和使用 Dropout（基础模型 0.1） | embedding、注意力权重及子层输出使用可配 `dropout` |
| 第 3.4 节共享两种 embedding 和 pre-softmax 投影权重 | **Weight Tying**：`wte.weight` 与 `lm_head.weight` 共享 |

几个关键点的解释：

1. **Pre-LN**：`Block.forward` 是 `x = x + attn(ln_1(x))`、`x = x + mlp(ln_2(x))`，
   即先归一化再进子层。相比论文的 Post-LN，Pre-LN 在深网络上更稳定，是 GPT-2 之后的主流写法。
2. **Weight Tying**：输入词嵌入和输出 `lm_head` 共用同一份权重。
   直觉上"token 的语义向量"和"预测该 token 的 logits 方向"是同一件事，共享可减少参数量并提升泛化。
3. **Causal Self-Attention** 就是论文 Decoder 里的 Masked Self-Attention：
   `is_causal=True`（或手写 `torch.tril` 下三角掩码）保证第 `t` 个位置只看 `≤ t` 的位置。

### 各模块的作用（结合代码）

- **Token Embedding** `wte`：把每个 token id（整数）查表得到一个 `n_embd` 维向量，是可学习的语义表示。
- **Position Embedding** `wpe`：给位置 0..block_size-1 各一个可学习向量，加在 token 向量上注入顺序信息。
- **Causal Self-Attention**：`c_attn` 一次算出 Q/K/V（`n_embd → 3·n_embd`），分头做注意力，`c_proj` 把多头结果拼回。
- **MLP**：`c_fc`（`n_embd → 4·n_embd`）→ GELU → `c_proj`（`4·n_embd → n_embd`），逐位置非线性加工。
- **LayerNorm**：对隐藏维归一化（`model.py` 里用 `F.layer_norm`），稳定激活。
- **LM Head** `lm_head`：把最后一层隐藏状态线性映射到 `vocab_size` 维，得到每个位置对"下一个 token"的 logits。
- **Final LayerNorm** `ln_f`：所有 Block 之后、进 LM Head 之前再做一次归一化。

---

## 3. 一个 token 的完整旅程：从原始文本到 next-token prediction

这是理解整个项目最关键的一条主线：

```
raw text（莎士比亚剧本原文，约 111 万字符）
    ↓  data/shakespeare_char/prepare.py
tokenization（字符级：每个字符 = 一个 token）
    ↓  建立 stoi/itos 映射，vocab_size = 65
token ids（存成 train.bin / val.bin，uint16 数组）
    ↓  train.py 的 get_batch：随机取 block_size 个连续 id 作为 x，向后一个 token 起点的窗口作为 y
(B, T) 的 x、y 张量（B=64, T=block_size=128）
    ↓  model.forward(x)
Token Embedding + Position Embedding（相加，再 Dropout）
    ↓
Transformer Block × 4（每个 = Pre-LN → Causal Self-Attention → +残差 → Pre-LN → MLP → +残差）
    ↓
Final LayerNorm
    ↓
LM Head（n_embd → vocab_size）
    ↓
logits，形状 (B, T, vocab_size) = (64, 128, 65)
    ↓
训练：与真实下一个 token y 计算 Cross-Entropy Loss
推理：取最后一个位置的 logits → softmax → 采样下一个 token
```

逐步说明：

1. **tokenization（字符级）**：`prepare.py` 把整篇文本 `set` 出所有唯一字符（65 个，含大小写字母、
   标点、空格、换行），建立 `stoi`（char→int）和 `itos`（int→char）。所谓"token"在这里就是一个字符。
2. **token ids**：把整篇文本编码成一串整数，前 90% 存为 `train.bin`、后 10% 存为 `val.bin`。
3. **get_batch**：随机选 batch 个起点，每个起点取连续 `block_size` 个 id 作为输入 `x`，
   从下一个 token 起点取窗口（`data[i+1:i+1+block_size]`）作为标签 `y`，对所有位置做 next-token 预测。
4. **embedding**：`wte(x) + wpe(pos)`，得到 (B, T, C) 的隐藏状态，注入语义 + 位置信息。
5. **Transformer Blocks**：4 个 Block 反复"自注意力混合 + 残差 + MLP 加工"。
6. **LM Head**：最后一层隐藏状态过 `lm_head` 得到 logits。第 `t` 个位置的 logits 就是在给定
   前 `t` 个字符时，对第 `t+1` 个字符的**未归一化分数**。
7. **损失/采样**：训练时 `cross_entropy(logits.view(-1, V), y.view(-1))`，把 (B,T) 全部位置都当样本求平均；
   推理时模型只计算**最后一个位置**的 LM Head logits，softmax 后按概率采样一个 token，
   拼回序列，重复直到生成 `max_new_tokens` 个字符。

---

## 4. 【进阶】关键 Tensor 的 Shape 追踪

设定一组合理的超参数：

```
B      = batch size      = 64
T      = sequence length = 128   (block_size)
C      = n_embd          = 128   (embedding 维度)
n_head = 4                      (注意力头数)
hs     = C // n_head    = 32    (每个头的维度)
V      = vocab_size     = 65
```

### 4.1 输入与 Embedding

| Tensor | 形状 | 来源 |
|---|---|---|
| input token ids `idx` | `(B, T) = (64, 128)` | `get_batch` 取出的整数 |
| token embedding `tok_emb` | `(B, T, C) = (64, 128, 128)` | `wte(idx)`，查表 |
| position ids `pos` | `(T,) = (128,)` | `torch.arange(0, T)` |
| position embedding `pos_emb` | `(T, C) = (128, 128)` | `wpe(pos)` |
| 相加后 `x` | `(B, T, C)` | `tok_emb + pos_emb`（`(T,C)` 自动广播到 `(B,T,C)`） |

> 注意 `pos_emb` 是 `(T, C)`，因为位置嵌入对所有 batch 共享，广播即可。

### 4.2 Causal Self-Attention 内部（`model.py` 第 52–76 行）

进入时 `x` 形状是 `(B, T, C)`。

1. **一次线性变换算出 Q/K/V**：`c_attn(x)`，`c_attn` 是 `Linear(C, 3C)`，输出 `(B, T, 3C)`。
   再 `split(C, dim=2)` 沿最后一维切成三段，得到 `q, k, v` 各 `(B, T, C)`。
   > 为什么不分别用三个 Linear？把三个投影合并成一个 `3C` 的 Linear，只做一次大矩阵乘，更快。

2. **分头**：`q.view(B, T, n_head, hs).transpose(1, 2)`
   - `view(B, T, n_head, hs)`：把 `C=128` 维**重排**成 `(n_head=4, hs=32)`，注意这里是
     "前 32 个是头 0、接着 32 个是头 1……" 的**连续切片**，而不是交错切片。这个 view 能成立，
     正是因为 `c_attn` 输出的 Q/K/V 是按头**连续存放**的（等价于每个头有自己独立的投影）。
   - `transpose(1, 2)`：把 T 维和 n_head 维交换，得到 `(B, n_head, T, hs) = (64, 4, 128, 32)`。
     这样每个头独立、且可以在最后两维直接做矩阵乘。

| Tensor | 形状 |
|---|---|
| `q`（分头后） | `(B, n_head, T, hs) = (64, 4, 128, 32)` |
| `k`（分头后） | `(64, 4, 128, 32)` |
| `v`（分头后） | `(64, 4, 128, 32)` |

3. **注意力分数矩阵**：`q @ k.transpose(-2, -1)`
   `(B, nh, T, hs) × (B, nh, hs, T) → (B, nh, T, T) = (64, 4, 128, 128)`。
   `k.transpose(-2,-1)` 交换最后两维，把 `hs` 挪到中间用于点积求和。
   再 `* (1/√hs)` 做缩放（`hs=32`，即 `d_k`）。
   （代码里实际走 `F.scaled_dot_product_attention`，`is_causal=True` 内部做了同样的计算 + 因果掩码。）

| Tensor | 形状 |
|---|---|
| attention matrix（未归一化分数） | `(B, n_head, T, T) = (64, 4, 128, 128)` |
| 因果掩码 `bias`（非 flash 路径） | `(1, 1, block_size, block_size)`，`torch.tril` 下三角 |

   掩码把上三角（未来）位置置 `-inf`，softmax 后为 0，实现"只看过去"。

4. **注意力输出**：`softmax(...) @ v`
   `(B, nh, T, T) × (B, nh, T, hs) → (B, nh, T, hs) = (64, 4, 128, 32)`。

5. **拼回头**：`y.transpose(1, 2).contiguous().view(B, T, C)`
   - `transpose(1,2)`：`(B, nh, T, hs) → (B, T, nh, hs)`。
   - `.contiguous()`：transpose 后的张量在内存里不连续，先整理成连续存储，否则 `view` 会报错。
   - `.view(B, T, C)`：把 `(nh, hs)` 拼回 `(128,)`，恢复到 `(B, T, C)`。
   - 最后 `c_proj`（`Linear(C, C)`）+ dropout，输出仍 `(B, T, C)`。

### 4.3 MLP 与 Block

| Tensor | 形状 |
|---|---|
| MLP 输入（`ln_2` 之后） | `(B, T, C)` |
| `c_fc` 输出 | `(B, T, 4C) = (64, 128, 512)` |
| GELU 输出 | `(B, T, 4C)` |
| `c_proj` 输出（MLP 输出） | `(B, T, C)` |
| Block 输出（残差相加后） | `(B, T, C)` |

> MLP 里维度先膨胀 4 倍再缩回，对应论文 FFN 的 `d_model → 4·d_model → d_model`。
> 每个 Block 前后形状不变，所以可以任意堆叠 `n_layer` 层。

### 4.4 输出与损失

| Tensor | 形状 |
|---|---|
| Final LayerNorm 输出 | `(B, T, C)` |
| logits（`lm_head` 输出） | `(B, T, V) = (64, 128, 65)` |
| 训练时 `logits.view(-1, V)` | `(B·T, V) = (8192, 65)` |
| 训练时 `targets.view(-1)` | `(B·T,) = (8192,)` |
| 推理时 `logits[:, [-1], :]` | `(B, 1, V)`，只取最后一步 |

训练损失：`F.cross_entropy(logits.view(-1, V), targets.view(-1))`。
把 `(B, T)` 的每个位置都摊平，当作独立的分类样本——这是语言模型"预测下一个 token"的标准做法。
`logits` 的最后一维就是词表大小 65，正好对应 65 个字符的分类问题。

输入 `x=[a,b,c]` 时，目标是 `y=[b,c,d]`，并非往张量前面补一个占位符。
因果掩码使第 0 个位置只看 a 预测 b，第 1 个位置看 a,b 预测 c，第 2 个位置看 a,b,c 预测 d。
所以训练一次前向就对所有位置计算 next-token loss，并不是每个位置都看完整的 128 字符。

没有 targets 时，`GPT.forward` 返回的是 `(B,1,V)`，不是 `(B,T,V)` 后再由调用方切片。
注意力内变量 `self.flash` 实际表示 SDPA API 是否可用；CPU 有这个 API，不代表使用了 CUDA FlashAttention 内核。

## 6. 训练与生成中值得核对的细节

- **权重共享**：`transformer.wte.weight` 与 `lm_head.weight` 是同一 Parameter，输入词向量和输出分类共享参数。
- **训练目标**：交叉熵内部处理 logits，不需要手动先 softmax；损失是有效位置的平均负对数概率。
- **优化**：AdamW 更新参数，梯度裁剪限制梯度范数；梯度累积将多个 micro-batch 的贡献汇总到一次更新。
- **评估**：`estimate_loss()` 切换到 eval 模式关闭 dropout，评估后恢复 train 模式；每次随机抽取窗口，验证 loss 仍有采样波动。
- **checkpoint**：保存模型、优化器、模型配置、迭代编号、best_val_loss 与训练配置，resume 可以恢复训练状态。
- **配置**：`configurator.py` 用 exec 执行可信本地配置，再接受 `--key=value` 覆盖；未知配置键会报错。
- **生成**：最后位置 logits 除以正温度，再保留 top-k、softmax、multinomial 采样。词表仅 65，top_k=200 会被截到 65，实际上不做 top-k 筛选。
- **上下文裁剪**：生成可以超过 block_size，但每次仅将最近 block_size 个 token 输入模型，不会扩展位置嵌入表。
- **停止边界**：当前源码先保存编号 max_iters 的评估 checkpoint，再做该编号更新，然后在 iter_num > max_iters 时结束。

源码基准：`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`；对应本地 `model.py`、`train.py`、`sample.py`、`configurator.py`。

---

## 5. 回答："一个 token 从进入 nanoGPT 到产生 next-token prediction 经历了什么？"

1. 原始文本在 `prepare.py` 里被**字符级 tokenize**，变成整数 id，存进 `.bin` 文件。
2. `get_batch` 取出形状 `(B, T)` 的 id 张量 `x`。
3. `x` 经过 `wte` 变成 token embedding，加上 `wpe` 的 position embedding（相加，dropout）。
4. 依次穿过 `n_layer` 个 `Block`：每个 Block 内先 `ln_1` 再做 **Causal Self-Attention**（Q/K/V 投影 → 分头 →
   缩放点积 → 因果掩码 → softmax → 加权 V → 拼头 → 输出投影），残差相加；再 `ln_2` 进 **MLP**，残差相加。
5. 过 `ln_f` 做最终归一化。
6. 过 `lm_head` 得到 `(B, T, V)` 的 logits——每个位置都是对**下一个 token**的分数。
7. 训练：与真实标签 `y` 算交叉熵；推理：取最后位置 logits → softmax → 采样 → 拼回序列继续生成。
