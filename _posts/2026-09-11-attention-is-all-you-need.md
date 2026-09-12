---
title: "Attention is all you need, transformer 学习笔记"
date: 2026-09-11 17:00:00 +0800
categories: [论文精读, 深度学习]
tags: [transformer, attention, nlp, 论文精读, 大模型]
math: true
mermaid: false
toc: true
---

> 一份深入的中文阅读笔记。文中所有专有名词首次出现时附英文，文末附「专有名词中英对照附录」。论文全部 5 张配图（Figure 1–5）均已嵌入并配中文解读。

---

## 0. 论文基本信息

| 项目 | 内容 |
| :--- | :--- |
| 标题 | **Attention Is All You Need**（注意力就是你所需要的一切） |
| 作者 | Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin |
| 机构 | 谷歌大脑（Google Brain）、谷歌研究院（Google Research）、多伦多大学（University of Toronto） |
| 发表 | 第 31 届神经信息处理系统大会（**NIPS 2017**），美国加州长滩；arXiv:1706.03762 |
| 地位 | 提出 **Transformer** 架构，是现代大语言模型（LLM）的奠基性工作 |

---

## 1. 摘要（Abstract）

- **现状**：当时主流的序列转录模型（sequence transduction model）依赖复杂的循环神经网络（Recurrent Neural Network, RNN）或卷积神经网络（Convolutional Neural Network, CNN），最好的模型还会用注意力机制（attention mechanism）连接编码器（Encoder）与解码器（Decoder）。
- **创新**：提出一种全新的、简单的网络架构——**Transformer**，它**完全基于注意力机制**，彻底抛弃了循环（recurrence）与卷积（convolution）。
- **优势**：翻译质量更高、并行性更好、训练时间显著缩短。
- **SOTA 成绩（State-of-the-Art，最优结果）**：
  - **WMT 2014 英译德**：达到 **28.4 BLEU**，比包括集成模型（ensemble）在内的既有最优结果高出 2 个 BLEU 以上。
  - **WMT 2014 英译法**：单模型达到 **41.8 BLEU** 的新 SOTA，仅在 8 块 GPU 上训练 3.5 天，训练成本远低于文献最优模型。
- **泛化**：将 Transformer 成功迁移到英语成分句法分析（English constituency parsing），在大数据与小数据场景下均表现优异。

---

## 2. 引言（Introduction）：为什么要抛弃 RNN

**RNN 的根本缺陷——顺序计算无法并行。**

RNN（如 LSTM、GRU）沿序列位置逐步计算，把位置与计算时间步对齐，逐个生成隐藏状态：

$$
h_t = f(h_{t-1},\, x_t)
$$

- 这种**固有的顺序性（inherently sequential nature）**导致无法在单个训练样本内部并行——第 $t$ 步必须等第 $t-1$ 步算完。
- 序列越长，内存压力越大，跨样本批处理（batching）也受限，成为计算瓶颈。

**注意力机制的价值：** 注意力可以对依赖关系建模而**不受序列距离影响**，但过去几乎总是与 RNN 搭配使用，没能摆脱顺序计算的枷锁。

**Transformer 的主张：** 完全依赖注意力机制来刻画输入与输出之间的全局依赖（global dependencies），从而实现大规模并行。仅在 8 块 P100 GPU 上训练 12 小时，就能达到翻译质量的 SOTA。

---

## 3. 背景（Background）：与已有工作的对比

- **减少顺序计算的前人尝试**：Extended Neural GPU、ByteNet、ConvS2S 都用 CNN 作为基本模块并行计算隐藏表示。
  - **局限**：关联任意两个位置信号所需的操作数随距离增长——ConvS2S 是线性增长，ByteNet 是对数增长，这使得**学习远距离依赖（long-range dependencies）更加困难**。
  - **Transformer 的改进**：把该操作数降为**常数 $O(1)$**。代价是注意力加权平均会降低有效分辨率，但用多头注意力（Multi-Head Attention）来弥补。
- **自注意力（Self-Attention，又称 intra-attention）**：此前已用于阅读理解、摘要等任务，但 Transformer 是**第一个完全依赖自注意力**来计算输入输出表示、不使用任何序列对齐 RNN 或卷积的转录模型。

---

## 4. 模型架构（Model Architecture）

### 4.1 整体：编码器–解码器结构

Transformer 沿用编码器–解码器（Encoder-Decoder）的整体框架，但两侧都由堆叠的自注意力层和逐位置全连接层组成。下图左半为编码器，右半为解码器：

![Figure 1: Transformer 整体架构图](/assets/posts/attention/figure1_architecture.png)

**图 1 解读（The Transformer — model architecture）：**

- **左侧编码器（Encoder，左侧 N× 堆叠）**：
  - 底部：输入嵌入（Input Embedding）+ 位置编码（Positional Encoding）相加。
  - 每层含两个子层：**多头注意力（Multi-Head Attention）** → **前馈网络（Feed Forward）**。
  - 每个子层都套一层「Add & Norm」（残差连接 Residual Connection + 层归一化 Layer Normalization）。
- **右侧解码器（Decoder，右侧 N× 堆叠）**：
  - 底部：输出嵌入（Output Embedding，右移一位 shifted right）+ 位置编码。
  - 每层含**三个子层**：**带掩码多头注意力（Masked Multi-Head Attention）** → **编码器-解码器注意力（Multi-Head Attention，Q 来自解码器、K/V 来自编码器）** → **前馈网络**，每个子层同样套「Add & Norm」。
  - 顶部：Linear（线性层）→ Softmax → 输出概率（Output Probabilities）。

**结构参数：**

- 编码器：$N = 6$ 层相同结构堆叠。每层 = 多头自注意力 + 逐位置前馈网络。
- 解码器：$N = 6$ 层相同结构堆叠。比编码器多一个「编码器-解码器注意力」子层。
- 残差连接公式：每个子层输出为

$$
\mathrm{LayerNorm}\bigl(x + \mathrm{Sublayer}(x)\bigr)
$$

- 为方便残差相加，所有子层与嵌入层的输出维度统一为 $d_{model} = 512$。
- **解码器的掩码（Masking）机制**：修改解码器的自注意力，禁止某位置「看到」它后面的位置（把非法连接的注意力打分置为 $-\infty$）。配合输出嵌入右移一位，保证位置 $i$ 的预测只依赖于位置小于 $i$ 的已知输出——这就是自回归（auto-regressive）特性。

### 4.2 注意力机制（Attention）

注意力函数可以描述为：把一个查询（Query）和一组键值对（Key-Value pairs）映射为一个输出，其中查询、键、值、输出都是向量。输出是值的加权和，权重由查询与对应键的兼容性函数（compatibility function）计算得到。

![Figure 2: 缩放点积注意力（左）与多头注意力（右）](/assets/posts/attention/figure2_attention.png)

**图 2 解读：**

- **左图 缩放点积注意力（Scaled Dot-Product Attention）**：数据流为 Q、K 做矩阵乘（MatMul）→ 缩放（Scale）→ 可选掩码（Mask (opt.)）→ SoftMax → 再与 V 做矩阵乘（MatMul）。
- **右图 多头注意力（Multi-Head Attention）**：V、K、Q 各自经过 $h$ 组线性层（Linear）投影 → 并行送入多个缩放点积注意力 → 拼接（Concat）→ 再过一层线性层（Linear）。图中 `h` 标注表示这套注意力并行运行 $h$ 份。

#### 4.2.1 缩放点积注意力（Scaled Dot-Product Attention）

输入为维度 $d_k$ 的查询与键、维度 $d_v$ 的值。计算查询与所有键的点积，除以 $\sqrt{d_k}$，经 softmax 得到权重，再对值加权求和：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V
$$

- **为什么要除以 $\sqrt{d_k}$（缩放因子）**：当 $d_k$ 很大时，点积结果数值会变大，把 softmax 推入梯度极小的饱和区。除以 $\sqrt{d_k}$ 可以抵消这一效应，稳定梯度。
- **加性注意力 vs 点积注意力**：加性注意力（additive attention）用带单隐层的前馈网络计算兼容性；点积注意力（dot-product attention）理论复杂度相同，但可用高度优化的矩阵乘法实现，实践中更快、更省空间。

#### 4.2.2 多头注意力（Multi-Head Attention）

与其用 $d_{model}$ 维做单次注意力，不如把 Q、K、V 用不同的、可学习的线性投影分别映射到 $d_k$、$d_k$、$d_v$ 维，做 $h$ 次并行注意力，再拼接、投影：

$$
\mathrm{MultiHead}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\,W^{O}
$$

$$
\text{其中} \quad \mathrm{head}_i = \mathrm{Attention}\bigl(QW_i^{Q},\, KW_i^{K},\, VW_i^{V}\bigr)
$$

- 投影矩阵：$W_i^{Q} \in \mathbb{R}^{d_{model}\times d_k}$，$W_i^{K} \in \mathbb{R}^{d_{model}\times d_k}$，$W_i^{V} \in \mathbb{R}^{d_{model}\times d_v}$，$W^{O} \in \mathbb{R}^{hd_v\times d_{model}}$。
- **为什么用多头**：让模型能同时关注来自**不同表示子空间（different representation subspaces）**、不同位置的信息。单头注意力的加权平均会抑制这种能力。
- **参数设置**：$h = 8$，$d_k = d_v = d_{model}/h = 64$。因每个头维度降低，总计算量与全维度单头注意力相近。

#### 4.2.3 模型中三处注意力的用法

1. **编码器-解码器注意力（Encoder-Decoder Attention / Cross-Attention）**：查询 Q 来自解码器上一层，键 K 与值 V 来自编码器输出。让解码器每个位置都能关注输入序列的所有位置（模仿传统 seq2seq 的注意力）。
2. **编码器自注意力（Encoder Self-Attention）**：Q、K、V 均来自编码器上一层输出。每个位置可关注上一层所有位置。
3. **解码器带掩码自注意力（Masked Decoder Self-Attention）**：Q、K、V 来自解码器上一层，但只允许关注到当前位置及之前——通过掩码把「向右看」的非法连接置为 $-\infty$，保证自回归特性。

### 4.3 逐位置前馈网络（Position-wise Feed-Forward Networks, FFN）

每层还包含一个前馈网络，独立且相同地作用于每个位置（等价于两个 kernel size = 1 的卷积），中间用 ReLU 激活：

$$
\mathrm{FFN}(x) = \max(0,\, xW_1 + b_1)\,W_2 + b_2
$$

- 输入/输出维度 $d_{model} = 512$，内层维度 $d_{ff} = 2048$。

### 4.4 嵌入与 Softmax（Embeddings and Softmax）

- 用可学习的嵌入（embedding）把词元（token）转成 $d_{model}$ 维向量。
- **权重共享（weight tying）**：两个嵌入层与 softmax 前的线性变换**共享同一权重矩阵**。
- **缩放**：在嵌入层里把权重乘以 $\sqrt{d_{model}}$。

### 4.5 位置编码（Positional Encoding）

由于模型没有循环也没有卷积，必须显式注入序列的位置信息。做法是把位置编码（维度同为 $d_{model}$）与输入嵌入**相加**。论文采用不同频率的正弦、余弦函数：

$$
PE_{(pos,\, 2i)} = \sin\!\left(\frac{pos}{10000^{\,2i/d_{model}}}\right)
$$

$$
PE_{(pos,\, 2i+1)} = \cos\!\left(\frac{pos}{10000^{\,2i/d_{model}}}\right)
$$

- 其中 $pos$ 是位置，$i$ 是维度索引；波长构成从 $2\pi$ 到 $10000\cdot 2\pi$ 的等比数列。
- **为什么这样设计**：对任意固定偏移 $k$，$PE_{pos+k}$ 都能表示为 $PE_{pos}$ 的线性函数，便于模型学习「按相对位置关注」；且能外推到比训练时更长的序列。
- 实验（见表 3 的 E 行）表明，用可学习的位置嵌入（learned positional embedding）效果几乎一致，作者选正弦版本是因为它能外推更长序列。

---

## 5. 为什么用自注意力（Why Self-Attention）

作者从三个维度对比自注意力、循环、卷积三类层：

1. **每层总计算复杂度**（Complexity per Layer）
2. **可并行的计算量**——用「最少所需顺序操作数」（Sequential Operations）衡量，越少越好
3. **网络中长程依赖的最大路径长度**（Maximum Path Length）——路径越短，越容易学习长程依赖

**表 1（Table 1）：不同层类型的对比**

| 层类型（Layer Type） | 每层复杂度 | 顺序操作数 | 最大路径长度 |
| :--- | :--- | :--- | :--- |
| 自注意力（Self-Attention） | $O(n^2\cdot d)$ | $O(1)$ | $O(1)$ |
| 循环（Recurrent） | $O(n\cdot d^2)$ | $O(n)$ | $O(n)$ |
| 卷积（Convolutional） | $O(k\cdot n\cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| 受限自注意力（Self-Attention, restricted） | $O(r\cdot n\cdot d)$ | $O(1)$ | $O(n/r)$ |

> 记号：$n$ = 序列长度，$d$ = 表示维度，$k$ = 卷积核大小，$r$ = 受限自注意力的邻域窗口大小。

**结论**：当序列长度 $n$ 小于表示维度 $d$（在机器翻译中很常见）时，自注意力比循环层更快；其 $O(1)$ 的最大路径长度让长程依赖极易学习。此外，自注意力还带来更好的**可解释性**（见下方 Figure 3–5）。

---

## 6. 训练细节（Training）

- **数据集与批处理（Batching）**：
  - 英译德：WMT 2014，约 450 万句对，用字节对编码（Byte-Pair Encoding, BPE），共享词表约 37,000 个 token。
  - 英译法：WMT 2014，约 3600 万句，词片（word-piece）词表 32,000 个 token。
  - 每个批次约含 25,000 个源端 token 和 25,000 个目标端 token。
- **硬件与时间**：8 块 NVIDIA P100 GPU。基础模型（base）训练 10 万步（约 12 小时）；大模型（big）训练 30 万步（约 3.5 天）。
- **优化器（Optimizer）**：Adam，$\beta_1 = 0.9$，$\beta_2 = 0.98$，$\epsilon = 10^{-9}$。
- **学习率调度（Learning Rate Schedule，预热 + 衰减）**：

$$
lrate = d_{model}^{-0.5}\cdot \min\!\left(step\_num^{-0.5},\; step\_num\cdot warmup\_steps^{-1.5}\right)
$$

  前 $warmup\_steps = 4000$ 步线性升温，之后按步数平方根倒数衰减。

- **正则化（Regularization）**：
  - **残差 Dropout（Residual Dropout）**：作用于每个子层输出（在残差相加与归一化之前），以及嵌入与位置编码之和。基础模型 $P_{drop} = 0.1$。
  - **标签平滑（Label Smoothing）**：$\epsilon_{ls} = 0.1$。虽然会略微损害困惑度（perplexity），但提升了准确率与 BLEU。

---

## 7. 实验结果（Results）

### 7.1 机器翻译（表 2 / Table 2）

| 模型 | 英德 BLEU | 英法 BLEU | 训练成本（英德 FLOPs） |
| :--- | :--- | :--- | :--- |
| ByteNet | 23.75 | — | — |
| GNMT + RL | 24.6 | 39.92 | $2.3\times10^{19}$ |
| ConvS2S | 25.16 | 40.46 | $9.6\times10^{18}$ |
| GNMT + RL 集成（Ensemble） | 26.30 | 41.16 | $1.8\times10^{20}$ |
| ConvS2S 集成（Ensemble） | 26.36 | 41.29 | $7.7\times10^{19}$ |
| **Transformer（base）** | **27.3** | **38.1** | $3.3\times10^{18}$ |
| **Transformer（big）** | **28.4** | **41.8** | $2.3\times10^{19}$ |

**要点**：大模型在英德、英法双双刷新 SOTA，且训练成本仅为前人最优模型的一小部分。

### 7.2 模型变体消融实验（表 3 / Table 3）

基础模型：$N=6$，$d_{model}=512$，$d_{ff}=2048$，$h=8$，$d_k=d_v=64$，困惑度 PPL≈4.92，BLEU≈25.8，参数量 65M。

| 变体 | 改动 | 观察结论 |
| :--- | :--- | :--- |
| **(A)** 头数 $h$（保持计算量不变） | $h=1$→BLEU 24.9；$h=4$→25.5；$h=16$→25.8；$h=32$→25.4 | 单头明显不够，头数过多也会掉点 |
| **(B)** 减小 $d_k$ | $d_k=16$→25.1；$d_k=32$→25.4 | 点积兼容性函数需要足够维度 |
| **(C)** 模型规模 | $d_{model}=256$→24.5；$d_{model}=1024$→26.0 | 模型越大越好 |
| **(D)** Dropout | $P_{drop}=0.0$→24.6；$P_{drop}=0.2$→25.5 | Dropout 对抑制过拟合至关重要 |
| **(E)** 位置编码 | 用可学习嵌入替代正弦→25.7 | 两种方案效果几乎一致 |
| **big** | $d_{model}=1024$，$d_{ff}=4096$，$h=16$，$P_{drop}=0.3$ | PPL≈4.33，BLEU≈26.4，参数量 213M |

### 7.3 英语成分句法分析（表 4 / Table 4）

| 训练设置 | Transformer 结果（F1） | 说明 |
| :--- | :--- | :--- |
| 仅 WSJ（判别式，discriminative） | **91.3** | 4 层 Transformer，接近当时 SOTA（91.7） |
| 半监督（semi-supervised） | **92.7** | 超越所有此前模型（前 SOTA 92.1） |

**结论**：即便不做任务专属调优，Transformer 依然展现出很强的泛化能力。

---

## 8. 注意力可视化（Attention Visualizations，图 3–5）

论文附录用真实句子展示了自注意力学到的可解释行为。三张图均来自编码器第 5 层（共 6 层）的自注意力。

### 图 3（Figure 3）：捕捉长距离依赖

![Figure 3: 编码器自注意力捕捉长距离依赖](/assets/posts/attention/figure3_longdistance.png)

- **场景**：例句中许多注意力头对动词「making」建立了远距离依赖，补全短语「making … more difficult」。
- **解读**：图中只展示「making」一词的注意力连线，不同颜色代表不同的头（head）。可见「making」清晰地连向远处的「more」「difficult」——说明自注意力能跨越很长距离直接建立句法/语义联系，正是表 1 中 $O(1)$ 路径长度的直观体现。

### 图 4（Figure 4）：疑似完成指代消解（Anaphora Resolution）

![Figure 4: 两个注意力头疑似完成指代消解](/assets/posts/attention/figure4_anaphora.png)

- **场景**：同为第 5 层的两个注意力头，似乎在做指代消解。
- **解读**：
  - **上半**：头 5（head 5）的完整注意力分布。
  - **下半**：只抽取「its」一词、在头 5 与头 6 上的注意力。可以看到「its」的注意力非常「锐利（sharp）」，明确指向「Law」「application」等先行词——这正是指代消解所需的行为。

### 图 5（Figure 5）：注意力头学到句子结构

![Figure 5: 不同注意力头学到不同的句子结构行为](/assets/posts/attention/figure5_structure.png)

- **场景**：来自编码器第 5 层的两个不同的头（图中绿、红两色）。
- **解读**：这两个头明显学会了执行**不同的任务**、呈现出与句子结构相关的不同注意力模式——印证了「多头注意力关注不同表示子空间」的设计初衷。

---

## 9. 结论（Conclusion）

- Transformer 是**第一个完全基于注意力**的序列转录模型，用多头自注意力替代了编码器-解码器架构中常用的循环层。
- 在翻译任务上训练速度远快于 RNN/CNN，并刷新了英德、英法翻译的 SOTA。
- **未来展望**：
  1. 将 Transformer 扩展到文本以外的模态（图像、音频、视频）；
  2. 研究局部/受限注意力（local/restricted attention）以高效处理超大输入输出；
  3. 让生成过程更少依赖顺序性（less sequential）。

> 事后来看，这三条展望几乎全部成真：Transformer 已成为 NLP、CV、语音、多模态乃至整个大模型时代的通用骨干。

---

## 10. 一句话总结

用**多头自注意力**取代循环与卷积，既解决了 RNN 无法并行的痛点，又用 $O(1)$ 的最短路径轻松捕捉长程依赖——**注意力，确实就是（几乎）你所需要的一切**。

---

## 附录：专有名词中英对照表

| 中文 | 英文 | 简要说明 |
| :--- | :--- | :--- |
| 变换器 / 转换器 | Transformer | 本文提出的核心架构 |
| 注意力机制 | Attention Mechanism | 通过查询-键-值加权聚合信息 |
| 自注意力 | Self-Attention（intra-attention） | 序列内部各位置互相注意 |
| 缩放点积注意力 | Scaled Dot-Product Attention | 点积后除以 $\sqrt{d_k}$ 再 softmax |
| 多头注意力 | Multi-Head Attention | 并行多组注意力，关注不同子空间 |
| 带掩码多头注意力 | Masked Multi-Head Attention | 解码器中禁止关注未来位置 |
| 编码器-解码器注意力 | Encoder-Decoder / Cross-Attention | Q 来自解码器、K/V 来自编码器 |
| 查询 / 键 / 值 | Query / Key / Value | 注意力的三类输入向量 |
| 编码器 | Encoder | 将输入序列编码为表示 |
| 解码器 | Decoder | 自回归地生成输出序列 |
| 循环神经网络 | Recurrent Neural Network (RNN) | 顺序计算的序列模型 |
| 卷积神经网络 | Convolutional Neural Network (CNN) | 用卷积核提取局部特征 |
| 长短期记忆网络 | Long Short-Term Memory (LSTM) | 一种门控 RNN |
| 门控循环单元 | Gated Recurrent Unit (GRU) | 一种门控 RNN |
| 序列转录 | Sequence Transduction | 序列到序列的映射任务 |
| 残差连接 | Residual Connection | 子层输入直接加到输出 |
| 层归一化 | Layer Normalization | 对特征维做归一化 |
| 逐位置前馈网络 | Position-wise Feed-Forward Network (FFN) | 对每个位置独立作用的两层全连接 |
| 位置编码 | Positional Encoding | 注入序列位置信息 |
| 词嵌入 | Embedding | 将 token 映射为稠密向量 |
| 词元 | Token | 文本的基本单元 |
| 字节对编码 | Byte-Pair Encoding (BPE) | 子词切分方法 |
| 词片 | Word-piece | 另一种子词切分方法 |
| 权重共享 | Weight Tying | 嵌入层与输出投影共享权重 |
| 自回归 | Auto-regressive | 逐位置依赖已生成结果 |
| 长程依赖 | Long-range Dependencies | 序列中相距较远元素的关联 |
| 表示子空间 | Representation Subspace | 各注意力头关注的不同特征空间 |
| 兼容性函数 | Compatibility Function | 衡量 Q 与 K 匹配度的函数 |
| 加性注意力 | Additive Attention | 用前馈网络计算兼容性 |
| 点积注意力 | Dot-Product Attention | 用点积计算兼容性 |
| 集成模型 | Ensemble | 多模型融合 |
| 困惑度 | Perplexity (PPL) | 语言模型评价指标，越低越好 |
| 双语评估替补 | BLEU | 机器翻译质量指标，越高越好 |
| 标签平滑 | Label Smoothing | 软化标签的正则化技术 |
| 学习率预热 | Warmup | 训练初期线性增大学习率 |
| 成分句法分析 | Constituency Parsing | 分析句子的成分结构 |
| 指代消解 | Anaphora Resolution | 确定代词指向的先行词 |
| 最优结果 | State-of-the-Art (SOTA) | 当前最佳水平 |
| 浮点运算次数 | FLOPs | 衡量计算量 |
| 全局依赖 | Global Dependencies | 跨整个序列的依赖关系 |

---

## 附录：原始论文（Original Paper）

- **论文标题**：Attention Is All You Need
- **arXiv 地址**：<https://arxiv.org/abs/1706.03762>
- **PDF 原文**：[点击查看 / 下载 PDF](/assets/posts/attention/attention-is-all-you-need.pdf)

> 本文所有配图与数据均出自上述原始论文，如需引用请以原文为准。
