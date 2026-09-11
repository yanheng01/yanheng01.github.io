---
title: "FlashAttention: 快速且省内存的精确注意力（IO 感知）"
date: 2026-09-11 18:00:00 +0800
permalink: /posts/flashattention/
categories: [技术笔记, 深度学习]
tags: [flashattention, attention, transformer, gpu, io-aware, cuda, 论文精读, 大模型, Infra]
math: true
mermaid: false
toc: true
---

> **原文标题（Original Title）**：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
> **作者（Authors）**：Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré（Stanford University; University at Buffalo, SUNY）
> **发表时间**：2022 年 6 月（arXiv:2205.14135v2）
> **一句话总结**：把注意力（Attention）计算的优化视角从「减少浮点运算量（FLOPs）」转向「减少 GPU 显存的读写次数（IO）」，通过**分块（Tiling）** 与**重计算（Recomputation）** 融合成单个 GPU 内核，在**不损失任何精度**（精确注意力）的前提下大幅加速训练、降低显存占用，并让 Transformer 首次能处理 16K/64K 的超长序列。

---

## 目录

1. [核心思想与动机](#1-核心思想与动机)
2. [背景：GPU 硬件与标准注意力](#2-背景gpu-硬件与标准注意力)
3. [FlashAttention 算法](#3-flashattention-算法)
4. [IO 复杂度分析](#4-io-复杂度分析)
5. [扩展：块稀疏 FlashAttention](#5-扩展块稀疏-flashattention)
6. [实验结果](#6-实验结果)
7. [关键要点回顾](#7-关键要点回顾)
8. [附录 A：专有名词中英对照表](#附录-a专有名词中英对照表)
9. [附录 B：原文与 PDF 链接](#附录-b原文与-pdf-链接)

---

## 1. 核心思想与动机

### 问题的根源

Transformer（转换器）架构的核心是自注意力（Self-Attention）模块，但它在处理长序列时既**慢**又**吃内存**——时间和内存复杂度都随序列长度 $N$ 呈**二次方增长**（$O(N^2)$）。这一直是限制模型处理长上下文（Long Context）的瓶颈。

### 现有方法为什么不够好

过去大量的**近似注意力（Approximate Attention）** 方法（如稀疏近似、低秩近似及其组合）试图把计算复杂度降到线性或近线性。但作者尖锐地指出一个普遍现象：

> 这些方法虽然减少了理论浮点运算量（FLOPs），却**很少带来实际的墙钟时间（Wall-clock Time）加速**，因此没有被广泛采用。

原因在于它们只盯着 FLOPs，却**忽略了内存访问（Memory Access / IO）的开销**。

### 缺失的关键原则：IO 感知（IO-Awareness）

作者提出的核心论点是：现代 GPU 的**计算速度已经远远超过了内存速度**，因此 Transformer 中的大多数操作其实是**内存受限（Memory-bound）** 的，而非计算受限。真正高效的注意力算法必须做到 **IO 感知（IO-Awareness）**——即精细地考虑不同层级显存（快而小的片上 SRAM 与大而慢的 HBM）之间的读写。

**FlashAttention** 正是基于这一原则设计的**精确（Exact）** 注意力算法。它的目标非常明确：**避免把巨大的 $N \times N$ 注意力矩阵在 HBM 中读来写去**。为此它使用了两项成熟技术：

1. **分块（Tiling）**：把输入切成小块，多次遍历，在片上增量地完成 Softmax 归约，从而无需实例化完整的 $N \times N$ 矩阵。
2. **重计算（Recomputation）**：前向传播不保存中间注意力矩阵，只保存 Softmax 的归一化统计量；反向传播时在片上 SRAM 里快速重算，这比从 HBM 读回中间矩阵还要快。

整套算法通过 CUDA 的**内核融合（Kernel Fusion）** 合并为一个 GPU 内核。

![Figure 1：左图为 GPU 内存层级与 FlashAttention 的 Tiling 机制示意；右图为 GPT-2 上注意力计算相对 PyTorch 的 7.6 倍加速](/assets/posts/flashattention/figure1.png)
_Figure 1：（左）FlashAttention 用分块避免在慢速 HBM 中实例化大 $N \times N$ 注意力矩阵（虚线框）。外循环（红色箭头）遍历 K、V 块并加载到快速 SRAM；内循环（蓝色箭头）遍历 Q 块，把注意力输出写回 HBM。（右）由于不再读写大矩阵，注意力计算在 GPT-2 上获得约 7.6 倍加速。_

### 主要贡献

- **更快的训练**：BERT-large（序列长度 512）比 MLPerf 1.1 训练速度纪录快 **15%**；GPT-2（序列长度 1K）比标准实现快 **3 倍**；长程竞技场（LRA，1K–4K）快 **2.4 倍**。
- **更高的模型质量**：支持更长序列，GPT-2 困惑度（Perplexity）降低 **0.7**；长文档分类提升最高 **6.4 个点**；**首次**让 Transformer 在 Path-X（16K）和 Path-256（64K）任务上取得超越随机猜测的成绩。
- **更全面的基准测试**：在 512 长度内比所有现有方法都更快、更省内存；更长序列下，块稀疏版本超越所有近似注意力方法。

---

## 2. 背景：GPU 硬件与标准注意力

### 2.1 GPU 的内存层级

理解 FlashAttention 必须先理解 GPU 的显存分层：

| 存储层级 | 容量 | 带宽 | 特点 |
| --- | --- | --- | --- |
| **SRAM（片上静态内存，On-chip SRAM）** | 极小（约 20 MB，每个 SM 约 192KB） | 极高（约 19 TB/s） | 快，但装不下大矩阵 |
| **HBM（高带宽内存，High Bandwidth Memory）** | 大（如 40 GB） | 中（1.5–2.0 TB/s） | 就是我们常说的「显存」 |
| **DRAM（主存，CPU DRAM）** | 巨大（>1 TB） | 低（约 12.8 GB/s） | 最慢 |

**执行模型**：GPU 内核（Kernel）先把数据从 HBM 搬到 SRAM/寄存器，计算完再写回 HBM。当运算的算术强度（Arithmetic Intensity）低时，瓶颈就在 HBM 读写上，此时称为**内存受限（Memory-bound）**。

**内核融合（Kernel Fusion）** 是减少 HBM 读写的常用手段：把多个操作合并成一个内核，中间结果不落盘到 HBM。但在训练时，反向传播（Backward Pass）需要中间结果，朴素的融合收益受限——这正是 FlashAttention 用重计算解决的痛点。

### 2.2 标准注意力实现（Algorithm 0）

给定 $Q, K, V \in \mathbb{R}^{N \times d}$（$N$ 为序列长度，$d$ 为头维度 Head Dimension），标准实现分三步：

$$
S = QK^{\top} \in \mathbb{R}^{N \times N}, \quad P = \text{softmax}(S) \in \mathbb{R}^{N \times N}, \quad O = PV \in \mathbb{R}^{N \times d}
$$

**缺陷**：中间矩阵 $S$ 和 $P$ 的尺寸都是 $N \times N$，必须完整写入 HBM 再读出，产生了 $O(N^2)$ 的 HBM 访问。这正是标准注意力慢的真正原因。

---

## 3. FlashAttention 算法

### 3.1 分块（Tiling）：如何分块计算 Softmax

Softmax 的难点在于它需要**一整行的全局信息**（求最大值、求和），似乎无法分块。FlashAttention 借助**在线 Softmax（Online Softmax）** 的带缩放技巧解决了这个问题。

对向量 $x$，定义：

$$
m(x) = \max_i x_i, \quad f(x)_i = e^{x_i - m(x)}, \quad \ell(x) = \sum_i f(x)_i, \quad \text{softmax}(x) = \frac{f(x)}{\ell(x)}
$$

其中减去最大值 $m(x)$ 是为了数值稳定（避免指数溢出）。关键在于：当把两个分块 $x = [x^{(1)}, x^{(2)}]$ 拼接时，可以**增量更新**统计量：

$$
m(x) = \max\!\big(m(x^{(1)}), m(x^{(2)})\big)
$$

$$
\ell(x) = e^{m(x^{(1)}) - m(x)}\,\ell(x^{(1)}) + e^{m(x^{(2)}) - m(x)}\,\ell(x^{(2)})
$$

因此，只要额外维护两个统计量 $(m, \ell)$，就能一块一块地计算，并把结果累加到输出 $O$ 上——这被称为**代数聚合（Algebraic Aggregation）**。

### 3.2 重计算（Recomputation）：省掉中间矩阵

反向传播需要 $S$ 和 $P$ 来计算梯度。FlashAttention 的做法是：前向传播**只保存输出 $O$ 和统计量 $(m, \ell)$**（仅需 $O(N)$ 额外内存），不保存 $N \times N$ 的中间矩阵；反向传播时，利用保存的统计量和 $Q, K, V$ 在 SRAM 中**重新计算** $S$ 和 $P$。

这可以看作一种**选择性梯度检查点（Selective Gradient Checkpointing）**。虽然重算增加了 FLOPs，但由于彻底避免了 $O(N^2)$ 的 HBM 读写，**实际速度反而更快**。

### 3.3 完整算法（Algorithm 1）

![Algorithm 1：FlashAttention 前向传播伪代码](/assets/posts/flashattention/algorithm1.png)
_Algorithm 1：FlashAttention 前向传播。外循环遍历 K、V 块并载入 SRAM；内循环遍历 Q 块，在片上完成 $S_{ij}=Q_i K_j^{\top}$、局部 Softmax、统计量更新与输出累加，全程只维护 $O(N)$ 的 $(\ell, m)$。_

算法要点（配合上图）：

1. **初始化**：在 HBM 中设 $O = 0$，$\ell = 0$，$m = -\infty$；把 $Q, K, V$ 划分为大小为 $B_r, B_c$ 的块，块大小由 SRAM 容量 $M$ 决定（$B_c = \lceil M/4d \rceil$）。
2. **外循环**（遍历 $K_j, V_j$）：把 $K_j, V_j$ 从 HBM 载入 SRAM。
3. **内循环**（遍历 $Q_i$）：载入 $Q_i, O_i, \ell_i, m_i$；在片上计算 $S_{ij} = Q_i K_j^{\top}$、局部最大值 $\tilde{m}_{ij}$、局部指数 $\tilde{P}_{ij}$ 和局部和 $\tilde{\ell}_{ij}$；更新全局统计量 $m_i^{\text{new}}, \ell_i^{\text{new}}$，并按下式更新输出：

$$
O_i \leftarrow \text{diag}(\ell_i^{\text{new}})^{-1}\Big(\text{diag}(\ell_i)\,e^{m_i - m_i^{\text{new}}}\,O_i + e^{\tilde{m}_{ij} - m_i^{\text{new}}}\,\tilde{P}_{ij}\,V_j\Big)
$$

4. 把更新后的 $O_i, \ell_i, m_i$ 写回 HBM，最终返回 $O$。

> **Theorem 1（定理 1）**：Algorithm 1 返回**精确**的注意力输出 $O = \text{softmax}(QK^{\top})V$，FLOPs 为 $O(N^2 d)$，且在输入输出之外**仅需 $O(N)$ 的额外内存**。

**内核融合**：整个 Algorithm 1 被融合为**一个 CUDA 内核**——矩阵乘法、掩码（Masking）、Softmax、Dropout、再矩阵乘法，全部在片上一气呵成，彻底消除中间矩阵在 HBM 的实例化。（含掩码/Dropout 的完整前向见原文 Algorithm 2，反向见 Algorithm 4。）

---

## 4. IO 复杂度分析

这是本文的理论核心：证明 FlashAttention 在 HBM 访问次数上远优于标准实现，且是最优的。

> **Theorem 2（定理 2）**：设 $N$ 为序列长度，$d$ 为头维度，$M$ 为 SRAM 大小（$d \le M \le Nd$）。
> - 标准注意力（Algorithm 0）需要 $\Theta(Nd + N^2)$ 次 HBM 访问；
> - FlashAttention（Algorithm 1）只需 $\Theta(N^2 d^2 M^{-1})$ 次 HBM 访问。

由于典型情况下 $d$（64–128）使得 $d^2 \ll M$（$M \approx 100\text{KB}$，而 $d^2 = 4096$），FlashAttention 的 HBM 访问次数**成数量级减少**（论文实测最多减少约 9 倍），从而带来更快的执行速度和更低的显存占用。

> **Proposition 3（命题 3，下界）**：在所有 $M$ 取值范围内，不存在任何精确注意力算法能渐进地突破 $\Omega(N^2 d^2 M^{-1})$ 的 HBM 访问下界。也就是说，**FlashAttention 在 IO 复杂度上是最优的**。

![Figure 2：IO 是运行时间的主导因素；块大小影响；块稀疏加速](/assets/posts/flashattention/figure2.png)
_Figure 2：（左）GPT-2 medium 上标准注意力与 FlashAttention 的前向+反向对比——尽管 FlashAttention 的 GFLOPs 更高（75.2 vs 66.6），但 HBM 读写量骤降（4.4 GB vs 40.3 GB），运行时间从 41.7ms 降到 7.3ms。（中）块大小（Block Size）越大，HBM 访问越少、越快，但超过 256 后受限于算术运算和 SRAM 容量。（右）块稀疏 FlashAttention 的运行时间随非零块比例线性下降。_

---

## 5. 扩展：块稀疏 FlashAttention

作者进一步把 FlashAttention 扩展为**块稀疏（Block-Sparse）** 的近似注意力。给定一个块稀疏掩码矩阵 $\tilde{M} \in \{0,1\}^{N \times N}$，只计算掩码非零的块，直接跳过全零块（对应原文 Algorithm 5）。

> **Proposition 4（命题 4）**：块稀疏 FlashAttention 的 IO 复杂度为 $\Theta\!\big(Nd + N^2 d^2 M^{-1} s\big)$，其中 $s$ 是非零块的比例。

也就是说，**加速比与稀疏度成正比**。对于常见的稀疏模式（如 $s = N^{-1/2}$ 或 $N^{-1}\log N$），IO 复杂度可降到 $\Theta(N\sqrt{N})$ 或 $\Theta(N \log N)$。下游实验采用固定的蝶形稀疏模式（Butterfly Sparsity Pattern），已被证明能近似任意稀疏。

---

## 6. 实验结果

### 6.1 更快的模型训练

**BERT-large**（Table 1）：在 8×A100 上从 MLPerf 相同初始化训练到 72.0% 目标精度，FlashAttention 用时 **17.4 分钟**，比 Nvidia MLPerf 1.1 纪录的 20.0 分钟快 **15%**。

**GPT-2**（Table 2）：在 OpenWebText 上，困惑度与 HuggingFace、Megatron-LM 完全一致（不改模型定义），但训练更快：

| 模型实现 | OpenWebText 困惑度 | 训练时间（加速比） |
| --- | --- | --- |
| GPT-2 small - HuggingFace | 18.2 | 9.5 天（1.0×） |
| GPT-2 small - Megatron-LM | 18.2 | 4.7 天（2.0×） |
| **GPT-2 small - FlashAttention** | 18.2 | **2.7 天（3.5×）** |
| GPT-2 medium - HuggingFace | 14.2 | 21.0 天（1.0×） |
| GPT-2 medium - Megatron-LM | 14.3 | 11.5 天（1.8×） |
| **GPT-2 medium - FlashAttention** | 14.3 | **6.9 天（3.0×）** |

**长程竞技场（LRA，Table 3）**：FlashAttention 比标准注意力快 **2.4×**，块稀疏版本快 **2.8×**，且精度与标准注意力持平。

### 6.2 更长序列带来更高质量

**长上下文语言建模**（Table 4）：GPT-2 small 用 4K 上下文的 FlashAttention 训练，仍比 Megatron 用 1K 上下文**快 30%**，且困惑度更低（17.5 vs 18.2，**降低 0.7**）。

**长文档分类**（Table 5）：在 MIMIC-III 和 ECtHR 数据集上，把序列长度加长带来最高 **8.5 个点**的 Micro-F1 提升（如 ECtHR 从 512 的 72.2 到 8K 的 80.7）。

**Path-X / Path-256**（Table 6）：这是长程竞技场里最难的两个任务（把 128×128 或 256×256 的图像逐像素喂给 Transformer，判断两点是否连通，序列长度分别为 16K 和 64K）。此前所有 Transformer 变体要么爆内存，要么只能做到随机水平（约 50%/更低）。FlashAttention **首次**在 Path-X 上达到 **61.4%**，块稀疏 FlashAttention 在 Path-256 上达到 **63.1%**——都是首个超越随机猜测的 Transformer 结果。

### 6.3 注意力基准测试

![Figure 3：不同序列长度下的运行时间与显存占用对比](/assets/posts/flashattention/figure3.png)
_Figure 3：（左）前向+反向运行时间——FlashAttention（黑点线）在常见长度内比 PyTorch/Megatron 等基线快，块稀疏版本（绿虚线）全程最快；标注了各方法与 FlashAttention 的「交叉点（Crossover Points）」。（右）显存占用——FlashAttention 随序列长度**线性增长**，在 64K 时比标准注意力省约 **20×**，比 Linformer 省约 **2×**。_

**IO 瓶颈的直接验证**：在 GPT-2 配置下，标准注意力的 HBM 读写量为 40.3 GB，运行时间 41.7ms；FlashAttention 只有 **4.4 GB**（减少约 9 倍），运行时间降到 **7.3ms**——尽管它的 FLOPs（75.2 GFLOPs）反而比标准注意力（66.6 GFLOPs）更高。这有力证明了 **HBM 访问才是注意力运行时间的主导因素**。

### 6.4 数值一致性

![Figure 4：GPT-2 训练中的验证困惑度曲线，FlashAttention 与 HuggingFace 完全重合](/assets/posts/flashattention/figure4.png)
_Figure 4：GPT-2 small/medium 在两种实现下的验证困惑度（Validation Perplexity）曲线。FlashAttention 与 HuggingFace 基线的曲线**完全重合**，证明它是精确注意力，数值稳定性与标准实现一致，不会因重计算或分块引入误差。_

---

## 7. 关键要点回顾

- **视角转变**：优化注意力的关键不是减少 FLOPs，而是**减少 HBM 的读写（IO）**——因为现代 GPU 是内存受限的。
- **两大技术**：分块（Tiling，在线 Softmax）+ 重计算（Recomputation，选择性梯度检查点），把额外内存从 $O(N^2)$ 降到 $O(N)$。
- **精确无损**：FlashAttention 是**精确注意力**，输出与标准实现逐位一致，不牺牲任何模型质量。
- **理论最优**：HBM 访问从 $\Theta(Nd + N^2)$ 降到 $\Theta(N^2 d^2 M^{-1})$，并被证明在所有 SRAM 大小下是最优的。
- **工程落地**：全部融合为一个 CUDA 内核，掩码与 Dropout 也在片上完成。
- **能力解锁**：更快训练 + 更省显存 + 支持超长上下文，首次攻克 Path-X（16K）与 Path-256（64K）。
- **意义**：FlashAttention 是深度学习底层算子「IO 感知优化」的典范，在摩尔定律放缓、内存墙日益突出的时代，为后续基础设施优化指明了方向。

---

## 附录 A：专有名词中英对照表

| 中文 | 英文（English） |
| --- | --- |
| 注意力 / 自注意力 | Attention / Self-Attention |
| 精确注意力 | Exact Attention |
| 近似注意力 | Approximate Attention |
| IO 感知 | IO-Awareness |
| 内存访问 / 读写 | Memory Access / IO |
| 内存受限 | Memory-bound |
| 计算受限 | Compute-bound |
| 墙钟时间 | Wall-clock Time |
| 浮点运算量 | FLOPs（Floating Point Operations） |
| 高带宽内存 | HBM（High Bandwidth Memory） |
| 片上静态内存 | SRAM（On-chip SRAM） |
| 主存 | DRAM（CPU DRAM） |
| 流多处理器 | SM（Streaming Multiprocessor） |
| 内核 / CUDA 内核 | Kernel / CUDA Kernel |
| 内核融合 | Kernel Fusion |
| 分块 | Tiling |
| 重计算 | Recomputation |
| 选择性梯度检查点 | Selective Gradient Checkpointing |
| 在线 Softmax | Online Softmax |
| 代数聚合 | Algebraic Aggregation |
| 归一化统计量 | Normalization Statistics |
| 前向传播 / 反向传播 | Forward Pass / Backward Pass |
| 序列长度 | Sequence Length（$N$） |
| 头维度 | Head Dimension（$d$） |
| 块大小 | Block Size |
| 块稀疏 | Block-Sparse |
| 蝶形稀疏模式 | Butterfly Sparsity Pattern |
| 掩码 | Masking / Mask |
| 困惑度 | Perplexity |
| 长上下文 | Long Context |
| 长程竞技场 | Long-Range Arena（LRA） |
| 算术强度 | Arithmetic Intensity |
| IO 复杂度 | IO Complexity |
| 下界 | Lower Bound |

---

## 附录 B：原文与 PDF 链接

- **论文网址（Paper URL）**：<https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com>
- **论文标题**：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **发布机构**：Stanford University; University at Buffalo, SUNY

---

> 本阅读笔记由 AI 辅助精读整理，配图均从原文 PDF 提取。如与原文有出入，请以原文为准。
