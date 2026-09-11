---
title: "Megatron-LM 大模型分布式怎么训练"
date: 2026-09-11 19:00:00 +0800
categories: [技术笔记, 深度学习]
tags: [Megatron-LM, 模型并行, 张量并行, 分布式训练, 大模型, Infra, 论文精读]
math: true
mermaid: false
toc: true
---

> 一份深入的中文阅读笔记。文中所有专有名词首次出现时附对应英文，文末附「专有名词中英对照附录」。论文全部 8 张配图（Figure 1–8）与关键表格均已嵌入并配中文解读。原文标题：*Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism*。

---

## 0. 论文基本信息

| 项目 | 内容 |
| :--- | :--- |
| 标题 | **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism** |
| 作者 | Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, Bryan Catanzaro |
| 机构 | 英伟达（NVIDIA） |
| 发表 | arXiv:1909.08053（2019 年 9 月首发，2020 年 3 月更新至 v4），发表于 ICML 2020 相关 Proceedings |
| 地位 | 现代大模型**张量并行（Tensor Parallelism）**的奠基性工作，确立了工业界层内模型并行的事实标准 |

**一句话总结**：利用 Transformer 层内矩阵乘法的结构特点，通过「列切分（Column Parallel）+ 行切分（Row Parallel）」的巧妙组合，仅用极少量通信原语（All-Reduce），就能在原生 PyTorch 中高效训练数十亿参数的语言模型，无需自定义编译器。

---

## 1. 摘要（Abstract）

- **背景**：语言建模领域的近期工作表明，训练更大的 Transformer 模型能推动自然语言处理（Natural Language Processing, NLP）的技术前沿。但超大模型受限于**显存（memory constraints）**，难以训练。
- **核心方法**：提出一种简单、高效的**层内模型并行（intra-layer model parallelism）**方法，可训练数十亿参数级别的 Transformer 模型。该方法：
  - **不需要**新的编译器或库改动；
  - 与**流水线并行（pipeline model parallelism）正交互补**；
  - 只需在原生 PyTorch 中插入少量通信操作即可完整实现。
- **规模与算力**：用 **512 块 GPU** 训练了高达 **83 亿（8.3 billion）参数**的 Transformer 模型，全应用维持 **15.1 PetaFLOPs** 的算力，相较于单卡强基线（39 TeraFLOPs，达到理论峰值的 30%），扩展效率（scaling efficiency）达 **76%**。
- **SOTA 成果**：训练了类 GPT-2 的 83 亿参数模型和类 BERT 的 39 亿参数模型，在多个基准上刷新最优结果（State-of-the-Art, SOTA）：
  - **WikiText103**：困惑度（Perplexity）降至 **10.8**（此前 SOTA 为 15.8）；
  - **LAMBADA**：准确率 **66.5%**（此前 SOTA 为 63.2%）；
  - **RACE**：准确率 **90.9%**（此前 SOTA 为 89.4%）。
- **关键发现**：对于类 BERT 模型，**层归一化（Layer Normalization, LayerNorm）位置的摆放**对于模型规模增长时性能能否持续提升至关重要。

---

## 2. 引言与背景（Introduction & Background）

### 2.1 为什么需要模型并行

- **算力与数据推动大模型**：NLP 快速发展，很大程度上得益于算力与数据集规模的增长。经验表明，更大的语言模型在文章补全、问答、自然语言推理等任务上表现显著更好。
- **显存墙（Memory Wall）**：随着模型变大，其参数超出了现代处理器的显存上限。此外，像 **Adam** 这类优化器需要为每个参数额外存储动量（momentum）等**优化器状态（optimizer state）**，进一步压缩了可训练模型的规模。
- **已有方案的局限**：
  - **激活重计算（activation checkpointing）**：反向传播时重算激活值而非存储，能省显存，但**模型仍须完整放进单个 worker**。
  - **GPipe**（Huang et al., 2018）与 **Mesh-TensorFlow**（Shazeer et al., 2018）提供了模型并行框架，但它们**需要重写模型**，并依赖仍在开发中的自定义编译器和框架。
- **两种模型并行范式**：
  - **流水线并行（Pipeline Parallelism）**：按层切分，一组操作算完再传给下一设备。存在**流水线气泡（pipeline bubble）**导致效率下降，或需改动优化器本身（如 GPipe 用同步梯度下降来保证一致性）。
  - **分布式张量计算（Distributed Tensor Computation）**：把单个张量运算切分到多设备，既加速又扩容。本文正属于此范式，但**只对现有 PyTorch 实现做少量针对性修改**。

### 2.2 本文的贡献

- 通过对现有 PyTorch Transformer 实现做少量针对性修改，实现了简单高效的模型并行方法。
- 深入的实证分析，用 512 块 GPU 展示了高达 **76%** 的扩展效率。
- 证明了**层归一化位置**对类 BERT 模型随规模增长的重要性。
- 展示了 GPT-2（up to 8.3B）和 BERT（up to 3.9B）随规模增长带来的精度提升。
- 在 WikiText103、LAMBADA、RACE 上取得 SOTA。
- 开源了代码与训练 / 评估管线（<https://github.com/NVIDIA/Megatron-LM>）。

![Figure 1：模型并行（蓝色）与模型 + 数据并行（绿色）的 FLOPS 随 GPU 数量变化](/assets/posts/megatron/figure1_flops.png)

**Figure 1 解读**：横轴为 GPU 数量（对数刻度），纵轴为每秒 PetaFLOPs（对数刻度）。蓝色线为纯模型并行（弱扩展，weak scaling），最高做到 8 路模型并行、约每卡 10 亿参数（例如 2 GPU 训 20 亿、4 GPU 训 40 亿）。绿色线为模型 + 数据并行，在模型并行基础上叠加 64 路数据并行，最终逼近 15.1 PetaFLOPs。虚线为理想线性扩展参考线。

### 2.3 神经语言模型预训练（Neural Language Model Pretraining）

预训练语言模型已成为 NLP 研究者不可或缺的工具。研究脉络从早期的**词向量（word embedding）**，发展到能捕捉上下文的**上下文表示（contextual representation）**，再到如今**直接迁移整个数十亿参数语言模型**并在下游任务上端到端微调（finetune）。这一演进对硬件、系统技术和框架提出了越来越高的算力要求。

### 2.4 Transformer 与多头注意力（Multi-Head Attention）

- 当前 NLP 主流采用 **Transformer**（Vaswani et al., 2017），因其精度和计算效率俱佳。
- 原始 Transformer 是**编码器 - 解码器（Encoder-Decoder）**结构。近期工作按需求只用其中一部分：**GPT-2 用解码器（Decoder）**，**BERT 用编码器（Encoder）**。本文两者都研究。
- 值得注意：GPT-2 和 BERT 都把**层归一化**和**GeLU 非线性**用在多头注意力和前馈层的**输入端**，而原始 Transformer 用 ReLU 且把层归一化用在输出端。

![Figure 2：Transformer 架构示意。紫色块为全连接层（MLP），蓝色块表示一个被重复 N 次的 Transformer 层](/assets/posts/megatron/figure2_transformer.png)

**Figure 2 解读**：这是本文采用的模型结构。一个 Transformer 层自底向上依次为：输入嵌入（Input Embeddings，含 token、位置编码与 Dropout）→ LayerNorm → Self-Attention（含 Attention Dropout）→ Dropout → 残差相加（Add）→ LayerNorm → MLP（H→4H 线性、GeLU、4H→H 线性、Dropout）→ 残差相加（Add）。该层重复 N 次，最后接输出层、多头（Heads）与损失（Loss）。可见这是 **Pre-LN**（层归一化前置）的结构。

### 2.5 深度学习中的数据并行与模型并行

- **数据并行（Data Parallelism）**：把一个训练小批量（minibatch）切分到多个 worker 上。通过增大 minibatch 实现近线性的吞吐扩展（即弱扩展）。但**过大的 batch** 会给优化带来困难，可能降低精度或延长收敛时间。
- **模型并行（Model Parallelism）**：把模型的显存占用与计算分摊到多个 worker，从而突破单卡显存上限，还能提升单个微批（microbatch）的并行度。
- 本文借鉴了 Mesh-TensorFlow 的思路（在计算注意力头时利用并行性），但不实现新框架 / 编译器，只对 PyTorch 做少量修改。

---

## 3. 模型并行 Transformer（Model Parallel Transformers）

一个 Transformer 层由**自注意力块（Self-Attention Block）**和**两层多层感知机（MLP）**组成。作者对这两个块分别设计切分方案，核心思想是**用一列一行的组合，消除中间的同步点**。

### 3.1 MLP 块的切分

MLP 第一部分是一个通用矩阵乘法（General Matrix Multiply, GEMM）后接 GeLU 非线性：

$$
Y = \text{GeLU}(XA)
$$

**方案对比：**

- **按行切分（错误做法）**：若把权重 $A$ 按行切、输入 $X$ 按列切：

$$
X = [X_1, X_2],\quad A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}
$$

  则 $Y = \text{GeLU}(X_1 A_1 + X_2 A_2)$。由于 GeLU 是**非线性函数**，$\text{GeLU}(X_1A_1 + X_2A_2) \neq \text{GeLU}(X_1A_1) + \text{GeLU}(X_2A_2)$，因此**必须在 GeLU 之前插入一个同步点**（先把两部分加起来）。

- **按列切分（本文做法）**：把 $A$ 按列切分 $A = [A_1, A_2]$：

$$
[Y_1, Y_2] = [\text{GeLU}(XA_1), \text{GeLU}(XA_2)]
$$

  这样每个 GPU 独立算自己那一列并各自过 GeLU，**消除了同步点**。

- **第二层 GEMM 按行切分**：第二层权重 $B$ 按行切分 $B = \begin{bmatrix} B_1 \\ B_2 \end{bmatrix}$，直接接收上一层 GeLU 的输出，无需通信即可做矩阵乘法。其输出在送入 Dropout 前，跨 GPU 做**一次 All-Reduce**。

**结论**：整个 MLP 块把两个 GEMM 都切到多 GPU 上，**前向仅需 1 次 All-Reduce（g 算子），反向仅需 1 次 All-Reduce（f 算子）**。

### 3.2 Self-Attention 块的切分

- **QKV 投影按列切分**：利用多头注意力的天然并行性，把生成键（Key, K）、查询（Query, Q）、值（Value, V）的 GEMM **按列切分**，使得**每个 GPU 负责若干个完整的注意力头（attention head）**，无需立即通信即可完成自注意力计算。
- **输出投影按行切分**：自注意力之后的线性层**按行切分**，直接接收并行注意力层的输出，无需 GPU 间通信。
- **同样**：该块前向 1 次 All-Reduce、反向 1 次 All-Reduce。

![Figure 3：带模型并行的 Transformer 块。(a) MLP，(b) Self-Attention。f 与 g 互为共轭算子](/assets/posts/megatron/figure3_blocks.png)

**Figure 3 解读**：
- **(a) MLP**：输入 $X$ 经 f 算子广播到各 GPU，分别算 $XA_1$、$XA_2$ 并过 GeLU 得 $Y_1$、$Y_2$；再各自算 $Y_1B_1$、$Y_2B_2$ 得 $Z_1$、$Z_2$，经 g 算子做 All-Reduce 汇总，最后 Dropout 得 $Z$。权重按 $A=[A_1,A_2]$（列切）、$B=[B_1;B_2]$（行切）划分。
- **(b) Self-Attention**：把注意力头切分到各 GPU（$Q=[Q_1,Q_2]$、$K=[K_1,K_2]$、$V=[V_1,V_2]$），每个 GPU 独立完成 Softmax、Dropout 得 $Y_1$、$Y_2$，输出线性层按行切分后经 g 算子 All-Reduce 汇总。
- **f / g 算子**：这是本文实现的精髓，二者互为共轭（conjugate），只需几行代码继承 `torch.autograd.Function`。

### 3.3 f 与 g 算子（核心实现）

两个自定义算子互为共轭：

- **f 算子**：**前向传播为恒等操作（Identity）**，**反向传播为 All-Reduce**。
- **g 算子**：**前向传播为 All-Reduce**，**反向传播为恒等操作（Identity）**。

论文给出的 f 算子实现（Code 1）仅需数行 PyTorch 代码：

```python
class f(torch.autograd.Function):
    def forward(ctx, x):
        return x
    def backward(ctx, gradient):
        all_reduce(gradient)
        return gradient
# g 算子与 f 类似：前向 all-reduce，反向 identity
```

**通信量小结**：这种「两个 GEMM 融合、消除中间同步点」的策略，使得**一个模型并行 Transformer 层在前向传播中仅需 2 次 All-Reduce，反向传播中仅需 2 次 All-Reduce**，共 4 次通信。

![Figure 4：一个 Transformer 层中的通信操作。前向 + 反向共 4 次 All-Reduce](/assets/posts/megatron/figure4_comm.png)

**Figure 4 解读**：图中把一个模型并行 Transformer 层拆成两个「Model Parallel」区域——自注意力区（Self Attention + Linear + Dropout）和 MLP 区（Linear + GeLU + Linear + Dropout），二者之间及首尾都有 LayerNorm 与残差连接（图中 ⊕）。每个 Model Parallel 区各产生「2 次 All-Reduce（前向 + 反向）」，因此单层总计 4 次通信操作。LayerNorm、Dropout、残差都在模型并行区之外、被各 GPU 冗余计算。

### 3.4 词表并行（Embedding 切分）与融合损失

- **输入嵌入切分**：Transformer 语言模型的输出嵌入维度为「隐藏层大小 $H$ × 词表大小 $v$」。现代词表极大（GPT-2 用 50,257），因此把输入嵌入权重矩阵 $E_{H\times v}$ **按词表维度（列）切分** $E = [E_1, E_2]$。由于每个 GPU 只持有部分词表，查表后需**增加 1 次 All-Reduce（g 算子）**。
- **输出层与损失融合（Fused Cross-Entropy）**：
  - 输出嵌入与输入嵌入**共享权重**。若对并行 GEMM 的输出 logits 直接做 All-Gather，通信量将高达 $b \times s \times v$（batch × 序列长度 × 词表大小），因词表巨大而无法承受。
  - **解决方案**：把并行 GEMM 的输出**直接与交叉熵损失函数融合（fuse）**，把通信维度从 $b \times s \times v$ 骤降到 $b \times s$（只通信标量损失）。这是本文对通信量的巨大削减。

### 3.5 用冗余计算代替通信

对于 **LayerNorm、Dropout、残差连接（residual connection）**，为避免 GPU 间通信，作者选择在**所有 GPU 上复制参数并做冗余计算**。既然所有值要么是本地的、要么是复制的，就不需要通信更新后的参数值。优化器也允许每个模型并行 worker 独立更新自己那份参数。

**整体设计目标**：尽量减少通信、让 GPU 处于**计算受限（compute-bound）**而非通信受限的状态。整个方案实现简单，只需向前向 / 反向传播添加少量 All-Reduce，且与流水线并行正交互补。

---

## 4. 混合并行：模型并行 × 数据并行

- **正交组合**：模型并行与数据并行完全正交，可同时使用。总 GPU 数 = 模型并行度 × 数据并行度（例如 8 × 64 = 512）。
- **拓扑结构**：
  - 同一服务器内的若干 GPU（如 GPU 1–8）组成一个**模型并行组（model parallel group）**，共同持有一个模型实例的不同切分；
  - 所有服务器中处于**相同相对位置**的 GPU（如各服务器的第 1、9、…、505 号）组成一个**数据并行组（data parallel group）**，持有相同的模型参数。
- **梯度同步**：反向传播时，在每个数据并行组内独立并行地做多次梯度 All-Reduce。所有通信都通过 PyTorch 调用 **NCCL** 实现。

![Figure 8：8 路模型并行 + 64 路数据并行的 GPU 分组拓扑](/assets/posts/megatron/figure8_grouping.png)

**Figure 8 解读**（位于论文附录 B.1）：以 8 路模型并行 + 64 路数据并行为例。GPU 1–8 组成模型并行组 1，GPU 9–16 组成模型并行组 2，……，GPU 505–512 组成模型并行组 64。跨服务器、处于相同位置的 GPU（如所有组的第 1 号 GPU）组成数据并行组 1，第 8 号 GPU 组成数据并行组 8。模型并行组内做 All-Reduce（参数切分同步），数据并行组内做梯度 All-Reduce。

### 4.1 模型并行下的随机数生成（Random Number Generation）

Transformer 中存在两类 Dropout：**残差连接前的 Dropout（在模型并行区之外）**和**自注意力块内的 Dropout（在模型并行区之内）**，需要小心处理随机数生成：
- 为同步残差连接的 Dropout，在训练开始时用**相同种子**为各模型并行 worker 播种，使 Dropout 模式一致；
- 而模型并行区**内部**的 Dropout 需要各 worker 产生**不同**的随机模式，因此维护一个**单独且唯一播种**的随机数生成器。

---

## 5. 实验设置（Setup）

### 5.1 基础设施

- **硬件**：32 台 **DGX-2H** 服务器，共 **512 块 Tesla V100 SXM3 32GB GPU**。
- **带宽**：服务器内部通过 **NVSwitch** 提供 300 GB/s；服务器之间通过 8 个 **InfiniBand** 网卡提供 100 GB/s。

### 5.2 训练数据集（Training Dataset）

- 聚合了 Wikipedia、CC-Stories、RealNews、OpenWebtext 等大型语料。
- 为避免测试泄漏，移除了出现在 WikiText103 测试集里的 Wikipedia 文章，并清理 CC-Stories 的多余换行。BERT 训练额外加入 BooksCorpus，但 GPT-2 训练排除它（因与 LAMBADA 任务重叠）。
- 过滤掉内容长度小于 128 token 的文档；用**局部敏感哈希（Locality-Sensitive Hashing, LSH）**对 Jaccard 相似度大于 0.7 的内容去重。最终得到 **174 GB** 去重文本。

### 5.3 训练优化与超参数

- 使用**混合精度训练（mixed precision training）**配合动态损失缩放（dynamic loss scaling），发挥 V100 Tensor Core 性能。
- 权重初始化 $W \sim \mathcal{N}(0, 0.02)$，并在残差层前把权重按 $\frac{1}{\sqrt{2N}}$ 缩放（$N$ 为 Transformer 层数）。
- 优化器 **Adam**，权重衰减 $\lambda = 0.01$，全局梯度裁剪（gradient clipping）1.0，Dropout 0.1，每个 Transformer 层后做激活重计算。

---

## 6. 实验结果（Experiments）

所有实验最多使用 32 台 DGX-2H 服务器（共 512 块 V100）。

### 6.1 扩展性分析（Scaling Analysis）

- **强基线（Baseline）**：12 亿参数模型在**单卡**上维持 **39 TeraFLOPs**，是单卡理论峰值的 **30%**，是一个非常强的基线。
- 词表原为 50,257；为让 logit 层 GEMM 高效，把词表**填充（pad）到能被 $128 \times 8 = 1024$ 整除**，即 **51,200**。
- **模型并行 scaling**：固定 batch size = 8。
- **模型 + 数据并行 scaling**：固定全局 batch size = 512，对应 64 路数据并行。

**Table 1：扩展性研究所用的模型配置（每个注意力头的隐藏维度恒定为 96）**

| 隐藏层大小 (Hidden Size) | 注意力头数 (Attention heads) | 层数 (layers) | 参数量（十亿） | 模型并行 GPU | 模型 + 数据并行 GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1536 | 16 | 40 | 1.2 | 1 | 64 |
| 1920 | 20 | 54 | 2.5 | 2 | 128 |
| 2304 | 24 | 64 | 4.2 | 4 | 256 |
| 3072 | 32 | 72 | 8.3 | 8 | 512 |

![Figure 5：模型并行与模型 + 数据并行的弱扩展效率](/assets/posts/megatron/figure5_scaling.png)

**Figure 5 解读**：横轴为 GPU 数量，纵轴为弱扩展效率（Weak Scaling）。**纯模型并行（蓝色）**：1 GPU 为 100%，2 GPU 95%，4 GPU 82%，8 GPU 77%。**模型 + 数据并行（绿色）**：64 GPU 96%，128 GPU 83%，256 GPU 79%，512 GPU 74%。即 **8.3B 模型在 512 GPU 上仍达 74% 的线性扩展效率**（相对单卡 1.2B 强基线）。

### 6.2 GPT-2 语言建模结果

**Table 2：GPT-2 的模型配置（含单 Epoch 训练耗时）**

| 参数量 | 层数 | 隐藏层大小 | 注意力头数 | 每头隐藏维度 | 总 GPU 数 | 单 Epoch 耗时（天） |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 355M | 24 | 1024 | 16 | 64 | 64 | 0.86 |
| 2.5B | 54 | 1920 | 20 | 96 | 128 | 2.27 |
| 8.3B | 72 | 3072 | 24 | 128 | 512 | 2.10 |

> 注：355M 模型在配置上等同于 BERT-Large；8.3B 模型比历史上任何从左到右的 Transformer 语言模型都大。单 Epoch 相当于 68,507 次迭代。

![Figure 6：验证集困惑度收敛曲线（355M / 2.5B / 8.3B，均训练 300k 次迭代）](/assets/posts/megatron/figure6_perplexity.png)

**Figure 6 解读**：横轴为迭代次数（千次），纵轴为语言模型困惑度（LM Perplexity）。三条曲线自上而下为 355M（蓝）、2.5B（红）、8.3B（黄）。可见**模型越大，收敛越快，最终困惑度越低**，8.3B 最终收敛到约 **9.27** 的验证困惑度。

**Table 3：GPT-2 Zero-shot 评估结果**（WikiText103 SOTA 来自 Khandelwal et al. 2019，LAMBADA SOTA 来自 Radford et al. 2019）

| 模型 | WikiText103 困惑度 ↓ | LAMBADA 准确率 ↑ |
| :--- | ---: | ---: |
| 355M | 19.31 | 45.18% |
| 2.5B | 12.76 | 61.73% |
| **8.3B** | **10.81** | **66.51%** |
| Previous SOTA | 15.79 | 63.24% |

> 8.3B 模型在 WikiText103 上取得经恰当调整后 **10.81** 的困惑度，在 LAMBADA 上取得 **66.51%** 的完形填空准确率，双双超越此前 SOTA。作者还核算了测试集 8-gram 与训练集的重叠率（WikiText103 最多 10.8%，LAMBADA 最多 1.4%），确认无数据泄漏。微软后来用 Megatron 训练了 170 亿参数的 **Turing-NLG** 模型，进一步验证了大模型的价值。

### 6.3 BERT 双向 Transformer 结果

**Table 4：BERT 的模型配置**

| 参数量 | 层数 | 隐藏层大小 | 注意力头数 | 总 GPU 数 |
| ---: | ---: | ---: | ---: | ---: |
| 336M | 24 | 1024 | 16 | 128 |
| 1.3B | 24 | 2048 | 32 | 256 |
| 3.9B | 48 | 2560 | 40 | 512 |

**Table 5：BERT 下游任务结果（MNLI / QQP / SQuAD 1.1 / SQuAD 2.0 为验证集，RACE 为测试集）**

| 模型 | trained tokens 比 | MNLI m/mm (dev) | QQP (dev) | SQuAD 1.1 F1/EM (dev) | SQuAD 2.0 F1/EM (dev) | RACE m/h (test) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| RoBERTa | 2 | 90.2 / 90.2 | 92.2 | 94.6 / 88.9 | 89.4 / 86.5 | 83.2 (86.5/81.8) |
| ALBERT | 3 | 90.8 | 92.2 | 94.8 / 89.3 | 90.2 / 87.4 | 86.5 (89.0/85.5) |
| XLNet | 2 | 90.8 / 90.8 | 92.3 | 95.1 / 89.7 | 90.6 / 87.9 | 85.4 (88.6/84.0) |
| Megatron-336M | 1 | 89.7 / 90.0 | 92.3 | 94.2 / 88.0 | 88.1 / 84.8 | 83.0 (86.9/81.5) |
| Megatron-1.3B | 1 | 90.9 / 91.0 | 92.6 | 94.9 / 89.1 | 90.2 / 87.1 | 87.3 (90.4/86.1) |
| **Megatron-3.9B** | 1 | **91.4 / 91.4** | **92.7** | **95.5 / 90.0** | **91.2 / 88.5** | **89.5 (91.8/88.6)** |
| ALBERT ensemble | | | | 95.5 / 90.1 | 91.4 / 88.9 | 89.4 (91.2/88.6) |
| **Megatron-3.9B ensemble** | | | | **95.8 / 90.5** | **91.7 / 89.0** | **90.9 (93.1/90.0)** |

> Megatron-3.9B 单模型在多个验证集上超越 RoBERTa、ALBERT、XLNet；集成模型在 RACE 测试集达 **90.9%（93.1/90.0）**，刷新 SOTA。

### 6.4 关键改动：BERT 的 LayerNorm 位置重排

- **问题发现**：此前工作（Lan et al. 2019）发现，把 BERT-Large 从 3.36 亿参数继续增大到更大规模时，会出现**意外的性能退化（model degradation）**和训练不稳定。
- **解决方案（Rearrange）**：作者重排了 Transformer 层中**层归一化与残差连接的顺序**：
  - **原始架构 Post-LN**：$x_{out} = \text{LayerNorm}(x + \text{Sublayer}(x))$
  - **重排架构 Pre-LN**：$x_{out} = x + \text{Sublayer}(\text{LayerNorm}(x))$
- **效果**：这一改动消除了训练不稳定，降低了训练损失，并保证**下游任务性能随模型规模单调递增**，成功支撑了 3.9B BERT 的训练。这也是当今主流大模型普遍采用 Pre-LN 的原因之一。

![Figure 7：原始架构 (a) 与重排架构 (b) 的 BERT 训练损失对比](/assets/posts/megatron/figure7_bert_ln.png)

**Figure 7 解读**：
- 左侧两个结构图：**(a)** 为原始 Post-LN（LayerNorm 在残差相加之后）；**(b)** 为重排后的 Pre-LN（LayerNorm 前置到子层输入）。
- 右侧损失曲线：黄线为 336M 用架构 (a)，红线为 752M 用架构 (a)，蓝线为 752M 用架构 (b)。可见 **752M 用原始架构 (a)（红线）在约 220k 迭代处训练崩溃（loss 飙升到 6 以上）**，而 **752M 用重排架构 (b)（蓝线）训练稳定且损失最低**。

---

## 7. 附录中的补充数据

### 7.1 注意力头数量对扩展效率的影响

**Table 7：注意力头数对 8.3B 模型 8 路模型并行扩展效率的影响**

| 注意力头数 | 每头隐藏维度 | 扩展效率 |
| ---: | ---: | ---: |
| 16 | 192 | 82% |
| 24 | 128 | 80% |
| 32 | 96 | 77% |

> 头数增加会让自注意力层内的 GEMM 变小、Softmax 元素增多，导致扩展效率略降。设计大模型时需在模型速度与精度间权衡。

### 7.2 强扩展（Strong Scaling）

**Table 8：固定 12 亿参数模型、固定 batch size = 8 时的加速比**

| GPU 数 | 1 | 2 | 4 | 8 |
| ---: | ---: | ---: | ---: | ---: |
| 加速比 (Speedup) | 1.0 | 1.64 | 2.34 | 2.98 |

> 用 2 GPU 使训练快 64%；超过之后由于每卡计算量下降、带宽与通信开销占比上升，出现**收益递减（diminishing returns）**。

### 7.3 BERT 微调超参数（Table 6）

论文 Table 6 给出了 336M / 1.3B / 3.8B 模型在 MNLI、QQP、SQuAD 1.1/2.0、RACE 各任务上的 batch size、学习率、训练 epoch 数等超参数配置（例如 MNLI：batch 128、lr 1e-5、10 epochs）。

---

## 8. 结论与未来工作（Conclusion & Future Work）

- 本文仅通过对现有 PyTorch Transformer 实现做少量修改，就突破了「单卡一个模型」的传统限制，用 512 块 V100、8 路模型并行高效训练了 83 亿参数模型，维持 15.1 PetaFLOPs。
- 证明了对类 BERT 模型而言，**层归一化位置**对随规模增长获得精度提升至关重要。
- 在 WikiText103、LAMBADA、RACE 上确立新 SOTA，并开源了代码。
- **未来方向**：面向超过 160 亿参数模型的**层内 + 层间 + 节点间混合并行**；预训练不同模型族（XLNet、T5）；在更难、更多样的下游任务（生成式问答、摘要、对话）上评估；以及用**知识蒸馏（knowledge distillation）**从大教师模型训练小学生模型。

---

## 附录：专有名词中英对照表

| 中文 | 英文 |
| :--- | :--- |
| 模型并行 | Model Parallelism |
| 层内模型并行 | Intra-layer Model Parallelism |
| 张量并行 | Tensor Parallelism |
| 数据并行 | Data Parallelism |
| 流水线并行 | Pipeline Parallelism |
| 流水线气泡 | Pipeline Bubble |
| 分布式张量计算 | Distributed Tensor Computation |
| 列切分 / 列并行 | Column Parallel |
| 行切分 / 行并行 | Row Parallel |
| 通用矩阵乘法 | General Matrix Multiply (GEMM) |
| 归约通信 | All-Reduce |
| 聚集通信 | All-Gather |
| 恒等操作 | Identity |
| 共轭算子 | Conjugate Operator |
| 自注意力 | Self-Attention |
| 多头注意力 | Multi-Head Attention |
| 注意力头 | Attention Head |
| 查询 / 键 / 值 | Query (Q) / Key (K) / Value (V) |
| 前馈网络 / 多层感知机 | Multi-Layer Perceptron (MLP) |
| 层归一化 | Layer Normalization (LayerNorm) |
| 残差连接 | Residual Connection |
| 词表 | Vocabulary |
| 词嵌入 / 输入嵌入 | Embedding / Input Embedding |
| 融合交叉熵损失 | Fused Cross-Entropy Loss |
| 困惑度 | Perplexity (PPL) |
| 弱扩展 | Weak Scaling |
| 强扩展 | Strong Scaling |
| 扩展效率 | Scaling Efficiency |
| 收益递减 | Diminishing Returns |
| 计算受限 | Compute-bound |
| 混合精度训练 | Mixed Precision Training |
| 动态损失缩放 | Dynamic Loss Scaling |
| 激活重计算 | Activation Checkpointing |
| 优化器状态 | Optimizer State |
| 梯度裁剪 | Gradient Clipping |
| 局部敏感哈希 | Locality-Sensitive Hashing (LSH) |
| 零样本 | Zero-shot |
| 完形填空 | Cloze |
| 知识蒸馏 | Knowledge Distillation |
| 最优结果 | State-of-the-Art (SOTA) |
| 层归一化前置 / 后置 | Pre-LN / Post-LN |

---

## 附录：原始论文（Original Paper）

- **arXiv 地址**：<https://arxiv.org/abs/1909.08053>
- **标题**：Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism
- **开源代码**：<https://github.com/NVIDIA/Megatron-LM>
