---
title: "GPT-1: Improving Language Understanding by Generative Pre-Training"
date: 2026-09-11 09:00:00 +0800
permalink: /posts/gpt1-improving-language-understanding/
categories: [技术笔记, 深度学习]
tags: [gpt, gpt-1, transformer, 预训练, nlp, 论文精读, 大模型]
math: true
mermaid: false
toc: true
---

> 一份深入的中文阅读笔记。文中所有专有名词首次出现时附英文，文末附「专有名词中英对照附录」。论文全部配图（Figure 1–2）均已嵌入并配中文解读。这是 **GPT 系列的开山之作**，奠定了「生成式预训练 + 判别式微调」这一现代大模型范式。

---

## 0. 论文基本信息

| 项目 | 内容 |
| :--- | :--- |
| 标题 | **Improving Language Understanding by Generative Pre-Training**（通过生成式预训练提升语言理解） |
| 作者 | Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever |
| 机构 | OpenAI |
| 发表 | Preprint（2018），业界通称 **GPT-1** |
| 地位 | 首次系统性提出「生成式预训练（Generative Pre-Training）+ 判别式微调（Discriminative Fine-tuning）」两阶段范式，是 GPT 系列与现代大语言模型（Large Language Model, LLM）的起点 |

---

## 1. 摘要（Abstract）

- **痛点**：自然语言理解（Natural Language Understanding, NLU）涵盖文本蕴含、问答、语义相似度、文档分类等多样任务。无标签文本语料（unlabeled corpora）浩如烟海，但特定任务的**有标签数据（labeled data）稀缺**，判别式模型（discriminative model）难以充分训练。
- **方法**：先在多样化无标签语料上做**生成式预训练**，再在各具体任务上做**判别式微调**。
- **关键创新**：微调阶段引入**任务感知的输入转换（task-aware input transformations）**，把结构化输入转成单一连续序列，从而**几乎无需改动模型架构**即可迁移。
- **成绩**：在 12 个 NLU 基准中，**9 个任务上刷新最优结果（State-of-the-Art, SOTA）**。典型提升：常识推理 Story Cloze Test **+8.9%**，问答 RACE **+5.7%**，文本蕴含 MultiNLI **+1.5%**；GLUE 基准总分 **72.8**（此前最优 68.9）。

---

## 2. 引言（Introduction）：为什么要做无监督预训练

**核心动机**：深度学习依赖昂贵的人工标注，而利用海量无标签数据学习语言表征（representation）是破解数据稀缺的关键。即使有监督数据，无监督学习（如预训练词嵌入）也能带来显著增益。

**两大挑战**：

1. **优化目标怎么选**：什么样的训练目标能学到最利于迁移的文本表征？语言建模、机器翻译、语篇连贯性各有优劣，尚无定论。
2. **迁移方式怎么选**：如何把学到的表征有效迁移到目标任务？现有方法往往需要改架构、设计复杂学习方案或加辅助目标，缺乏通用范式。

**本文方案**：

- 采用**无监督预训练 + 有监督微调**的半监督（semi-supervised）两阶段方法，目标是学到只需极少适配就能迁移到广泛任务的**通用表征**。
- **架构选择 Transformer** 取代 RNN/LSTM：Transformer 提供更好的结构化记忆来处理长距离依赖（long-range dependencies），迁移更鲁棒。
- **输入适配用「遍历式（traversal-style）」方法**：把不同任务的结构化输入（句子对、文档+问题+选项）强制转成单一连续 token 序列，微调时**无需对模型做任何任务专属改动**。

---

## 3. 相关工作（Related Work）

- **NLP 半监督学习**：早期用无标签数据算词/短语级统计特征，后来广泛用词嵌入（Word2Vec、GloVe）。但这些方法主要迁移**词级**信息；本文旨在捕获**句子/段落级的更高层语义**。
- **无监督预训练**：预训练可视作一种正则化（regularization），提供更好的初始化。Dai 等人与 ULMFiT（Howard & Ruder）也用过「语言模型预训练 + 微调」，但用的是 **LSTM**，限制了长距离依赖能力；本文用 **Transformer** 克服这一点，并在更广任务上验证。相比把预训练隐层当辅助特征（需为每个任务引入大量新参数）的做法，本文迁移时**仅需极少额外参数**。
- **辅助训练目标（auxiliary training objectives）**：在监督任务里加无监督辅助目标（如语言模型目标）。本文在微调时也用了辅助 LM 目标，但强调**无监督预训练本身已学到大量对下游有用的语言知识**。

---

## 4. 框架（Framework）

整体两阶段：先在大语料上训练一个高容量语言模型，再在有标签数据上微调适配到判别式任务。下图左侧为 Transformer 架构与训练目标，右侧为不同任务的输入转换方式：

![Figure 1：（左）Transformer 架构与训练目标 （右）不同任务的输入转换](/assets/posts/gpt1/figure1_architecture.png)

**图 1 解读：**

- **左侧（架构与目标）**：底部 Text & Position Embed（词嵌入 + 位置嵌入）→ 堆叠 **12×** Transformer Block（每块含 Masked Multi Self-Attention → Layer Norm → Feed Forward → Layer Norm，均带残差相加 ⊕）→ 顶部分出 **Text Prediction**（语言模型目标）与 **Task Classifier**（任务分类目标）两个头。
- **右侧（输入转换）**：所有结构化输入都被拼成「Start（起始符）… Extract（抽取符/结束符）」的单一序列，送入同一个 Transformer 再接 Linear（线性层）：
  - **Classification（分类）**：`Start | Text | Extract`
  - **Entailment（蕴含）**：`Start | Premise | Delim | Hypothesis | Extract`
  - **Similarity（相似度）**：两种句子顺序各跑一遍，两路 Transformer 输出相加 ⊕ 后接 Linear
  - **Multiple Choice（多选问答）**：每个候选答案单独拼 `Start | Context | Delim | Answer_k | Extract`，各自过 Transformer+Linear 后经 Softmax 归一化

### 4.1 无监督预训练（Unsupervised Pre-training）

用标准自回归语言模型（auto-regressive language model）目标，最大化似然：

$$
L_1(\mathcal{U}) = \sum_i \log P\bigl(u_i \mid u_{i-k}, \dots, u_{i-1};\ \Theta\bigr)
$$

其中 $k$ 是上下文窗口大小，$\Theta$ 是神经网络参数，用随机梯度下降训练。

模型采用多层 **Transformer 解码器（Transformer Decoder，仅含 Masked Self-Attention）**。前向传播：

$$
h_0 = U W_e + W_p
$$

$$
h_l = \mathrm{transformer\_block}(h_{l-1}), \quad \forall l \in [1, n]
$$

$$
P(u) = \mathrm{softmax}(h_n W_e^{T})
$$

其中 $U=(u_{-k},\dots,u_{-1})$ 是上下文 token 向量，$n$ 是层数，$W_e$ 是词嵌入矩阵（token embedding matrix），$W_p$ 是位置嵌入矩阵（position embedding matrix）。

### 4.2 有监督微调（Supervised Fine-tuning）

对有标签数据集 $\mathcal{C}$，输入序列 $x^1,\dots,x^m$ 经预训练模型后，取最后一个 Transformer block 的激活 $h_l^m$，接一个线性输出层 $W_y$ 预测标签 $y$：

$$
P(y \mid x^1, \dots, x^m) = \mathrm{softmax}(h_l^m W_y)
$$

监督目标：

$$
L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y \mid x^1, \dots, x^m)
$$

**联合目标**（把语言模型目标作为辅助任务，权重 $\lambda$，可加速收敛、提升泛化）：

$$
L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C})
$$

**微调时唯一需要新增的参数**：线性分类层权重 $W_y$，以及分隔符（delimiter）token 的嵌入。

### 4.3 特定任务的输入转换（Task-specific Input Transformations）

用**遍历式**方法把结构化输入拼成单一连续序列，均含随机初始化的起始符 $\langle s \rangle$、结束符 $\langle e \rangle$ 和分隔符 `$`：

1. **文本蕴含（Textual Entailment）**：拼接前提 premise 与假设 hypothesis —— `<s> premise $ hypothesis <e>`
2. **语义相似度（Similarity）**：两句无固有顺序，故两种排列各输入一次，两路最后隐层状态逐元素相加再送分类器 —— `<s> sent1 $ sent2 <e>` 与 `<s> sent2 $ sent1 <e>`
3. **问答与常识推理（QA & Commonsense Reasoning）**：上下文 $z$、问题 $q$、候选答案集 $\{a_k\}$，每个答案单独拼 `<s> context $ question $ answer_k <e>`，各自过模型后经 softmax 输出答案概率分布

---

## 5. 实验（Experiments）

### 5.1 模型规格与超参数

**预训练数据**：**BooksCorpus**（7000+ 本未出版书籍，含冒险、奇幻、言情等，关键在于**长连续文本**利于学习长距离依赖）。模型在该语料上达到极低困惑度（Perplexity）**18.4**。

**模型架构**：12 层 Decoder-only Transformer

- 隐层维度（hidden size）：**768**
- 注意力头数（attention heads）：**12**
- 前馈内层维度（FFN inner dim）：**3072**

**分词与词表**：字节对编码（Byte-Pair Encoding, BPE），合并 **40,000** 次；用 `spaCy` 分词、`ftfy` 清洗。

**预训练超参数**：

| 项 | 值 |
| :--- | :--- |
| 优化器 | Adam，最大学习率 **2.5e-4** |
| 学习率调度 | 前 2000 步线性预热（warmup），后按余弦（cosine）退火到 0 |
| Batch size / 序列长度 | **64** / **512** tokens |
| 训练 Epochs | **100** |
| 正则化 | Dropout **0.1**；L2 权重衰减 $w=0.01$；权重初始化 $N(0,\ 0.02)$ |
| 激活函数 | **GELU**（Gaussian Error Linear Unit） |
| 位置编码 | **可学习（learned）** 位置嵌入（非正弦） |

**微调超参数**：分类器 Dropout **0.1**；学习率 **6.25e-5**；batch size **32**；多数任务 **3** 个 epoch 即收敛；线性学习率衰减，前 0.2% 步预热；辅助 LM 权重 $\lambda = 0.5$。

### 5.2 实验任务与数据集（表 1 / Table 1）

| 任务类型 | 数据集 |
| :--- | :--- |
| 自然语言推理（Natural Language Inference, NLI） | SNLI、MultiNLI、Question NLI (QNLI)、RTE、SciTail |
| 问答（Question Answering） | RACE、Story Cloze |
| 语义相似度（Sentence Similarity） | MSR Paraphrase Corpus (MRPC)、Quora Question Pairs (QQP)、STS Benchmark (STS-B) |
| 分类（Classification） | Stanford Sentiment Treebank-2 (SST-2)、CoLA |

### 5.3 自然语言推理结果（表 2 / Table 2，准确率 %）

| 方法 | MNLI-m | MNLI-mm | SNLI | SciTail | QNLI | RTE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ESIM + ELMo (5x) | – | – | 89.3 | – | – | – |
| CAFE (5x) | 80.2 | 79.0 | 89.3 | – | – | – |
| Stochastic Answer Network (3x) | 80.6 | 80.1 | – | – | – | – |
| CAFE (1x) | 78.7 | 77.9 | 88.5 | 83.3 | – | – |
| GenSen | 71.4 | 71.3 | – | – | 82.3 | 59.2 |
| Multi-task BiLSTM + Attn | 72.2 | 72.1 | – | – | 82.1 | **61.7** |
| **Finetuned Transformer LM（本文）** | **82.1** | **81.4** | **89.9** | **88.3** | **88.1** | 56.0 |

> 在 5 个 NLI 数据集中的 4 个上刷新 SOTA；仅 RTE（仅约 2490 训练样本）略逊于多任务 BiLSTM，作者推测小数据集可从多任务训练受益。

### 5.4 问答与常识推理结果（表 3 / Table 3，准确率 %）

| 方法 | Story Cloze | RACE-m | RACE-h | RACE |
| :--- | :---: | :---: | :---: | :---: |
| val-LS-skip | 76.5 | – | – | – |
| Hidden Coherence Model | 77.6 | – | – | – |
| Dynamic Fusion Net (9x) | – | 55.6 | 49.4 | 51.2 |
| BiAttention MRU (9x) | – | 60.2 | 50.3 | 53.3 |
| **Finetuned Transformer LM（本文）** | **86.5** | **62.9** | **57.4** | **59.0** |

> Story Cloze 提升高达 8.9%，RACE 整体提升 5.7%，凸显模型处理长距离上下文的能力。

### 5.5 语义相似度与分类结果（表 4 / Table 4）

> 评价指标：mc = Matthews 相关系数，acc = 准确率，pc = Pearson 相关系数。

| 方法 | CoLA (mc) | SST-2 (acc) | MRPC (F1) | STS-B (pc) | QQP (F1) | GLUE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Sparse byte mLSTM | – | 93.2 | – | – | – | – |
| TF-KLD | – | – | 86.0 | – | – | – |
| ECNU (mixed ensemble) | – | – | – | 81.0 | – | – |
| Single-task BiLSTM + ELMo + Attn | 35.0 | 90.2 | 80.2 | 55.5 | 66.1 | 64.8 |
| Multi-task BiLSTM + ELMo + Attn | 18.9 | 91.6 | 83.5 | 72.8 | 63.3 | 68.9 |
| **Finetuned Transformer LM（本文）** | **45.4** | 91.3 | 82.3 | **82.0** | **70.3** | **72.8** |

> CoLA 从 35.0 大幅跃升到 45.4，说明模型学到了内在的语言学归纳偏置（inductive bias）；GLUE 总分 72.8，超越此前 68.9 达 5.5%。整体在 12 个数据集里 9 个刷新 SOTA。

---

## 6. 分析（Analysis）

下图左为迁移层数的影响，右为零样本性能随预训练更新次数的演化：

![Figure 2：（左）迁移层数影响 （右）零样本性能随预训练更新的演化](/assets/posts/gpt1/figure2_analysis.png)

**图 2 解读：**

- **左图（迁移层数影响）**：横轴为迁移的层数（0–12），纵轴为准确率。RACE 与 MultiNLI 的 Dev/Train 曲线均**单调递增**——迁移的 Transformer 层越多性能越好，全量迁移相比只迁移嵌入层在 MultiNLI 上带来高达 **9%** 的提升，说明预训练模型**每一层都蕴含对下游有用的功能**。
- **右图（零样本行为）**：横轴为预训练更新次数（$10^3 \to 10^6$，对数轴），纵轴为归一化后的相对任务性能。多条实线为 Transformer 在各任务（情感分析、Winograd 指代消解、语言学可接受性、问答）上的零样本启发式表现，随训练**稳步上升**；虚线为 LSTM，**方差明显更大**，反衬 Transformer 的迁移稳定性。

### 6.1 迁移层数的影响（Impact of Number of Layers Transferred）

仅迁移嵌入层即有提升，随迁移层数增加性能单调递增；MultiNLI 上全量迁移带来最高 9% 增益。

### 6.2 零样本行为（Zero-shot Behaviors）

**假设**：生成式语言模型为了更好地预测下一个词，被迫学到许多下游任务所需的语言知识。**验证**：设计无需微调的启发式规则（heuristics）直接用 LM 概率预测：

- **CoLA**：算句子平均 token 对数概率，用阈值判断是否合语法。
- **SST-2**：句末追加 "very"，限制 LM 只输出 "positive"/"negative"，取概率高者。
- **RACE**：给定 context+question，算每个候选答案的平均 token 对数概率，取最高。
- **DPRD（Winograd）**：替换代词后算剩余序列概率，取概率高的指代对象。

**结论**：这些零样本表现随预训练步数稳步上升，直接证明**生成式预训练支撑了广泛任务相关能力的学习**；且 Transformer 的归纳偏置比 LSTM 更利于迁移。

### 6.3 消融研究（Ablation Studies，表 5 / Table 5）

- **去掉辅助 LM 目标（w/o aux LM）**：在 NLI、QQP 等**大数据集**上性能下降，小数据集上影响不大甚至微升。
- **Transformer 换成 LSTM**（单层 2048 单元）：平均分**下降 5.6 分**，仅在 MRPC 上 LSTM 胜出，证明 Transformer 架构的优越性。
- **去掉预训练（w/o pre-training，直接在监督数据上从头训 Transformer）**：所有任务性能**暴跌 14.8%**，凸显无监督预训练是核心价值所在。

---

## 7. 结论（Conclusion）

本文提出「生成式预训练 + 判别式微调」在**单一任务无关模型**上实现强大 NLU 的框架。通过在具有长距离依赖的多样连续文本（BooksCorpus）上预训练，模型获得丰富的世界知识与长文本处理能力，并成功迁移到问答、蕴含、相似度等判别式任务。研究证明：**Transformer 架构 + 长程依赖文本数据**是发挥无监督预训练威力的最佳组合，为后续 NLP 无监督学习指明方向。

---

## 8. 一句话总结

先用海量无标签文本把一个 Transformer 解码器「读」成通用语言模型，再用极少的任务专属改动微调——**生成式预训练 + 判别式微调**，正是 GPT 系列乃至整个大模型时代的起点。

---

## 附录 A：专有名词中英对照表

| 中文 | 英文 | 简要说明 |
| :--- | :--- | :--- |
| 生成式预训练 | Generative Pre-Training | 用语言模型目标在无标签文本上预训练 |
| 判别式微调 | Discriminative Fine-tuning | 在有标签任务上微调分类器 |
| 自然语言理解 | Natural Language Understanding (NLU) | 理解类 NLP 任务总称 |
| 半监督学习 | Semi-supervised Learning | 结合无标签与有标签数据 |
| 无标签语料 | Unlabeled Corpora | 没有人工标注的文本 |
| 语言模型 | Language Model (LM) | 建模文本序列概率 |
| 自回归 | Auto-regressive | 逐词依赖已生成内容 |
| Transformer 解码器 | Transformer Decoder | 仅含掩码自注意力的 Transformer |
| 掩码自注意力 | Masked Self-Attention | 禁止关注未来位置的注意力 |
| 词嵌入矩阵 | Token Embedding Matrix ($W_e$) | 词元到向量的映射 |
| 位置嵌入矩阵 | Position Embedding Matrix ($W_p$) | 注入位置信息 |
| 遍历式方法 | Traversal-style Approach | 把结构化输入拼成单一序列 |
| 任务感知输入转换 | Task-aware Input Transformations | 按任务把输入转成 token 序列 |
| 文本蕴含 | Textual Entailment | 判断前提是否蕴含假设 |
| 前提 / 假设 | Premise / Hypothesis | 蕴含任务的两段文本 |
| 语义相似度 | Sentence Similarity | 判断两句是否语义等价 |
| 问答 | Question Answering (QA) | 根据上下文回答问题 |
| 常识推理 | Commonsense Reasoning | 需要常识的推断 |
| 分隔符 / 起始符 / 结束符 | Delimiter / Start / Extract token | 拼接序列的特殊标记 |
| 辅助训练目标 | Auxiliary Training Objective | 主任务外附加的目标 |
| 困惑度 | Perplexity | 语言模型评价指标，越低越好 |
| 字节对编码 | Byte-Pair Encoding (BPE) | 子词切分方法 |
| 长距离依赖 | Long-range Dependencies | 序列中相距较远元素的关联 |
| 归纳偏置 | Inductive Bias | 模型对解空间的先验假设 |
| 零样本 | Zero-shot | 无需任务微调直接推断 |
| 消融研究 | Ablation Study | 移除组件以评估其贡献 |
| 正则化 | Regularization | 抑制过拟合的手段 |
| 高斯误差线性单元 | GELU | 一种激活函数 |
| 自然语言推理 | Natural Language Inference (NLI) | 判断句子对的蕴含/矛盾/中立关系 |
| 马修斯相关系数 | Matthews Correlation Coefficient (mc) | 分类评价指标 |
| 皮尔逊相关系数 | Pearson Correlation (pc) | 相关性评价指标 |
| 最优结果 | State-of-the-Art (SOTA) | 当前最佳水平 |
| 大语言模型 | Large Language Model (LLM) | 大规模预训练语言模型 |

---

## 附录 B：原始论文（Original Paper）

- **论文标题**：Improving Language Understanding by Generative Pre-Training
- **论文网址（PDF 原文）**：<https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf>
- **本站 PDF 副本**：[点击查看 / 下载 PDF](/assets/posts/gpt1/gpt1-improving-language-understanding.pdf)

> 本文所有配图与数据均出自上述原始论文，如需引用请以原文为准。
