---
title: "GPT-2: Language Models are Unsupervised Multitask Learners"
date: 2026-09-11 17:00:00 +0800
permalink: /posts/gpt2-language-models-unsupervised-multitask-learners/
categories: [论文精读, 深度学习]
tags: [gpt, gpt-2, transformer, 零样本, 预训练, nlp, 论文精读, 大模型]
math: true
mermaid: false
toc: true
---

> **原文标题（Original Title）**：Language Models are Unsupervised Multitask Learners
> **作者（Authors）**：Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever（OpenAI）
> **发表时间**：2019 年
> **一句话总结**：只要在足够大且足够多样的文本上训练一个足够大的语言模型（Language Model, LM），它就能在**零样本（Zero-shot）**、不改动任何参数或架构的前提下，开始自发地完成问答、翻译、摘要、阅读理解等多种任务。

---

## 目录

1. [核心思想与动机](#1-核心思想与动机)
2. [方法：把一切任务都变成语言建模](#2-方法把一切任务都变成语言建模)
3. [训练数据集：WebText](#3-训练数据集webtext)
4. [输入表示：字节级 BPE](#4-输入表示字节级-bpe)
5. [模型架构与四种规模](#5-模型架构与四种规模)
6. [实验结果](#6-实验结果)
7. [泛化还是记忆？](#7-泛化还是记忆)
8. [讨论与结论](#8-讨论与结论)
9. [关键要点回顾](#9-关键要点回顾)
10. [附录 A：专有名词中英对照表](#附录-a专有名词中英对照表)
11. [附录 B：原文与 PDF 链接](#附录-b原文与-pdf-链接)

---

## 1. 核心思想与动机

### 当前系统的局限

论文开篇指出，当时主流的机器学习（Machine Learning, ML）系统虽然在特定任务上表现出色，但本质上是**「脆弱的窄领域专家」（brittle narrow experts）**：它们高度依赖大规模数据、高容量模型和监督学习（Supervised Learning），一旦数据分布或任务定义发生细微变化，性能就会急剧下降，缺乏真正的泛化能力。

作者怀疑，**在单一领域数据集上做单任务训练**，正是导致系统泛化能力差的根本原因。

### 多任务学习的困境

多任务学习（Multitask Learning）是一个有前景的方向，但在自然语言处理（Natural Language Processing, NLP）中仍处于起步阶段。当时最先进的尝试（如 decaNLP、MQAN）也只用到 10–17 组「（数据集，目标）」对。要让系统真正泛化，可能需要成百上千组，而人工构造这些数据集的成本极高、难以扩展。

### 本文的主张

论文将两条研究路线连接起来：

- **路线一**：预训练 + 微调（Pre-training + Fine-tuning）
- **路线二**：零样本语言模型（Zero-shot Language Model）

核心主张是：**语言模型可以在零样本设置下直接执行下游任务，无需任何参数更新或架构修改**。自然语言本身就提供了一种灵活的方式来描述任务、输入和输出，因此一个容量足够大的 LM 能够从文本序列中**推断并执行**其中所演示的任务。

---

## 2. 方法：把一切任务都变成语言建模

语言建模的本质是对一组变长符号序列 $(s_1, s_2, ..., s_n)$ 做无监督分布估计。由于语言天然具有顺序性，通常将联合概率分解为条件概率之积：

$$
p(x) = \prod_{i=1}^{n} p(s_n \mid s_1, ..., s_{n-1})
$$

关键洞察在于：一个通用系统应该能对**同一个输入**执行**许多不同任务**。也就是说，它建模的不应只是 $p(\text{output} \mid \text{input})$，而应是 $p(\text{output} \mid \text{input}, \text{task})$——即**以任务为条件（Task Conditioning）**。

而语言提供了一种优雅的方式来「顺带」指定任务，例如：

- 翻译训练样本可以写成序列：`(translate to french, english text, french text)`
- 阅读理解样本可以写成序列：`(answer the question, document, question, answer)`

因此，只要 LM 学会了续写这类自然语言序列，它其实就在无监督地学习执行这些任务。

---

## 3. 训练数据集：WebText

为了收集尽可能多样的「任务演示」，作者构建了一个全新的大规模数据集 **WebText**：

- **数据来源**：不使用质量参差的 Common Crawl，而是抓取社交平台 **Reddit** 上获得至少 **3 个 karma（点赞）** 的外链——用人类的点赞作为「内容有价值」的启发式过滤信号。
- **数据清洗**：使用 Dragnet 与 Newspaper 提取正文；移除维基百科（Wikipedia）内容以避免与常见测试集重叠；移除 2017 年 12 月之后的链接。
- **最终规模**：约 4500 万个链接 → 去重清洗后得到 **800 万个文档**，共 **40GB** 文本。

---

## 4. 输入表示：字节级 BPE

### 痛点

- 纯字节级（Byte-level）语言模型在大规模数据集上的表现不如词级（Word-level）模型。
- 传统的字节对编码（Byte Pair Encoding, BPE）作用在 Unicode 码点上，会导致词表爆炸（超过 130,000）。
- 若直接对字节做 BPE，会产生次优的合并——例如 `dog`、`dog.`、`dog!` 被合并成不同 token，白白浪费词表容量。

### 解决方案：字节级 BPE（Byte-level BPE）

- 基础词表仅需 **256**（字节的全部取值）。
- **关键改进**：禁止 BPE 跨越不同**字符类别（character categories）**进行合并，但对空格做例外处理。
- **效果**：既保留了词级模型的性能优势，又保留了字节级的通用性——无需有损预处理或分词，可以对任意 Unicode 字符串、任意数据集计算概率。

---

## 5. 模型架构与四种规模

模型基于 **Transformer**，在原始 GPT 的基础上做了如下修改：

- 将层归一化（Layer Normalization, LayerNorm）移到每个子块的**输入端**（类似 Pre-LN）；
- 在最后一个自注意力块之后额外增加一个 LayerNorm；
- 残差层权重按 $1/\sqrt{N}$ 缩放初始化（$N$ 为残差层数量）；
- 词表大小（Vocabulary）扩展到 **50,257**；
- 上下文长度（Context size）从 512 提升到 **1024** 个 token；
- 批大小（Batch size）为 **512**。

### 四种模型规模（对应原文 Table 2）

| 参数量（Parameters） | 层数（Layers） | 隐藏维度 $d_{model}$ | 备注 |
| :--- | :---: | :---: | :--- |
| 117M | 12 | 768 | 相当于原始 GPT |
| 345M | 24 | 1024 | 相当于 BERT-large |
| 762M | 36 | 1280 | — |
| **1542M（即 GPT-2）** | **48** | **1600** | 比 GPT 大一个数量级 |

> 最大的模型即通常所说的 **GPT-2**，拥有 15 亿（1.5B）参数。

---

## 6. 实验结果

> ⚠️ 以下所有结果均为**零样本（Zero-shot）**：没有做任何微调或特定任务训练。

### 6.0 总览图：性能随模型规模增长（Figure 1）

![Figure 1：WebText LM 在多个 NLP 任务上的零样本表现随模型规模变化](/assets/posts/gpt2/figure1.png)

> **Figure 1**. WebText LM 在多个 NLP 任务上的零样本表现，作为模型规模的函数。四个子图分别为阅读理解（CoQA）、翻译（WMT-14 Fr-En）、摘要（CNN and Daily Mail）、问答（Natural Questions）。横轴为参数量（117M → 1542M），可见随模型增大，各任务性能均呈**对数线性（log-linear）**提升趋势，并逐步逼近或超越各类基线（如 Human、DrQA+PGNet、Unsupervised Statistical MT、Lead-3 等）。

### 6.1 语言建模（Language Modeling）

在 8 个语言建模数据集上测试零样本领域迁移，**其中 7 个刷新了当时的 SOTA**（对应原文 Table 3）：

| 数据集 | 指标 | 先前 SOTA | GPT-2 (1542M) |
| :--- | :--- | :---: | :---: |
| LAMBADA | 困惑度 PPL | 99.8 | **8.63** |
| LAMBADA | 准确率 ACC | — | **63.24%** |
| CBT-CN | ACC | 85.7 | **93.30%** |
| CBT-NE | ACC | 82.3 | **89.05%** |
| WikiText-2 | PPL | 39.14 | **18.34** |
| PTB (Penn Treebank) | PPL | 46.54 | **35.76** |
| enwik8 | BPB | 0.99 | **0.93** |
| text8 | BPC | 1.08 | **0.98** |
| WikiText-103 | PPL | 18.3 | **17.48** |
| 1BW (1 Billion Word) | PPL | **21.8** | 42.16 ❌ |

> 唯一未达 SOTA 的是 **1BW**——因为该数据集在预处理时做了破坏性的句子级打乱（sentence-level shuffling），摧毁了所有长程结构，对擅长长程依赖的 GPT-2 极为不利。

### 6.2 儿童图书测试（Children's Book Test, CBT / Figure 2）

![Figure 2：儿童图书测试性能随模型容量变化](/assets/posts/gpt2/figure2.png)

> **Figure 2**. GPT-2 在儿童图书测试上的表现随模型容量增长的曲线（左：普通名词 Common Nouns；右：命名实体 Named Entities）。人类表现（Human）来自 Bajgar et al. (2016)。可见性能随规模稳步提升，逐步缩小与人类的差距。GPT-2 在普通名词上达 **93.3%**、命名实体上达 **89.1%**，均为 SOTA。

### 6.3 LAMBADA

LAMBADA 测试长程依赖建模能力（需至少 50 个 token 的上下文才能预测最后一个词）：

- 困惑度从先前的 99.8 骤降至 **8.6**；
- 准确率从 19% 提升到 **52.66%**；
- 加入**停用词过滤器（stop-word filter）**后进一步升至 **63.24%**，整体 SOTA 提升约 4%。

### 6.4 Winograd 模式挑战（Winograd Schema Challenge, WSC / Figure 3）

![Figure 3：Winograd 模式挑战性能随模型容量变化](/assets/posts/gpt2/figure3.png)

> **Figure 3**. GPT-2 在 Winograd 模式挑战上的表现随模型容量增长的曲线。蓝线为部分打分（Partial Scoring），橙线为完整打分（Full Scoring），灰色虚线为先前 SOTA。GPT-2 将 SOTA 提升约 7%，达到 **70.70%**。注意该数据集仅有 273 个样本，规模很小，结果需谨慎解读。

### 6.5 阅读理解（Reading Comprehension, CoQA）

- **输入构造**：`[文档] + [对话历史] + "A:"`，采用贪心解码（Greedy decoding）。
- **结果**：在开发集上达到 **55 F1**。
- **意义**：在**完全没有使用** CoQA 的 127,000+ 人工标注问答对的情况下，就匹配或超过了 4 个基线系统中的 3 个（人类约为 89 F1）。
- **局限**：GPT-2 常使用简单的检索式启发（例如遇到 "who" 问题就从文档中抽取一个名字）。

### 6.6 摘要生成（Summarization, CNN and Daily Mail / Table 4）

- **输入构造**：在文章末尾添加提示词 `TL;DR:`，再用 Top-k 随机采样（k=2）生成 100 个 token，取前 3 句作为摘要。
- **结果（ROUGE F1）**：

| 方法 | R-1 | R-2 | R-L | R-AVG |
| :--- | :---: | :---: | :---: | :---: |
| Bottom-Up Sum（SOTA） | 41.22 | 18.68 | 38.34 | 32.75 |
| Lede-3 | 40.38 | 17.66 | 36.62 | 31.55 |
| Seq2Seq + Attn | 31.33 | 11.81 | 28.83 | 23.99 |
| **GPT-2 `TL;DR:`** | 29.34 | 8.27 | 26.58 | **21.40** |
| Random-3 | 28.78 | 8.63 | 25.52 | 20.98 |
| GPT-2 无提示词 | 21.58 | 4.03 | 19.47 | **15.03** |

> 关键对照：一旦**移除 `TL;DR:` 提示词**，聚合指标骤降 6.4 分（21.40 → 15.03），有力证明了「用自然语言提示即可唤起语言模型的特定任务行为」。

### 6.7 机器翻译（Translation, WMT-14）

- **输入构造**：给出若干个 `english sentence = french sentence` 的示例，然后给 `english sentence =` 让模型续写。
- **结果**：
  - 英译法（En-Fr）：**5 BLEU**（较弱）；
  - 法译英（Fr-En）：**11.5 BLEU**（借助其强大的英语建模能力，优于多个无监督翻译基线）。
- **惊人之处**：WebText 在构建时**特意移除了非英语网页**，其中法语数据仅约 10MB（比先前工作小约 500 倍），但模型依然「涌现」出了翻译能力。

### 6.8 问答（Question Answering, Natural Questions / Table 5）

- **输入构造**：提供若干短答案风格的问答示例作为上下文。
- **结果**：精确匹配（Exact Match）准确率为 **4.1%**（远超「只回答最常见答案」的 1.0% 基线）。
- **置信度校准**：在模型**最自信的前 1%** 问题上，准确率高达 **63.1%**。
- Table 5 列出了置信度最高的 30 个问题，例如：「Who wrote the book the origin of species?」→ Charles Darwin（83.4%）。

---

## 7. 泛化还是记忆？

为排除「GPT-2 只是背下了测试集（数据泄露）」的质疑，作者做了严格的 **8-gram 重叠分析**：

- **方法**：用布隆过滤器（Bloom filters）检测 WebText 训练集与各测试集之间的 8-gram 重叠，假阳性率 < $10^{-8}$。
- **结果（对应原文 Table 6）**：常见 LM 测试集与 WebText 的重叠率仅约 **1%–6%**（平均 3.2%）；相比之下，许多数据集**自身**的 train/test 重叠率反而更高（平均 5.9%，如 1BW 自身高达 13.19%）。
- **逐任务排查**：
  - **WSC**：仅 10 个 schema 有重叠，其中仅 1 个泄露答案；
  - **CoQA**：新闻领域有约 15% 重叠（带来约 3 F1 提升），但**问答对本身零重叠**（CoQA 发布时间晚于 WebText 链接截止日期）；
  - **LAMBADA**：仅 1.2% 重叠，排除重叠样本后 PPL 仅从 8.6 变为 8.7，准确率从 63.2% 降到 62.9%，影响微乎其微。

### 训练/测试困惑度同步下降（Figure 4）

![Figure 4：WebText 训练/测试困惑度随模型规模同步下降](/assets/posts/gpt2/figure4.png)

> **Figure 4**. 在 WebText 上训练的 LM 的困惑度随模型规模变化的曲线（蓝：WebText 训练集；橙：WebText 测试集）。两条曲线随模型增大**同步下降**，说明 GPT-2 **仍在欠拟合（underfitting）** WebText——其能力来自真正的泛化学习，而非记忆。

### 样本与训练集的 8-gram 重叠分布（Figure 5）

![Figure 5：样本与测试集的 8-gram 训练集重叠 CDF](/assets/posts/gpt2/figure5.png)

> **Figure 5**. WebText 测试集与 GPT-2 生成样本（以 WebText 测试集为条件、top-k 截断随机采样，k=40）分别与训练集的 8-gram 重叠百分比的累积分布函数（Empirical CDF）。大多数样本重叠 **低于 1%**，其中超过 30% 的样本**零重叠**；而测试集自身的重叠中位数为 2.6%。说明 GPT-2 复制训练集文本的频率**低于**留出文章的基线率。

---

## 8. 讨论与结论

- **理论意义**：无监督任务学习是一个极具前景的方向。这也解释了为何「预训练 + 微调」在 NLP 中如此成功——在极限情况下，预训练本身就已经开始直接学习任务，无需监督适配。
- **实际局限**：尽管 CoQA 上的表现令人兴奋，但在摘要等任务上定量指标仍很初级（rudimentary）。就实用性而言，GPT-2 的零样本性能**离可用仍有很大距离**。
- **容量瓶颈**：在问答、翻译等任务上，只有当模型容量足够大时，才开始超越平凡基线。
- **未来工作**：计划在 decaNLP 和 GLUE 上探索**微调（Fine-tuning）**的潜力，以弥补单向表示相较 BERT 双向表示的不足。
- **最终结论**：当大型语言模型在足够大且足够多样的数据集上训练时，它能在众多领域和任务上表现出色。**最大化多样化文本语料似然的高容量模型，会开始学习执行数量惊人的任务，且无需显式监督。**

---

## 9. 关键要点回顾

1. **零样本（Zero-shot）是核心贡献**：无需微调、无需改架构，仅靠自然语言提示唤起任务能力。
2. **规模就是关键**：模型容量增大，各任务性能呈对数线性提升——这直接启发了后续 GPT-3 的「规模法则（Scaling Laws）」路线。
3. **数据质量与多样性**：用 Reddit karma 做启发式过滤，构建了高质量、多样的 WebText。
4. **字节级 BPE**：兼顾通用性与性能，无需有损预处理。
5. **泛化而非记忆**：严格的 8-gram 重叠分析证明，模型能力来自泛化，且仍在欠拟合数据。
6. **提示（Prompt）的力量**：`TL;DR:` 实验首次清晰展示了「提示词」对唤起任务行为的决定性作用，是 Prompt Engineering 的早期雏形。

---

## 附录 A：专有名词中英对照表

| 中文 | 英文（English） |
| :--- | :--- |
| 语言模型 | Language Model (LM) |
| 无监督多任务学习者 | Unsupervised Multitask Learners |
| 零样本 | Zero-shot |
| 监督学习 | Supervised Learning |
| 多任务学习 | Multitask Learning |
| 预训练 + 微调 | Pre-training + Fine-tuning |
| 自然语言处理 | Natural Language Processing (NLP) |
| 机器学习 | Machine Learning (ML) |
| 脆弱的窄领域专家 | brittle narrow experts |
| 任务条件化 | Task Conditioning |
| 字节对编码 | Byte Pair Encoding (BPE) |
| 字节级 BPE | Byte-level BPE |
| 字符类别 | character categories |
| 词表 | Vocabulary |
| 上下文长度 | Context size |
| 批大小 | Batch size |
| 层归一化 | Layer Normalization (LayerNorm) |
| 隐藏维度 | $d_{model}$ |
| 困惑度 | Perplexity (PPL) |
| 每字节比特 | Bits Per Byte (BPB) |
| 每字符比特 | Bits Per Character (BPC) |
| 儿童图书测试 | Children's Book Test (CBT) |
| 命名实体 | Named Entities (NE) |
| 普通名词 | Common Nouns (CN) |
| Winograd 模式挑战 | Winograd Schema Challenge (WSC) |
| 常识推理 | commonsense reasoning |
| 阅读理解 | Reading Comprehension |
| 对话式问答数据集 | Conversation Question Answering (CoQA) |
| 贪心解码 | Greedy decoding |
| 摘要生成 | Summarization |
| 随机采样 | random sampling |
| 提示词 | Prompt / hint |
| 机器翻译 | Machine Translation |
| 双语评估替补（翻译指标） | BLEU |
| 面向摘要的评估指标 | ROUGE |
| 问答 | Question Answering (QA) |
| 精确匹配 | Exact Match |
| 自然问题数据集 | Natural Questions |
| 泛化 | Generalization |
| 记忆 | Memorization |
| 数据泄露 | data leakage |
| 布隆过滤器 | Bloom filters |
| 累积分布函数 | Empirical CDF |
| 欠拟合 | underfitting |
| 过拟合 | overfitting |
| 对数线性 | log-linear |
| 规模法则 | Scaling Laws |
| 十亿词基准 | 1 Billion Word Benchmark (1BW) |
| 宾州树库 | Penn Treebank (PTB) |

---

## 附录 B：原文与 PDF 链接

- **论文 PDF（Paper PDF）**：<https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>
- **论文标题**：Language Models are Unsupervised Multitask Learners
- **发布机构**：OpenAI

---

> 本阅读笔记由 AI 辅助精读整理，配图均从原文 PDF 提取。如与原文有出入，请以原文为准。
