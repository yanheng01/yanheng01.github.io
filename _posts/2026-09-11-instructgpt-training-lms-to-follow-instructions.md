---
title: "InstructGPT: Training language models to follow instructions with human feedback"
date: 2026-09-11 19:00:00 +0800
permalink: /posts/instructgpt-training-lms-to-follow-instructions/
categories: [技术笔记, 深度学习]
tags: [instructgpt, gpt-3, rlhf, 对齐, 强化学习, 人类反馈, 大模型, 论文精读, nlp]
math: true
mermaid: false
toc: true
---

> **原文标题（Original Title）**：Training language models to follow instructions with human feedback
> **作者（Authors）**：Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, Ryan Lowe（OpenAI）
> **发表时间**：2022 年 3 月（arXiv:2203.02155）
> **一句话总结**：单纯把语言模型（Language Model, LM）做得更大，并不会让它更懂用户意图；通过**基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）** 微调 GPT-3，得到的 **InstructGPT** 在人类偏好上大幅超越原版 GPT-3——**仅 1.3B 参数的 InstructGPT 就击败了 175B 的 GPT-3**，同时更真实、更少有毒输出，且几乎不损失公开 NLP 基准性能。

---

## 目录

1. [核心问题：对齐（Alignment）](#1-核心问题对齐alignment)
2. [方法总览：RLHF 三步走](#2-方法总览rlhf-三步走)
3. [数据集与标注](#3-数据集与标注)
4. [三个训练阶段的技术细节](#4-三个训练阶段的技术细节)
5. [实验结果](#5-实验结果)
6. [定性结果：泛化与简单错误](#6-定性结果泛化与简单错误)
7. [讨论、局限与广泛影响](#7-讨论局限与广泛影响)
8. [关键要点回顾](#8-关键要点回顾)
9. [附录 A：专有名词中英对照表](#附录-a专有名词中英对照表)
10. [附录 B：原文与 PDF 链接](#附录-b原文与-pdf-链接)

---

## 1. 核心问题：对齐（Alignment）

### 「更大」不等于「更懂你」

大语言模型（Large Language Models, LMs）可以通过「提示（Prompting）」执行各种自然语言处理（NLP）任务，但它们常常表现出**非预期行为（unintended behaviors）**：编造事实、生成带偏见或有毒的文本、不遵循用户指令。

根本原因在于**目标错位（Misalignment）**：大多数 LM 的预训练目标是「预测互联网网页上的下一个词元（token）」，这与用户真正想要的「有帮助且安全地遵循我的指令」是两个不同的目标。因此论文说——这些模型与用户**未对齐（misaligned）**。

### 对齐的三个标准：3H 原则

论文借用 Askell et al. (2021) 的语言，希望语言模型做到：

- **有用（Helpful）**：帮助用户完成任务，同时遵循显式指令与隐式意图。
- **诚实（Honest）**：不捏造信息、不误导用户（即真实性 Truthfulness）。
- **无害（Harmless）**：不对人或环境造成生理、心理或社会伤害，避免偏见与有毒内容。

本文的关键取舍：**不改动预训练过程**，而是聚焦于**微调阶段**的对齐，并且刻意使用**真实 API 用户提交的提示（Prompt）分布**来做对齐，让模型贴近真实使用场景，而非仅在学术数据集上表现好。

![Figure 1：各模型在 API 提示分布上的人类偏好胜率](/assets/posts/instructgpt/figure1.png)

> **Figure 1**. 各模型在我们的 API 提示分布上的人类评估结果，纵轴为「相对 175B SFT 模型的胜率（Win rate against SFT 175B）」，横轴为模型规模（1.3B / 6B / 175B）。五条曲线自上而下为 PPO-ptx、PPO、SFT、GPT (prompted)、GPT。可见 InstructGPT（PPO-ptx）及其无预训练混合的变体（PPO）显著优于 GPT-3 基线；**即便是 1.3B 的 PPO-ptx，其输出也比 175B 的 GPT-3 更受偏好**。全文误差棒均为 95% 置信区间。

---

## 2. 方法总览：RLHF 三步走

论文基于 GPT-3 架构，训练了 **1.3B、6B、175B** 三种规模的模型。核心流程是一个由三步组成的闭环：

![Figure 2：方法的三个步骤示意图](/assets/posts/instructgpt/figure2.png)

> **Figure 2**. 方法三步走示意图：(1) 监督微调（Supervised Fine-Tuning, SFT）；(2) 奖励模型（Reward Model, RM）训练；(3) 通过近端策略优化（Proximal Policy Optimization, PPO）针对奖励模型做强化学习。蓝色箭头表示该数据被用于训练某个模型。在第 2 步中，A–D 是模型生成、供标注员排序的样本。

三步概括如下：

1. **Step 1 —— 监督微调（SFT）**：从提示数据集中采样提示，让标注员写出「期望的输出」（示范数据 Demonstrations），用监督学习微调 GPT-3。
2. **Step 2 —— 奖励模型训练（RM）**：对同一提示采样多个模型输出，让标注员从好到坏排序（Ranking），用这些排序数据训练一个能预测「人类更偏好哪个输出」的奖励模型。
3. **Step 3 —— PPO 强化学习**：把奖励模型当作奖励函数，用 PPO 算法进一步微调 SFT 模型，使其输出最大化奖励。

---

## 3. 数据集与标注

### 提示（Prompt）从哪来

- **早期冷启动**：由标注员亲自撰写的提示（Plain、Few-shot、User-based 三类），用于训练最初的 InstructGPT 模型。
- **主体来源**：OpenAI API Playground 上真实用户提交给早期 InstructGPT 模型的提示，过滤掉含个人身份信息（PII）的内容，并对每个用户 ID 限制提示数量以保证多样性。

三个数据集的规模大致为：**SFT 数据约 13k 提示、RM 数据约 33k 提示、PPO 数据约 31k 提示**（PPO 阶段只用提示，不需要人工标注答案）。

### 真实用户到底在用模型做什么

论文对 API 提示做了用例分类，发现**开放式生成（Generation）占绝对多数**，而非传统的分类或问答任务：

| 用例类别（Use-case） | 占比 |
| :--- | :--- |
| 生成（Generation） | 45.6% |
| 开放式问答（Open QA） | 12.4% |
| 头脑风暴（Brainstorming） | 11.2% |
| 聊天（Chat） | 8.4% |
| 改写（Rewrite） | 6.6% |
| 摘要（Summarization） | 4.2% |
| 分类（Classification） | 3.5% |
| 其他 / 封闭式问答 / 抽取等 | 其余 |

> 对应论文 **Table 1（Distribution of use case categories）** 与 **Table 2（Illustrative prompts）**。这解释了为什么在公开 NLP 数据集（多为分类/QA）上微调，无法很好覆盖真实使用分布——分类与 QA 合计仅约 18%，而开放式生成与头脑风暴合计约 57%。

### 标注团队与标注维度

- 雇佣约 **40 名承包商（contractors）** 作为标注员，经过筛选测试挑选。
- 标注员对输出打分维度见 **Table 3**：整体质量（1–7 分 Likert 量表）、是否遵循指令、是否幻觉（Hallucination）、是否含性/暴力/仇恨等敏感内容等。
- 训练与评估时采用不同侧重：**训练时优先「有用性」**，评估时同时衡量有用、真实与无害。

---

## 4. 三个训练阶段的技术细节

### Step 1：监督微调（SFT）

在 GPT-3 预训练模型上，用标注员撰写的约 13k 条示范数据做监督学习微调。论文指出 SFT 在 1 个 epoch 后就在验证损失上过拟合，但继续训练更多 epoch 反而对 RM 分数和人类偏好有益。

### Step 2：奖励模型（RM）

- 移除原 GPT-3 的反嵌入层（unembedding layer），换成输出**标量奖励**的头，训练一个 **6B 参数的 RM**（作者提到 175B RM 训练不稳定，故选 6B）。
- 对每个提示让模型生成 **K 个（4–9 个）**输出，标注员排序；把同一提示下的 $\binom{K}{2}$ 个「两两比较对」放进**同一个 batch** 联合训练，避免过拟合并大幅提速。
- 损失函数基于 Bradley-Terry 偏好模型的成对交叉熵：

$$
\text{loss}(\theta) = -\frac{1}{\binom{K}{2}} \, \mathbb{E}_{(x,\, y_w,\, y_l)} \left[ \log \sigma\big( r_\theta(x, y_w) - r_\theta(x, y_l) \big) \right]
$$

其中 $y_w$ 是标注员更偏好的输出，$y_l$ 是较差的输出，$r_\theta$ 是奖励模型给出的标量分数。

### Step 3：PPO 与「对齐税」的解法

用 PPO 算法在 Bandit 环境中优化策略：输入提示，模型产出输出，RM 给出奖励。为防止模型为「刷奖励」而偏离语言能力（Reward Hacking），奖励中加入一个相对 SFT 模型的 **KL 散度惩罚项**：

$$
\text{objective}(\phi) = \mathbb{E}_{(x,y)\sim D_{\pi_\phi^{RL}}}\!\left[ r_\theta(x,y) - \beta \log\!\frac{\pi_\phi^{RL}(y\mid x)}{\pi^{SFT}(y\mid x)} \right] + \gamma\, \mathbb{E}_{x\sim D_{\text{pretrain}}}\!\left[ \log \pi_\phi^{RL}(x) \right]
$$

- **前两项** = RM 奖励 − β × KL 惩罚（约束模型不要偏离 SFT 太远）。
- **最后一项（PPO-ptx 关键创新）**：在 PPO 梯度里**混入预训练数据的对数似然梯度**（Pretraining Mix，系数 γ）。这是为了解决**对齐税（Alignment Tax）**——纯 PPO 会让模型在 SQuAD、DROP、HellaSwag、翻译等公开 NLP 任务上退化；混入预训练更新后即可修复这些退化，甚至在部分基准上超过 GPT-3，而几乎不损失人类偏好得分。

---

## 5. 实验结果

### 5.1 标注员显著偏好 InstructGPT

在测试集上，标注员在各个规模都明显更偏好 InstructGPT 的输出：

- **175B InstructGPT 对比 175B GPT-3：85% ± 3% 的情况下胜出**；
- 对比「精心设计少样本提示的 GPT-3（GPT prompted）」：**71% ± 4%** 胜出。
- 效果阶梯清晰：GPT（最差）< GPT (prompted) < SFT < PPO ≈ PPO-ptx（最好）。

![Figure 3：不同提示分布、不同标注群体下的偏好胜率](/assets/posts/instructgpt/figure3.png)

> **Figure 3**. 各模型相对 175B SFT 的胜率细分。左列为「提交给 GPT-3 的提示（GPT distribution）」，右列为「提交给 InstructGPT 的提示（Instruct distribution）」；上排为**留出标注员（Held-out workers）**，下排为**训练标注员（Training workers）**。结论：即便在为 GPT-3 设计的提示上（左），InstructGPT 依旧领先（仅在大模型上优势略收窄）；且在**从未参与训练数据生产的留出标注员**上结论一致，说明模型并非只是过拟合了训练标注员的偏好。

### 5.2 更细颗粒的元数据表现

![Figure 4：API 分布上的元数据表现](/assets/posts/instructgpt/figure4.png)

> **Figure 4**. API 分布上的元数据结果（因数据量原因跨模型规模合并）。四个面板分别为：**尝试执行正确指令（Attempts correct instruction）**、**遵循显式约束（Follows explicit constraints，如「用两段以内作答」）**、**幻觉（Hallucinations，越低越好）**、**输出语言适合客服场景（Uses language appropriate for customer assistant）**。除幻觉外 InstructGPT（PPO/PPO-ptx）全面高于 GPT；在**幻觉面板上 SFT/PPO 明显更低**——封闭域任务的编造率从 GPT-3 的约 **41% 降到约 21%**。

### 5.3 公开 NLP 微调数据集不足以替代真实分布

![Figure 5：与 FLAN、T0 的 Likert 打分对比](/assets/posts/instructgpt/figure5.png)

> **Figure 5**. 在 InstructGPT 提示分布上，用 1–7 分 Likert 量表对比各模型。FLAN 和 T0（在公开 NLP 数据集上做指令微调的模型）表现优于默认 GPT-3，与「加了好提示的少样本 GPT-3」相当，但**明显不如 InstructGPT（PPO-ptx）**。头对头比较中，175B InstructGPT 相对 FLAN 胜率 **78% ± 4%**、相对 T0 胜率 **79% ± 4%**。原因：公开数据集偏重易自动评测的分类/QA，缺乏真实用户输入的多样性。

### 5.4 真实性（Truthfulness）

![Figure 6：TruthfulQA 数据集结果](/assets/posts/instructgpt/figure6.png)

> **Figure 6**. TruthfulQA 数据集结果。**灰色条**表示「真实（truthful）」的比例，**彩色条**表示「既真实又有信息量（truthful and informative）」的比例；左右分别为「QA 提示」和「Instruction + QA 提示」两种设置。PPO 模型在真实性上小幅但显著优于 GPT-3（大致约两倍于 GPT-3 生成真实且有信息量答案的概率）。有趣的例外：1.3B PPO-ptx 略差于同规模 GPT-3。在「Instruction+QA」设置中，模型会在不确定时选择「我无可奉告」，倾向真实而非自信地说假话。

### 5.5 毒性（Toxicity）与偏见（Bias）

![Figure 7：RealToxicityPrompts 上的人类与自动评估](/assets/posts/instructgpt/figure7.png)

> **Figure 7**. RealToxicityPrompts 上对比人类评估（Human eval）与自动评估（Perspective API score）。横轴按提示类型分为「无提示（None）」和「要求尊重（Respectful）」，纵轴为毒性（Toxicity）。共标注 1,729 条提示、三个 175B 模型。结论：**在被要求「保持尊重」时，InstructGPT（PPO-ptx）毒性比 GPT-3 约低 25%**；但**在无提示时二者相近**，且若被恶意要求生成有毒内容，InstructGPT 反而更「听话」地生成更毒的内容。在 Winogender、CrowS-Pairs 等偏见数据集上，InstructGPT **并未显著改善偏见**。

### 5.6 减小对齐税（Alignment Tax）

- 纯 PPO 会在 SQuAD、DROP、HellaSwag、WMT 2015 法译英等任务上性能退化；
- **PPO-ptx（混入预训练更新）** 修复了大部分退化，甚至在 HellaSwag 上超过 GPT-3，且几乎不牺牲人类偏好。附录中还给出预训练混合系数、KL 系数、学习率等超参数的消融（对应论文附录的 Figure 33–38）。

---

## 6. 定性结果：泛化与简单错误

### 6.1 惊人的零样本泛化

![Figure 8：175B PPO-ptx 模型的泛化示例](/assets/posts/instructgpt/figure8.png)

> **Figure 8**. 175B PPO-ptx（InstructGPT 175B）与 175B GPT-3（无额外前缀）的对比示例（示例为精选以展示行为，但输出非精选）。(1) InstructGPT 能**遵循非英语（如法语）指令**，尽管有时仍会用英语作答；(2) 能更可靠地**对代码做问答与摘要**。而 GPT-3 需要更精细的提示工程。这说明对齐可以泛化到微调数据中占比极小的分布（非英语与代码）。

### 6.2 仍会犯的简单错误

![Figure 9：175B PPO-ptx 模型的简单错误示例](/assets/posts/instructgpt/figure9.png)

> **Figure 9**. 175B PPO-ptx 仍会犯的简单错误示例。(1) **虚假前提（false premise）**：当指令暗含错误前提（如「冥想后为什么要吃袜子」），模型有时会顺着假前提一本正经地编内容；(2) **过度端水（over-hedging）**：对简单问题（如「高速向南瓜射炮弹会怎样」）过度谨慎、给出模棱两可的多种答案，而不直接作答。此外，当指令含多个显式约束时性能也会下降。

---

## 7. 讨论、局限与广泛影响

### 7.1 我们究竟在对齐「谁」（Who are we aligning to?）

论文诚实地反思：模型对齐的并非抽象的「人类价值观」，而是**一小群特定人群的偏好**——负责标注的承包商（多为讲英语的美国或东南亚人群）、给出标注指南的 OpenAI 研究员，以及 API 早期候补名单（Waitlist）用户。这带来关于代表性、公平性与问责的深刻问题。

### 7.2 局限性

- **有害指令依然照做**：训练时优先「有用性」，导致用户若恶意要求生成有毒/危险内容，模型仍可能服从。
- **过度端水**：因标注员偏好「认知谦逊」，模型学会对简单问题也含糊其辞。
- **偏见改善有限**：在标准偏见基准上未见显著提升。
- 模型仍会被虚假前提误导、难以同时满足多个复杂约束。

### 7.3 广泛影响与成本

- **成本效益极高**：训练 175B SFT 约 4.9 Petaflops/s-days、175B PPO-ptx 约 60 Petaflops/s-days，相比 GPT-3 预训练的约 **3640 Petaflops/s-days** 只是零头。**「对齐现有模型」远比「盲目把模型做更大」更划算**。
- **双刃剑**：让模型更会听指令，也意味着恶意用户更容易引导它生成虚假信息或仇恨言论。对齐不是安全万能药，需与部署端限制、监管协同。

---

## 8. 关键要点回顾

1. **对齐 > 规模**：1.3B 的 InstructGPT 在人类偏好上击败 100 倍大的 175B GPT-3；175B InstructGPT 对 GPT-3 胜率 85%。
2. **RLHF 三步走**：SFT → 训练奖励模型（RM）→ PPO 强化学习，是本文奠定的经典范式。
3. **KL 惩罚防跑偏，PPO-ptx 消对齐税**：混入预训练梯度可在保住人类偏好的同时修复公开 NLP 基准退化。
4. **真实用户分布很重要**：真实 API 提示以开放式生成为主，公开数据集（FLAN/T0）不足以替代。
5. **更真实、更少毒（有条件）**：真实性约翻倍、封闭域幻觉 41%→21%、被要求尊重时毒性降约 25%；但偏见改善不明显，且对齐的是特定人群偏好。
6. **仍有短板**：虚假前提、过度端水、复杂多约束、恶意指令照做等。

---

## 附录 A：专有名词中英对照表

| 中文 | 英文（English） |
| :--- | :--- |
| 对齐 | Alignment |
| 未对齐 | Misaligned |
| 基于人类反馈的强化学习 | Reinforcement Learning from Human Feedback (RLHF) |
| 语言模型 | Language Model (LM) |
| 大语言模型 | Large Language Models (LMs) |
| 自然语言处理 | Natural Language Processing (NLP) |
| 词元 | Token |
| 提示 | Prompt |
| 提示工程 | Prompt Engineering |
| 有用 | Helpful |
| 诚实 | Honest |
| 无害 | Harmless |
| 真实性 | Truthfulness |
| 监督微调 | Supervised Fine-Tuning (SFT) |
| 奖励模型 | Reward Model (RM) |
| 近端策略优化 | Proximal Policy Optimization (PPO) |
| 预训练混合（PPO 变体） | PPO-ptx / Pretraining Mix |
| 对齐税 | Alignment Tax |
| 示范数据 | Demonstrations |
| 排序 | Ranking |
| 成对比较 | Pairwise Comparison |
| 布拉德利-特里模型 | Bradley-Terry Model |
| 交叉熵 | Cross-Entropy |
| KL 散度 | KL Divergence |
| 奖励作弊 | Reward Hacking |
| 老虎机环境 | Bandit Environment |
| 策略 | Policy |
| 标量奖励 | Scalar Reward |
| 反嵌入层 | Unembedding Layer |
| 李克特量表 | Likert Scale |
| 幻觉 | Hallucination |
| 个人身份信息 | Personally Identifiable Information (PII) |
| 承包商 / 标注员 | Contractors / Labelers |
| 留出标注员 | Held-out Workers |
| 训练标注员 | Training Workers |
| 候补名单 | Waitlist |
| 毒性 | Toxicity |
| 偏见 | Bias |
| 用例类别 | Use-case Categories |
| 生成 | Generation |
| 开放式问答 | Open QA |
| 封闭域问答 | Closed-domain QA |
| 头脑风暴 | Brainstorming |
| 改写 | Rewrite |
| 摘要 | Summarization |
| 分类 | Classification |
| 虚假前提 | False Premise |
| 过度端水 / 过度谨慎 | Over-hedging |
| 认知谦逊 | Epistemic Humility |
| 泛化 | Generalization |
| 过拟合 | Overfitting |
| 零样本 | Zero-shot |
| 少样本 | Few-shot |
| 浮点运算量单位 | Petaflops/s-days |

---

## 附录 B：原文与 PDF 链接

- **论文标题（Paper Title）**：Training language models to follow instructions with human feedback
- **论文网址（Paper URL）**：<https://arxiv.org/abs/2203.02155>
- **论文 PDF（Paper PDF，arXiv）**：<https://arxiv.org/pdf/2203.02155>
- **本站存档 PDF（Local PDF）**：[/assets/posts/instructgpt/instructgpt-paper.pdf](/assets/posts/instructgpt/instructgpt-paper.pdf)
- **arXiv 编号**：arXiv:2203.02155
- **发布机构**：OpenAI
