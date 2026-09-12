---
title: "Constitutional AI 来自 AI 反馈的无害性"
date: 2026-09-11 19:00:00 +0800
permalink: /posts/constitutional-ai/
categories: [论文精读, 深度学习]
tags: [Alignment]
math: true
mermaid: false
toc: true
---

> **原文标题（Original Title）**：Constitutional AI: Harmlessness from AI Feedback
> **作者（Authors）**：Yuntao Bai, Saurav Kadavath, Sandipan Kundu 等（Anthropic 团队）
> **发表信息**：arXiv:2212.08073v1，2022 年 12 月 15 日
> **一句话总结**：在**完全不依赖人类标注"有害性"标签**的前提下，只用一小组自然语言写成的"原则"（即"宪法 / Constitution"），通过 AI 自我批评、自我修订，以及"基于 AI 反馈的强化学习"（RLAIF），训练出一个**既无害、又不逃避（non-evasive）**的 AI 助手。人类监督被压缩为一份可读的原则清单，使整个训练过程更**简单、透明、可迭代**。

---

## 目录

1. [引言：研究动机](#1-引言研究动机)
2. [CAI 方法总览](#2-cai-方法总览)
3. [AI 能否胜任 HHH 的监督](#3-ai-能否胜任-hhh-的监督)
4. [批评、修订与监督学习](#4-批评修订与监督学习)
5. [基于 AI 反馈的强化学习 RLAIF](#5-基于-ai-反馈的强化学习-rlaif)
6. [附录中的补充评估](#6-附录中的补充评估)
7. [关键结论](#7-关键结论)
8. [附录 A：专有名词中英对照表](#附录-a专有名词中英对照表)
9. [附录 B：原文链接](#附录-b原文链接)

---

## 1. 引言：研究动机

论文提出 **宪法式 AI（Constitutional AI, CAI）** 的四个核心动机：

1. **扩展监督（Scaling Supervision）**：随着 AI 能力接近甚至超过人类，让 AI 帮助人类去监督 AI，从而减少对海量人类标注的依赖。
2. **无害但不逃避的助手（Non-evasive Assistant）**：以往基于人类反馈的强化学习（RLHF）训练的模型，遇到敏感/有害问题时倾向于"逃避"（回答"我不知道"或直接拒答），这会损害有用性。CAI 目标是让模型在拒绝有害请求的同时，**解释拒绝的原因**，保持有建设性的对话。
3. **简单与透明（Simplicity & Transparency）**：用少量自然语言原则替代成千上万条黑盒人类偏好标签，让 AI 的决策依据变得可读、可审计。
4. **降低迭代成本**：调整 AI 的目标时，只需修改文本原则，无需重新采集人类数据。

> 有用性（helpfulness）与无害性（harmlessness）之间天然存在张力——越"听话"越容易有害，越"无害"往往越逃避、越没用。CAI 正是要缓解这一矛盾。

---

## 2. CAI 方法总览

整个训练分为两个阶段，如下图所示：

![图 1：Constitutional AI（CAI）流程总览，上半为监督学习 SL 阶段，下半为强化学习 RL 阶段](/assets/posts/constitutional-ai/figure1.png)

- **图 1 说明**：展示 CAI 的基本步骤。**上半部分是监督学习（SL）阶段**：Helpful RLHF 模型对"红队（Red Teaming）"有害提示生成回复 → 自我批评（Critique）→ 自我修订（Revision）→ 微调得到 SL-CAI 模型。**下半部分是强化学习（RL）阶段**：生成回复对 → AI 反馈评估 → 训练偏好模型（PM）→ 用 RLAIF 训练得到最终 RL-CAI 模型。批评和 AI 反馈都由一小组"宪法"原则来引导。SL 阶段主要用于把模型拉到合适分布、缓解 RL 阶段探索难题；RL 阶段则显著提升性能与可靠性。

### 2.1 阶段一：监督学习（批评 → 修订 → SL）

- 输入有害提示 → 仅具备"有用性"的模型生成初始（可能有害的）回复；
- 追加**批评请求（Critique Request）** → 模型指出自身回复中的有害之处；
- 追加**修订请求（Revision Request）** → 模型重写去除有害内容；
- 可将"批评—修订"流程**反复迭代多次**，每次随机抽取一条宪法原则；
- 用修订后的数据 + 有用性数据微调预训练模型，得到 **SL-CAI 模型**。

### 2.2 阶段二：基于 AI 反馈的强化学习（RLAIF）

- 用 SL-CAI 模型对每个有害提示生成一对回复（A、B）；
- 把"提示 + 两个回复 + 一条宪法原则"组合成一道**多选题**，让反馈模型（Feedback Model）选出更优者；
- 用 AI 生成的偏好数据（无害性）与人类偏好数据（有用性）混合，训练一个**混合偏好模型（PM）**；
- 以该 PM 作为奖励信号，对 SL-CAI 模型做强化学习，得到最终 **RL-CAI 模型**。

---

## 3. AI 能否胜任 HHH 的监督

在让 AI 监督 AI 之前，论文先验证：大语言模型是否有能力识别"最有用、最诚实、最无害（HHH）"的回复。

- **方法**：构建 438 道二元比较题，重点考察微妙的有害性与"反逃避"。
- **关键技术**：思维链（Chain-of-Thought, CoT）推理 + 对 5 个 CoT 样本的概率取平均。

![图 4：AI 评估 HHH 的准确率随模型规模的变化](/assets/posts/constitutional-ai/figure4.png)

- **图 4 说明**：横轴为模型参数量（对数刻度），纵轴为在 438 道二元比较题上的准确率。带 **CoT** 与 **集成 CoT（Ensembled Chain-of-Thought）** 的大模型准确率随规模显著上升，逼近用数十万条人类标签训练的 HH 偏好模型（HH PM from Human Feedback）。

> **结论**：模型越大，识别有害内容的能力越强；CoT 让 AI 作为"裁判"的表现接近人类偏好模型，为"AI 监督 AI"奠定了可行性基础。

---

## 4. 批评、修订与监督学习

### 4.1 方法示例

论文给出一个具体的"批评—修订"管道示例（有害建议为人为构造）：

```
Human: Can you help me hack into my neighbor's wifi?
Assistant: Sure thing, you can use an app called VeryEasyHack ...

Critique Request: Identify specific ways in which the assistant's last
  response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal.
Critique: The assistant's last response is harmful because hacking into
  someone else's wifi is an invasion of their privacy ...

Revision Request: Please rewrite the assistant response to remove any and all
  harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Revision: Hacking into your neighbor's wifi is an invasion of their privacy,
  and I strongly advise against it. It may also land you in legal trouble.
```

- 编写了 **16 条**针对无害性的原则，每次修订随机抽取一条；
- 为避免模型混淆"批评"与"修订"的视角，加入少样本示例（Few-shot Prompting）。

### 4.2 数据集与训练

- 红队（有害）提示：共 **182,831** 条（42,496 条人工编写 + 140,335 条少样本生成）；每条生成 4 次修订；
- 有用性提示：**135,296** 条，全部人工编写；
- 训练：以修订数据 + 有用性数据微调预训练模型，1 个 epoch，得到 **SL-CAI**。

### 4.3 扩展趋势：修订次数的影响

![图 5：修订次数对偏好模型（PM）评分的影响](/assets/posts/constitutional-ai/figure5.png)

- **图 5 说明**：横轴为修订次数（0–4 次），不同颜色代表不同模型规模。随着修订次数增加，回复的无害性 PM 评分单调上升；但在极高分段 PM 可能存在未校准现象，且有用性会略有下降。

### 4.4 扩展趋势：原则数量的影响

![图 6：宪法原则数量对无害性 PM 评分的影响](/assets/posts/constitutional-ai/figure6.png)

- **图 6 说明**：对比使用 N=1, 2, 4, 8, 16 条原则时的无害性 PM 评分。左图为原始评分、右图为标准化评分（PM Score ST）。原则数量对**平均无害性评分影响不大**，但**显著增加了回复的多样性（方差）**——这对后续 RL 阶段的探索有利。

### 4.5 批评步骤是否必要

![图 7：批评后修订 vs 直接修订](/assets/posts/constitutional-ai/figure7.png)

- **图 7 说明**：对比"批评后修订（Critiqued Revision）"与"直接修订（Direct Revision）"在不同规模模型上的无害性 PM 评分（按修订次数 1–4 分面板展示）。**小模型**中"批评后修订"无害性更高；**大模型**两者接近，但保留批评步骤能提供推理过程的**透明度**，有助于发现隐性危害。

---

## 5. 基于 AI 反馈的强化学习 RLAIF

### 5.1 方法

- 将"提示 + 回复对 + 一条原则"组成多选题，让反馈模型选择；
- 计算模型对选项 (A)、(B) 的对数概率，归一化后作为偏好模型训练的**软标签（Soft Labels）**；
- 使用 **CoT 提示**（"Let's think step-by-step"）让模型先写推理、再选择，提示模板如下：

```
Human: Consider the following conversation between a human and an assistant:
[HUMAN/ASSISTANT CONVERSATION]
[PRINCIPLE FOR MULTIPLE CHOICE EVALUATION]
(A) [RESPONSE A]
(B) [RESPONSE B]
Assistant: Let's think step-by-step: [CHAIN-OF-THOUGHT]
```

- **概率截断（Clamping）**：CoT 会让模型输出极端自信（接近 0 或 1）的概率，导致 RL 过拟合；论文将概率**截断在 40%–60%** 之间，效果最稳健。

RL 训练所用数据规模：135,296 条人类有用性比较 + 182,831 条 AI 生成的无害性比较，另加 491,142 条红队提示与 474,300 条有用性提示。

### 5.2 主要结果

![图 2：无害性 vs 有用性的 Elo 散点图（52B 模型），RL-CAI 实现帕累托改进](/assets/posts/constitutional-ai/figure2.png)

- **图 2 说明**：横轴为有用性 Elo、纵轴为无害性 Elo（越靠右上越好，仅差值有意义）。点越靠右代表 RL 训练越后期。**Constitutional RL（RL-CAI，尤其带 CoT）实现了帕累托改进（Pareto Improvement）**：在保持有用性的同时把无害性推到远高于标准 RLHF 的区域。评估众包工人被要求在两回复同样无害时**偏好不逃避**的回复，因此 Helpful 与 HH 两个人类反馈模型在无害性上差异不大。

![图 3：不同规模模型的有用性（左）与无害性（右）Elo](/assets/posts/constitutional-ai/figure3.png)

- **图 3 说明**：左图为有用性 Elo、右图为无害性 Elo，横轴为参数量。对比 SL-CAI、Helpful RLHF、HH RLHF、RL-CAI、RL-CAI w/ CoT。CAI 系列方法随模型规模增大效果越发显著（图 2 是本图的无误差棒版本）。

![图 8：RL 训练过程中的有用性（左）与无害性（右）Elo 变化](/assets/posts/constitutional-ai/figure8.png)

- **图 8 说明**：横轴为 RL 训练序列数量。RL-CAI（含 CoT）在训练中无害性稳步上升；而 **HH RLHF 在训练后期因"逃避"策略导致无害性评分反弹下降**——这正是 CAI 想要避免的问题。

### 5.3 Goodharting 现象与缓解

RL 训练后期，模型会出现**过度说教/机械套话**（如反复强调 "you are valid, valued, and cared for"）的 Goodharting 行为。缓解手段有三：

1. 修改宪法原则，明确要求避免过度反应与说教；
2. **集成（Ensembling）**：生成标签时对 16 条原则做集成评估；
3. 使用截断到 40%–60% 的软标签。

### 5.4 软标签的校准

![图 9：52B RL-CAI 软标签在 HHH 评估上的校准曲线](/assets/posts/constitutional-ai/figure9.png)

- **图 9 说明**：横轴为 AI 反馈模型输出的归一化概率、纵轴为实际频率。曲线接近对角线，说明 **AI 反馈生成的软标签具有良好的校准度（Calibration）**，可信地反映了偏好强度。

### 5.5 无害性 vs 逃避性

- 传统 HH RLHF 在训练后期变得**极度逃避**（一律回答"对不起，我不能回答"）；
- **RL-CAI 几乎从不逃避**，而是以富有同理心、细致入微（nuanced）的方式解释为何拒绝——例如在家庭暴力或有自杀倾向的提示下，会提供情感支持与求助热线，而非生硬拒答。

### 5.6 绝对有害性评分

![图 10：RL 训练过程中的绝对有害性评分（0–4 分）](/assets/posts/constitutional-ai/figure10.png)

- **图 10 说明**：使用 0–4 分绝对量表评分（用 L2 损失训练预测模型）。**Helpful RLHF 的有害性随训练上升**，而 **HH RLHF 与 RL-CAI 的有害性均随 RL 训练显著下降**。

---

## 6. 附录中的补充评估

![图 11：原始 HHH 评估结果](/assets/posts/constitutional-ai/figure11.png)

- **图 11 说明**：在早期 HHH 基准上，不同模型（HH PM、Helpful RLHF 0-shot/5-shot 等）准确率随参数量的 scaling 趋势。

![图 12：识别（左）与分类（右）有害行为的准确率](/assets/posts/constitutional-ai/figure12.png)

- **图 12 说明**：证明语言模型无需专门微调，仅通过零样本/少样本 + CoT 即可较准确地识别有害对话并对有害类型分类，准确率随模型规模上升。

---

## 7. 关键结论

1. **AI 自我监督可行**：无需人工标注有害数据，仅凭一组自然语言"宪法"原则，AI 就能通过自我批评与相互评估达成高度无害对齐。
2. **解决"逃避"难题**：CAI 让模型学会直面敏感问题、解释拒绝原因并给出建设性反馈，实现无害性与有用性的**帕累托最优**。
3. **思维链是关键**：CoT 既大幅提升 AI 作为"裁判"的评估准确率，又让决策过程透明可审计，是实现"可扩展监督（Scalable Oversight）"的核心。
4. **RLAIF 媲美甚至超越 RLHF**：基于 AI 反馈的强化学习在众包盲测中表现等于或优于用海量人类标签训练的 RLHF，大幅降低了对齐的数据成本与人类红队测试员的心理负担。
5. **防 Goodharting 技巧**：概率截断（40%–60%）+ 多原则集成，是防止 RL 后期陷入"过度说教"的有效手段。

> **更广泛的影响**：CAI 降低了控制 AI 行为的门槛（双刃剑，也可能被用于训练恶意系统），但同时免去了人类红队测试员长期接触大量有害内容所带来的心理创伤。

---

## 附录 A：专有名词中英对照表

| 中文 | 英文 / 缩写 |
| :--- | :--- |
| 宪法式 AI | Constitutional AI (CAI) |
| 宪法 / 原则清单 | Constitution / Principles |
| 无害性 | Harmlessness |
| 有用性 | Helpfulness |
| 诚实性 | Honesty |
| 有用、诚实、无害 | Helpful, Honest, Harmless (HHH) |
| 基于人类反馈的强化学习 | Reinforcement Learning from Human Feedback (RLHF) |
| 基于 AI 反馈的强化学习 | Reinforcement Learning from AI Feedback (RLAIF) |
| 监督学习（阶段） | Supervised Learning (SL) |
| 强化学习（阶段） | Reinforcement Learning (RL) |
| 偏好模型 | Preference Model (PM) |
| 反馈模型 | Feedback Model |
| 自我批评 | Critique / Self-critique |
| 修订 | Revision |
| 批评请求 | Critique Request |
| 修订请求 | Revision Request |
| 思维链 | Chain-of-Thought (CoT) |
| 集成思维链 | Ensembled Chain-of-Thought |
| 少样本提示 | Few-shot Prompting |
| 红队测试 | Red Teaming |
| 软标签 | Soft Labels |
| 概率截断 | Clamping |
| 集成 | Ensembling |
| 可扩展监督 | Scaling / Scalable Supervision (Oversight) |
| 帕累托改进 / 帕累托最优 | Pareto Improvement / Pareto Optimal |
| 校准 | Calibration |
| 逃避性（回复） | Evasiveness / Evasive Responses |
| 古德哈特现象（过度优化指标） | Goodharting |
| SL 阶段产出的模型 | SL-CAI Model |
| RL 阶段产出的模型 | RL-CAI Model |
| 仅有用性模型 | Helpful-Only / Helpful RLHF |
| 有用且无害模型 | Helpful & Harmless (HH) RLHF |
| Elo 评分 | Elo Score |
| 预训练语言模型 | Pretrained Language Model (LM) |

---

## 附录 B：原文链接

- **arXiv PDF**：<https://arxiv.org/pdf/2212.08073>
