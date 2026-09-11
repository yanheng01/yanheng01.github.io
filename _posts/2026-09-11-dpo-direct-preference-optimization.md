---
title: "DPO 简化 Preference Optimization"
date: 2026-09-11 18:45:00 +0800
permalink: /posts/dpo/
categories: [技术笔记, 深度学习]
tags: [Alignment]
math: true
mermaid: false
toc: true
---

> **原文标题（Original Title）**：Direct Preference Optimization: Your Language Model is Secretly a Reward Model
> **作者（Authors）**：Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn（Stanford University; CZ Biohub）
> **发表信息**：NeurIPS 2023（第 37 届神经信息处理系统大会），arXiv:2305.18290v3
> **一句话总结**：通过对奖励模型（Reward Model）做一次巧妙的**重参数化（Reparameterization）**，证明"你的语言模型其实暗中就是一个奖励模型"，从而把复杂、不稳定的 RLHF 强化学习流程，转化为一个**简单的二元交叉熵分类问题**——无需显式奖励模型、无需强化学习、无需在训练中采样。

---

## 目录

1. [引言：为什么需要 DPO](#1-引言为什么需要-dpo)
2. [相关工作](#2-相关工作)
3. [核心方法：DPO 的完整推导](#3-核心方法dpo-的完整推导)
4. [理论分析](#4-理论分析)
5. [实验设置](#5-实验设置)
6. [实验结果与结论](#6-实验结果与结论)
7. [论文配图详解](#7-论文配图详解)
8. [讨论与局限性](#8-讨论与局限性)
9. [附录 A：专有名词中英对照表](#附录-a专有名词中英对照表)
10. [附录 B：原文链接](#附录-b原文链接)

---

## 1. 引言：为什么需要 DPO

大规模语言模型（Language Model, **LM**）在无监督预训练中学到了极其广博的知识，但也正因为训练完全是无监督的，它**难以被精确控制**。训练数据中混杂了人类各种各样的目标、偏好与技能，其中不乏我们**不希望模型模仿**的行为（例如常见的编程错误、普遍存在的错误认知）。因此，如何从模型广博的知识中筛选出安全、可控、符合期望的行为——也就是**对齐（Alignment）**——至关重要。

**当前主流方案 RLHF 的痛点**：目前最成功的对齐方法是**基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）**。它分两步走：

1. 收集人类偏好标签，拟合一个**奖励模型（Reward Model）**；
2. 用**强化学习（Reinforcement Learning, RL）**（通常是 **PPO**，Proximal Policy Optimization / 近端策略优化）微调 LM 以最大化奖励，同时用 **KL 散度（Kullback–Leibler Divergence）** 约束模型不要偏离原始预训练分布太远。

但这条流水线**极其复杂**：需要训练多个 LM，还要在训练循环中不断从 LM 采样，计算成本高、且训练容易不稳定。

**本文要回答的问题**：能否**绕过显式的奖励建模和强化学习**，直接用人类偏好数据优化语言模型策略（Policy）？

**核心贡献——DPO（Direct Preference Optimization, 直接偏好优化）**：通过一次**变量代换（change of variables）**，把偏好损失直接写成策略的函数。DPO 隐式地优化了与 RLHF 完全相同的目标（带 KL 约束的奖励最大化），但只需要一个**简单的二元交叉熵（Binary Cross-Entropy）目标**即可训练。它稳定、高效、计算轻量，**不需要在微调时从 LM 采样，也不需要繁琐的超参数调优**。

---

## 2. 相关工作

- **指令微调与偏好学习**：指令微调（Instruction-tuning）能提升 LM 的泛化能力；而收集人类的**相对偏好标签（relative preference）** 通常比收集专家示范更容易。已有方法多基于 **Bradley-Terry** 等偏好模型先训练奖励函数，再用 RL（如 REINFORCE、PPO）优化策略。
- **非语言领域的偏好学习**：包括**上下文对决赌博机（Contextual Dueling Bandits, CDB）** 和**基于偏好的强化学习（Preference-based RL, PbRL）**。这些方法通常需要先**显式估计潜在的评分/奖励函数**，再进行优化。DPO 则提出了一种**单阶段（single-stage）** 的策略学习方法，直接从偏好中优化策略。

---

## 3. 核心方法：DPO 的完整推导

### 3.1 标准 RLHF 目标与最优策略的闭式解

RLHF 的强化学习微调阶段，目标是**最大化奖励**并**约束策略偏离参考策略**（参考策略 $\pi_{ref}$ 通常是监督微调后的 SFT 模型）：

$$
\max_{\pi_\theta}\ \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y|x)}\big[r_\phi(x,y)\big] - \beta\, D_{KL}\big(\pi_\theta(y|x)\,\|\,\pi_{ref}(y|x)\big)
$$

其中 $\beta$ 控制偏离程度。通过变分法可以证明，该"带 KL 约束的奖励最大化"问题的**最优策略** $\pi_r$ 具有如下**闭式解（closed form）**：

$$
\pi_r(y|x) = \frac{1}{Z(x)}\,\pi_{ref}(y|x)\,\exp\!\left(\frac{1}{\beta}\, r(x,y)\right)
$$

其中**配分函数（Partition Function）** 为：

$$
Z(x) = \sum_y \pi_{ref}(y|x)\,\exp\!\left(\frac{1}{\beta}\, r(x,y)\right)
$$

$Z(x)$ 需要对所有可能的回答 $y$ 求和，**难以计算**——这正是无法直接使用该闭式解的障碍。

### 3.2 奖励函数的重参数化（Reparameterization）

DPO 的关键一招：对闭式解两边取对数并重排，把**真实奖励函数** $r^*(x,y)$ 反过来表示成**最优策略** $\pi^*$ 与**参考策略** $\pi_{ref}$ 的函数：

$$
r^*(x,y) = \beta\,\log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta\,\log Z(x)
$$

### 3.3 结合 Bradley-Terry 偏好模型

人类偏好数据常用 **Bradley-Terry（BT）模型**建模，即"偏好某个回答的概率"只取决于两个回答的**奖励之差**：

$$
p^*(y_1 \succ y_2 \mid x) = \sigma\big(r^*(x,y_1) - r^*(x,y_2)\big) = \frac{\exp\big(r^*(x,y_1)\big)}{\exp\big(r^*(x,y_1)\big) + \exp\big(r^*(x,y_2)\big)}
$$

其中 $\sigma$ 是 **Sigmoid 函数**。把 3.2 中重参数化后的 $r^*(x,y)$ 代入 BT 模型，**难算的配分函数 $Z(x)$ 会被完美消去**：

$$
p^*(y_1 \succ y_2 \mid x) = \frac{1}{1 + \exp\!\left(\beta\log\dfrac{\pi^*(y_2|x)}{\pi_{ref}(y_2|x)} - \beta\log\dfrac{\pi^*(y_1|x)}{\pi_{ref}(y_1|x)}\right)}
$$

这意味着：**人类偏好概率可以完全用最优策略 $\pi^*$ 和参考策略 $\pi_{ref}$ 表达，不再需要显式的奖励模型。**

### 3.4 DPO 损失函数（DPO Loss）

基于上述推导，可以直接对参数化策略 $\pi_\theta$ 构造**最大似然（二元交叉熵）** 损失：

$$
\mathcal{L}_{DPO}(\pi_\theta;\ \pi_{ref}) = -\,\mathbb{E}_{(x,\,y_w,\,y_l)\sim\mathcal{D}}\left[\log \sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]
$$

其中：$y_w$ 是**被偏好的回答（preferred / winner）**，$y_l$ 是**不被偏好的回答（dispreferred / loser）**。

### 3.5 DPO 梯度的直观理解

对 $\mathcal{L}_{DPO}$ 求梯度：

$$
\nabla_\theta \mathcal{L}_{DPO} = -\beta\, \mathbb{E}_{(x,\,y_w,\,y_l)\sim\mathcal{D}}\Big[\ \sigma\big(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w)\big)\ \big(\nabla_\theta \log\pi_\theta(y_w|x) - \nabla_\theta \log\pi_\theta(y_l|x)\big)\ \Big]
$$

其中**隐式奖励（implicit reward）** 定义为：

$$
\hat{r}_\theta(x,y) = \beta\,\log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}
$$

**直观含义**：梯度会**提高 $y_w$ 的概率、降低 $y_l$ 的概率**。关键在于那个**动态权重** $\sigma(\hat{r}_\theta(x,y_l) - \hat{r}_\theta(x,y_w))$：当隐式奖励模型**错误地给 $y_l$ 打了更高分**时，权重会变大、更新更猛。正是这一权重机制，避免了朴素"概率比"目标常见的模型退化（degeneration）。

### 3.6 与 RLHF / PPO 的关系

- **目标等价**：DPO 优化的数学目标与 PPO-based RLHF **完全相同**（带 KL 惩罚的奖励最大化）。
- **角色合并**：在 DPO 中，语言模型 $\pi_\theta$ **同时扮演策略（Policy）和隐式奖励模型（Reward Model）** 两个角色——这正是标题「你的语言模型其实暗中就是一个奖励模型」的含义。
- **流程简化**：彻底去掉了显式奖励模型的训练，以及 RL 的 Actor-Critic 采样循环。

---

## 4. 理论分析

### 4.1 隐式奖励模型的表达能力（Theorem 1）

作者定义了奖励函数的**等价类（equivalence class）**：若两个奖励函数满足 $r(x,y) - r'(x,y) = f(x)$（差只与 $x$ 有关），则两者等价（因为 BT 模型只依赖奖励差）。

- **Lemma 1 & 2**：同一等价类中的奖励函数，会产生**相同的偏好分布**和**相同的最优策略**。
- **Theorem 1**：在温和假设下，所有与 **Plackett-Luce / Bradley-Terry** 模型一致的奖励等价类，都可以用 $r(x,y) = \beta\log\frac{\pi(y|x)}{\pi_{ref}(y|x)}$ 来表示。
- **结论**：DPO 的重参数化**没有损失任何奖励模型的表达能力**，它只是在一个等价类中选择了那个"能让最优策略解析可解"的代表元。

### 4.2 Actor-Critic 算法的不稳定性诊断

在标准 RLHF（如 PPO）中，目标里含有一个**归一化项**（参考策略的软价值函数 soft value function）。若缺少这个**基线（baseline）**，策略梯度的**方差会非常大**，导致训练不稳定。PPO 通常靠额外训练一个**价值模型（Value Model）** 或用人类回答做 baseline 来缓解。而 **DPO 的重参数化天然内含了这个归一化基线**，因此**无需额外的 Value Model** 也能保持训练稳定。

---

## 5. 实验设置

### 5.1 任务与数据集

1. **受控情感生成（Controlled Sentiment Generation）**：使用 **IMDb** 电影评论前缀，借助一个预训练情感分类器作为**真值奖励模型（ground-truth reward）** 生成偏好对。
2. **摘要（Summarization）**：使用 **Reddit TL;DR** 数据集及其人类偏好标签。
3. **单轮对话（Single-turn Dialogue）**：使用 **Anthropic Helpful and Harmless（HH）** 数据集。

### 5.2 评估方法

- **情感生成**：因为有真值奖励，评估 **Reward–KL 前沿（frontier）**——即在给定 KL 散度下能达到的最高奖励。
- **摘要与对话**：使用 **GPT-4 作为代理评估者（proxy evaluator）**，计算相对基线（人类写的摘要，或数据集中被选中的回答）的**胜率（Win Rate）**。

### 5.3 基线方法（Baselines）

- **Prompting**：Zero-shot GPT-J、Few-shot Pythia-2.8B。
- **SFT**：监督微调（Supervised Fine-Tuning）模型。
- **Preferred-FT**：仅在被偏好（chosen）数据上做 SFT。
- **Unlikelihood**：最大化 $y_w$ 概率、最小化 $y_l$ 概率。
- **PPO / PPO-GT**：用学习到的奖励（learned reward）或真值奖励（ground-truth reward）做 PPO。
- **Best of N**：从 SFT 模型采样 N 次、选奖励最高的一个（推理成本极高）。

---

## 6. 实验结果与结论

- **优化效率（Reward–KL 前沿）**：在 IMDb 任务中，DPO 的 Reward–KL 前沿**严格优于 PPO**，甚至优于使用真值奖励的 PPO-GT——能以更小的 KL 代价拿到更高的奖励。
- **扩展性与胜率**：在 TL;DR 与 Anthropic-HH 上，DPO 的胜率**超过 PPO 和计算昂贵的 Best of N**。
- **鲁棒性**：DPO 对**采样温度（sampling temperature）** 的变化比 PPO 更鲁棒；PPO 在高温下容易退化到基础模型水平。
- **分布外泛化（Out-of-Distribution, OOD）**：在 **CNN/DailyMail** 新闻摘要上测试（训练集为 Reddit），DPO 的胜率显著高于 PPO，说明其泛化能力良好。
- **人类与 GPT-4 评估一致性**：人类研究表明，GPT-4 的判断与人类的一致性，几乎等于人类评估者彼此之间的一致性，验证了使用 GPT-4 评估的可靠性。

---

## 7. 论文配图详解

### Figure 1：DPO 在优化人类偏好的同时避免了强化学习

![Figure 1：左侧为传统 RLHF 两阶段流程（先训练显式奖励模型，再用 RL/PPO 优化策略并反复采样）；右侧为 DPO 单阶段流程（直接用最大似然从偏好数据优化出 final LM）](/assets/posts/dpo/figure1.png)

- **内容**：对比传统 RLHF 与 DPO 的流程概念图。
- **左侧（RLHF）**：两阶段流程。① 由提示词和人类偏好数据训练出**显式奖励模型（reward model）**；② 奖励模型 + 参考策略，用 **RL（如 PPO）** 优化 LM 策略（需在循环中 "label rewards / sample completions" 反复采样）。
- **右侧（DPO）**：单阶段流程。直接由偏好数据**用最大似然（maximum likelihood）优化出 final LM**，无需奖励模型、无需 RL 循环。
- **含义**：直观展示 DPO 如何砍掉复杂的 RL 循环和独立的奖励模型训练步骤。

### Figure 2：优化效率与摘要胜率

![Figure 2：左图为 IMDb 情感生成的 Reward–KL 前沿，DPO 严格支配 PPO；右图为 TL;DR 摘要胜率 vs 采样温度，DPO 更高且更鲁棒](/assets/posts/dpo/figure2.png)

- **左图（IMDb 情感生成的 Reward–KL 前沿）**：横轴为 KL 散度 $KL(\pi_\theta\|\pi_{ref})$，纵轴为期望奖励 Reward。散点为不同算法、不同超参数下的表现。**DPO（黄色）在所有 KL 值下都取得最高的期望奖励**，其前沿严格支配 PPO（含使用真值奖励的 PPO-GT）。
- **右图（TL;DR 摘要胜率 vs 采样温度）**：横轴为采样温度（0.0→1.0），纵轴为相对人类参考摘要的胜率。DPO 在 temp=0.0 时胜率约 **61%**，超过 PPO 的最佳约 **57%**；DPO 曲线更平缓、对温度更鲁棒，而 PPO 在高温下急剧下降。图中虚线为 0.5 的平局线。

### Figure 3：对话胜率及训练动态

![Figure 3：左图为 Anthropic-HH 单轮对话胜率 vs Chosen，DPO 是唯一稳定超过 0.5 的高效方法；右图为训练过程中胜率随微调步数的演变，收敛稳定](/assets/posts/dpo/figure3.png)

- **左图（Anthropic-HH 单轮对话胜率 vs Chosen）**：横轴为采样温度，纵轴为相对数据集中被选回答（Chosen）的胜率。**DPO 是唯一能稳定超过 Chosen 基线（胜率 > 0.5）的计算高效方法**；Best of 128 虽好但推理成本不现实。
- **右图（训练过程中胜率的演变）**：横轴为微调步数（fine-tuning step），纵轴为胜率。展示 DPO 在温度 1.0 与 0.7 下的收敛过程，说明其性能提升**稳定**，训练中没有明显崩溃或剧烈震荡。

### Figure 4（附录）：Best of N 基线的表现

![Figure 4：Best of N（N = 1, 4, 16, 64, 128, 256）在 Anthropic-HH 与 TL;DR 上的胜率，性能在 N≈64~128 后趋于饱和](/assets/posts/dpo/figure4.png)

- **内容**：在 Anthropic-HH 与 TL;DR 上，从 SFT 模型采样 N 次（N = 1, 4, 16, 64, 128, 256）取奖励最高者的胜率，横轴为采样温度。
- **结论**：性能在 **N ≈ 64~128** 左右趋于饱和（plateau），说明 Best of N 推理成本高且**收益递减**。

### Figure 5（附录）：人类研究问卷界面

![Figure 5：使用 SurveyMonkey 进行人类评估的问卷界面截图，受访者需对比 Summary A / Summary B 并选择更好的摘要](/assets/posts/dpo/figure5.png)

- **内容**：使用 **SurveyMonkey** 进行人类评估的网页截图。受访者需对比不同模型生成的摘要（Summary A / Summary B / "I can't tell"）并选择更好的一个。
- **含义**：说明人类评估的实验界面设计，用于校验 GPT-4 胜率与人类判断的一致性。

---

## 8. 讨论与局限性

- **核心总结**：DPO 建立了一种映射关系，使语言模型可以直接用简单的交叉熵损失去满足人类偏好，**无需强化学习、且不损失一般性**，大幅降低了偏好对齐的门槛。
- **局限性与未来工作**：
  1. **分布外泛化**：DPO 策略在 OOD 上的泛化能力仍需更全面的研究；能否像 PPO 那样利用无标签提示做**自标注（self-labeling）**？
  2. **奖励过度优化（Reward Over-optimization）**：在直接偏好优化设定下，奖励过度优化会如何表现？（Figure 3 右图后期的轻微下降可能就是一个实例。）
  3. **规模扩展**：本文最大只评估到 **6B** 参数模型，把 DPO 扩展到数十亿/上百亿参数的 SOTA 模型很值得期待。
  4. **评估方法**：GPT-4 的胜率受提示词（Prompt）影响较大，未来需研究如何从自动化系统中引出更高质量的判断。
  5. **跨模态应用**：DPO 不限于语言模型，也可用于训练图像、视频等其他模态的生成模型。

---

## 附录 A：专有名词中英对照表

| 中文 | 英文 |
| --- | --- |
| 对齐 | Alignment |
| 语言模型 | Language Model (LM) |
| 基于人类反馈的强化学习 | Reinforcement Learning from Human Feedback (RLHF) |
| 强化学习 | Reinforcement Learning (RL) |
| 直接偏好优化 | Direct Preference Optimization (DPO) |
| 近端策略优化 | Proximal Policy Optimization (PPO) |
| 奖励模型 | Reward Model |
| 策略 | Policy |
| 参考策略 | Reference Policy ($\pi_{ref}$) |
| 监督微调 | Supervised Fine-Tuning (SFT) |
| 闭式解 | Closed Form |
| 配分函数 | Partition Function ($Z(x)$) |
| 重参数化 | Reparameterization |
| KL 散度 | Kullback–Leibler Divergence |
| Bradley-Terry 偏好模型 | Bradley-Terry (BT) Model |
| Plackett-Luce 模型 | Plackett-Luce Model |
| 二元交叉熵 | Binary Cross-Entropy |
| 隐式奖励 | Implicit Reward |
| 等价类 | Equivalence Class |
| 价值模型 | Value Model |
| 基线 | Baseline |
| 上下文对决赌博机 | Contextual Dueling Bandits (CDB) |
| 基于偏好的强化学习 | Preference-based RL (PbRL) |
| 采样温度 | Sampling Temperature |
| 胜率 | Win Rate |
| 分布外（泛化） | Out-of-Distribution (OOD) |
| 奖励过度优化 | Reward Over-optimization |
| 指令微调 | Instruction-tuning |
| Sigmoid 函数 | Sigmoid Function |

---

## 附录 B：原文链接

- 原论文（arXiv）：<https://arxiv.org/abs/2305.18290?utm_source=chatgpt.com>
