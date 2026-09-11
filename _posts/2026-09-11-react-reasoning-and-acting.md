---
title: "ReAct，Reasoning + Tool"
date: 2026-09-11 18:53:00 +0800
categories: [技术笔记, 深度学习]
tags: [react, reasoning, tool, llm, agent, 论文精读, Agent]
math: true
mermaid: false
toc: true
---

> 一份关于 **ReAct（Reasoning + Acting）** 的中文精读笔记。文中专有名词首次出现时附英文，文末附「专有名词中英对照附录」。论文正文与附录中的全部关键配图（Figure 1–5、Table 1–5）均已嵌入并配中文解读，文末附论文原文链接。

---

## 0. 论文基本信息

| 项目 | 内容 |
| :--- | :--- |
| 标题 | **ReAct: Synergizing Reasoning and Acting in Language Models** |
| 作者 | Shunyu Yao、Jeffrey Zhao、Dian Yu、Nan Du、Izhak Shafran、Karthik Narasimhan、Yuan Cao |
| 机构 | 普林斯顿大学（Princeton University）& 谷歌研究院 Brain 团队（Google Research, Brain team） |
| 发表 | 第 11 届国际学习表征大会（**ICLR 2023**） |
| 地位 | 现代 Agent「推理 + 工具调用」范式的奠基性工作之一 |

---

## 一句话总结

ReAct 让大语言模型（LLM）以**交错穿插**的方式，同时产出「推理轨迹」（Reasoning trace，即模型的内心思考）和「动作」（Action，即与外部环境/工具的交互），从而实现「用推理来指导动作、用动作来支撑推理」的闭环。它是现代 Agent 范式的奠基性工作之一。

---

## 1. 背景与动机

人类智能的一个独特之处，在于能把**面向任务的动作**与**语言推理（内心独白，inner speech）**无缝结合起来：我们一边动手做事，一边在脑子里跟踪进度、处理意外、临时调整计划、必要时主动去查资料。

在此之前，LLM 的「推理」和「动作」是被**割裂**研究的两条线：

- **纯推理路线（以思维链 Chain-of-Thought 为代表）**：推理过程是一个**静态的黑盒**，完全依赖模型内部的参数化知识，不接触外部世界（not grounded）。因此容易产生**事实幻觉**（hallucination）和**错误累积传播**（error propagation）——一步错，步步错。
- **纯动作路线（以 WebGPT 等为代表）**：模型能与外部环境交互获取信息，但**缺乏高层次的抽象推理和工作记忆**（working memory），因此难以做长程规划，也难以从错误中恢复，容易迷失方向。

**ReAct 的核心主张**：把两者结合，做到

- **Reason to act**（推理以指导动作）：用思考来创建、维护、调整行动计划；
- **Act to reason**（动作以辅助推理）：用与外部环境（如维基百科）的交互，把新信息补充进推理过程。

### 📊 图 1（Figure 1）：四种方法的轨迹直观对比

![Figure 1：四种提示方法在 HotpotQA 与 ALFWorld 上的轨迹对比](/assets/posts/react/figure1.png)

这张图是全文的「灵魂图」，直观对比了四种方法在两类任务上的实际轨迹：

- **上半部分（HotpotQA 问答任务）**，对比 Standard / CoT / Act-only / ReAct：
  - **Act-only**：搜索失败后没有思考能力，盲目重复搜索、陷入死循环。
  - **CoT**：凭空编造事实（幻觉），推理链看似合理但答案错误。
  - **ReAct**：先用 Thought 规划「我需要搜索 Apple Remote」，搜索不到时能**重构查询**（如把 `Front Row` 改写为 `Front Row (software)`），最终给出正确且**可追溯**的答案。
- **下半部分（ALFWorld 决策任务）**：
  - **Act-only**：找不到胡椒瓶（pepper shaker），且无法理解环境反馈，产生幻觉动作。
  - **ReAct**：用 Thought 记录常识推理（「胡椒瓶更可能出现在柜子或台面上」），系统性地探索并完成任务。

> **关键结论**：ReAct 打通了「推理 ↔ 外部反馈」的闭环，同时解决了幻觉和盲目探索两大问题。

---

## 2. 方法：ReAct 到底是怎么做的

**核心形式化**：把 Agent 的动作空间从原来的环境动作空间 $A$，扩展为

$$
\hat{A} = A \cup L
$$

其中 $L$ 是**语言空间**（Language space）。

- 落在 $L$ 里的「动作」就是 **Thought（思考/推理轨迹）**：它**不直接改变外部环境**，也不会带来环境的 Observation（观察）反馈，而是用来重新组织当前上下文 $c_t$ 中的有用信息、更新上下文，从而支撑后续的推理或真实动作。
- 落在 $A$ 里的就是真实动作（如调用搜索 API、在环境里移动）。

**Thought 能承担的多种功能**：分解任务目标、注入常识知识、从观察中抽取关键信息、跟踪进度、处理异常并调整计划等等。

**实现方式**：用一个**冻结参数**的大模型（主实验用 PaLM-540B），通过 **少样本上下文示例**（few-shot in-context examples）来提示——示例里包含人工标注的「思考—动作—观察」（Thought–Action–Observation）轨迹。

### ReAct 的四大优势

1. **直观、易设计**（Intuitive and easy to design）：只需在动作旁边补上人类自然语言的思考即可。
2. **通用、灵活**（General and flexible）：Thought 空间无限大，适配问答、事实验证、文字游戏、网页导航等各种任务。
3. **高效、鲁棒**（Performant and robust）：仅需 1~6 个示例就能泛化，性能稳健。
4. **对齐人类、可控**（Human aligned and controllable）：人类可以直接编辑 Thought 来纠正 Agent 的行为（人在回路，Human-in-the-loop）。

---

## 3. 知识密集型推理任务（HotpotQA & FEVER）

### 3.1 实验设置

- **数据集**：HotpotQA（多跳问答，multi-hop QA）、FEVER（事实验证，fact verification）。模型只拿到问题，必须靠内部知识或外部检索来推理。
- **动作空间**：设计了一个简易的维基百科 API：
  - `search[entity]`：返回实体摘要，或返回相似实体列表；
  - `lookup[string]`：在当前页面内查找字符串；
  - `finish[answer]`：给出最终答案。

### 3.2 对比方法（Baselines）

- **Standard**：直接生成答案（无 Thought、无 Action）。
- **CoT**（思维链）：只推理不动作（"Let's think step by step"）。
- **CoT-SC**（自洽性，Self-Consistency）：采样 21 条 CoT 轨迹后取多数投票。
- **Act**：只动作不思考（类似 WebGPT）。

**内外知识结合的两种回退策略**：

- **ReAct → CoT-SC**：ReAct 在给定步数内找不到答案时，回退到 CoT-SC，依赖模型内部知识。
- **CoT-SC → ReAct**：CoT-SC 多数投票不够自信（出现次数少于 n/2）时，回退到 ReAct，依赖外部检索。

**微调（Finetuning）**：用 ReAct 生成的 3000 条正确轨迹，去微调更小的模型（PaLM-8B / 62B）。

### 3.3 结果与分析

#### 📊 表 1（Table 1）：HotpotQA 与 FEVER 的提示结果

![Table 1：HotpotQA 与 FEVER 的提示结果](/assets/posts/react/table1.png)

- ReAct 显著优于 Act（27.4 vs 25.7），证明**推理对动作有指导价值**。
- 在 FEVER 上，ReAct（60.9）优于 CoT（56.3）——因为 FEVER 更需要精确、接地气的事实。
- 在 HotpotQA 上 ReAct（27.4）略逊于 CoT（29.4）。
- **组合方法拿下提示范式下的最佳（SOTA）**：HotpotQA 上 CoT-SC→ReAct 达 34.2、ReAct→CoT-SC 达 35.1；FEVER 上 CoT-SC→ReAct 达 64.6。

#### 📊 图 2（Figure 2）：随 CoT-SC 采样数量变化的性能曲线

![Figure 2：随 CoT-SC 采样数量变化的性能曲线](/assets/posts/react/figure2.png)

- 组合方法（ReAct→CoT-SC、CoT-SC→ReAct）在各种采样数下都稳定优于纯 CoT-SC。
- **仅需 3~5 个样本**，组合方法就能达到纯 CoT-SC 需要 **21 个样本**才能达到的性能，体现了内外知识结合的高效性。

#### 📊 表 2（Table 2）：HotpotQA 上成功/失败模式的人工分析

![Table 2：HotpotQA 上成功/失败模式的人工分析](/assets/posts/react/table2.png)

作者随机抽取 50 条正确 + 50 条错误轨迹（ReAct 与 CoT 各 200 例），人工标注模式：

- **CoT 的致命伤——幻觉**：失败案例中 **56%** 是幻觉（Hallucination），甚至成功案例里也有 **14%** 是靠幻觉「蒙对」的假阳性（false positive）。
- **ReAct 更接地气、更可信**：幻觉导致的失败为 **0%**，成功里的假阳性只有 **6%**。
- **ReAct 的短板**：推理错误（Reasoning error）高达 **47%**（常陷入「重复搜索」的死循环）；另有 **23%** 是搜索结果错误（Search result error，即检索返回空或无用信息）。

#### 📊 图 3（Figure 3）：提示 vs. 微调的规模化（Scaling）效果

![Figure 3：提示与微调的规模化效果对比](/assets/posts/react/figure3.png)

- 在**纯提示**阶段（左图），ReAct 反而表现**最差**——因为同时学会推理和动作对模型来说更难。
- 但**用 3000 条数据微调后**（右图），ReAct **逆袭成为四种方法中的最佳**。
- 更惊人的是：**微调后的 8B ReAct 甚至超越了所有 62B 提示方法**，微调后的 62B ReAct 超越所有 540B 提示方法。
- **启示**：ReAct 是一种「可学习的通用技能」，微调能极大释放其潜力；相比之下微调 Standard/CoT 只是在让模型「死记硬背知识」。

#### 📊 图 4（Figure 4）：数据集标签过时的案例（附录 A.2）

![Figure 4：数据集标签过时的案例](/assets/posts/react/figure4.png)

- 问题问「太阳马戏团某演出所在酒店有多少房间」，HotpotQA 的原始标签已经**过时**。
- Standard 和 CoT 因幻觉给出错误答案，Act-only 也无法正确检索。
- **只有 ReAct 能通过推理指导检索，拿到真实世界中最新的知识**（up-to-date knowledge）。

---

## 4. 交互式决策任务（ALFWorld & WebShop）

### 4.1 ALFWorld（文字环境交互）

- **任务**：在模拟家居环境里完成长程任务（如「把干净的生菜放到餐桌上」），需要常识推理和子目标跟踪。单个任务实例可能涉及 50+ 个地点、50+ 个步骤。
- **对比基线**：BUTLER（模仿学习，用 10⁵ 条专家轨迹训练）。

#### 📊 表 3 & 表 4（Table 3 & Table 4）：ALFWorld 与 WebShop 结果

![Table 3 与 Table 4：ALFWorld 与 WebShop 结果](/assets/posts/react/table3_4.png)

**表 3（ALFWorld，成功率 %）**：

- **ReAct（best of 6）平均成功率 71%**，远超 Act（45%）和 BUTLER（37%），绝对提升达 **34 个百分点**。
- **关键对比 ReAct vs. ReAct-IM**：内心独白式（Inner Monologue, IM）提示——只有密集的外部环境反馈式思考，缺乏内部常识推理和子目标分解——成功率仅 **53%**。
  → **证明「内部高层推理」比「单纯的外部反馈」更重要**。

**表 4（WebShop，Score 与成功率 SR）**：见下一节。

### 4.2 WebShop（网页导航购物）

- **任务**：在含 **118 万**真实商品、12k 条人类指令的电商网站里，根据复杂自然语言指令搜索并购买满足**所有属性**的商品。
- **对比基线**：IL（模仿学习）、IL+RL（模仿学习 + 强化学习）。

从表 4 可见：

- One-shot ReAct 的成功率达 **40.0%**，比之前最佳的 IL+RL（28.7%）**绝对提升超 10 个百分点**。
- ReAct 能通过推理，弥合「嘈杂的网页观察」与「正确动作」之间的鸿沟（例如推理出「这个省空间的沙发凳有 39x18x18 英寸、蓝色选项，看起来符合要求」）。
- 但离人类专家（59.6%）仍有明显差距。

#### 📊 图 5（Figure 5）：人在回路的行为纠正示例（附录 A.3）

![Figure 5：人在回路的行为纠正示例](/assets/posts/react/figure5.png)

- 在 ALFWorld 中，Agent 因为一句幻觉 Thought（第 17 步）导致任务失败。
- 人类**只需删除那句错误 Thought、并在第 23 步补一句提示 Thought**，Agent 立刻改变行为、成功完成任务。
- **意义**：ReAct 具备极强的可解释性和人机协作潜力——修改几个词的思考，远比修改强化学习策略或整段动作序列容易。

---

## 5. 核心消融与方法对比总结

论文用严密的消融实验，证明「推理」与「动作」缺一不可：

| 方法 | 核心机制 | 优势 | 劣势 / 失败模式 |
| :--- | :--- | :--- | :--- |
| **Standard** | 直接端到端输出答案 | 简单 | 完全黑盒、无过程、严重幻觉 |
| **CoT**（思维链） | 仅内部推理（Thought） | 逻辑结构清晰，无需外部 API | 静态黑盒，重度依赖内部知识，**幻觉率极高（占失败的 56%）**，无法处理过时信息 |
| **Act-only** | 仅外部交互（Action） | 能获取实时/外部事实 | 缺乏工作记忆与高层规划，容易迷失、陷入死循环 |
| **ReAct** | Thought 与 Action 交错 | **接地气、可解释**；推理指导动作，观察修正推理 | 受限于外部 API 检索能力；贪心解码下易陷入「重复思考/动作」死循环（推理错误占 47%） |
| **CoT-SC** | CoT + 多次采样投票 | 提升 CoT 鲁棒性 | 计算成本高（需 21 次采样），且无法根治事实幻觉 |
| **ReAct + CoT-SC** | 动态回退机制 | 自信时用内部知识、不自信时用外部检索 | —— 取得提示范式下的 SOTA 性能 |

---

## 6. 局限性与未来工作

1. **提示的上下文长度限制**：对于动作空间极大、程 horizon 极长的复杂决策任务（如 ALFWorld），少样本示例很容易超出模型输入长度上限。
2. **推理死循环**：贪心解码（greedy decoding）下，ReAct 容易陷入重复此前思考与动作的局部最优。作者推测更好的解码策略（如集束搜索 beam search）可能缓解。
3. **未来方向**：
   - 用更多高质量人工标注数据做**微调**（已初步证明 3k 数据即可让 8B 模型超越 62B 提示表现）；
   - 将 ReAct 与**强化学习（RL）**结合，用环境奖励进一步优化思考与动作的生成；
   - 多任务训练（multi-task training）以提升泛化能力。

---

## 7. 结论

ReAct 是一个**简单而有效**的通用范式：通过在 LLM 中协同「推理」与「动作」，它在知识密集型问答、事实验证和交互式决策任务上都取得了卓越性能和极高的可解释性。

它传递的一个更深层观点是：**语言不只是生成答案的工具，更是智能体进行自我调节、规划、并与环境交互的认知机制（cognitive mechanism）**。这正是后续 Agent 生态（工具调用、ReAct-style prompting）的思想源头之一。

---

## 8. 附录：其它值得关注的图表

- **表 5（Table 5，附录 A.1）**：用 GPT-3（text-davinci-002）复现 ReAct，在 HotpotQA（30.8）和 ALFWorld（78.4%）上表现甚至优于 PaLM-540B，证明 **ReAct 范式具有跨模型通用性**。

  ![Table 5：PaLM-540B 与 GPT-3 的 ReAct 结果对比](/assets/posts/react/table5.png)

- **表 6（Table 6，附录 C.3）**：WebShop 的提示示例——Act 只有机械的 `search`/`click`，ReAct 额外加入 `think` 步骤来评估商品属性是否匹配需求。
- **表 7/8/9（附录 C.4）**：ALFWorld 提示示例，对比 Act（无思考）、ReAct（含常识推理与子目标追踪）、ReAct-IM（仅含类似「我要找刀」的机械外部反馈）。
- **表 10（附录 D.3）**：WebShop 轨迹对比——Act 盲目点了不匹配口味的商品导致低分；ReAct 通过思考排除错误选项、精准选中符合「apple cinnamon」和「16 pack」的商品，拿到满分（Score 1.0）。

---

## 附录：专有名词中英对照

| 中文 | 英文 |
| :--- | :--- |
| 大语言模型 | Large Language Models (LLMs) |
| 推理轨迹 / 思考 | Reasoning trace / Thought |
| 动作 | Action |
| 观察 | Observation |
| 思维链 | Chain-of-Thought (CoT) |
| 自洽性 | Self-Consistency (CoT-SC) |
| 仅动作 | Act-only |
| 标准提示 | Standard prompting |
| 事实幻觉 | Hallucination |
| 错误传播 | Error propagation |
| 接地气 / 有事实依据 | Grounded |
| 工作记忆 | Working memory |
| 少样本上下文学习 | Few-shot in-context learning |
| 提示 | Prompting |
| 微调 | Finetuning |
| 强化学习 | Reinforcement Learning (RL) |
| 模仿学习 | Imitation Learning (IL) |
| 人在回路 | Human-in-the-loop |
| 内心独白 | Inner Monologue (IM) |
| 多跳问答 | Multi-hop QA |
| 事实验证 | Fact verification |
| 精确匹配 | Exact Match (EM) |
| 成功率 | Success Rate (SR) |
| 贪心解码 | Greedy decoding |
| 集束搜索 | Beam search |
| 假阳性 | False positive |
| 认知机制 | Cognitive mechanism |
| 最新知识 | Up-to-date knowledge |

---

## 附录：论文链接

原文地址：<https://arxiv.org/abs/2210.03629?utm_source=chatgpt.com>
