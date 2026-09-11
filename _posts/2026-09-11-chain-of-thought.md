---
title: "Chain-of-Thought，LLM 怎么产生复杂推理"
date: 2026-09-11 18:47:00 +0800
categories: [技术笔记, 深度学习]
tags: [chain-of-thought, cot, llm, 推理, prompting, 论文精读, Agent]
math: true
mermaid: false
toc: true
---

> 一份关于「思维链提示（Chain-of-Thought Prompting）」的中文精读笔记。文中所有专有名词首次出现时附英文，文末附「专有名词中英对照附录」。论文正文全部 8 张配图（Figure 1–8）均已嵌入并配中文解读。

---

## 0. 论文基本信息

| 项目 | 内容 |
| :--- | :--- |
| 标题 | **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**（思维链提示激发大语言模型的推理能力） |
| 作者 | Jason Wei、Xuezhi Wang、Dale Schuurmans、Maarten Bosma、Brian Ichter、Fei Xia、Ed H. Chi、Quoc V. Le、Denny Zhou |
| 机构 | 谷歌研究院·大脑团队（Google Research, Brain Team） |
| 发表 | 第 36 届神经信息处理系统大会（**NeurIPS 2022**）；arXiv:2201.11903 |
| 地位 | 提出**思维链提示（Chain-of-Thought Prompting）**，是"提示激发推理"这一范式的奠基性工作 |

---

## 1. 一句话总结

只要让大语言模型（Large Language Model, LLM）在给出最终答案之前，先把「中间推理步骤」一步步写出来（即一条**思维链 / Chain of Thought**），它在算术、常识、符号等复杂推理任务上的表现就会**大幅跃升**——而且这件事**几乎不需要额外训练**，只需在提示（Prompt）里放几个带推理过程的示例即可。

---

## 2. 摘要（Abstract）

- **核心发现**：生成"思维链"（chain of thought，一系列中间推理步骤）能显著提升大语言模型执行复杂推理的能力。
- **提出方法**：**思维链提示（Chain-of-Thought Prompting）**——一种极其简单的方法，只需在提示中提供少量包含思维链的示例（exemplar），推理能力便会在足够大的模型中**自然涌现**。
- **惊人结果**：PaLM 540B 仅用 **8 个**思维链示例，就在 GSM8K（数学应用题基准）上取得当时的最新最优（State-of-the-Art, SOTA），甚至**超越了经过微调并带验证器（verifier）的 GPT-3**。

---

## 3. 研究动机（Introduction）

作者观察到两个此前相互独立的现象，并尝试把它们的优点结合起来：

1. **单纯堆规模不够用。** 把模型做大（scaling up）确实能提升很多常规 NLP 任务，但在**算术、常识、符号推理**这类需要"多步思考"的任务上，光靠扩大规模远远不够。
2. **两条已有思路各有短板：**
   - **生成推理依据（rationale）**：让模型输出自然语言推理步骤能帮助算术推理，但传统做法要么从头训练、要么微调，**成本高、需大量标注数据**。
   - **上下文少样本学习（in-context few-shot learning）**：大模型可只靠提示里的几个示例学会新任务，无需微调；但**标准少样本提示（standard prompting）在推理任务上表现很差**，且不太随规模增长而改善。

**本文的做法**：把两者结合，提出思维链提示——在少样本示例中，用 `⟨输入, 思维链, 输出⟩`（`⟨input, chain of thought, output⟩`）三元组代替传统的 `⟨输入, 输出⟩` 对。无需任何梯度更新，单一模型检查点即可胜任多种复杂推理任务。

---

## 4. 什么是思维链提示（Chain-of-Thought Prompting）

**核心思想**：模仿人类解决复杂问题（比如多步数学应用题）的思路——先把问题拆解成一连串中间步骤，逐步推导，最后再得出答案。

下图直观对比了标准提示与思维链提示的差异：标准提示直接让模型猜答案（结果算错，得到 27）；思维链提示则让模型先写出中间推理过程，最终得到正确答案 9。

![Figure 1: 标准提示 vs 思维链提示的对比示意](/assets/posts/chain-of-thought/figure1.png)

*图 1（Figure 1）：思维链提示让大语言模型能够处理复杂的算术、常识和符号推理任务。图中高亮部分即为"思维链"推理过程。*

### 思维链的四大优势

1. **动态分配计算量（compute allocation）**：把多步问题拆开，让模型对更难的问题自然地生成更多中间 token、投入更多"思考"。
2. **可解释性（interpretability）**：推理过程提供了一扇观察模型行为的窗口，便于调试推理链在哪一步出了错。
3. **通用性（wide applicability）**：原则上适用于任何人类能用语言分步解决的任务——数学、常识、符号操作皆可。
4. **易于激发（easy to elicit）**：只要在现成的超大模型的少样本示例里加入思维链即可触发，**完全不需要重新训练模型**。

---

## 5. 实验设置（Experimental Setup）

- **算术推理数据集**：GSM8K、SVAMP、ASDiv、AQuA、MAWPS
- **常识推理数据集**：CSQA、StrategyQA、Date Understanding（日期理解）、Sports Understanding（体育理解）、SayCan（机器人指令）
- **符号推理任务**：Last Letter Concatenation（末位字母拼接）、Coin Flip（硬币翻转状态追踪）
- **对照基线（baseline）**：标准提示（standard prompting），只给 `⟨输入, 输出⟩` 对。
- **思维链提示**：为每个任务人工编写 8 个（AQuA 为 4 个）带推理步骤的少样本示例。
- **评估模型**：GPT-3（350M–175B）、LaMDA（422M–137B）、PaLM（8B–540B）、UL2 20B、Codex。均采用贪心解码（greedy decoding）。

下图展示了各类任务所用的 `⟨输入, 思维链, 输出⟩` 示例，高亮部分即人工编写的思维链：

![Figure 3: 算术、常识、符号推理任务的思维链示例集合](/assets/posts/chain-of-thought/figure3.png)

*图 3（Figure 3）：涵盖算术（数学应用题）、常识（CSQA、StrategyQA、日期、体育、SayCan）、符号（末位字母拼接、硬币翻转）任务的思维链示例，证明该框架可适配多种异构任务。*

---

## 6. 核心结果

### 6.1 算术推理（Arithmetic Reasoning）

在 GSM8K（数学应用题基准）上，PaLM 540B 仅用 8 个思维链示例就取得了当时的**最新最优（SOTA）**成绩，甚至超越了**经过微调并带验证器（verifier）的 GPT-3**。

![Figure 2: PaLM 540B 在 GSM8K 上依靠思维链达到 SOTA](/assets/posts/chain-of-thought/figure2.png)

*图 2（Figure 2）：在 GSM8K 上，PaLM 540B 仅凭思维链提示（57%）就超越了微调 GPT-3 175B（33%）与此前最优（Prior best，55%），而标准提示只有 18%。*

**三大关键结论：**

1. **思维链是"规模的涌现能力"（emergent ability）。** 小模型用思维链甚至**不如**标准提示（会生成"流畅但荒谬"的步骤）；只有当模型规模达到约 **100B（千亿）参数** 时，思维链才带来性能的陡峭飞跃。
2. **问题越复杂，思维链收益越大。** 在最难的 GSM8K 上性能翻倍以上；而在只需单步计算的 MAWPS（SingleOp）子集上，提升极小甚至为负。
3. **无需微调即达 SOTA。** PaLM 540B + 思维链在 GSM8K、SVAMP、MAWPS 上超越了此前需要专门微调的最优模型。

下图清晰展示了"涌现"现象：标准提示曲线几乎平坦，而思维链曲线在约 100B 参数后急剧上升。

![Figure 4: 思维链是随模型规模增长而涌现的能力](/assets/posts/chain-of-thought/figure4.png)

*图 4（Figure 4）：横轴为模型规模（0.4B–540B），纵轴为解题率（solve rate）。三行分别为 GSM8K / SVAMP / MAWPS，三列分别为 LaMDA / GPT / PaLM。标准提示（黑线）平坦，思维链提示（蓝线）在 ~100B 后陡峭涌现，橙色虚线为此前监督学习最优。*

**错误分析**：在得出正确答案的思维链中，逻辑几乎全对；在得出错误答案的思维链中，约 **46%** 只是微小错误（如计算器算错、漏一步），另 **54%** 是语义理解或逻辑连贯性的重大错误。

### 6.2 常识推理（Commonsense Reasoning）

思维链的**语言通用性**在这里得到验证——它不仅适用于数学，也适用于需要背景知识和多跳推理（multi-hop reasoning）的常识任务。PaLM 540B 在 StrategyQA 上超越此前 SOTA（**75.6% vs 69.4%**），在 Sports Understanding 上甚至超越了资深体育迷（**95.4% vs 84%**）。

![Figure 7: 思维链同样提升常识推理能力](/assets/posts/chain-of-thought/figure7.png)

*图 7（Figure 7）：PaLM 在 CSQA、StrategyQA、Date、Sports、SayCan 上的规模曲线。思维链（蓝线）在多数任务上显著提升，并在 StrategyQA、Sports 上超越 Prior best 与人类（绿色虚线）。注意在 CSQA 上收益较小。*

### 6.3 符号推理（Symbolic Reasoning）

任务设计区分**域内（in-domain，与示例同步数）**与**域外（out-of-domain, OOD，序列更长）**两种测试。

**关键结论**：思维链不仅让模型学会符号操作（同样在 100B+ 规模涌现），更重要的是极大促进了**长度泛化（length generalization）**——模型能够处理比提示示例**更长**的推理序列。

![Figure 8: 思维链促进符号推理中的长度泛化](/assets/posts/chain-of-thought/figure8.png)

*图 8（Figure 8）：PaLM 在末位字母拼接（Letter Concat）与硬币翻转（Coin Flip）任务上的表现，分为 in-domain（2 步）与 OOD（4 步）。标准提示在 OOD 上几乎完全失败，思维链则维持了上升的规模曲线，展现出长度泛化能力。*

---

## 7. 消融实验（Ablation Study）与鲁棒性分析（Robustness）

### 7.1 消融实验：思维链到底为什么有效？

作者设计了三个变体来排除其他解释：

| 变体 | 做法 | 结论 |
| :--- | :--- | :--- |
| **仅输出方程（Equation only）** | 只让模型先写数学方程再答 | 对复杂题（GSM8K）**无效**（语义太难无法直接转方程），对简单题有效 |
| **仅占位计算（Variable compute only）** | 用等量的 `…` 点号占位，只增计算量不含语义 | 效果**等同基线**，证明思维链的成功**不只是因为多算了几步 token** |
| **先答后推（Reasoning after answer）** | 先给答案，再补推理过程 | 效果**等同基线**，证明起作用的是"**顺序推理这个过程本身**"，而非单纯激活预训练知识 |

![Figure 5: 不同提示变体的消融实验](/assets/posts/chain-of-thought/figure5.png)

*图 5（Figure 5）：LaMDA 137B 与 PaLM 540B 在 GSM8K 上，对比标准提示、仅方程、仅占位计算、先答后推、完整思维链。只有完整思维链（橙色）带来巨大提升，排除了"仅增计算量"和"仅激活知识"这两种假设。*

### 7.2 鲁棒性分析（Robustness）

- **不同标注者/风格**：由不同作者（Annotator A / B / C）编写、或采用更简洁风格的思维链，虽有方差，但**都大幅超越基线**——说明思维链不依赖某种特定的语言学风格。
- **不同示例来源**：从 GSM8K 训练集中随机采样示例（α / β / γ），效果与人工精编相当——说明示例**无需与测试集同分布**。
- **其他**：对示例的排列顺序、数量（在一定范围内）都表现稳健。

![Figure 6: 思维链对不同标注者与示例的鲁棒性](/assets/posts/chain-of-thought/figure6.png)

*图 6（Figure 6）：LaMDA 137B 在 GSM8K 与 MAWPS 上，对比标准提示与多种思维链变体（不同标注者 B/C、简洁风格、来自 GSM8K 的示例 α/β/γ）。虽有方差，但所有思维链变体都远超标准基线。*

---

## 8. 讨论、局限与结论

### 8.1 讨论（Discussion）

- 思维链提示是一种简单、通用的机制，用于激发大语言模型的多步推理行为。
- **标准提示只是 LLM 能力的下限**；思维链极大地扩展了 LLM 能成功解决的任务集合。
- 引出新问题：随着模型规模继续增大，推理能力还能提升多少？还有哪些提示方法能进一步拓展任务边界？

### 8.2 局限性（Limitations）

1. **"推理"的本质存疑**：思维链虽模仿人类思考，但神经网络是否真的在"推理"仍是开放问题。
2. **标注成本**：少样本场景下手写思维链成本很低，但若用于**微调（fine-tuning）**，构建大规模高质量思维链数据集的成本极高。
3. **事实准确性**：思维链无法保证推理路径绝对正确（存在幻觉/事实错误），可能得出错误答案，甚至"碰巧答对"。
4. **部署成本**：思维链的涌现依赖超大模型（100B+），在现实中的小模型上部署成本过高。

### 8.3 结论（Conclusions）

思维链提示是一种**简单、通用**的增强 LLM 推理的方法，且是**模型规模的涌现特性（emergent property）**——它让原本规模曲线（scaling curve）平坦的推理任务获得了陡峭的性能提升。

---

## 9. 专有名词中英对照附录（Glossary）

| 英文术语 | 中文翻译 | 说明 |
| :--- | :--- | :--- |
| Chain-of-Thought (CoT) Prompting | 思维链提示 | 本文核心方法 |
| Large Language Model (LLM) | 大语言模型 | — |
| Emergent Ability / Property | 涌现能力 / 涌现特性 | 小模型不具备、大模型突然具备的能力 |
| Standard Prompting | 标准提示 | 只含 `⟨输入, 输出⟩` 的基线 |
| Few-shot Learning | 少样本学习 | 靠少量示例学会任务 |
| In-context Learning | 上下文学习 | 在提示上下文中学习，无需微调 |
| Exemplar / Demonstration | 示例 / 演示 | 提示中给出的样例 |
| Rationale | 推理依据 | 自然语言的推理步骤 |
| Arithmetic Reasoning | 算术推理 | 如数学应用题 |
| Math Word Problems | 数学应用题 | — |
| Commonsense Reasoning | 常识推理 | 依赖世界知识与多跳推理 |
| Multi-hop Reasoning | 多跳推理 | 需多步串联的推理 |
| Symbolic Reasoning | 符号推理 | 如字母拼接、硬币翻转追踪 |
| Last Letter Concatenation | 末位字母拼接 | 符号推理玩具任务 |
| Coin Flip | 硬币翻转 | 状态追踪玩具任务 |
| In-domain | 域内 | 测试与示例同分布/同步数 |
| Out-of-Domain (OOD) | 域外 / 分布外 | 测试序列更长/分布不同 |
| Length Generalization | 长度泛化 | 处理比示例更长的序列 |
| Ablation Study | 消融实验 | 逐一移除组件以定位关键因素 |
| Robustness | 鲁棒性 | 对扰动的稳定性 |
| Equation only | 仅输出方程 | 消融变体 |
| Variable compute only | 仅占位计算 | 消融变体（用 `…` 占位） |
| Reasoning after answer | 先答后推 | 消融变体 |
| Annotator | 标注者 | 编写思维链的人 |
| Greedy Decoding | 贪心解码 | 每步取最高概率 token |
| Verifier | 验证器 | 微调 GPT-3 用于校验答案 |
| Solve rate | 解题率 | 评估指标 |
| Scaling / Scaling up | 规模扩展 | 增大模型参数量 |
| State-of-the-Art (SOTA) | 最新最优 | 当前最佳水平 |
| Fine-tuning | 微调 | 在下游数据上继续训练 |

主要涉及的模型与数据集：**PaLM 540B**、**GPT-3 175B**、**LaMDA 137B**、**UL2 20B**、**Codex**；**GSM8K**、**SVAMP**、**ASDiv**、**AQuA**、**MAWPS**、**CSQA**、**StrategyQA**、**SayCan**、**BIG-bench**。

---

## 附录：论文链接

- 原论文（arXiv）：<https://arxiv.org/abs/2201.11903?utm_source=chatgpt.com>
