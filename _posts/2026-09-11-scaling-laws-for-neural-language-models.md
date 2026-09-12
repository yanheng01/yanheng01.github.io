---
title: "Scaling Laws for Neural Language Models 缩放定律学习笔记"
date: 2026-09-11 18:00:00 +0800
categories: [论文精读, 深度学习]
tags: [scaling-laws, 缩放定律, nlp, 论文精读, 大模型, openai]
math: true
mermaid: false
toc: true
---

> 一份深入的中文阅读笔记。文中专有名词附英文，文末附「专有名词中英对照附录」。论文全部 24 张配图（Figure 1–24）均已嵌入并配中文解读。

> **原文标题**：Scaling Laws for Neural Language Models
> **作者**：Jared Kaplan、Sam McCandlish、Tom Henighan、Tom B. Brown、Benjamin Chess、Rewon Child、Scott Gray、Alec Radford、Jeffrey Wu、Dario Amodei
> **机构**：约翰霍普金斯大学（Johns Hopkins University）、OpenAI
> **发表时间**：2020 年 1 月 23 日（arXiv:2001.08361v1）

---

## 目录

- [一、一句话总结](#一一句话总结)
- [二、研究动机与背景](#二研究动机与背景)
- [三、符号与术语约定](#三符号与术语约定)
- [四、核心缩放定律与公式](#四核心缩放定律与公式)
- [五、逐节精读](#五逐节精读)
- [六、关键结论提炼](#六关键结论提炼)
- [七、讨论与局限性](#七讨论与局限性)
- [八、全部配图与解读](#八全部配图与解读)
- [附录：术语对照表与原文链接](#附录术语对照表与原文链接)

---

## 一、一句话总结

语言模型的交叉熵损失（Cross-entropy Loss）与**模型规模**、**数据集规模**、**训练计算量**这三个因素之间存在跨越 **七个数量级** 的、精确而平滑的 **幂律关系（Power-law）**；而网络的具体架构形状（深度、宽度）几乎不影响性能。由此可以推导出固定算力预算下的最优分配策略：**在适量数据上训练一个非常大的模型，并在其收敛之前显著提前停止训练**。

---

## 二、研究动机与背景

深度学习中语言建模是一个理想的研究对象：数据近乎无限，模型可以任意扩大，损失函数（交叉熵）连续可测。本文系统性地实证研究了当模型规模、数据规模、计算量在极宽范围内变化时，模型性能如何随之变化，并发现这些变化遵循简单、可预测的幂律规律。

这一发现的深远意义在于：性能不再是"炼丹式"的不可预测结果，而是可以**提前定量预测**的——只要知道要投入多少参数、多少数据、多少算力，就能相当准确地预估最终损失。这直接为后续 GPT-3 等超大模型的研发提供了理论依据。

---

## 三、符号与术语约定

论文建立了一套严密的符号系统，是理解全文的基础：

| 符号 | 含义 | 英文/说明 |
|------|------|-----------|
| $L$ | 交叉熵损失（单位：nats），通常在上下文 token 上取平均 | Cross-entropy Loss |
| $N$ | **非嵌入（non-embedding）** 模型参数量，核心规模指标 | Number of Parameters |
| $D$ | 数据集大小（单位：tokens） | Dataset Size |
| $C$ | 总非嵌入训练计算量（单位：PF-days），估算式 $C \approx 6NBS$ | Compute |
| $B$ | 批次大小（单位：tokens） | Batch Size |
| $S$ | 训练步数（参数更新次数） | Number of Steps |
| $B_\text{crit}$ | 临界批次大小，决定速度与算力效率的权衡点 | Critical Batch Size |
| $C_\text{min}$ | 达到给定损失所需的**最小非嵌入计算量**（假设极小 batch） | Minimum Compute |
| $S_\text{min}$ | 达到给定损失所需的**最小训练步数**（假设极大 batch） | Minimum Steps |
| $\alpha_X$ | 幂律指数，$L(X) \propto 1/X^{\alpha_X}$ | Power-law Exponent |

**关键约定**：模型规模 $N$ 严格定义为**不含词嵌入（embedding）与位置嵌入的参数量**，近似为：

$$
N \approx 12\, n_\text{layer}\, d_\text{model}^2
$$

排除嵌入参数后，缩放定律才会呈现出惊人的平滑与一致。

---

## 四、核心缩放定律与公式

以下所有常数均基于 WebText2 数据集拟合得出，是全文的灵魂。

### 4.1 单一瓶颈下的三大幂律

**① 受参数量约束**（数据充足且训练至收敛）：

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076,\ N_c \approx 8.8\times10^{13}
$$

**② 受数据量约束**（大模型且提前停止）：

$$
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095,\ D_c \approx 5.4\times10^{13}
$$

**③ 受计算量约束**（最优分配、极小批次）：

$$
L(C_\text{min}) = \left(\frac{C_c^\text{min}}{C_\text{min}}\right)^{\alpha_C^\text{min}}, \quad \alpha_C^\text{min} \approx 0.050,\ C_c^\text{min} \approx 3.1\times10^{8}\ \text{PF-days}
$$

### 4.2 临界批次大小（Critical Batch Size）

$$
B_\text{crit}(L) = \frac{B_*}{L^{1/\alpha_B}}, \quad B_* \approx 2\times10^{8}\ \text{tokens},\ \alpha_B \approx 0.21
$$

> 注意：临界批次大小**只由目标损失 $L$ 决定，与模型大小 $N$ 无关**。

### 4.3 联合依赖公式

**④ 模型与数据的联合依赖**（刻画过拟合）：

$$
L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D}
$$

**⑤ 模型与训练步数的联合依赖**（学习曲线）：

$$
L(N, S) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_\text{min}(S)}\right)^{\alpha_S}, \quad S_c \approx 2.1\times10^{3},\ \alpha_S \approx 0.76
$$

### 4.4 最优计算分配法则

在固定算力预算 $C$ 下，最优的 $N,B,S$ 增长规律为：

$$
N \propto C^{\alpha_C^\text{min}/\alpha_N}, \quad B \propto C^{\alpha_C^\text{min}/\alpha_B}, \quad S \propto C^{\alpha_C^\text{min}/\alpha_S}, \quad D = B\cdot S
$$

综合指数：

$$
\alpha_C^\text{min} = \frac{1}{1/\alpha_S + 1/\alpha_B + 1/\alpha_N} \approx 0.050
$$

**经验拟合结果**：

$$
N \propto C_\text{min}^{0.73}, \quad B \propto C_\text{min}^{0.24}, \quad S \propto C_\text{min}^{0.03}
$$

即：算力增加时，**绝大部分应投入到扩大模型规模**，批次大小其次，而训练步数几乎不变。

---

## 五、逐节精读

### Section 2：背景与方法（Background and Methods）

- **Transformer 计算量估算**：单次前向传播的非嵌入计算量约为 $C_\text{forward} \approx 2N + 2\,n_\text{layer}\,n_\text{ctx}\,d_\text{model}$。计入反向传播后，**每个训练 token 的总计算量估算为 $C \approx 6N$ FLOPs**。
- **参数定义的关键性**：只有将 $N$ 定义为非嵌入参数，缩放曲线才会平滑；若把嵌入参数计入，则模型深度会严重干扰趋势。

### Section 3：实证结果与基本幂律（Empirical Results and Basic Power Laws）

- **架构无关性**：固定 $N$ 时，改变模型"形状"（深度、宽度、注意力头数、前馈层比例），损失变化极小（通常在 10% 以内）。
- **嵌入参数的陷阱**：只有**排除嵌入参数**后，不同深度的模型才收敛于同一条完美的幂律曲线（见 Figure 6）。
- **Transformer vs LSTM**：LSTM 在上下文早期 token 上表现尚可，但在长上下文后期完全无法匹敌 Transformer，后者具有渐近的长上下文优势。
- **迁移/泛化**：模型在其他数据分布（书籍、维基百科等）上的损失与训练集损失存在**恒定的常数偏移**；泛化能力**只取决于训练集上的表现**，与是否收敛、网络深度无关。

### Section 4：无限数据极限与过拟合（Infinite Data Limit and Overfitting）

- **过拟合的普遍规律**：过拟合的惩罚程度主要取决于比率 $N^{\alpha_N/\alpha_D}/D \approx N^{0.74}/D$。
- **数据需求法则**：为把过拟合造成的损失波动控制在 0.02 以内，数据量需满足 $D \gtrsim (5\times10^3)\,N^{0.74}$。即**数据量的增长可以慢于参数量的增长（次线性）**。

### Section 5：模型规模与训练时间的缩放（Scaling with Model Size and Training Time）

- **临界批次大小 $B_\text{crit}$**：随损失降低呈幂律增长；在 $B_\text{crit}$ 下训练能同时兼顾时间效率与算力效率。
- **学习曲线的普适性**：所有模型在初始瞬态之后，学习曲线都可用统一公式 $L(N, S_\text{min})$ 完美拟合，暗示损失曲面（Loss Landscape）的 Hessian 特征值密度大致与模型规模无关。

### Section 6：算力预算的最优分配（Optimal Allocation of the Compute Budget）

- **大模型即正义**：算力增加时，约 **73%** 的指数应投入扩大参数 $N$，**24%** 投入增大批次 $B$，而训练步数 $S$ 仅占 **3%**（几乎不增加）。
- **计算高效训练的本质**：固定算力下，训练大模型并**提前停止**，远优于把小模型训练到收敛。
- **终极极限猜想**：计算高效所需的数据增速（$D \propto C^{0.26}$）慢于避免过拟合所需的数据增速（$D \propto C^{0.54}$）。两者交点（约 $10^4$ PF-days、$10^{12}$ 参数、$L \approx 1.7$ nats/token）可能标志着 Transformer 架构的极限，而 $L^* \approx 1.7$ nats/token 可能就是**自然语言的真实熵**。

---

## 六、关键结论提炼

1. **性能平滑可预测**：损失随 $N,D,C$ 平滑幂律下降，在触及语言熵之前没有明显天花板或突变。
2. **大模型比大数据更"样本高效"**：大模型收敛更快，达到相同损失所需的样本量与训练步数远少于小模型。
3. **算力分配黄金法则**：给定算力，"训练超大模型 + 极少训练步数（提前停止）"是最优解。
4. **架构不重要**：只要总参数量 $N$ 相同，"深而窄"或"浅而宽"对最终损失影响微乎其微。

---

## 七、讨论与局限性

### 核心讨论

- **呼唤统计力学式理论**：目前的缩放定律是纯经验的（类似热力学），缺乏底层理论推导（类似统计力学）。
- **"量变引起质变"（More is different）**：损失的平滑下降可能掩盖了模型在特定任务上突然涌现的定性能力（即后来所说的"涌现能力"）。
- **模型并行的重要性**：既然扩大规模比增加数据更划算，未来**模型并行（Model Parallelism）** 技术将至关重要。

### 局限性（Caveats）

1. **缺乏理论基础**：尚无坚实理论能推导这些定律，$N$ 与 $C$ 的缩放机制仍是谜。
2. **小数据区域未充分探索**：数据极小时 $L(N,D)$ 拟合较差，可能处于不同机制区域。
3. **未联合调优正则化**：未系统研究 Dropout、数据增强等对过拟合边界的影响。
4. **计算量估算简化**：$C \approx 6NBS$ 忽略了与上下文长度 $n_\text{ctx}$ 相关的注意力计算，在极长上下文下会失效。
5. **超参数盲区**：可能忽略了初始化尺度、动量等潜在影响因素。
6. **学习率局限**：大模型需极小学习率防发散，但在算力受限的短训练中，更大学习率可能更优，本文未做实验。
7. **临界批次外推风险**：对极端损失下的 $B_\text{crit}$ 预测缺乏把握，会影响训练时间估算。

---

## 八、全部配图与解读

### Figure 1｜三大因素的幂律关系

![Figure 1](/assets/posts/scaling-laws/figure01.png)

**横轴**：分别为计算量（Compute, PF-days）、数据集大小（Dataset Size, tokens）、参数量（Parameters, non-embedding）；**纵轴**：测试损失（Test Loss）。三条曲线均呈平滑幂律下降，分别对应 $L=(C_\text{min}/2.3\times10^8)^{-0.050}$、$L=(D/5.4\times10^{13})^{-0.095}$、$L=(N/8.8\times10^{13})^{-0.076}$，跨越多个数量级。这是全文的核心示意图。

### Figure 2｜样本效率与计算高效性

![Figure 2](/assets/posts/scaling-laws/figure02.png)

展示一系列训练曲线：**大模型只需极少的样本（tokens）即可达到与小模型相同的损失**；最优模型大小随算力平滑增长；计算高效的训练应在损失曲线远未收敛时就停止。

### Figure 3｜算力扩展策略

![Figure 3](/assets/posts/scaling-laws/figure03.png)

当算力增加 100 万倍时的最优分配：模型规模应增大约 100 万倍（0.73 次方主导），批次大小增大约 100 倍，而**串行训练步数仅增加不到 10 倍**。

### Figure 4｜联合依赖关系

![Figure 4](/assets/posts/scaling-laws/figure04.png)

**左**：$L(N,D)$ 公式拟合，展示固定数据量 $D$ 时盲目增大 $N$ 带来的过拟合惩罚；**右**：$L(N,S)$ 学习曲线拟合，不同规模模型的学习曲线可被统一公式刻画。

### Figure 5｜模型形状的无关性

![Figure 5](/assets/posts/scaling-laws/figure05.png)

固定参数量，改变前馈层比例（FFN ratio）、长宽比（深度 vs 宽度）、注意力头维度，损失变化极小（< 10%），证明性能几乎只取决于总参数量。

### Figure 6｜嵌入参数的干扰

![Figure 6](/assets/posts/scaling-laws/figure06.png)

**左**：包含嵌入参数时，层数对趋势干扰极大，曲线杂乱；**右**：排除嵌入参数后，不同深度的模型完美收敛于同一条幂律曲线。

### Figure 7｜Transformer vs LSTM

![Figure 7](/assets/posts/scaling-laws/figure07.png)

LSTM 在上下文前几个 token 表现尚可，但随位置推移迅速恶化；Transformer 在整个长上下文中表现优异，并渐近超越 LSTM。

### Figure 8｜泛化与迁移学习

![Figure 8](/assets/posts/scaling-laws/figure08.png)

**左**：在书籍、维基百科等外部数据集上的损失随 $N$ 平滑下降，与训练集损失保持恒定偏移；**右**：泛化能力仅与训练集表现相关，与训练所处阶段（是否收敛）无关。

### Figure 9｜过拟合的普遍性

![Figure 9](/assets/posts/scaling-laws/figure09.png)

**左**：固定数据量 $D$，盲目增加 $N$ 会导致过拟合（损失回升）；**右**：过拟合程度精确依赖于比率 $N^{0.74}/D$。

### Figure 10｜临界批次大小

![Figure 10](/assets/posts/scaling-laws/figure10.png)

临界批次大小 $B_\text{crit}$ 随损失降低呈幂律增长，且**只由损失决定，与模型规模 $N$ 无关**。

### Figure 11｜固定算力/步数下的最优 N

![Figure 11](/assets/posts/scaling-laws/figure11.png)

**左**：固定算力 $C$，存在使损失最低的"最优模型大小"；**右**：固定步数 $S$，同样存在一个最优 $N$。

### Figure 12｜次优模型大小的惩罚

![Figure 12](/assets/posts/scaling-laws/figure12.png)

**左**：使用 0.6×～2.2× 最优大小的模型，仅需多耗费约 20% 的算力；大模型虽略微浪费算力，却能大幅减少训练步数（更利于并行）。

### Figure 13｜调整后的计算量趋势

![Figure 13](/assets/posts/scaling-laws/figure13.png)

将经验数据调整到极小批次（即 $C_\text{min}$）后，$L(C_\text{min})$ 的幂律拟合更完美，消除了批次大小带来的噪声。

### Figure 14｜最优参数与步数随算力的增长

![Figure 14](/assets/posts/scaling-laws/figure14.png)

**左**：最优参数量 $N$ 随算力急剧增长（0.73 次方）；**右**：最优训练步数 $S_\text{min}$ 随算力几乎不增长（0.03 次方）。

### Figure 15｜缩放定律的极限交点

![Figure 15](/assets/posts/scaling-laws/figure15.png)

计算高效所需的数据增长曲线 $L(C_\text{min})$ 与避免过拟合所需的曲线 $L(D(C))$ 发生交叉。交点预测了 Transformer 的终极极限（约 $10^{12}$ 参数、$1.7$ nats/token），且该交点对幂律参数十分敏感。

### Figure 16｜早停下界与过拟合曲线

![Figure 16](/assets/posts/scaling-laws/figure16.png)

**左**：理论推导的早停步数下界（early stopping step）；**右**：不同数据量下的训练/测试损失曲线，展示测试损失何时开始偏离并回升。

### Figure 17｜Universal Transformers（参数复用）

![Figure 17](/assets/posts/scaling-laws/figure17.png)

参数复用的循环 Transformer 在同等参数量下略好，但在同等计算量下略差。

### Figure 18｜批次大小扫描

![Figure 18](/assets/posts/scaling-laws/figure18.png)

通过不同批次大小的实验，实证测量并验证临界批次大小 $B_\text{crit}$ 的理论公式（对应公式 5.1）。

### Figure 19｜样本效率的另一视角

![Figure 19](/assets/posts/scaling-laws/figure19.png)

达到固定损失所需的最小串行步数与最小样本量，随模型规模 $N$ 增大而断崖式下降。

### Figure 20｜上下文位置依赖

![Figure 20](/assets/posts/scaling-laws/figure20.png)

**左**：损失随上下文位置 $T$ 呈幂律变化；**右**：模型先学会短距离依赖，随后逐渐学会长距离依赖。

### Figure 21｜不同位置损失随 N 的变化

![Figure 21](/assets/posts/scaling-laws/figure21.png)

大模型不仅在长上下文末端表现更好，在短上下文（早期 token）上的收敛速度也更快。

### Figure 22｜学习率调度扫描

![Figure 22](/assets/posts/scaling-laws/figure22.png)

测试多种学习率调度（Cosine、Linear 等）；只要总学习率积分足够且带 Warmup，具体衰减曲线对最终损失影响不大。

### Figure 23｜幂律 vs 对数拟合

![Figure 23](/assets/posts/scaling-laws/figure23.png)

定性对比证明，幂律函数（Power-law）对 $L(N)$ 的拟合远优于对数函数（Logarithm）。

### Figure 24｜泛化与深度的关系

![Figure 24](/assets/posts/scaling-laws/figure24.png)

固定参数量、改变网络深度时，泛化到外部数据集的表现几乎无差异，仅取决于训练集损失。

---

## 附录：术语对照表与原文链接

### 专有名词中英对照

| 中文 | 英文 |
|------|------|
| 缩放定律 | Scaling Laws |
| 幂律 | Power-law |
| 交叉熵损失 | Cross-entropy Loss |
| 参数量 | Number of Parameters (N) |
| 数据集大小 | Dataset Size (D) |
| 计算量 | Compute (C) |
| 批次大小 | Batch Size (B) |
| 训练步数 | Number of Steps (S) |
| 临界批次大小 | Critical Batch Size |
| 提前停止 | Early Stopping |
| 样本效率 | Sample Efficiency |
| 过拟合 | Overfitting |
| 泛化 | Generalization |
| 词嵌入 | Embedding |
| 上下文长度 | Context Length ($n_\text{ctx}$) |
| 前馈网络 | Feed-Forward Network (FFN) |
| 注意力头 | Attention Head |
| 损失曲面 | Loss Landscape |
| 海森矩阵 | Hessian |
| 涌现能力 | Emergent Ability |
| 模型并行 | Model Parallelism |
| 学习率预热 | Learning Rate Warmup |
| 自然语言的熵 | Entropy of Natural Language |
| PF-days（千万亿次浮点运算·天） | Petaflop-days |

### 原文链接

- **论文地址**：<https://arxiv.org/abs/2001.08361?utm_source=chatgpt.com>

---

*本笔记为个人精读整理，图表均取自论文原文，仅供学习交流使用。*
