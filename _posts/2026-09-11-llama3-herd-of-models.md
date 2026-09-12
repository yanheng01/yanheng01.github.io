---
title: "The Llama 3 Herd of Models 精读笔记"
date: 2026-09-11 18:00:00 +0800
categories: [论文精读, 深度学习]
tags: [llama3, meta, llm, 论文精读, 大模型, 多模态]
math: true
mermaid: false
toc: true
---

> 论文题目：**The Llama 3 Herd of Models**（Llama 3 模型群）
> 作者：Llama Team, AI @ Meta
> 发布日期：2024 年 7 月 23 日
> 本笔记基于论文全文（92 页）系统精读整理，配图均取自论文原图。

---

## 摘要与核心结论（Abstract）

现代人工智能（Artificial Intelligence, AI）系统由**基础模型（foundation models）**驱动。本文提出一组新的基础模型 **Llama 3**，这是一个**原生支持多语言（multilinguality）、代码（coding）、推理（reasoning）与工具使用（tool usage）**的语言模型群。

核心要点：

- **旗舰模型**：稠密（dense）Transformer，**405B 参数（parameters）**，上下文窗口（context window）最长 **128K tokens**。
- **性能**：在大量任务上与业界领先模型（如 GPT-4）质量相当，并接近最先进水平（state-of-the-art, SOTA）。
- **开源发布**：公开发布 405B 语言模型的预训练（pre-trained）与后训练（post-trained）版本，以及用于输入/输出安全的 **Llama Guard 3** 模型。
- **多模态（multimodal）扩展**：论文还展示了通过**组合式方法（compositional approach）**将图像（image）、视频（video）、语音（speech）能力集成到 Llama 3 的实验结果，性能与 SOTA 相当竞争，但这些多模态模型尚在开发中、尚未广泛发布。

三大高质量基础模型开发杠杆：**数据（data）、规模（scale）、复杂度管理（managing complexity）**。

| 关键数字 | Llama 3 | Llama 2 对比 |
|---|---|---|
| 预训练语料 | 约 **15T** 多语言 tokens | 1.8T tokens |
| 旗舰模型训练算力 | **3.8 × 10²⁵ FLOPs** | 约为 Llama 2 最大版本的 ~50 倍 |
| 旗舰模型参数 | **405B**，训练于 **15.6T** 文本 tokens | — |

> **Llama 3 模型群（Herd）成员**（论文所有结果均针对 Llama 3.1 系列，全文简称 Llama 3）：8B / 70B / 405B 三种规模，各有预训练版与 Instruct（指令）版。8B/70B 于 2024 年 4 月发布，Llama 3.1 全系列（含多语言、长上下文、工具使用）于 2024 年 7 月发布。

---

## 1. 引言（Introduction）

基础模型是语言、视觉、语音等模态的通用模型。现代基础模型开发包含两个主要阶段：

1. **预训练阶段（pre-training stage）**：用海量数据、以简单任务（如下一词预测 next-word prediction、图像描述 captioning）大规模训练模型。
2. **后训练阶段（post-training stage）**：微调模型使其遵循指令（follow instructions）、对齐人类偏好（align with human preferences）、并提升特定能力（如编码、推理）。

**复杂度管理的关键设计选择**：

- 选用**标准稠密 Transformer 架构**（Vaswani et al., 2017），仅做少量改动，而非**混合专家模型（Mixture-of-Experts, MoE）**——以最大化训练稳定性。
- 采用相对简单的后训练流程：**监督微调（Supervised Finetuning, SFT）+ 拒绝采样（Rejection Sampling, RS）+ 直接偏好优化（Direct Preference Optimization, DPO）**，而非更复杂、更不稳定的强化学习（Reinforcement Learning, RL）算法（如 PPO）。

> **规模化观察**：旗舰模型按缩放定律（scaling laws）约为**算力最优（compute-optimal）**规模；而较小模型训练时长远超算力最优点，从而在相同推理预算下表现更优。旗舰模型还用于在后训练阶段进一步提升小模型质量。

---

## 2. 总体概览（General Overview）

Llama 3 语言模型的整体架构如 **图 1** 所示。

### 图 1：Llama 3 整体架构与训练示意（原文第 3–4 页）

> Llama 3 是一个 Transformer 语言模型，训练目标是预测文本序列的下一个 token（next token）。输入文本 tokens → token 嵌入（embeddings）→ 多层自注意力（Self-attention）+ 前馈网络（Feedforward network）→ 输出文本 token，通过**自回归解码（autoregressive decoding）**循环生成。

![图 1：Llama 3 整体架构与训练示意](/assets/posts/llama3/fig01_architecture.png)

开发分两大阶段：

- **语言模型预训练**：将大规模多语言文本语料转换为离散 tokens，预训练大语言模型（Large Language Model, LLM）做下一 token 预测。405B 模型在 15.6T tokens 上、以 8K tokens 上下文窗口预训练，随后经**持续预训练（continued pre-training）**将上下文扩展到 128K tokens。
- **语言模型后训练**：通过多轮 SFT + DPO 与人类反馈对齐，并集成工具使用等新能力，同时在后训练阶段加入安全缓解（safety mitigations）。

**多模态组合式方法**（详见 **图 28**）包含三个额外阶段：

- **多模态编码器预训练（Multi-modal encoder pre-training）**：分别为图像和语音训练独立编码器。图像编码器用海量图文对训练；语音编码器用自监督（self-supervised）方法（掩码重建）训练。
- **视觉适配器训练（Vision adapter training）**：训练一系列**交叉注意力层（cross-attention layers）**将图像编码器表示喂入语言模型；训练时更新图像编码器参数但**不更新语言模型参数**。视频适配器在图像适配器之上训练。
- **语音适配器训练（Speech adapter training）**：通过适配器将语音编码转换为 token 表示喂入微调后的语言模型。

---

## 3. 预训练（Pre-Training）

预训练涉及：(1) 大规模训练语料的整理与过滤；(2) 模型架构与对应缩放定律的开发；(3) 大规模高效预训练技术；(4) 预训练配方（recipe）的开发。

### 3.1 预训练数据（Pre-Training Data）

数据知识截止到 **2023 年底**。移除含大量个人可识别信息（Personally Identifiable Information, PII）与已知成人内容的域名。

#### 3.1.1 网络数据整理（Web Data Curation）

- **PII 与安全过滤**：过滤不安全内容、高 PII 域名、有害域名、成人内容。
- **文本提取与清洗**：自建 HTML 解析器（parser），优化去样板（boilerplate removal）精度与内容召回；保留数学与代码结构；保留图片 `alt` 属性文本（数学常以预渲染图片呈现）。**发现 markdown 标记对主要基于网络数据训练的模型有害，因此移除所有 markdown 标记。**
- **去重（De-duplication）**（三级）：
  - **URL 级**：保留每个 URL 的最新版本。
  - **文档级**：全局 **MinHash**（Broder, 1997）去除近重复文档。
  - **行级**：类似 ccNet，移除在每 30M 文档桶中出现超过 6 次的行。
- **启发式过滤（Heuristic filtering）**：重复 n-gram 覆盖率去除日志/错误信息；"脏词"计数过滤成人网站；token 分布 KL 散度（Kullback-Leibler divergence）过滤异常 token 过多的文档。
- **基于模型的质量过滤**：使用 fasttext（识别是否会被维基百科引用）与基于 Roberta 的分类器（DistilRoberta，基于 Llama 2 预测训练）。
- **代码与推理数据**：类似 DeepSeek，构建领域专用流水线提取代码与数学网页，分类器为基于 Llama 2 标注训练的 DistilRoberta。
- **多语言数据**：用基于 fasttext 的语言识别模型分类为 **176 种语言**；做文档级、行级去重；用多语言 Llama 2 分类器做质量排序。

#### 3.1.2 确定数据配比（Determining the Data Mix）

通过**知识分类（knowledge classification）**下采样过度呈现的类别（如艺术娱乐），并用**缩放定律实验**确定最佳配比。

**最终数据配比（Data mix summary）**：

| 类别 | 占比 |
|---|---|
| 通用知识（general knowledge） | 约 **50%** |
| 数学与推理（mathematical and reasoning） | 约 **25%** |
| 代码（code） | **17%** |
| 多语言（multilingual） | **8%** |

#### 3.1.3 退火数据（Annealing Data）

在少量高质量代码和数学数据上做**退火（annealing）**可提升关键基准表现。退火使 8B 模型在 GSM8k 和 MATH 验证集上分别提升 **24.0%** 和 **6.4%**；但对 405B 模型提升微乎其微，说明旗舰模型已具备强大的上下文学习与推理能力。退火还可用于评估小型领域数据集的价值（对 50% 训练进度的 8B 模型在 40B tokens 上线性退火学习率至 0，新数据权重 30%）。

### 3.2 模型架构（Model Architecture）

Llama 3 使用标准稠密 Transformer，相比 Llama/Llama 2 架构无重大偏离，性能提升主要来自数据质量/多样性与训练规模。相比 Llama 2 的少量改动：

- **分组查询注意力（Grouped Query Attention, GQA）**：8 个键值头（key-value heads），提升推理速度、减小解码时 KV cache 大小。
- **文档内注意力掩码（attention mask）**：防止同一序列内不同文档间自注意力——在长序列持续预训练中尤为重要。
- **词表（vocabulary）128K tokens**：100K 来自 tiktoken tokenizer + 28K 额外 tokens 更好支持非英语；英语压缩率从 3.17 提升到 **3.94 字符/token**。
- **RoPE（旋转位置编码）基频（base frequency）提升到 500,000**：更好支持长上下文。

#### 表 3：Llama 3 关键超参数（原文第 7 页）

| 超参数 | 8B | 70B | 405B |
|---|---|---|---|
| 层数（Layers） | 32 | 80 | 126 |
| 模型维度（Model Dimension） | 4,096 | 8,192 | 16,384 |
| FFN 维度（FFN Dimension） | 14,336 | 28,672 | 53,248 |
| 注意力头数（Attention Heads） | 32 | 64 | 128 |
| 键值头数（Key/Value Heads） | 8 | 8 | 8 |
| 峰值学习率（Peak Learning Rate） | 3×10⁻⁴ | 1.5×10⁻⁴ | 8×10⁻⁵ |
| 激活函数（Activation Function） | SwiGLU | | |
| 词表大小（Vocabulary Size） | 128,000 | | |
| 位置编码（Positional Embeddings） | RoPE (θ = 500,000) | | |

#### 3.2.1 缩放定律（Scaling Laws）

开发缩放定律（Hoffmann/Kaplan 等）以在给定算力预算下确定最优模型规模，并预测下游基准表现。**两阶段方法**：

1. 建立算力最优模型在下游任务上的**负对数似然（negative log-likelihood, NLL）**与训练 FLOPs 的相关性。
2. 用缩放定律模型与 Llama 2 模型建立 NLL 与任务准确率的关系。

- 缩放定律实验用 **6×10¹⁸ 到 10²² FLOPs** 算力预算，模型规模 **40M 到 16B 参数**。
- 幂律关系 $N^{\*}(C) = A C^{\alpha}$，拟合得 **(α, A) = (0.53, 0.29)**。
- 外推到 3.8×10²⁵ FLOPs 建议训练 **402B 参数模型于 16.55T tokens**；因 IsoFLOPs 曲线在最小值附近变平，最终选择训练 **405B 参数**旗舰模型。

##### 图 2 & 图 3：缩放定律 IsoFLOPs 曲线 与 算力最优模型训练 tokens 数（原文第 8 页）

> **图 2**：6×10¹⁸ 到 10²² FLOPs 之间的缩放定律 IsoFLOPs 曲线，损失为留出验证集上的 NLL，用二次多项式拟合。
> **图 3**：算力最优模型的训练 tokens 数与预训练算力预算的关系，含拟合的缩放定律预测；算力最优模型对应图 2 中各抛物线的最小值。

##### 图 4：ARC Challenge 缩放定律预测（原文第 8–9 页）

> 左：ARC Challenge 上正确答案的归一化 NLL 与预训练 FLOPs 的关系；右：ARC Challenge 准确率与归一化 NLL 的关系。该两步预测跨越四个数量级，相当准确，仅略微低估旗舰模型的最终表现。

### 3.3 基础设施、规模化与效率（Infrastructure, Scaling, and Efficiency）

#### 3.3.1 训练基础设施（Training Infrastructure）

Llama 1/2 在 Meta 的 AI Research SuperCluster 训练；Llama 3 迁移到 Meta 生产集群。

- **算力（Compute）**：405B 训练于最多 **16K 块 H100 GPU**，每块 700W TDP、80GB HBM3，采用 Meta 的 **Grand Teton** AI 服务器平台。每台服务器 8 块 GPU + 2 块 CPU，服务器内 GPU 经 **NVLink** 互联。调度用 **MAST**（Meta 全球训练调度器）。
- **存储（Storage）**：**Tectonic** 分布式文件系统，提供 **240 PB** 存储（7,500 台 SSD 服务器），可持续吞吐 **2 TB/s**、峰值 **7 TB/s**。检查点（checkpoint）每 GPU 状态 1 MB–4 GB。
- **网络（Network）**：405B 用基于 Arista 7800 与 Minipack2 OCP 交换机的 **RoCE（RDMA over Converged Ethernet）** 架构；小模型用 Nvidia Quantum2 Infiniband。两者均为 **400 Gbps** GPU 间互联。
  - **网络拓扑**：24K GPU 的三层 Clos 网络；每机架 16 GPU（跨两台服务器，单 Minipack2 ToR 交换机）；192 机架经 Cluster Switch 组成 **3,072 GPU 的 pod**（全等分带宽）；8 个 pod 经 Aggregation Switch 组成 **24K GPU 集群**（聚合层过订阅比 1:7）。**Llama 3 预训练仅用其中最多 16K GPU。**
  - **负载均衡**：集合通信库在两 GPU 间建 16 条网络流；Enhanced-ECMP（E-ECMP）协议基于 RoCE 头额外字段哈希均衡。
  - **拥塞控制**：脊层用深缓冲交换机；成功运行 24K GPU 集群而无需传统 DCQCN。

#### 3.3.2 模型规模化的并行策略（Parallelism for Model Scaling）

采用 **4D 并行（4D parallelism）**分片模型，组合四种并行方式：

- **张量并行（Tensor Parallelism, TP）**：将单个权重张量切成多块分布到不同设备。
- **流水线并行（Pipeline Parallelism, PP）**：按层将模型纵向切成阶段（stages）。
- **上下文并行（Context Parallelism, CP）**：沿序列维度切分输入上下文，降低超长序列的内存瓶颈。
- **数据并行（Data Parallelism, DP）**：采用**完全分片数据并行（Fully Sharded Data Parallelism, FSDP）**，分片模型、优化器状态、梯度。

##### 图 5：4D 并行示意（原文第 11 页）

> GPU 按 **[TP, CP, PP, DP]** 顺序划分并行组（DP 即 FSDP）。示例中 16 块 GPU 配置为 |TP|=2, |CP|=2, |PP|=2, |DP|=2。GPU 在 4D 并行中的位置表示为向量 [D₁, D₂, D₃, D₄]。

![图 5：4D 并行示意](/assets/posts/llama3/fig05_4d_parallelism.png)

**GPU 利用率**：BF16 模型 FLOPs 利用率（Model FLOPs Utilization, MFU）达 **38–43%**。

#### 表 4：Llama 3 405B 预训练各阶段的规模化配置与 MFU（原文第 10 页）

| GPUs | TP | CP | PP | DP | 序列长度 | 每 DP 批大小 | 每批 tokens | TFLOPs/GPU | BF16 MFU |
|---|---|---|---|---|---|---|---|---|---|
| 8,192 | 8 | 1 | 16 | 64 | 8,192 | 32 | 16M | 430 | 43% |
| 16,384 | 8 | 1 | 16 | 128 | 8,192 | 16 | 16M | 400 | 41% |
| 16,384 | 8 | 16 | 16 | 8 | 131,072 | 16 | 16M | 380 | 38% |

**流水线并行改进**：解决批大小约束、内存不均衡、计算不均衡问题，使 N（连续微批数）可灵活设置；从首尾阶段各减一个 Transformer 层以平衡流水线；采用交错调度（interleaved schedule）减少流水线气泡（bubble）。流水线气泡比为 $\frac{PP-1}{V \* M}$。通过这些优化可在 8K tokens 序列上无需激活检查点（activation checkpointing）预训练。

##### 图 6：Llama 3 流水线并行示意（原文第 11–12 页）

> 将 8 个流水线阶段（0–7）分布到 4 个流水线 rank（PP rank 0–3），rank 0 的 GPU 运行阶段 0 和 4，依此类推。彩色块（0–9）代表微批序列，M 为总微批数，N 为同阶段连续微批数——关键洞见是让 N 可调。

**长序列上下文并行**：将输入序列切为 2×CP 块以更好负载均衡；采用**基于 all-gather 的方法**（先 all-gather 键 K 与值 V 张量，再计算本地查询 Q 的注意力输出）。因 GQA 使 K/V 张量远小于 Q，且注意力计算复杂度 O(S²) 远大于 all-gather 的 O(S)，故 all-gather 开销可忽略。

**数值稳定性（Numerical stability）**：反向传播中用 FP32 梯度累积；FSDP 中跨 DP worker 用 FP32 reduce-scatter 梯度。

#### 3.3.3 集合通信（Collective Communication）

集合通信库 **NCCLX**（基于 Nvidia NCCL 的分支），显著改善高延迟网络下的性能。

#### 3.3.4 可靠性与运维挑战（Reliability and Operational Challenges）

- 实现 **>90% 有效训练时间（effective training time）**。
- 54 天预训练快照期间共 **466 次任务中断**：47 次计划中断，419 次意外中断。约 **78%** 的意外中断归因于确认或疑似的硬件问题。**GPU 问题**是最大类别（约 58.7%）。仅 3 次需要重大人工干预。
- 观察到 405B 训练存在**昼夜 1–2% 吞吐波动**（中午高温影响 GPU 动态电压/频率调节）；数万 GPU 同时增减功耗可导致数据中心数十兆瓦级瞬时功率波动。

#### 表 5：Llama 3 405B 预训练 54 天期间意外中断的根因分类（原文第 13 页）

| 组件 | 类别 | 中断次数 | 占比 |
|---|---|---|---|
| 故障 GPU（Faulty GPU） | GPU | 148 | 30.1% |
| GPU HBM3 内存 | GPU | 72 | 17.2% |
| 软件 Bug | 依赖 | 54 | 12.9% |
| 网络交换机/线缆 | 网络 | 35 | 8.4% |
| 主机维护 | 计划外维护 | 32 | 7.6% |
| GPU SRAM 内存 | GPU | 19 | 4.5% |
| GPU 系统处理器 | GPU | 17 | 4.1% |
| ...（余略，含 NIC、静默数据损坏、SSD、电源等） | | | |

> 约 **78%** 的意外中断归因于确认或疑似硬件问题。

### 3.4 训练配方（Training Recipe）

405B 预训练配方三阶段：**(1) 初始预训练；(2) 长上下文预训练；(3) 退火**。

#### 3.4.1 初始预训练（Initial Pre-Training）

- 优化器 **AdamW**，峰值学习率 **8×10⁻⁵**，线性预热（warmup）8,000 步，余弦学习率调度衰减到 **8×10⁻⁷**（历时 1,200,000 步）。
- **批大小递增策略**：初始 4M tokens（序列长 4,096）→ 预训练 252M tokens 后翻倍到 8M（序列长 8,192）→ 预训练 2.87T tokens 后再翻倍到 **16M**。训练非常稳定，几乎无损失尖峰（loss spikes）。
- **调整数据配比**：训练中增加非英语数据比例、上采样数学数据、后期加入更新的网络数据推进知识截止点、下采样后识别的低质量数据。

#### 3.4.2 长上下文预训练（Long Context Pre-Training）

- 在预训练最后阶段将上下文从 8K 扩展到 **128K tokens**（分**六个阶段**逐步增加）。
- 成功适配的判据：(1) 短上下文评估性能完全恢复；(2) 完美解决"大海捞针（needle in a haystack）"任务。
- 长上下文预训练阶段用约 **800B 训练 tokens**。

#### 3.4.3 退火（Annealing）

- 在最后 **40M tokens** 上将学习率线性退火到 0，保持 128K 上下文长度。
- 上采样极高质量数据源；对退火期间的模型检查点做**平均（Polyak averaging）**得到最终预训练模型。

---

## 4. 后训练（Post-Training）

每轮后训练 = **SFT（监督微调）→ DPO（直接偏好优化）**，共进行**六轮（six rounds）**。

### 4.1 建模（Modeling）

后训练骨架 = **奖励模型（Reward Model, RM）+ 语言模型**。流程如 **图 7** 所示。

#### 图 7：Llama 3 整体后训练方法示意（原文第 15 页）

> 后训练策略涉及拒绝采样、监督微调、直接偏好优化。收集提示 → 每个提示 K 次生成 → 拒绝采样（用奖励模型选最佳）→ SFT 数据 → SFT 模型 → DPO 训练 → 最终 DPO 模型。前几轮的最佳模型反馈到下一轮。

![图 7：Llama 3 后训练方法示意](/assets/posts/llama3/fig07_posttraining.png)

- **4.1.1 对话格式（Chat Dialog Format）**：设计新的多消息聊天协议，用特殊 header/termination tokens 标识消息来源和目的（支持工具使用等多消息场景）。
- **4.1.2 奖励建模（Reward Modeling）**：训练目标与 Llama 2 相同但移除边际项（margin term）。除标准偏好对（chosen, rejected），还引入第三个"编辑后回复（edited response）"，形成清晰排序 **edited > chosen > rejected**。
- **4.1.3 监督微调（SFT）**：用 RM 对人工标注提示做拒绝采样，配合合成数据，用标准交叉熵损失微调（掩码提示 token 上的损失）。最大模型学习率 **10⁻⁵**，训练 **8.5K 到 9K 步**。
- **4.1.4 直接偏好优化（DPO）**：学习率 **10⁻⁵**，β = **0.1**。两个算法改动：**(1) 在 DPO 损失中掩码格式 token**（header/termination）稳定训练；**(2) 在 chosen 序列上加系数为 0.2 的 NLL 正则项**。发现 DPO 比 PPO 计算量更少、性能更好（尤其在 IFEval 等指令遵循基准上）。
- **4.1.5 模型平均（Model Averaging）**：对各 RM/SFT/DPO 阶段用不同数据/超参得到的模型做平均。

#### 表 6：人类偏好数据统计（原文第 17 页）

| 数据集 | 对比占比 | 平均对话轮数 | 平均 token 数/样本 | 提示平均 token | 回复平均 token |
|---|---|---|---|---|---|
| General English（通用英语） | 81.99% | 4.1 | 1,000.4 | 36.4 | 271.2 |
| Coding（编码） | 6.93% | 3.2 | 1,621.0 | 113.8 | 462.9 |
| Multilingual（多语言） | 5.19% | 1.8 | 1,299.4 | 77.1 | 420.9 |
| Reasoning and tools（推理与工具） | 5.89% | 1.6 | 707.7 | 46.6 | 129.9 |
| **总计** | 100% | 3.8 | 1,041.6 | 44.5 | 284.0 |

### 4.2 后训练数据（Post-training Data）

- **4.2.1 偏好数据（Preference Data）**：每个提示从两个不同模型采样两个回复；标注四级偏好强度（significantly better / better / slightly better / marginally better）；加入编辑步骤形成三级排序。
- **4.2.2 SFT 数据**：来源为拒绝采样回复 + 合成数据 + 少量人工整理数据。**拒绝采样（RS）**：每个提示采样 K 个输出（通常 10–30），用 RM 选最佳。采用 **PagedAttention** 提升拒绝采样吞吐 **2× 以上**。
- **4.2.3 数据处理与质量控制**：规则清洗（如去除过度道歉语气"I'm sorry"）；数据剪枝（主题分类、质量打分、难度打分 Instag、语义去重 semantic deduplication）。

#### 表 7：SFT 数据统计（原文第 18 页）

| 数据集 | 样本占比 | 平均轮数 | 平均 token 数 | 上下文平均 token | 最终回复平均 token |
|---|---|---|---|---|---|
| General English | 52.66% | 6.3 | 974.0 | 656.7 | 317.1 |
| Code | 14.89% | 2.7 | 753.3 | 378.8 | 374.5 |
| Multilingual | 3.01% | 2.7 | 520.5 | 230.8 | 289.7 |
| Exam-like（考试类） | 8.14% | 2.3 | 297.8 | 124.4 | 173.4 |
| Reasoning and tools | 21.19% | 3.1 | 661.6 | 359.8 | 301.9 |
| Long context（长上下文） | 0.11% | 6.7 | 38,135.6 | 37,395.2 | 740.5 |
| **总计** | 100% | 4.7 | 846.1 | 535.7 | 310.4 |

### 4.3 能力（Capabilities）

#### 4.3.1 代码（Code）

- 高优先编程语言：Python, Java, Javascript, C/C++, Typescript, Rust, PHP, HTML/CSS, SQL, bash/shell。
- **代码专家（code expert）**：分支主预训练，在 **1T token**（>85% 代码）上继续预训练，末段做 16K 上下文长上下文微调（LCFT）。
- **合成数据生成**（共生成超 **270 万条**合成样本）：
  1. **执行反馈（execution feedback）**：约 100 万条合成编码对话——问题生成 → 解答生成 → 正确性分析（静态分析 + 单元测试执行）→ 错误反馈与迭代自纠正。约 **20% 初始错误的解答能自纠正**。
  2. **编程语言翻译**：将常见语言（Python/C++）代码翻译到冷门语言（Typescript/PHP），见 **图 8**。
  3. **回译（backtranslation）**：约 120 万条关于代码解释、生成、文档、调试的合成对话（生成 → 回译 → 过滤）。

##### 图 8 & 图 9：代码翻译示例 与 系统提示改善代码质量（原文第 20–21 页）

> **图 8**：用 Llama 3 将 Python 代码（左）翻译为 PHP 代码（右），扩充 SFT 数据集的编程语言范围。
> **图 9**：系统提示（system prompt）改善生成代码质量——左：无系统提示；右：有系统提示（增加必要注释、更有意义的变量名、节省内存等）。

- 用 **"模型即裁判（model-as-judge）"**方法（早期 Llama 3 按代码正确性和风格给 0/1 分，仅保留满分 2 的样本）过滤训练数据。

#### 4.3.2 多语言（Multilinguality）

- 支持德语、法语、意大利语、葡萄牙语、印地语、西班牙语、泰语。
- **多语言专家**：分支预训练，在 **90% 多语言 token** 数据上继续预训练。
- 多语言 SFT 数据分布：**2.4% 人工标注 + 44.2% 其他 NLP 任务数据 + 18.8% 拒绝采样数据 + 34.6% 翻译推理数据**。
- 避免使用机器翻译数据以防"翻译腔（translationese）"、姓名/性别/文化偏见；唯一例外是翻译合成的定量推理数据（在 MGSM 上有强提升）。

#### 4.3.3 数学与推理（Math and Reasoning）

挑战：提示缺乏、缺乏真值思维链（chain of thought）、中间步骤错误、教模型用外部工具、训练推理不一致。

方法：从数学语料构造问答对；用 Llama 3 生成逐步推理迹（reasoning traces）并按正确答案过滤；训练结果和逐步奖励模型（stepwise reward models）过滤错误中间步骤；对更难的提示用**蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS）**+ 逐步奖励模型；交错代码与文本推理；从错误中学习（错误纠正）。

#### 4.3.4 长上下文（Long Context）

- SFT 中主要依赖合成数据：长文档问答、层次化摘要（hierarchical summarization）、长上下文代码推理（识别 Python 文件依赖）。
- 按序列长度（16K/32K/64K/128K）分类。**混合 0.1% 合成长上下文数据**与原短上下文数据可同时优化短、长上下文基准表现。
- **DPO**：仅用短上下文数据不损害长上下文性能（前提是 SFT 模型在长上下文任务上质量高）。

#### 4.3.5 工具使用（Tool Use）

Llama 3 训练使用以下工具：

- **搜索引擎**：Brave Search（回答超出知识截止的近期事件）。
- **Python 解释器**：生成执行代码做复杂计算、读取上传文件。
- **数学计算引擎**：Wolfram Alpha API。

支持多轮对话中的多次工具调用、逐步规划；支持**零样本（zero-shot）工具使用**（函数调用 function calling）。数据收集依赖人工标注和偏好（在消息级标注）。工具数据集：单步工具使用、多步工具使用（类似 ReAct，见 **图 10**）、文件上传（见 **图 11**）。零样本工具使用支持单一、嵌套、并行函数调用及多轮函数调用。

##### 图 10 & 图 11：多步工具使用 与 处理文件上传（原文第 25–27 页）

> **图 10**：Llama 3 执行多步规划、推理和工具调用完成任务的示例。
> **图 11**：Llama 3 对上传文件进行分析和可视化的示例。

#### 4.3.6 事实性（Factuality）

采取"幻觉优先（hallucination-first）"方法，原则是让模型"知道自己知道什么（know what it knows）"而非增加知识。**知识探测（knowledge probing）**流程：提取预训练数据片段 → 生成事实性问题 → 采样回复 → 用 Llama 3 作裁判评正确性和信息量 → 对一贯有信息量但错误的回复生成拒答（refusal）。

#### 4.3.7 可引导性（Steerability）

通过系统提示的自然语言指令增强可引导性（回复长度、格式、语气、角色/人设）。论文给出了一个"家庭膳食计划助手"的定制系统提示示例。

---

## 5. 结果（Results）

评估三方面：(1) 预训练语言模型；(2) 后训练语言模型；(3) 安全特性。

### 5.1 预训练语言模型（Pre-trained Language Model）

#### 5.1.1 标准基准（Standard Benchmarks）

八大类：常识推理、知识、阅读理解、数学推理与问题求解、长上下文、代码、对抗评估、聚合评估。用 **95% 置信区间（Confidence Intervals, CIs）**报告方差：

$$
CI(S) = 1.96 \times \sqrt{\frac{S \times (1 - S)}{N}}
$$

##### 图 12：预训练 Llama 3 8B 和 70B 在预训练基准上的表现（原文第 29 页）

> 按能力类别聚合准确率。Llama 3 8B 几乎在每个类别都超越同类竞品；70B 大幅超越前代 Llama 2 70B，并超越 Mixtral 8×22B。

**代表性预训练基准结果**（含 95% 置信区间，节选自表 9–13）：

| 任务 | Llama 3 8B | Llama 3 70B | Llama 3 405B | GPT-4 |
|---|---|---|---|---|
| HumanEval（代码） | 37.2 | 58.5 | 61.0 | 67.0 |
| GSM8K（数学） | 57.2 | 83.7 | 89.0 | 92.0 |
| MATH | 20.3 | 41.4 | 53.8 | — |
| ARC-C（推理） | 79.7 | 92.9 | 96.1 | 96.3 |
| MMLU（通用） | 66.7 | 79.3 | 85.2 | 86.4 |
| MMLU-Pro | 37.1 | 53.8 | 61.6 | — |

#### 5.1.2 模型鲁棒性（Model Robustness）

用 MMLU 评估对多选题（Multiple-Choice Question, MCQ）设置的鲁棒性：**少样本标签偏差、标签变体、答案顺序、提示格式**。结果显示预训练模型对这些设计选择非常鲁棒，**405B 尤为突出**（见 **图 13、图 14**）。

##### 图 13 & 图 14：MMLU 基准不同设计选择的鲁棒性（原文第 32–33 页）

> **图 13**：左——不同标签变体的性能；右——少样本示例中不同标签的性能。
> **图 14**：左——不同答案顺序的性能；右——不同提示格式的性能。

#### 5.1.3 对抗基准（Adversarial Benchmarks）

##### 图 15：对抗与非对抗性能对比（原文第 33 页）

> 问答、数学推理、复述检测三领域，左：预训练模型；右：后训练模型。在复述检测（PAWS）上模型不受对抗性影响；但在数学推理和问答上，对抗性能显著低于非对抗性能。

#### 5.1.4 污染分析（Contamination Analysis）

基于 **8-gram 重叠**判断评估数据在预训练语料中的污染程度。PiQA 和 HellaSwag 污染和性能增益估计都高；Natural Questions 估计 52% 污染但对性能几乎无影响（见表 15）。

#### 表 14：预训练模型长上下文任务表现（原文第 33 页）

| 任务 | 8B | 70B | 405B |
|---|---|---|---|
| QuALITY (5-shot) | 56.0 | 82.8 | 87.6 |
| GSM8K (16-shot) | 60.0 | 83.0 | 90.0 |

### 5.2 后训练语言模型（Post-trained Language Model）

- **通用知识**：405B 在 MMLU、MMLU-Pro 上超越 GPT-4 和 Nemotron 4 340B，Claude 3.5 Sonnet 在更大模型中领先。
- **指令遵循**：IFEval（约 500 条可验证指令）。
- **代码**：HumanEval、MBPP EvalPlus、MultiPL-E（非 Python 语言，见表 19）。
- **数学与推理、多语言（MGSM）、工具使用（BFCL/Nexus/API-Bank）、长上下文（ZeroSCROLLS/InfiniteBench/大海捞针）**。
- 熟练度考试（表 17，含 LSAT/SAT/GMAT/AP）。

#### 图 16：Llama 3 405B vs GPT-4o 代码执行任务人类评估（原文第 40 页）

> 涵盖绘图（plotting）和文件上传的代码执行任务。Llama 3 405B 在代码执行（无绘图/文件上传）和绘图生成上超越 GPT-4o，但在文件上传用例上落后。

![图 16：Llama 3 405B vs GPT-4o 代码执行人类评估](/assets/posts/llama3/fig16_humaneval_code.png)

### 5.3 人类评估（Human Evaluations）

#### 图 17：Llama 3 405B 人类评估结果（原文第 40 页）

> 采用 7 点量表进行成对人类评估。左：与 GPT-4 对比；中：与 GPT-4o 对比；右：与 Claude 3.5 Sonnet 对比。405B 与 GPT-4（0125 API 版）大致持平；在多轮推理和编码上超越 GPT-4，但在多语言上落后于 GPT-4；在编码和推理上落后于 Claude 3.5 Sonnet。

### 5.4 安全（Safety）

优化两大指标：**违规率（Violation Rate, VR）**与**误拒率（False Refusal Rate, FRR）**。

- **5.4.1 基准构建**：基于 ML Commons 危害分类，每个能力/语言超 **4000 条提示**（对抗性 + 边界性 borderline）。
- **5.4.2 安全预训练**：过滤 PII，关注**可发现记忆（discoverable memorization）**。405B 逐字记忆率低（n=50 时 1.13%，n=1000 时 3.91%），与同规模 Llama 2 大致相当（表 24）。
- **5.4.3 安全微调**：质量比数量更重要；引入边界数据集降低 FRR；安全 SFT + 安全 DPO。**图 18** 显示小模型需要更高比例的安全数据。
- **5.4.4 安全结果**：**图 19（多语言短上下文）、图 20（工具使用与长上下文）、图 21（各模型各能力的 VR-FRR 权衡）**。Llama 3 平衡了低违规率和低误拒率。长上下文能有效抵御 **256-shot 多样本越狱（many-shot jailbreaking）**攻击。
- **5.4.5 网络安全与化生武器安全**：
  - 代码解释器滥用：405B 有 **10.4%** 概率遵从恶意提示，70B 为 **3.8%**。
  - 文本提示注入（prompt injection）：405B 攻击成功率 **21.7%**（图 22）。
  - 鱼叉式网络钓鱼（spear phishing）：70B 成功率 **24%**，405B 为 **14%**（图 23）。
  - 网络攻击提升测试（62 名志愿者，31 专家 + 31 新手）与化生武器提升测试：**均无显著提升**——评估 Llama 3 发布不会显著增加相关生态风险。
- **5.4.6 红队（Red Teaming）**：跨能力发现风险（多轮拒绝抑制、假设场景、角色扮演、逐步升级违规、多语言混合、不安全工具链等）。含儿童安全风险评估。
- **5.4.7 系统级安全**：
  - **Llama Guard 3**：基于 Llama 3 8B 微调的安全分类器，训练于 **13 类危害** + 代码解释器滥用类别。平均减少 **65% 违规**；提供 int8 量化版（体积减小 40% 以上，见表 27）。
  - **Prompt Guard**：基于 mDeBERTa-v3-base（86M 参数）的多标签分类器，检测直接越狱和间接提示注入（表 28）。
  - **Code Shield**：基于静态分析（Insecure Code Detector, ICD，覆盖 7 种语言）的推理时过滤。

##### 图 18–23：安全相关图表（原文第 43–46 页）

> **图 18**：模型规模对安全配比设计（VR vs FRR）的影响。
> **图 19**：英语及核心多语言短上下文基准的违规率与误拒率（Llama 3 405B 有/无 Llama Guard vs 竞品）。
> **图 20**：工具使用和长上下文基准的违规率与误拒率。
> **图 21**：各模型各能力的整体误拒率与违规率散点。
> **图 22**：各模型各提示注入策略的文本提示注入成功率。
> **图 23**：各鱼叉钓鱼模型和目标的平均说服力得分（Llama 3 70B 裁判评估）。

#### 表 25：使用 Llama Guard 3 后相对 Llama 3 的 VR 与 FRR 变化（各语言，原文第 50 页）

| 能力/语言 | Full Llama Guard VR | Full Llama Guard FRR |
|---|---|---|
| English | -86% | +102% |
| French | -59% | +29% |
| German | -77% | +37% |
| Hindi | -71% | +62% |
| Italian | -48% | +29% |
| Portuguese | -65% | +39% |
| Spanish | -60% | +27% |
| Thai | -51% | +39% |

> -X% 的 VR 表示违规率下降 X%；添加系统防护会增加对良性提示的误拒。

---

## 6. 推理（Inference）

两大高效推理技术：**(1) 流水线并行；(2) FP8 量化（quantization）**（FP8 量化实现已公开）。

### 6.1 流水线并行（Pipeline Parallelism）

- BF16 表示下 405B 无法装入单机 8 块 H100 的显存，故用 **BF16 精度跨两台机器（16 GPU）**并行：机内用张量并行（高 NVLink 带宽），跨节点用流水线并行。
- 推理时无反向传播故无流水线气泡问题，用**微批处理（micro-batching）**提升吞吐（4,096 输入 token + 256 输出 token，见 **图 24**）。

##### 图 24：微批处理对推理吞吐和延迟的影响（原文第 52 页）

> 左：预填充（pre-filling）阶段；右：解码（decoding）阶段。图中数字对应（微）批大小。微批处理带来更好的吞吐-延迟权衡。

### 6.2 FP8 量化（FP8 Quantization）

利用 H100 原生 FP8 支持做低精度推理，对**前馈网络层（约占推理计算的 50%）**的大部分矩阵乘法做 FP8 量化，**不量化自注意力层参数**。质量提升三改动：

1. **不量化第一和最后一个 Transformer 层**。
2. 将动态缩放因子（dynamic scaling factors）上界设为 **1200**（避免高困惑度 token 如日期导致的下溢错误）。
3. 用**行级量化（row-wise quantization）**（对参数和激活矩阵按行计算缩放因子），优于张量级量化（tensor-wise），见 **图 25**。

##### 图 25：张量级与行级 FP8 量化示意（原文第 53 页）

> 右：行级量化可用比左侧张量级量化更细粒度的激活因子。

![图 25：张量级与行级 FP8 量化](/assets/posts/llama3/fig25_fp8_quant.png)

##### 图 26：Llama 3 405B 用 BF16 和 FP8 推理的奖励分布（原文第 53 页）

> 对 100,000 条回复分析奖励模型分数分布。FP8 量化方法对模型回复影响极小（negligible）。

##### 图 27：FP8 推理的吞吐-延迟权衡（原文第 54 页）

> 与 BF16 推理对比（不同流水线并行设置）。左：预填充；右：解码。FP8 推理在预填充阶段带来最高 **50%** 的吞吐提升，并在解码阶段带来显著更优的吞吐-延迟权衡。

---

## 7. 视觉实验（Vision Experiments）

通过**组合式方法**为 Llama 3 加入视觉识别能力，两大阶段：(1) 组合预训练图像编码器与语言模型，引入并训练交叉注意力层；(2) 引入时间聚合层（temporal aggregator）和视频交叉注意力层。整体多模态架构见 **图 28**。

### 图 28：为 Llama 3 添加多模态能力的组合式方法示意（原文第 55 页）

> 五个训练阶段：**(1) 语言模型预训练；(2) 多模态编码器预训练；(3) 视觉适配器训练；(4) 模型微调；(5) 语音适配器训练**。图中展示图像编码器（Image Encoder）、视频聚合器（Video Aggregator）、语言模型（Language Model，含交叉注意力）和语音编码器（Speech Encoder，Conformer block + Speech adapter）。

![图 28：多模态组合式方法架构](/assets/posts/llama3/fig28_multimodal.png)

**组合式方法的优势**：(1) 并行开发视觉与语言能力；(2) 规避联合预训练的复杂性；(3) 保证纯文本任务性能不受影响；(4) 交叉注意力架构无需将全分辨率图像传过整个 LLM 骨干，推理更高效。

### 7.1 数据（Data）

- **图像数据**：图文对，四阶段流水线——质量过滤（CLIP score）、感知去重（SSCD 拷贝检测模型，512 维表示 + FAISS 近邻搜索）、重采样（n-gram 平衡低频类别）、光学字符识别（Optical Character Recognition, OCR）。安全上用 PhotoDNA 等扫描 CSAM，做人脸模糊（face blurring）。退火数据集约 **3.5 亿**样本 + **1.5 亿**额外样本（视觉定位、截图解析、问答对、合成描述、合成结构化图像）。
- **视频数据**：视频文本对，多阶段过滤（语言识别、OCR、CLIP 风格对齐、运动分数过滤）。平均时长 **21 秒**，中位数 **16 秒**，99% 短于一分钟；70% 以上短边 >720 像素。

### 7.2 模型架构（Model Architecture）

- **图像编码器（Image encoder）**：标准视觉 Transformer（Vision Transformer, ViT）**ViT-H/14** 变体，**630M 参数**，在 **25 亿图文对**上训练 5 轮。224×224 分辨率，16×16 patch（每 patch 14×14 像素）。多层特征提取（第 4/8/16/24/31 层）+ 8 个门控自注意力层（共 40 个 Transformer block），最终 **850M 参数**，每个 patch 产生 **7680 维**表示。
- **图像适配器（Image adapter）**：在图像编码器视觉 token 与语言模型 token 之间引入交叉注意力层（**每四个自注意力层后应用一次**），用 GQA。405B 的交叉注意力层约 **100B 参数**。分初始预训练（约 60 亿图文对）和退火（约 5 亿图像）两阶段。
- **视频适配器（Video adapter）**：最多输入 64 帧；时间聚合器（temporal aggregator，实现为 perceiver resampler）将 32 连续帧合并为一帧；每四个图像交叉注意力层前加视频交叉注意力层。8B/70B 的视频聚合器和交叉注意力层分别为 **0.6B/4.6B 参数**。

### 7.3 模型规模化（Model Scaling）

新增三挑战：**模型异构性（model heterogeneity）、数据异构性（data heterogeneity）、数值不稳定性（numerical instabilities）**。

- 模型异构性：每个流水线阶段含 5 层（4 个自注意力 + 1 个交叉注意力），在所有阶段复制图像编码器做负载均衡。
- 数据异构性：图像平均 **2,308 token**，关联文本仅平均 **192 token**；引入图像编码器序列并行化；用更大微批（8 而非 1）。
- 数值不稳定性：图像 token 经所有交叉注意力层引入语言骨干，误差累积——用 **FP32 梯度累积**解决。

### 7.4–7.5 预训练与后训练

- **图像预训练**：从预训练文本模型和视觉编码器权重初始化，视觉编码器解冻、文本模型冻结。全局批大小 16,384，余弦学习率，初始 10×10⁻⁴。
- **视频预训练**：从图像预训练权重开始，仅训练视频专用参数；16 帧、每帧四块 448×448，聚合因子 16，全局批大小 4,096。
- **后训练**：SFT（学术数据集 + 人工标注 + 合成数据）→ DPO（EMA 更新参考模型）→ 拒绝采样 → **质量微调（Quality-Tuning, QT）**。视觉奖励模型（vision RM）冻结自注意力层（来自语言 RM），解冻视觉编码器和交叉注意力层。

### 7.6 图像识别结果（Image Recognition Results）

#### 表 29：视觉模块图像理解表现（原文第 61 页）

| 基准 | Llama 3-V 8B | Llama 3-V 70B | Llama 3-V 405B | GPT-4V | GPT-4o | Gemini 1.5 Pro | Claude 3.5 |
|---|---|---|---|---|---|---|---|
| MMMU (val, CoT) | 49.6 | 60.6 | 64.5 | 56.4 | 69.1 | 62.2 | 68.3 |
| VQAv2 (test-dev) | 78.0 | 79.1 | 80.2 | 77.2 | — | 80.2 | — |
| AI2 Diagram (test) | 84.4 | 93.0 | 94.1 | 78.2 | 94.2 | 94.4 | 94.7 |
| ChartQA (test, CoT) | 78.7 | 83.2 | 85.8 | 78.4 | 85.7 | 87.2 | 90.8 |
| TextVQA (val) | 78.2 | 83.4 | 84.8 | 78.0 | — | 78.7 | — |
| DocVQA (test) | 84.4 | 92.2 | 92.6 | 88.4 | 92.8 | 93.1 | 95.2 |

> Llama 3-V 405B 在所有基准上超越 GPT-4V，略逊于 Gemini 1.5 Pro 和 Claude 3.5 Sonnet；在**文档理解**任务上尤为有竞争力。

### 7.7 视频识别结果（Video Recognition Results）

#### 表 30：视频理解表现（零样本，原文第 62 页）

| 基准 | Llama 3-V 8B | Llama 3-V 70B | Gemini 1.0 Pro | Gemini 1.0 Ultra | Gemini 1.5 Pro | GPT-4V | GPT-4o |
|---|---|---|---|---|---|---|---|
| PerceptionTest (test) | 53.8 | 60.8 | 51.1 | 54.7 | — | — | — |
| TVQA (val) | 82.5 | 87.9 | — | — | — | 87.3 | — |
| NExT-QA (test) | 27.3 | 30.3 | 28.0 | 29.9 | — | — | — |
| ActivityNet-QA (test) | 52.7 | 56.3 | 49.8 | 52.2 | 57.5 | — | 61.9 |

> 所有结果均为**零样本**。Llama 3（仅评估 8B/70B）在视频识别上很有竞争力，在 PerceptionTest 上表现最佳（强时间推理能力）；在 ActivityNet-QA 长视频上即使每 3 秒仅处理一帧也有强结果。

---

## 8. 语音实验（Speech Experiments）

采用类似视觉的组合式方法。输入侧加入编码器 + 适配器处理语音信号；通过文本系统提示启用不同模式。若无系统提示，模型作为通用口语对话模型。也支持**自动语音识别（Automatic Speech Recognition, ASR）**和**自动语音翻译（Automatic Speech Translation, AST）**。**语音接口支持最多 34 种语言**。语音生成用流式**文本转语音（Text-to-Speech, TTS）**系统（不微调语言模型，仅在推理时利用 Llama 3 嵌入）。

### 图 29：Llama 3 语音接口架构（原文第 63 页）

> 展示 Llama 3 语音接口的整体架构（语音编码器 → 适配器 → 语言模型，以及语音生成的 TTS 组件）。

### 8.1 数据

- **语音理解预训练数据**：约 **1500 万小时（15M hours）**语音录音（多语言），VAD 阈值 >0.7 筛选，去 PII（Presidio Analyzer）。
- **ASR 训练数据**：**23 万小时（230K hours）**人工转录语音，覆盖 34 种语言。
- **AST 训练数据**：**9 万小时（90K hours）**双向翻译（33 语言↔英语），含 NLLB 工具生成的合成数据。语音片段最长 60 秒。
- **口语对话数据**：合成生成（60K 小时 ASR 子集 + 25K 小时 Voicebox TTS 合成）。
- **语音生成数据**：文本归一化（Text Normalization, TN）55K 样本；韵律模型（Prosody Model, PM）50K 小时 TTS 数据。**Llama 3 嵌入取自第 16 层解码器输出**（仅用 8B 模型）。

### 8.2 模型架构

- **语音编码器（Speech encoder）**：**Conformer** 模型，**1B 参数**。输入 80 维梅尔频谱（mel-spectrogram），stride-4 堆叠层降帧长到 40ms；24 个 Conformer 层，潜维 1536，两个 Macaron 式前馈网络（维度 4096），卷积核 7，旋转注意力 24 头。
- **语音适配器（Speech adapter）**：约 **100M 参数**（卷积层 + 旋转 Transformer 层 + 线性层），将语音帧长降到 80ms。与视觉模块不同，语音模块直接生成可与文本 token 无缝集成的嵌入（而非用交叉注意力）。
- **语音生成**：TN 模块（流式 LSTM 序列标注）+ PM 模块（decoder-only Transformer，6 注意力头，双交叉注意力——一层给语言输入、一层给 Llama 嵌入）。PM 预测三个韵律特征：每音素对数时长、对数 F0 均值、对数功率均值。

### 8.3 训练配方

- **语音预训练**：用自监督 **BEST-RQ** 算法；编码器训练 **500K 步**，全局批 2,048 utterances。
- **监督微调**：预训练语音编码器 + 随机初始化适配器与 Llama 3 联合优化（**语言模型保持冻结**）。8B 语音模型训练 650K 步（批 512，学习率 10⁻⁴）；70B 训练 600K 步（批 768，学习率 4×10⁻⁵）。设计的 ASR 系统提示：`Repeat after me in {language}:`；AST 系统提示：`Translate the following sentence into {language}:`。
- **语音生成训练**：韵律模型用前瞻（lookahead）机制支持流式；批 1,024 utterances（最长 500 音素），学习率 9×10⁻⁴（AdamW），训练 100 万步。

### 8.4 语音理解结果

#### 表 31：语音识别词错误率（Word Error Rate, WER，越低越好，原文第 67 页）

| 基准 | Llama 3 8B | Llama 3 70B | Whisper | SeamlessM4T v2 | Gemini 1.0 Ultra | Gemini 1.5 Pro |
|---|---|---|---|---|---|---|
| MLS (English) | 4.9 | 4.4 | 6.2 (v2) | 6.5 | 4.4 | 4.2 |
| LibriSpeech (test-other) | 3.4 | 3.1 | 4.9 (v2) | 6.2 | — | — |
| VoxPopuli (English) | 6.2 | 5.7 | 7.0 (v2) | 7.0 | — | — |
| FLEURS (34 languages) | 9.6 | 8.2 | 14.4 (v3) | 11.7 | — | — |

> Llama 3 在所有基准上超越专门的语音模型 Whisper 和 SeamlessM4T；MLS 英语上与 Gemini 相当。

#### 表 32：语音翻译 BLEU 分数（越高越好，原文第 67 页）

| 基准 | Llama 3 8B | Llama 3 70B | Whisper v2 | SeamlessM4T v2 |
|---|---|---|---|---|
| FLEURS (33 lang. → English) | 29.5 | 33.7 | 21.9 | 28.6 |
| Covost 2 (15 lang. → English) | 34.4 | 38.8 | 33.8 | 37.9 |

**口语问答**：可零样本理解语码转换（code-switched）语音；虽仅在单轮对话上训练，却能进行多轮对话（见 **图 30**）。

#### 表 33：MuTox 数据集上的语音毒性（原文第 68 页）

| 语言 | Llama 3 8B AT(↓) / LT(↑) | Llama 3 70B AT / LT | Gemini 1.5 Pro AT / LT |
|---|---|---|---|
| English | 0.84 / 15.09 | 0.68 / 15.46 | 1.44 / 13.42 |
| Overall（21 语言均值） | 2.31 / 9.89 | 2.00 / 10.29 | 2.06 / 10.94 |

> AT = 新增毒性（added toxicity，输入安全但输出有毒），LT = 丢失毒性（lost toxicity，输入有毒但回复安全）。Llama 3 语音模型英语新增毒性最低（<1%）。

##### 图 30：Llama 3 语音接口转录对话示例（原文第 68 页）

> 展示零样本多轮和语码转换能力的对话示例。

### 8.5 语音生成结果

#### 表 34：文本归一化准确率（原文第 68 页）

| 模型 | 上下文 | 准确率 |
|---|---|---|
| 无 Llama 3 8B 嵌入 | 3 | 73.6% |
| 无 Llama 3 8B 嵌入 | ∞（全双向） | 88.0% |
| **有 Llama 3 8B 嵌入** | 3 | **90.7%** |

> 有 Llama 3 嵌入的模型即使只用 3-token 右上下文也超越所有其他模型，实现 token 级流式输入/输出。

#### 表 35：韵律建模（PM）人类偏好评估（原文第 69 页）

| 对比 | PM for Llama 3 8B | 基线 |
|---|---|---|
| vs 流式音素基线 | **60.0%** | 40.0% |
| vs 非流式音素基线 | **63.6%** | 36.4% |

> Llama 3 8B PM 在流式能力下仍显著优于基线，兼顾自然度、表现力和低延迟。

---

## 9. 相关工作（Related Work）

- **语言（Language）**：Llama 3 遵循"简单方法 + 不断增大规模"的趋势（405B 用近 50 倍于 Llama 2 70B 的算力）；参数量少于 PALM 等早期模型（得益于更好的缩放定律理解）；小模型通过远超算力最优的训练换取推理效率。相比 MoE 架构（Mixtral、Arctic），稠密架构非限制因素。开源权重模型（Mistral、Falcon、MPT、Pythia、Qwen、Gemma、Grok、Phi 等）快速进步，Llama3-405B 已与闭源 SOTA 竞争。
- **多模态（Multimodality）**：图像（CLIP、Flamingo 等）、视频（多用适配器方法）、语音（AudioPaLM、VioLA 等）。Llama 3 选择**不为语音任务微调语言模型本身**（避免在非语音任务上产生竞争冲突）。

---

## 10. 结论（Conclusion）

- 高质量基础模型开发仍处早期，Llama 3 经验表明进一步显著改进指日可待。
- **对高质量数据、规模、简洁性的强烈聚焦持续带来最佳结果**；更复杂的架构和训练配方的收益不足以抵消其引入的复杂度。
- **组织决策同样关键**：由独立团队采购处理预训练数据以防基准污染；仅让不参与模型开发的少数研究者执行人类评估以保证可信。
- 公开发布 Llama 3 语言模型以加速社会相关用例的 AI 系统开发，并让研究界审视改进；相信基础模型的公开发布对负责任地发展通用人工智能（Artificial General Intelligence, AGI）至关重要。

---

## 附录：专有名词中英对照（Glossary）

| 中文 | 英文 |
|---|---|
| 基础模型 | Foundation Model |
| 稠密 Transformer | Dense Transformer |
| 混合专家模型 | Mixture-of-Experts (MoE) |
| 上下文窗口 | Context Window |
| 预训练 / 后训练 | Pre-training / Post-training |
| 监督微调 | Supervised Finetuning (SFT) |
| 拒绝采样 | Rejection Sampling (RS) |
| 直接偏好优化 | Direct Preference Optimization (DPO) |
| 奖励模型 | Reward Model (RM) |
| 分组查询注意力 | Grouped Query Attention (GQA) |
| 旋转位置编码 | Rotary Position Embedding (RoPE) |
| 缩放定律 | Scaling Laws |
| 算力最优 | Compute-optimal |
| 负对数似然 | Negative Log-Likelihood (NLL) |
| 张量并行 / 流水线并行 / 上下文并行 / 数据并行 | Tensor / Pipeline / Context / Data Parallelism (TP/PP/CP/DP) |
| 完全分片数据并行 | Fully Sharded Data Parallelism (FSDP) |
| 模型 FLOPs 利用率 | Model FLOPs Utilization (MFU) |
| 退火 | Annealing |
| 违规率 / 误拒率 | Violation Rate (VR) / False Refusal Rate (FRR) |
| 光学字符识别 | Optical Character Recognition (OCR) |
| 视觉 Transformer | Vision Transformer (ViT) |
| 交叉注意力 | Cross-attention |
| 时间聚合器 | Temporal Aggregator |
| 自动语音识别 / 自动语音翻译 | ASR / AST |
| 文本转语音 | Text-to-Speech (TTS) |
| 文本归一化 / 韵律模型 | Text Normalization (TN) / Prosody Model (PM) |
| 词错误率 | Word Error Rate (WER) |
| 大海捞针 | Needle in a Haystack |
| 思维链 | Chain of Thought (CoT) |
| 个人可识别信息 | Personally Identifiable Information (PII) |
| 通用人工智能 | Artificial General Intelligence (AGI) |

---

## 附录：论文链接

- **论文标题**：The Llama 3 Herd of Models
- **arXiv 链接**：<https://arxiv.org/abs/2407.21783>
- **官方网站**：<https://llama.meta.com/>
