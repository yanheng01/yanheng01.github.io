---
title: "ZeRO / DeepSpeed 万亿参数怎么管理显存"
date: 2026-09-11 18:42:00 +0800
permalink: /posts/zero-deepspeed/
categories: [论文精读, 深度学习]
tags: [zero, deepspeed, 显存优化, 数据并行, 大模型, 分布式训练, 论文精读, Infra]
math: true
mermaid: false
toc: true
---

> **原文标题（Original Title）**：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
> **作者（Authors）**：Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He（微软 Microsoft）
> **论文精读笔记 · 中文**

---

## 目录

1. [一句话总结](#一句话总结)
2. [背景：显存都去哪了？](#背景显存都去哪了)
3. [核心思想：消除数据并行中的显存冗余](#核心思想消除数据并行中的显存冗余)
4. [ZeRO-DP：模型状态优化的三个阶段](#zero-dp模型状态优化的三个阶段)
5. [ZeRO-R：残余状态优化](#zero-r残余状态优化)
6. [通信量分析：为什么几乎不增加通信](#通信量分析为什么几乎不增加通信)
7. [迈向万亿参数](#迈向万亿参数)
8. [实验与评估](#实验与评估)
9. [核心贡献与结论](#核心贡献与结论)
10. [个人思考与延伸](#个人思考与延伸)
11. [专有名词中英文对照附录](#专有名词中英文对照附录)
12. [论文来源附录](#论文来源附录)

---

## 一句话总结

**ZeRO（零冗余优化器）** 通过把训练所需的"模型状态"（优化器状态、梯度、参数）在数据并行的多个设备之间**切分（partition）而不是复制（replicate）**，在保持数据并行高计算效率和低通信量的同时，让可训练的模型规模随着设备数量**线性增长**——理论上可支撑**万亿（1 Trillion）参数**模型的训练。作者的实现 **ZeRO-100B** 在 400 张 V100 上训练了超过 1000 亿参数的模型，达到 **15 Petaflops** 的吞吐量，相比业界最优（SOTA）实现了模型规模 **8 倍**、性能 **10 倍** 的提升。

---

## 背景：显存都去哪了？

大模型（BERT、GPT-2、T5 等）的参数量持续暴涨，但单卡显存（如 32GB 的 V100）成为硬约束。作者首先系统性地拆解了显存到底被什么占用。

### 两大类显存占用

论文把训练时的显存分成两大类：

- **模型状态（Model States）**：优化器状态（Optimizer States）、梯度（Gradients）、参数（Parameters）。这是**最大头**。
- **残余状态（Residual States）**：激活值（Activations）、临时缓冲区（Temporary Buffers）、显存碎片（Fragmented Memory）。

### 混合精度训练下的显存账本

在使用 Adam 优化器的 **混合精度训练（Mixed-Precision Training，fp16/fp32）** 下，设参数量为 $\Psi$：

- fp16 参数（Parameters）：$2\Psi$ 字节
- fp16 梯度（Gradients）：$2\Psi$ 字节
- fp32 优化器状态（Adam）：动量（Momentum，$4\Psi$）+ 方差（Variance，$4\Psi$）+ 主权重（Master Weights，$4\Psi$）= $12\Psi$ 字节

因此总显存为：

$$
2\Psi + 2\Psi + 12\Psi = 16\Psi \text{ 字节}
$$

论文中把优化器部分的系数记作 $K = 12$。

> 举例：一个 **15 亿（1.5B）参数** 的模型，仅"模型状态"就需要约 **24GB** 显存——已经把一张卡塞满。这解释了为什么朴素的数据并行在约 **14 亿（1.4B）参数** 时就会 OOM。

### 残余状态的隐患

- **激活值**可能占用几十 GB。
- **临时缓冲区**（如 All-Reduce 用的缓冲）会随模型规模增长而膨胀。
- **显存碎片**：短生命周期张量与长生命周期张量交错分配，导致即使理论上还有空闲显存，也会因为无法找到连续块而 OOM。

---

## 核心思想：消除数据并行中的显存冗余

传统方案各有硬伤：

- **数据并行（Data Parallelism, DP）**：计算粒度高、通信效率好，但**每张卡都完整复制一份模型状态**，显存冗余极其严重。
- **模型并行（Model Parallelism, MP）**：把模型"竖切"分到多卡，显存效率高，但**降低了计算粒度**，且跨节点通信开销巨大——跨节点时效率会跌到峰值算力的 5% 以下。
- **流水线并行（Pipeline Parallelism, PP）**：需要很大的 micro-batch 才能填满流水线、隐藏"气泡（bubble）"，这会影响收敛并占用显存。

**ZeRO 的关键洞察**：在数据并行中，**并非所有模型状态在任何时刻都需要**。因此可以把它们切分开、按需通过通信动态收集，从而在保持 DP 计算/通信优势的同时获得 MP 级别的显存效率。

ZeRO 由两部分组成：

- **ZeRO-DP**：针对**模型状态**，切分优化器状态、梯度、参数。
- **ZeRO-R**：针对**残余状态**，做激活值切分、恒定大小缓冲区、显存去碎片化。

---

## ZeRO-DP：模型状态优化的三个阶段

这是全文最核心的图，展示了三个渐进阶段如何逐步把每卡显存从 120GB 压到 1.88GB（以 7.5B 模型、$N_d=64$、$K=12$ 为例）。

![Figure 1: 三阶段 ZeRO-DP 的每卡模型状态显存对比](/assets/posts/zero-deepspeed/figure1.png)

*图 1（Figure 1）：对比三阶段 ZeRO-DP 下每设备的模型状态显存消耗。以 7.5B 参数模型、数据并行度 $N_d=64$、优化器系数 $K=12$ 为例，显存从基线的 120GB 依次下降。*

设 $N_d$ 为**数据并行度（Data Parallel Degree）**，即参与切分的进程/设备数：

### 阶段一 $P_{os}$：优化器状态切分（Optimizer State Partitioning）

把优化器状态均分到 $N_d$ 个进程，每个进程只维护自己那一份。

$$
\text{显存} = 4\Psi + \frac{K\Psi}{N_d}
$$

当 $N_d$ 很大时，约有 **4 倍** 显存缩减。

### 阶段二 $P_{os}+P_g$：优化器状态 + 梯度切分（Gradient Partitioning）

用 **Reduce-Scatter** 替代 **All-Reduce**：梯度只被规约到"负责更新该参数分区"的那个进程上。

$$
\text{显存} = 2\Psi + \frac{(K+2)\Psi}{N_d} = 2\Psi + \frac{14\Psi}{N_d}
$$

约有 **8 倍** 显存缩减。

### 阶段三 $P_{os}+P_g+P_p$：再加参数切分（Parameter Partitioning）

参数也被切分，在前向/反向传播过程中按需 **广播（Broadcast）** 收集。

$$
\text{显存} = \frac{16\Psi}{N_d}
$$

显存随 $N_d$ **线性下降**，而通信量只增加 **1.5 倍**（详见通信分析）。

### 三阶段显存对比表

| 阶段 | 切分内容 | 每卡显存 | 缩减倍数 |
| --- | --- | --- | --- |
| 基线（Baseline DP） | 无（全复制） | $16\Psi$ | 1× |
| $P_{os}$（ZeRO-1） | 优化器状态 | $4\Psi + \dfrac{12\Psi}{N_d}$ | ~4× |
| $P_{os}+P_g$（ZeRO-2） | + 梯度 | $2\Psi + \dfrac{14\Psi}{N_d}$ | ~8× |
| $P_{os}+P_g+P_p$（ZeRO-3） | + 参数 | $\dfrac{16\Psi}{N_d}$ | $N_d$× |

> **表 1（Table 1）** 进一步给出了 7.5B、128B、1T 三种模型在 $N_d = 1 \sim 1024$ 下的每卡显存（GB）矩阵，从数学上证明了 **$N_d=1024$ 配合 ZeRO-3 可以把万亿参数模型塞进 32GB 显存**。

---

## ZeRO-R：残余状态优化

针对模型状态之外的"残余"显存，ZeRO-R 提供三项技术：

### $P_a$：分区激活检查点（Partitioned Activation Checkpointing）

模型并行会强制**复制激活值**。ZeRO 把激活值跨 MP 的 GPU 切分保存，只在反向传播需要时用 **All-Gather** 重新拼回来。可选地还能把激活值卸载（offload）到 CPU 内存。

### $C_B$：恒定大小缓冲区（Constant Size Buffers）

把临时缓冲区大小**固定为一个常数上限**，避免它随着模型规模线性膨胀，同时保证足够大以维持通信/计算效率。

### $M_D$：显存去碎片化（Memory Defragmentation）

为长生命周期张量（如激活检查点、梯度）**预分配连续显存块**，把长短生命周期张量分开，实时做去碎片化，从而避免"明明有空闲显存却因碎片而 OOM"的情况。

---

## 通信量分析：为什么几乎不增加通信

ZeRO 最反直觉、也最漂亮的结论：切分了这么多状态，通信量却几乎不增加。

### ZeRO-DP 的通信量（每个训练步，$\Psi$ 为参数量）

| 方案 | 通信操作 | 总通信量 | 相对标准 DP |
| --- | --- | --- | --- |
| 标准数据并行 | All-Reduce | $2\Psi$ | 1× |
| ZeRO $P_{os}+P_g$（ZeRO-2） | Reduce-Scatter($\Psi$) + All-Gather($\Psi$) | $2\Psi$ | **1×（零额外通信）** |
| ZeRO $P_{os}+P_g+P_p$（ZeRO-3） | 前/反向广播参数 + 梯度规约 | $3\Psi$ | **1.5×** |

关键结论：

- **ZeRO-2（$P_{os}+P_g$）相比标准 DP 不增加任何通信量**（多出的 0 字节）。
- **ZeRO-3（$P_{os}+P_g+P_p$）只增加 1.5 倍通信量**，却换来了显存的线性下降。

### ZeRO-R 的通信量

- $P_a$（激活切分）给 MP 增加的通信开销 **< 10%**。
- 但由于显存占用大幅下降，可以把 batch size 提升 MP 度那么多倍，这反而让**数据并行的通信量下降一个数量级**——整体是净收益。

---

## 迈向万亿参数

ZeRO 让"在 1024 张 GPU 上装下一个万亿参数模型"在**显存层面**成为可能。

但作者也坦诚指出 **"算力鸿沟（Compute Power Gap）"**：即使显存够了，用 1024 张 V100 训练一个万亿参数模型也需要**一年以上**。ZeRO 解决的是**显存与显存带宽瓶颈**；真正训练万亿模型还需要等待未来的 **Exascale（百亿亿次）** 超算系统。当那样的算力到来时，ZeRO 提供了高效利用它的系统架构。

---

## 实验与评估

### 硬件环境

- **400 张 NVIDIA V100 GPU**（32GB 显存）
- **25 个 DGX-2 节点**
- 节点间带宽 **800 Gbps**

### 关键结果

![Figure 2: ZeRO 吞吐量与相对 SOTA 基线的加速比](/assets/posts/zero-deepspeed/figure2.png)

*图 2（Figure 2）：不同模型规模下 ZeRO 的训练吞吐量及相对 SOTA 基线的加速比。横轴为模型规模（1.5B~170B）。ZeRO 保持高吞吐（最高 15 Petaflops），而 Megatron-LM 基线在超过 40B 后因跨节点 MP 通信瓶颈吞吐量急剧下滑。*

![Figure 3: 60B 模型的超线性可扩展性](/assets/posts/zero-deepspeed/figure3.png)

*图 3（Figure 3）：60B 参数模型从 64 卡扩展到 400 卡的每卡吞吐量。呈现**超线性（superlinear）**扩展——GPU 翻倍时吞吐量增长超过一倍，因为更低的每卡显存占用允许更大、更高效的 batch size。*

![Figure 4: 仅用数据并行时 ZeRO-DP 的最大模型吞吐量](/assets/posts/zero-deepspeed/figure4.png)

*图 4（Figure 4）：ZeRO-DP 的最大模型吞吐量。展示 ZeRO **仅靠数据并行（无需 MP/PP）** 就能训练高达 13B 参数的模型，远超标准 PyTorch DDP 的 1.4B 上限。*

![Figure 5: ZeRO 支撑的 SOTA 模型 Turing-NLG](/assets/posts/zero-deepspeed/figure5.png)

*图 5（Figure 5）：ZeRO 使能的 SOTA 模型 Turing-NLG。展示 170 亿参数 Turing-NLG 模型在 30 万次迭代上的验证困惑度（perplexity）曲线，超越此前 SOTA（Megatron 8.3B）。*

![Figure 6: 不同配置 C1-C5 下的最大模型规模](/assets/posts/zero-deepspeed/figure6.png)

*图 6（Figure 6）：五种 ZeRO 配置（C1~C5）下的最大可训练模型规模柱状图。凸显 $P_a$（激活切分）与 $P_{os}+P_g$ 如何大幅抬高显存天花板——从约 40B 提升到约 150B。*

![Figure 7: 不同配置下的最大缓存显存](/assets/posts/zero-deepspeed/figure7.png)

*图 7（Figure 7）：40B 与 100B 模型在不同配置下 PyTorch 缓存的最大显存。证明 ZeRO-R 技术有效抑制了显存膨胀。*

![Figure 8: 每卡吞吐量](/assets/posts/zero-deepspeed/figure8.png)

*图 8（Figure 8）：每卡吞吐量。显示显存缩减与性能提升直接相关——更低显存允许更大 batch size（更高算术强度）。*

### 数据速览

| 指标 | 数值 |
| --- | --- |
| 最大训练模型规模 | **170B**（约为 SOTA 基线 ~20B 的 8 倍） |
| 纯数据并行可训练规模 | **13B**（标准 DDP 仅 1.4B） |
| 万亿参数理论上限 | **1 Trillion**（1024 卡上证明可行） |
| 聚合吞吐量 | **15 Petaflops**（100B 模型） |
| 单卡平均吞吐 | **38 TFlops/GPU**（超过硬件峰值的 30%） |
| 相对 Megatron-LM 加速 | **>10×**（>40B 模型） |
| 可扩展性 | 64→400 卡 **超线性** 加速 |
| Turing-NLG 困惑度 | **10.21**（Webtext-103，当时 SOTA），41.4 TFlops/GPU |
| ZeRO-2 额外通信 | **0 字节** |
| ZeRO-3 额外通信 | **1.5×** |

> **表 3（Table 3）** 给出了消融实验用的 C1~C5 配置组合（如 C4 = $P_{os}+P_g$ + $C_B$ + $M_D$ + $P_a$）；**表 4（Table 4）** 列出了 1.5B~170B 各模型的层数与隐藏维度；附录 **表 5~10** 详列了各图对应的 batch size、MP 度、GPU 数等，保证可复现。

---

## 核心贡献与结论

### 核心贡献

1. **ZeRO-DP（模型状态显存优化）**：一种新颖的切分策略，消除数据并行的巨大显存冗余，同时不牺牲其计算粒度和通信效率。
2. **ZeRO-R（残余状态显存优化）**：激活切分、恒定缓冲区、去碎片化三件套，解决困扰超大模型的次要显存瓶颈。
3. **大模型民主化（Democratization）**：让数据并行也能处理 10B+ 参数模型，数据科学家无需为了适配复杂的模型/流水线并行拓扑而重写模型。
4. **开源与真实影响**：作为微软 **DeepSpeed** 库的一部分开源，并直接支撑了当时世界最大的语言模型 **Turing-NLG（170 亿参数）**。

### 结论

- **系统性突破**：ZeRO 把模型规模与"单卡显存 + 节点内通信"两大瓶颈解耦。
- **面向 Exascale 未雨绸缪**：当前硬件算力尚不足以在合理时间内训练万亿模型，但 ZeRO 已经解决了显存与显存带宽问题，为未来百亿亿次超算铺好了系统架构。
- **以用户为中心的设计**：ZeRO 是标准 PyTorch DDP 的**即插即用替代**，零模型重构成本，大幅降低了大规模 AI 研究的门槛。

---

## 个人思考与延伸

- **ZeRO 的分层设计（1/2/3 阶段）非常务实**：它把"显存节省"和"通信开销"做成了可调档位，工程上可以按集群带宽和模型规模选择合适的 stage。这也是今天 DeepSpeed / PyTorch FSDP 的思想源头。
- **ZeRO 与激活检查点、CPU Offload 是正交的**，可以叠加——这为后续的 ZeRO-Offload、ZeRO-Infinity 打下了基础。
- **"不改变优化数学过程"** 是 ZeRO 相对某些内存高效优化器的关键差异：它只是重新安排了数据的存放和通信方式，不影响收敛性，这让它极易被信任和采用。

---

## 专有名词中英文对照附录

| 中文 | 英文 | 简要说明 |
| --- | --- | --- |
| 零冗余优化器 | ZeRO (Zero Redundancy Optimizer) | 本文提出的核心技术 |
| 深度加速库 | DeepSpeed | 微软开源的大规模训练库，ZeRO 是其组成部分 |
| 数据并行 | Data Parallelism (DP) | 每卡复制完整模型，切分数据 |
| 模型并行 | Model Parallelism (MP) | 把模型竖切分到多卡 |
| 流水线并行 | Pipeline Parallelism (PP) | 按层分段做流水线 |
| 数据并行度 | Data Parallel Degree ($N_d$) | 参与切分的进程/设备数 |
| 模型状态 | Model States | 优化器状态 + 梯度 + 参数 |
| 残余状态 | Residual States | 激活值 + 临时缓冲区 + 显存碎片 |
| 优化器状态 | Optimizer States | 如 Adam 的动量与方差 |
| 梯度 | Gradients | 反向传播计算的梯度 |
| 参数 | Parameters | 模型权重 |
| 混合精度训练 | Mixed-Precision Training | fp16 计算 + fp32 主权重 |
| 主权重 | Master Weights | fp32 精度的权重副本 |
| 动量 | Momentum | Adam 一阶矩 |
| 方差 | Variance | Adam 二阶矩 |
| 激活值 | Activations | 前向传播的中间输出 |
| 激活检查点 | Activation Checkpointing | 用重计算换显存 |
| 分区激活检查点 | Partitioned Activation Checkpointing ($P_a$) | 跨 MP 卡切分激活值 |
| 优化器状态切分 | Optimizer State Partitioning ($P_{os}$) | ZeRO 阶段一 |
| 梯度切分 | Gradient Partitioning ($P_g$) | ZeRO 阶段二 |
| 参数切分 | Parameter Partitioning ($P_p$) | ZeRO 阶段三 |
| 恒定大小缓冲区 | Constant Size Buffers ($C_B$) | 固定临时缓冲区上限 |
| 显存去碎片化 | Memory Defragmentation ($M_D$) | 预分配连续显存块 |
| 临时缓冲区 | Temporary Buffers | 通信/计算用的临时空间 |
| 显存碎片 | Fragmented Memory | 显存分配碎片化 |
| 显存不足 | Out-Of-Memory (OOM) | 显存耗尽错误 |
| 全规约 | All-Reduce | 集合通信：规约并广播 |
| 规约散射 | Reduce-Scatter | 集合通信：规约并分发 |
| 全收集 | All-Gather | 集合通信：收集拼接 |
| 广播 | Broadcast | 集合通信：一对多分发 |
| 卸载 | Offload | 把数据搬到 CPU 内存/NVMe |
| 计算粒度 | Computational Granularity | 单次计算的规模，影响效率 |
| 流水线气泡 | Pipeline Bubble | 流水线并行的空闲等待 |
| 微批次 | Micro-batch | 流水线并行切分的小批次 |
| 困惑度 | Perplexity | 语言模型评估指标（越低越好） |
| 超线性扩展 | Superlinear Scalability | 资源翻倍时性能增长超过一倍 |
| 算力鸿沟 | Compute Power Gap | 显存够但算力不足的差距 |
| 百亿亿次 | Exascale | 每秒 10^18 次浮点运算的算力量级 |
| 业界最优 | State-Of-The-Art (SOTA) | 当时最优水平 |

---

## 论文来源附录

- **论文标题**：ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
- **论文地址**：<https://arxiv.org/abs/1910.02054>
- **作者**：Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He (Microsoft)

> 本文所有配图均提取自原论文，仅用于学习笔记。
