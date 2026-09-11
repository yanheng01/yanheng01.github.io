---
title: "PagedAttention / vLLM：LLM Serving 怎么提高吞吐"
date: 2026-09-11 18:44:00 +0800
permalink: /posts/pagedattention-vllm/
categories: [技术笔记, 深度学习]
tags: [pagedattention, vllm, kv-cache, llm-serving, gpu, 论文精读, 大模型, Inference]
math: true
mermaid: false
toc: true
---

> **原文标题（Original Title）**：Efficient Memory Management for Large Language Model Serving with PagedAttention
> **作者（Authors）**：Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, Ion Stoica（UC Berkeley; Stanford University; UC San Diego; Independent Researcher）
> **发表**：SOSP '23（ACM 操作系统原理研讨会 Symposium on Operating Systems Principles）
> **开源地址**：<https://github.com/vllm-project/vllm>

---

## 一句话总结

大语言模型（LLM）推理吞吐的瓶颈不在算力，而在**显存管理**——具体说，是 KV 缓存（KV Cache）被内存碎片和冗余复制严重浪费。本文借鉴操作系统（OS）**虚拟内存（Virtual Memory）与分页（Paging）**的思想，提出注意力算法 **PagedAttention**，并在其上构建服务系统 **vLLM**，把 KV 缓存的浪费从 60%~80% 降到接近零，从而把可批处理的请求数大幅提高，在相同延迟下把主流 LLM 的吞吐量提升 **2～4 倍**。

---

## 1. 背景与问题：为什么 LLM 推理是「显存受限」的

### 1.1 自回归生成的两个阶段

LLM 的核心是自回归 Transformer（Autoregressive Transformer）。给定输入提示词（Prompt），模型逐个 token 生成输出，每个新 token 都依赖此前所有 token。生成过程分两个阶段：

- **提示词阶段（Prompt Phase / Prefill）**：一次性并行处理整个 prompt，计算所有位置的 Query / Key / Value，可用矩阵-矩阵乘法（matrix-matrix multiplication）充分利用 GPU 并行度，计算利用率高。
- **自回归生成阶段（Autoregressive Generation Phase / Decode）**：逐 token 生成。每一步只计算新 token 的 Key/Value，与此前缓存的历史 Key/Value 做注意力计算。这一步是矩阵-向量乘法（matrix-vector multiplication），无法并行，严重浪费 GPU 算力，是**内存受限（Memory-bound）**的，构成单请求延迟的主要部分。

### 1.2 KV 缓存（KV Cache）是什么

为避免自回归阶段重复计算历史 token 的 Key/Value 向量，系统会把它们缓存下来，即 **KV Cache**。注意：同一个 token 在序列中不同位置，其 KV 缓存是不同的（因为依赖它之前的所有 token）。

### 1.3 显存都花在哪了

![图1：13B 模型在 A100 上的显存布局，以及 vLLM 平滑 KV 缓存增长曲线后带来的吞吐提升](/assets/posts/pagedattention-vllm/memory-distribution.svg)

**图 1（Figure 1）**：在 NVIDIA A100（40GB）上服务一个 13B 参数模型时的显存布局。
- **左图**：模型权重（灰色，静态）约占 **65%**；KV 缓存（红色，随请求动态分配/释放）约占近 **30%**；激活值（activation，黄色，临时）占少量。由于权重恒定、激活占比小，**KV 缓存的管理方式直接决定了最大批大小（batch size）与吞吐量**。
- **右图**：现有系统的 KV 缓存增长曲线陡峭、易撞显存上限；vLLM 平滑了这条曲线，显著提升了服务吞吐。

### 1.4 现有系统的浪费有多严重

![图2：不同 LLM 服务系统的平均显存浪费占比](/assets/posts/pagedattention-vllm/memory_breakdown.svg)

**图 2（Figure 2）**：对比不同 LLM 服务系统的 KV 缓存有效利用率。现有系统（如 Orca、FasterTransformer）中，真正存储有效 token 状态的显存只占 **20.4%～38.2%**，其余 60%~80% 被浪费；vLLM 的浪费接近 0。

### 1.5 三类显存浪费（核心痛点）

现有系统要求 KV 缓存存放在**连续内存（contiguous memory）**中（大多数深度学习框架的张量都要求连续存储），并按请求可能的最大长度（如 2048 tokens）预分配。这带来三类浪费：

![图3：现有系统的 KV 缓存管理，展示保留、内部碎片、外部碎片三类浪费](/assets/posts/pagedattention-vllm/baseline-memory-management.svg)

**图 3（Figure 3）**：现有系统的 KV 缓存管理。三类浪费导致其他请求无法挤进显存：
- **保留（Reserved）**：为未来将生成的 token 预留、但当前尚未用到的连续空间；这段空间在请求整个生命周期内被占用，其他更短的请求也无法借用。
- **内部碎片（Internal Fragmentation）**：预分配的最大长度与实际生成长度之间的差额。请求实际长度往往远小于最大长度。
- **外部碎片（External Fragmentation）**：不同请求预分配大小不一，在显存中留下无法利用的空隙。

### 1.6 无法共享内存

LLM 服务常用**并行采样（Parallel Sampling）**、**束搜索（Beam Search）**等解码算法，一个请求会产生多个序列，这些序列的 prompt 甚至部分生成内容是相同的、本可共享 KV 缓存。但由于连续内存的限制，现有系统各序列的 KV 缓存被分别存放，**无法共享**。

---

## 2. 核心思想：把操作系统的虚拟内存搬进注意力

作者的洞察：KV 缓存管理面临的「碎片 + 共享」难题，正是操作系统几十年前用**虚拟内存 + 分页**解决过的问题。类比关系是：

| 操作系统概念 | vLLM 中的对应 |
| --- | --- |
| 页（Page） | 块（Block，固定数量 token 的 KV） |
| 字节（Byte） | token |
| 进程（Process） | 请求（Request） |
| 页表（Page Table） | 块表（Block Table） |

---

## 3. 方法：PagedAttention 算法

### 3.1 分块存储、非连续

PagedAttention 把每个序列的 KV 缓存切分成固定大小的 **KV 块（KV Block）**，每块保存固定数量 token 的 Key/Value 向量。关键点：**这些块在物理显存中无需连续存放**。

![图5：PagedAttention 算法示意——Query 向量与非连续物理块逐块做注意力计算](/assets/posts/pagedattention-vllm/pagedattention.svg)

**图 5（Figure 5）**：PagedAttention 算法示意。注意力计算被转化为**逐块（block-wise）**计算：kernel 依据块表找到分散在显存各处的 KV 块（Block 0、1、2……），依次取出参与注意力运算。

- 用较小的块 + 按需分配，缓解**内部碎片**；
- 所有块大小一致，消除**外部碎片**；
- 支持以块为粒度在序列之间、乃至请求之间**共享**。

### 3.2 KV 缓存管理器（KV Cache Manager）：逻辑块与物理块

vLLM 仿照 OS 虚拟内存，把「逻辑」和「物理」分离：

![图6：vLLM 的块表翻译机制——逻辑块到物理块的映射与已填充计数](/assets/posts/pagedattention-vllm/logical-and-physical-block-table.svg)

**图 6（Figure 6）**：块表（Block Table）翻译机制。
- **逻辑 KV 块（Logical KV Block）**：从序列视角看，是「连续」的 KV 缓存。
- **物理 KV 块（Physical KV Block）**：GPU/CPU 显存里实际分配的固定大小物理块，可分散存放。
- **块表（Block Table）**：记录逻辑块 → 物理块的映射，以及每个逻辑块中**已填充（# filled）**的槽位数。
- **按需分配（on-demand allocation）**：不预分配最大长度的显存，只有当新 token 写入且当前块被填满时才分配新物理块。于是浪费被严格限制在**最后一个未填满的块内**。

![图7：vLLM 中同时存储两个请求的 KV 缓存，各自逻辑块映射到分散的物理块](/assets/posts/pagedattention-vllm/multi-sequence-block-mapping.svg)

**图 7（Figure 7）**：同时存储两个请求的 KV 缓存。两个请求各自的逻辑块被映射到 GPU 显存中分散的物理块上，从而把碎片空间高效拼接利用起来。

---

## 4. 复杂解码场景下的内存共享

### 4.1 并行采样（Parallel Sampling）与写时复制（Copy-on-Write, CoW）

![图8：并行采样示例——多个样本共享 prompt 物理块，写入时触发 CoW](/assets/posts/pagedattention-vllm/parallel-decoding.svg)

**图 8（Figure 8）**：并行采样示例。一个请求生成多个输出，它们共享同一份 prompt 的 KV 缓存。物理块引入**引用计数（Reference Count）**：
- 当某序列要写入一个被共享的物理块（引用计数 > 1）时，触发**写时复制（Copy-on-Write, CoW）**：系统新分配一个物理块、复制原块数据，再对新块写入，并更新块表。
- 这样多个样本能最大限度共享 prompt 部分，只有产生分歧时才复制。

### 4.2 束搜索（Beam Search）

![图9：束搜索示例——候选束之间树状共享物理块，被淘汰的束释放引用](/assets/posts/pagedattention-vllm/beam-search.svg)

**图 9（Figure 9）**：束搜索示例。不同候选束（beam candidate）不仅共享 prompt，还在生成过程中动态共享相同的分支前缀，形成**树状共享**。被淘汰的候选束，其物理块引用计数减为 0 后被释放。同样用 CoW 处理分歧点写入，**避免了传统系统中大量的 KV 缓存拷贝**。

### 4.3 共享前缀（Shared Prefix）

![图10：机器翻译中的共享前缀示例——系统提示被多个请求共享](/assets/posts/pagedattention-vllm/share-prompt.svg)

**图 10（Figure 10）**：共享前缀示例（机器翻译场景，few-shot 示例来自 Brown et al., 2020）。类似 OS 的**共享库（Shared Library）**：系统提示词（System Prompt）的 KV 缓存被预先计算并固化在物理块中，新请求只需把逻辑块映射到这些已有物理块，再对自己的任务输入部分执行 prompt 阶段即可。

### 4.4 混合解码（Mixed Decoding）

统一的块表抽象层屏蔽了底层复杂的共享逻辑，使不同解码算法（贪心、并行采样、束搜索……）的请求能被**无缝混合批处理**。

---

## 5. vLLM 系统设计

### 5.1 整体架构

![图4：vLLM 系统总览——集中式调度器 + KV 缓存管理器 + 多个 GPU Worker](/assets/posts/pagedattention-vllm/system-overview.png)

**图 4（Figure 4）**：vLLM 系统总览。包含 FastAPI 前端、集中式**调度器（Scheduler）**、**KV 缓存管理器（KV Cache Manager）**，以及多个带**缓存引擎（Cache Engine）**的 GPU Worker。调度器与 PagedAttention 协同设计，做块级内存管理和抢占式请求调度。

### 5.2 调度与抢占（Scheduling and Preemption）

- **调度策略**：先来先服务（FCFS, First-Come-First-Serve），保证公平。
- **全有或全无驱逐（All-or-nothing Eviction）**：因为注意力计算需要一个序列的全部历史 KV 块，抢占时要么驱逐该序列的**所有**块，要么都不驱逐。同一请求内的多个序列（如各束候选）作为**序列组（Sequence Group）**一起被抢占。
- **两种恢复机制**：
  - **交换（Swapping）**：把 GPU 显存里的物理块换出到 CPU 内存（RAM），需要时再换回。
  - **重计算（Recomputation）**：把已生成的 token 拼回 prompt 之后，重新跑一次 prompt 阶段来恢复 KV 缓存，避免 PCIe 传输开销。

### 5.3 分布式执行（Distributed Execution）

- 支持 **Megatron-LM 风格的张量模型并行（Tensor Model Parallelism）**。
- **全局单一 KV 缓存管理器**：所有 GPU Worker 共享同一份块表映射。
- **局部 KV 存储**：虽然每个 Worker 收到相同的块表，但各自只存储它负责的那部分**注意力头（Attention Head）**对应的 KV 缓存切片。

### 5.4 内核级实现优化（Kernel-level Implementation）

vLLM 针对 PagedAttention 的非连续访问，做了多项 GPU kernel 融合：
- **融合 reshape 与块写入（Fused reshape and block write）**：把 KV 缓存的分块、重塑、写入融合成单个 kernel，减少启动开销。
- **融合块读取与注意力（Fusing block read and attention）**：按块表读取非连续块并直接算注意力；用一个 GPU warp 读取一个块，保证**合并内存访问（Coalesced Memory Access）**。
- **融合块拷贝（Fused block copy）**：把 CoW 里离散的块拷贝批处理成单次 kernel 调用，避免频繁小数据 `cudaMemcpyAsync` 的开销。
- **接口原语**：提供 `fork`（复制/共享前缀）、`append`（追加新 token）、`free`（释放序列）三个基本操作。

---

## 6. 实验评估

### 6.1 实验设置

- **模型**：OPT（13B / 66B / 175B）、LLaMA-13B。
- **硬件**：NVIDIA A100 集群。
- **数据集**：ShareGPT（长文本、方差大）、Alpaca（短文本）。
- **对比基线（Baseline）**：FasterTransformer；Orca 的三种内存预留策略（Orca (Max)、Orca (Pow2)、Orca (Oracle)）。

![图11：ShareGPT 与 Alpaca 数据集的输入/输出长度分布](/assets/posts/pagedattention-vllm/sharegpt_hist.svg)
![图11（续）：Alpaca 数据集长度分布](/assets/posts/pagedattention-vllm/alpaca_hist.svg)

**图 11（Figure 11）**：两个数据集的输入/输出长度分布直方图。ShareGPT 序列更长、方差更大；Alpaca 偏短。

### 6.2 基础采样（单序列生成）性能

![图12：OPT 各模型在 ShareGPT 数据集上单序列生成的归一化延迟-请求率曲线](/assets/posts/pagedattention-vllm/n1-sharegpt.svg)
![图12（续）：OPT 各模型在 Alpaca 数据集上的曲线](/assets/posts/pagedattention-vllm/n1-alpaca.svg)

**图 12（Figure 12）**：OPT 模型在 ShareGPT / Alpaca 上单序列生成时，归一化延迟随请求率（request rate）的变化曲线。vLLM 能承受明显更高的请求率而不使延迟爆炸。

![图13：vLLM 与各基线平均可批处理的请求数对比（ShareGPT）](/assets/posts/pagedattention-vllm/batched_requests_sharegpt.svg)
![图13（续）：Alpaca 数据集的平均批处理请求数](/assets/posts/pagedattention-vllm/batched_requests_alpaca.svg)

**图 13（Figure 13）**：平均能同时批处理的请求数。vLLM 远超 Orca 系列——这正是吞吐提升的直接原因。

**核心结果**：
- vLLM 吞吐量比 **Orca (Oracle)** 高 **1.7×～2.7×**，比 **Orca (Max)** 高 **2.7×～8×**；相比 **FasterTransformer** 最高提升可达 **22×**。

### 6.3 并行采样与束搜索

![图14：OPT-13B 在 Alpaca 上并行生成与束搜索的性能](/assets/posts/pagedattention-vllm/parallel.svg)
![图14（续）：束搜索性能](/assets/posts/pagedattention-vllm/beam.svg)

**图 14（Figure 14）**：并行生成与束搜索场景（OPT-13B, Alpaca）。由于内存共享，vLLM 优势进一步扩大；束宽（beam width）越大，共享收益越明显。束搜索（width=6）下比 Orca (Oracle) 提升约 **2.3×**。

![图15：并行采样与束搜索带来的内存节省比例](/assets/posts/pagedattention-vllm/mem_saving_parallel_gen.svg)
![图15（续）：束搜索内存节省](/assets/posts/pagedattention-vllm/mem_saving_beam.svg)

**图 15（Figure 15）**：共享机制带来的内存节省。并行采样节省约 **6%～10%**，束搜索节省高达 **37%～55%**。

### 6.4 共享前缀（Shared Prefix）

![图16：输入共享公共前缀的翻译工作负载，1-shot 与 5-shot 前缀](/assets/posts/pagedattention-vllm/prefix.svg)

**图 16（Figure 16）**：输入共享公共前缀的翻译工作负载。前缀分别为 1 个和 5 个 few-shot 示例。vLLM 通过共享前缀 KV 缓存，吞吐分别提升约 **1.67×（1-shot）**和 **3.58×（5-shot）**，前缀越长优势越大。

### 6.5 聊天机器人（Chatbot）

![图17：聊天机器人工作负载下的性能](/assets/posts/pagedattention-vllm/chat-sharegpt.svg)

**图 17（Figure 17）**：长上下文（约 1024 tokens）多轮对话场景，vLLM 相比 Orca 提升约 **2×**。

### 6.6 消融实验（Ablation Studies）

![图18：注意力内核延迟对比，以及块大小对端到端延迟的影响](/assets/posts/pagedattention-vllm/micro_latency.svg)
![图18（续）：不同块大小（Block Size）对端到端延迟的影响](/assets/posts/pagedattention-vllm/n1-block-size.svg)

**图 18（Figure 18）**：
- **左**：PagedAttention 因额外的块表寻址与分支，注意力 kernel 延迟比高度优化的 FasterTransformer 高 **20%～26%**；但端到端吞吐的巨大提升完全掩盖了这点开销。
- **右**：块大小（Block Size）的影响。默认取 **16**：太小（如 1）无法用好 GPU 并行读取；太大（如 256）会增加内部碎片、降低共享概率。

![图19：重计算 vs 交换的微基准与端到端对比](/assets/posts/pagedattention-vllm/micro-swap.svg)
![图19（续）：重计算与交换的端到端性能](/assets/posts/pagedattention-vllm/recompute-vs-swap.svg)

**图 19（Figure 19）**：交换（Swapping）与重计算（Recomputation）对比。
- 块较小时：**重计算更优**（大量离散小块换出会让 PCIe 带宽碎片化）。
- 块较大时：**交换更优**（单次大块传输能更好利用 PCIe 带宽）。
- 中等块大小（16～64）时两者性能相当。

---

## 7. 关键数据与结论

1. **显存利用率飞跃**：传统系统 KV 缓存有效利用率仅 **20.4%～38.2%**；vLLM 把浪费限制在最后一个块内，利用率接近 **100%**。
2. **吞吐提升**：相同延迟下，vLLM 比 SOTA 系统（FasterTransformer、Orca）吞吐提升 **2×～4×**。
3. **优势放大条件**：序列越长、模型越大、解码算法越复杂（并行采样、束搜索、共享前缀），vLLM 的相对优势越明显。
4. **不损精度**：吞吐提升完全来自内存管理优化，**不影响模型输出精度**。

---

## 8. 表格：模型与服务器配置

**表 1（Table 1）**：论文列出了 OPT-13B / 66B / 175B 的参数量、所需 GPU 数量、总显存、参数占用显存、可用于 KV 缓存的显存及最大 KV 槽位数，用于说明不同规模模型下 KV 缓存对可批处理请求数的约束。

---

## 9. 术语中英对照（附录：专有名词）

| 中文 | 英文 |
| --- | --- |
| 分页注意力 | PagedAttention |
| 键值缓存 | KV Cache (Key-Value Cache) |
| 大语言模型 | Large Language Model (LLM) |
| 虚拟内存 | Virtual Memory |
| 分页 | Paging |
| 页 / 页表 | Page / Page Table |
| 块 / 块表 | Block / Block Table |
| 逻辑 KV 块 | Logical KV Block |
| 物理 KV 块 | Physical KV Block |
| 已填充（槽位数） | # filled |
| 按需分配 | On-demand Allocation |
| 写时复制 | Copy-on-Write (CoW) |
| 引用计数 | Reference Count |
| 内部碎片 | Internal Fragmentation |
| 外部碎片 | External Fragmentation |
| 保留（内存） | Reserved |
| 自回归 Transformer | Autoregressive Transformer |
| 自注意力 | Self-Attention |
| 提示词阶段 / 预填充 | Prompt Phase / Prefill |
| 自回归生成阶段 / 解码 | Autoregressive Generation Phase / Decode |
| 内存受限 | Memory-bound |
| 迭代级调度 | Iteration-level Scheduling |
| 连续批处理 | Continuous Batching |
| 并行采样 | Parallel Sampling |
| 束搜索 / 束宽 | Beam Search / Beam Width |
| 共享前缀 | Shared Prefix |
| 混合解码 | Mixed Decoding |
| 序列组 | Sequence Group |
| 全有或全无驱逐 | All-or-nothing Eviction |
| 抢占 | Preemption |
| 交换 | Swapping |
| 重计算 | Recomputation |
| 先来先服务 | FCFS (First-Come-First-Serve) |
| 张量模型并行 | Tensor Model Parallelism |
| 注意力头 | Attention Head |
| 合并内存访问 | Coalesced Memory Access |
| 缓存引擎 | Cache Engine |
| 调度器 | Scheduler |
| 共享库 | Shared Library |

---

## 附录：论文链接

- 原始论文（arXiv PDF）：<https://arxiv.org/pdf/2309.06180>

---

*标签：Inference*
