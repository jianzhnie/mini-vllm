# mini-vLLM：轻量级大语言模型推理引擎

## 目录

- [mini-vLLM：轻量级大语言模型推理引擎](#mini-vllm轻量级大语言模型推理引擎)
  - [目录](#目录)
  - [一、仓库概览](#一仓库概览)
    - [1.1 项目背景与定位](#11-项目背景与定位)
    - [1.2 核心特性](#12-核心特性)
    - [1.3 项目信息](#13-项目信息)
    - [1.4 依赖环境](#14-依赖环境)
    - [1.5 系统要求](#15-系统要求)
    - [1.6 安装步骤](#16-安装步骤)
    - [1.7 仓库目录结构](#17-仓库目录结构)
  - [二、代码主要说明](#二代码主要说明)
    - [2.1 核心流程概述](#21-核心流程概述)
    - [2.2 核心优化技术详解](#22-核心优化技术详解)
    - [2.3 架构设计原则](#23-架构设计原则)
  - [三、代码文件详细解释](#三代码文件详细解释)
    - [3.1 入口和配置文件](#31-入口和配置文件)
    - [3.2 引擎核心模块 (`engine/`)](#32-引擎核心模块-engine)
    - [3.3 模型实现 (`models/`)](#33-模型实现-models)
    - [3.4 神经网络层 (`models/layers/`)](#34-神经网络层-modelslayers)
    - [3.5 采样模块 (`sampling/`)](#35-采样模块-sampling)
    - [3.6 工具模块 (`utils/`)](#36-工具模块-utils)
    - [3.7 使用示例](#37-使用示例)
  - [四、代码逻辑流程（调用关系）](#四代码逻辑流程调用关系)
    - [4.1 初始化流程](#41-初始化流程)
    - [4.2 生成流程](#42-生成流程)
    - [4.3 张量并行流程（多 GPU）](#43-张量并行流程多-gpu)
    - [4.4 前缀缓存工作流](#44-前缀缓存工作流)
  - [五、关键优化总结](#五关键优化总结)
  - [六、性能特点](#六性能特点)
  - [七、使用建议与最佳实践](#七使用建议与最佳实践)
  - [八、多设备支持指南](#八多设备支持指南)
  - [九、许可证与致谢](#九许可证与致谢)
  - [十、总结](#十总结)

## 一、仓库概览

### 1.1 项目背景与定位

**什么是 mini-vLLM？**

mini-vLLM 是一个**从零开始构建的轻量级大语言模型推理引擎**，为开发者提供简单、高效、可扩展的推理解决方案。与其他复杂推理框架不同，mini-vLLM 特别注重**代码的可读性和理解性**，同时保持与主流推理引擎相当的性能。

**为什么需要 mini-vLLM？**

现有主流推理框架（如 vLLM、TGI 等）虽然性能优异，但存在以下痛点：

1. **代码复杂**：追求极致性能导致代码结构复杂，依赖众多，不利于学习理解
2. **学习曲线陡峭**：初学者理解框架内部原理需要大量时间
3. **定制困难**：代码复杂度高，根据需求定制修改困难
4. **资源消耗大**：部署运行需要大量计算资源和内存

mini-vLLM 正是为解决这些问题而生，提供**轻量级、易理解、高性能**的推理引擎。

**项目定位**

- **教育平台**：学习理解大语言模型推理原理的绝佳资源
- **开发工具**：清晰的模块化实现，便于二次开发和定制
- **轻量级部署**：适用于需要高效推理但不想承担复杂依赖的场景
- **研究实验台**：研究新推理优化技术的理想平台
- **快速原型开发**：便于快速验证和测试新想法

**核心价值主张**

- 🔬 **易理解**：代码结构清晰，注释详细，适合学习推理原理
- ⚡ **高性能**：集成多种优化技术，推理速度与主流框架相当
- 🛠️ **易扩展**：模块化设计，便于添加新模型和优化技术
- 📦 **轻量级**：依赖少，部署简单，资源消耗低
- 🌟 **易定制**：代码可读性高，便于根据特定需求修改

### 1.2 核心特性

#### 🔥 高性能推理

- **CUDA Graph 优化**：将 GPU 操作录制为计算图，减少调度开销，提升 decode 阶段速度
- **智能批处理**：动态批处理和序列长度感知调度，充分利用 GPU 资源
- **KV 缓存优化**：块式管理和前缀缓存技术，减少内存占用

#### 🧠 先进架构

- **多设备支持**：CUDA、NPU、XPU、MPS、MLU、MUSA
- **张量并行**：支持多 GPU 分布式推理（1-8 张量并行度）
- **两阶段调度**：预填充（Prefill）和解码（Decode）阶段分离调度
- **FlashAttention**：集成高性能注意力计算库

#### 🛠️ 灵活配置

- **内存管理**：可配置设备内存利用率（10%-100%）
- **缓存配置**：可调节 KV 缓存块大小和数量
- **采样控制**：支持温度调节、top-p、top-k、min-p 等采样参数

#### 📖 易读代码库

- 模块化设计，代码结构清晰
- 详细的注释和文档
- 适合作为学习教材，也便于二次开发

### 1.3 项目信息

| 属性 | 值 |
|------|-----|
| 项目名称 | mini-vLLM |
| 项目描述 | 轻量级 vLLM 实现，专注可读性和学习 |
| 主要用途 | 大语言模型离线推理 |
| 典型应用场景 | 研究实验、教学演示、轻量级部署 |
| 许可证 | Apache 2.0 |

### 1.4 依赖环境

```
torch>=2.4.0          # PyTorch 深度学习框架
transformers>=4.51.0  # HuggingFace Transformers 库
triton>=3.0.0         # Triton 优化编译器（可选）
flash-attn            # 高性能注意力实现（可选）
xxhash                # 高效哈希算法库
```

### 1.5 系统要求

- Python 3.10 - 3.12
- PyTorch 2.2+
- CUDA 11.8+ (推荐用于 GPU)
- 至少 8GB GPU 内存（推荐）

### 1.6 安装步骤

```bash
# 克隆项目
git clone https://github.com/jianzhnie/mini-vllm.git
cd mini-vllm

# 基础安装
pip install -e .

# 安装 CUDA 优化版本（推荐）
pip install -e ".[cuda]"

# 安装 NPU 版本
pip install -e ".[npu]"

# 安装开发依赖
pip install -e ".[dev]"
```

**验证安装**

```bash
python -c "import minivllm; print('Installation successful!')"
```

### 1.7 仓库目录结构

```
mini-vllm/
├── minivllm/                   # 主要包目录
│   ├── __init__.py            # 导出核心类和版本信息
│   ├── llm.py                 # LLM 类入口（用户接口）
│   ├── config.py              # 配置类定义
│   ├── sampling_params.py     # 采样参数定义（用户接口）
│   ├── engine/                # 推理引擎核心模块
│   │   ├── __init__.py
│   │   ├── llm_engine.py      # 推理引擎主类
│   │   ├── model_runner.py    # 模型运行器
│   │   ├── scheduler.py       # 批次调度器
│   │   ├── sequence.py        # 序列数据结构
│   │   ├── block_manager.py   # KV缓存块管理器
│   │   ├── inference_executor.py  # 推理执行器
│   │   └── distributed_manager.py # 分布式管理器
│   ├── models/                # 模型实现
│   │   ├── __init__.py
│   │   ├── qwen2.py           # Qwen2 模型实现
│   │   ├── qwen3.py           # Qwen3 模型实现
│   │   ├── opt.py             # OPT 模型实现
│   │   ├── manager.py         # 模型管理器
│   │   └── layers/            # 神经网络层
│   │       ├── __init__.py
│   │       ├── attention.py   # 注意力机制
│   │       ├── attention_backend.py  # 注意力后端抽象
│   │       ├── linear.py      # 张量并行线性层
│   │       ├── embed_head.py  # 嵌入和输出层
│   │       ├── rotary_embedding.py   # 旋转位置编码
│   │       ├── layernorm.py   # RMSNorm 层
│   │       ├── activation.py  # 激活函数
│   │       └── npu_flash_attention.py # NPU Flash Attention
│   ├── sampling/              # 采样模块
│   │   ├── __init__.py
│   │   ├── sampler.py         # 统一采样器
│   │   ├── config.py          # 采样配置
│   │   ├── functional.py      # 采样函数实现
│   │   └── mirostat.py        # Mirostat 采样
│   └── utils/                 # 工具函数
│       ├── __init__.py
│       ├── context.py         # 全局上下文管理
│       ├── device.py          # 设备检测和管理
│       ├── loader.py          # 模型权重加载
│       ├── memory_pool.py     # 内存池管理
│       ├── random_utils.py    # 随机工具
│       └── logger_utils.py    # 日志工具
├── tests/                     # 测试目录
├── examples/                  # 示例代码
├── README.md                  # 项目说明
├── intro.md                   # 本文件
└── pyproject.toml             # 项目配置
```

## 二、代码主要说明

### 2.1 核心流程概述

mini-vLLM 的推理流程分为六个阶段：

```
┌─────────────────────────────────────────────────────────────────┐
│                        推理流程                                  │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│  初始化阶段  │ 请求处理阶段 │   调度阶段   │   推理阶段   │ 采样阶段 │ 输出阶段 │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────┤
│ 加载模型权重 │ 接收推理请求 │ PREFILL阶段 │ 执行前向传播 │ 应用采样 │ 解码文本 │
│ 初始化KV缓存 │ 文本转token │ DECODE阶段  │ 存储KV缓存  │ 生成token│ 返回结果 │
│ 准备CUDA图  │ 创建序列对象 │  调度序列    │  计算logits │         │         │
│ 初始化分布式 │ 添加到队列   │  分配资源    │             │         │         │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────┘
```

1. **初始化阶段**：加载模型权重、初始化 KV 缓存、准备 CUDA 图、初始化分布式环境
2. **请求处理阶段**：接收请求、文本转 token IDs、创建序列对象、添加到待处理队列
3. **调度阶段**：
   - **Prefill**：首次处理提示文本，生成初始 KV 缓存
   - **Decode**：基于已有 KV 缓存，每次生成一个新 token
4. **推理阶段**：执行前向传播、存储/更新 KV 缓存
5. **采样阶段**：应用温度、top-p、top-k 等采样策略，生成下一个 token
6. **输出阶段**：token IDs 转文本、返回结果、释放资源

### 2.2 核心优化技术详解

#### 前缀缓存 (Prefix Caching)

当多个请求具有相同前缀时，共享这部分的计算结果：

- 对每个 KV 缓存块计算哈希值
- 新请求检查前缀是否已存在缓存中
- 复用已有缓存块，避免重复计算
- 通过引用计数管理缓存块生命周期

#### 块化 KV 缓存 (Block-wise KV Cache)

- 将 KV 缓存分成固定大小的块（默认 256 tokens）
- 每个序列分配块表记录使用的缓存块 ID
- 动态添加/释放缓存块，支持变长序列
- 减少内存碎片化，提高利用率

#### 张量并行 (Tensor Parallelism)

- 将模型权重分割到多个 GPU
- 输入数据同时发送到所有 GPU 并行处理
- 通过 all-reduce 聚合输出结果
- 支持 1-8 个 GPU，突破单卡内存限制

#### CUDA 图优化

- 捕获 decode 阶段的 GPU 操作为静态计算图
- 后续直接复用计算图，避免重复调度
- 为不同 batch size 录制不同图形
- 显著减少 CPU-GPU 同步开销

#### FlashAttention

- 分块计算注意力矩阵
- 优化内存访问模式
- 减少 GPU 内存带宽瓶颈
- 支持变长序列处理

### 2.3 架构设计原则

1. **模块化设计**：清晰的模块划分，便于维护和扩展
2. **可扩展性**：支持多种模型架构和优化技术
3. **高性能**：集成多种优化技术，保持与主流框架相当的性能
4. **可读性**：简洁清晰的代码，便于学习和理解
5. **灵活性**：丰富的配置选项，适应不同使用场景

## 三、代码文件详细解释

### 3.1 入口和配置文件

#### llm.py - LLM 类入口

```python
class LLM(LLMEngine):
    """用户直接使用的接口类，继承自 LLMEngine"""
    pass
```

**功能**：提供统一 API，隐藏底层复杂实现，主要通过 `generate()` 方法进行批量推理。

#### config.py - 配置类

```python
@dataclass
class Config:
    model: str                          # 模型路径或名称
    max_num_batched_tokens: int = 16384 # 每批次最大 token 数
    max_num_seqs: int = 512             # 每批次最大序列数
    max_model_len: int = 4096           # 模型最大序列长度
    device_memory_utilization: float = 0.9  # 设备内存利用率
    tensor_parallel_size: int = 1       # 张量并行 GPU 数量
    enforce_eager: bool = False         # 是否强制 eager 模式
    kvcache_block_size: int = 256       # KV 缓存块大小（必须被256整除）
    num_kvcache_blocks: int = -1        # KV 缓存块数量（-1自动）
    trust_remote_code: bool = False     # 是否信任远程代码
    dtype: str = 'auto'                 # 模型权重数据类型
    seed: Optional[int] = None          # 随机种子
```

**关键约束**：
- `kvcache_block_size` 必须被 256 整除
- `tensor_parallel_size` 范围 [1, 8]
- `device_memory_utilization` 范围 [0.1, 1.0]
- `max_num_batched_tokens >= max_model_len`

#### sampling_params.py - 采样参数

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0    # 采样温度 (> 1e-10)
    top_p: float = 1.0          # Nucleus 采样阈值 (0-1]
    top_k: int = -1             # Top-k 采样 (-1禁用)
    min_p: float = 0.0          # 最小相对概率 [0-1]
    max_tokens: int = 64        # 最大生成 token 数
    ignore_eos: bool = False    # 是否忽略结束符
```

### 3.2 引擎核心模块 (`engine/`)

#### llm_engine.py - 推理引擎主类

**核心职责**：
1. **初始化**：创建 Config、启动多进程、初始化 ModelRunner、Tokenizer、Scheduler
2. **请求处理**：`add_request()` 将提示转为 token，创建 Sequence，添加到调度队列
3. **推理循环**：`step()` 调度序列 → 执行推理 → 更新状态
4. **生成接口**：`generate()` 批量处理提示，循环调用 step() 直到完成

#### sequence.py - 序列数据结构

```python
class Sequence:
    seq_id: int                    # 序列唯一标识
    status: SequenceStatus         # 状态：WAITING/RUNNING/FINISHED
    token_ids: list[int]           # 所有 token IDs
    num_cached_tokens: int         # 已缓存 token 数（前缀缓存）
    block_table: list[int]         # 分配的 KV 缓存块 ID 列表
    # 采样参数
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    max_tokens: int
    ignore_eos: bool
```

#### scheduler.py - 批次调度器

**两阶段调度策略**：

1. **Prefill 阶段**（首次处理）
   - 从 `waiting` 队列选择序列
   - 检查 `max_num_seqs` 和 `max_num_batched_tokens` 限制
   - 分配 KV 缓存块（利用前缀缓存）
   - 状态改为 `RUNNING`

2. **Decode 阶段**（生成阶段）
   - 从 `running` 队列选择序列
   - 检查是否需要追加 KV 缓存块
   - 若块不足，考虑抢占其他序列

3. **Postprocess 阶段**
   - 添加新生成的 token
   - 检查是否完成（达到 `max_tokens` 或生成 EOS）
   - 释放资源，状态改为 `FINISHED`

#### block_manager.py - KV 缓存块管理器

**核心功能**：
- 管理 KV 缓存的分配、释放和共享
- 实现前缀缓存优化

```python
class Block:
    block_id: int          # 块唯一标识
    ref_count: int         # 引用计数
    hash: int              # 块哈希值（前缀匹配）
    token_ids: list[int]   # 块中包含的 token IDs
```

**核心方法**：
- `allocate(seq)`：分配 KV 缓存，优先复用前缀缓存
- `deallocate(seq)`：释放 KV 缓存，引用计数为 0 时回收
- `may_append(seq)`：Decode 阶段检查是否需要新块

#### model_runner.py - 模型运行器

**核心职责**：
1. **初始化**：分布式初始化、加载模型、分配 KV 缓存、捕获 CUDA 图
2. **多进程支持**：Rank 0 协调执行，Rank > 0 接收命令执行计算
3. **KV 缓存分配**：根据可用内存计算块数，创建 KV 缓存张量
4. **数据准备**：
   - Prefill：处理变长序列，准备 `cu_seqlens`、`slot_mapping`
   - Decode：提取每个序列最后一个 token
5. **CUDA 图优化**：为不同 batch size 捕获计算图

#### distributed_manager.py - 分布式管理器

**新增模块**，替代原有的简单多进程通信：

```python
class DistributedManager:
    """管理分布式推理协调和通信"""
    config: Config
    rank: int                      # 当前进程 rank
    world_size: int                # 总进程数
    backend: str                   # 分布式后端（nccl/hccl/gloo）
    is_distributed: bool           # 是否分布式模式
```

**核心功能**：
- 自动检测并设置分布式后端（NCCL for CUDA，HCCL for NPU）
- 提供 `broadcast_data()`、`gather_data()`、`all_gather_data()` 数据通信
- 进程同步和协调
- 支持 pickle 序列化传输复杂数据

### 3.3 模型实现 (`models/`)

#### qwen3.py - Qwen3 模型实现

**架构组成**：

1. **Qwen3Attention** - 多头注意力层
   - QKV 投影
   - 分组查询注意力 (GQA)
   - 旋转位置编码 (RoPE)
   - FlashAttention 加速
   - 输出投影

2. **Qwen3MLP** - 前馈网络
   - 门控上升投影
   - SiLU 激活 + Mul 操作
   - 下降投影

3. **Qwen3DecoderLayer** - Transformer 解码块
   ```
   Input -> RMSNorm -> Self-Attention -> Add & Norm ->
   RMSNorm -> MLP -> Add & Norm -> Output
   ```

4. **Qwen3ForCausalLM** - 因果语言模型
   - 完整模型结构
   - LM Head 将隐藏状态映射到词汇表
   - 权重绑定（词嵌入与输出层共享）

### 3.4 神经网络层 (`models/layers/`)

#### attention.py - 注意力机制

```python
class Attention(nn.Module):
    def forward(self, q, k, v, k_cache, v_cache, slot_mapping):
        # 1. 存储 K, V 到 KV 缓存
        store_kvcache_kernel(k, v, k_cache, v_cache, slot_mapping)

        # 2. Prefill 阶段
        if is_prefill:
            if block_tables is not None:  # 有前缀缓存
                o = flash_attn_varlen_func(q, cached_k, cached_v, ...)
            else:
                o = flash_attn_varlen_func(q, k, v, ...)
        # 3. Decode 阶段
        else:
            o = flash_attn_with_kvcache(q, k_cache, v_cache, ...)
        return o
```

#### linear.py - 张量并行线性层

```
LinearBase (基类)
├── ReplicatedLinear          # 复制线性层（所有 GPU 相同权重）
├── ColumnParallelLinear      # 列并行（输出分片）
│   ├── MergedColumnParallelLinear  # 合并列并行
│   └── QKVParallelLinear    # QKV 投影专用
└── RowParallelLinear         # 行并行（输入分片，all-reduce）
```

#### rotary_embedding.py - 旋转位置编码

```python
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096):
        # 预计算所有位置的 cos/sin 值
        self.cos_sin_cache = self._compute_cos_sin_cache(...)

    def forward(self, positions, q, k):
        # 查找预计算的 cos/sin 值
        cos_sin = self.cos_sin_cache[positions]
        # 应用旋转位置编码
        q = apply_rotary_emb(q, cos_sin[:, 0], cos_sin[:, 1])
        k = apply_rotary_emb(k, cos_sin[:, 0], cos_sin[:, 1])
        return q, k
```

#### npu_flash_attention.py - NPU Flash Attention

**NPU 专用优化**：
- 使用 transformers 库中的 NPU flash attention 内核
- BNSD 数据布局（Batch, Num heads, Sequence, Dim）
- 自动检测 NPU 设备并启用优化

### 3.5 采样模块 (`sampling/`)

#### sampler.py - 统一采样器

**重构后的统一采样器**，替代原有的分散实现：

```python
class Sampler(nn.Module):
    """
    统一采样器，支持多种采样策略：
    - 贪婪采样 (Greedy)
    - 随机采样 (Random)
    - Top-K 采样
    - Top-P (Nucleus) 采样
    - Min-P 采样
    - Typical 采样
    - 重复惩罚
    """

    def forward(self, logits, config=None,
                temperatures=None, top_ks=None, top_ps=None,
                min_ps=None, typical_ps=None, generator=None):
        # 1. 应用惩罚
        if prev_tokens is not None:
            logits = apply_repetition_penalty(...)
            logits = apply_frequency_penalty(...)

        # 2. 应用温度
        logits = apply_temperature(logits, temp)

        # 3. 应用各种过滤
        logits = apply_typical_filtering(logits, typ_p)
        logits = apply_top_k(logits, k)
        logits = apply_top_p(logits, p)
        logits = apply_min_p(logits, mp)

        # 4. 采样
        return sample_from_logits(logits, generator=generator)
```

**采样策略说明**：

| 策略 | 参数 | 说明 |
|------|------|------|
| Greedy | temperature→0 | 选择概率最高的 token |
| Top-K | top_k | 只考虑概率最高的 K 个 token |
| Top-P | top_p | 只考虑累积概率达到 P 的 token |
| Min-P | min_p | 只考虑相对最大概率不小于 P 的 token |
| Typical | typical_p | 基于熵的典型采样 |
| 重复惩罚 | repetition_penalty | 降低重复 token 的概率 |

#### config.py - 采样配置

```python
@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 0           # 0 表示禁用
    top_p: float = 1.0       # 1.0 表示禁用
    min_p: float = 0.0       # 0.0 表示禁用
    typical_p: float = 1.0   # 1.0 表示禁用
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    avoid_top_k: int = 0
    seed: int | None = None
```

### 3.6 工具模块 (`utils/`)

#### device.py - 设备检测和管理

```python
def get_device_type() -> str:
    """自动检测设备类型：cuda/npu/xpu/mps/mlu/musa/cpu"""

def get_distributed_backend() -> str:
    """获取分布式后端：nccl(cuda)/hccl(npu)/gloo(cpu)"""

def get_device_module(device_type: str):
    """获取设备模块（torch.cuda/torch_npu等）"""
```

#### context.py - 全局上下文管理

```python
@dataclass
class Context:
    is_prefill: bool         # 当前是否 Prefill 阶段
    cu_seqlens_q: Tensor     # Query 累积序列长度
    cu_seqlens_k: Tensor     # Key 累积序列长度
    max_seqlen_q: int        # 最大 Query 长度
    max_seqlen_k: int        # 最大 Key 长度
    slot_mapping: Tensor     # KV 缓存槽位映射
    context_lens: Tensor     # 每个序列的上下文长度
    block_tables: Tensor     # 块表（前缀缓存查找）
```

### 3.7 使用示例

#### 基本使用

```python
from minivllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    max_num_seqs=256,
    device_memory_utilization=0.9
)

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_tokens=128
)

# 生成文本
outputs = llm.generate(
    ["Once upon a time", "The future of AI is"],
    sampling_params
)

for output in outputs:
    print(f"Prompt: {output['prompt']}")
    print(f"Generated: {output['text']}")
```

#### 高级配置

```python
# 多 GPU 配置
llm = LLM(
    model="Qwen/Qwen2-72B-Instruct",
    tensor_parallel_size=4,      # 4 张 GPU
    max_num_seqs=512,
    device_memory_utilization=0.85,
    enforce_eager=False,          # 启用 CUDA Graph
    max_model_len=4096
)

# 自定义采样参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    min_p=0.05,                   # 新增参数
    max_tokens=256,
    ignore_eos=True
)
```

## 四、代码逻辑流程（调用关系）

### 4.1 初始化流程

```
User: llm = LLM(model_path, ...)
  │
  └─> LLMEngine.__init__()
      ├─> Config(model, **kwargs)
      │   ├─> 验证模型路径
      │   ├─> 验证参数范围
      │   └─> 加载 HuggingFace config
      │
      ├─> DistributedManager(config, rank=0)  [新增]
      │   └─> 初始化分布式后端
      │
      ├─> 创建 ModelRunner 进程 (rank > 0)
      │
      ├─> ModelRunner(config, rank=0)
      │   ├─> 加载模型
      │   ├─> warmup_model()
      │   ├─> allocate_kv_cache()
      │   └─> capture_cudagraph() (if not enforce_eager)
      │
      ├─> AutoTokenizer.from_pretrained()
      │
      └─> Scheduler(config)
          └─> BlockManager(num_blocks, block_size)
```

### 4.2 生成流程

```
User: outputs = llm.generate(prompts, sampling_params)
  │
  └─> LLMEngine.generate()
      ├─> Tokenize prompts
      │
      ├─> For each prompt:
      │   └─> add_request(prompt, sp)
      │       └─> Sequence(token_ids, sp)
      │       └─> scheduler.add(seq)
      │
      └─> While not is_finished():
          │
          ├─> step()
          │   │
          │   ├─> scheduler.schedule()
          │   │   ├─> PREFILL: 从 waiting 队列选择序列，分配 KV 缓存
          │   │   └─> DECODE: 从 running 队列选择序列，检查块空间
          │   │
          │   ├─> model_runner.call("run", seqs, is_prefill)
          │   │   ├─> prepare_prefill/decode()
          │   │   ├─> run_model()
          │   │   │   ├─> model.forward()
          │   │   │   │   ├─> embed_tokens
          │   │   │   │   ├─> For each layer:
          │   │   │   │   │   ├─> self_attn (FlashAttention)
          │   │   │   │   │   └─> mlp
          │   │   │   │   └─> compute_logits
          │   │   │   └─> (CUDA Graph if enabled)
          │   │   ├─> sampler(logits, temperatures)
          │   │   └─> reset_context()
          │   │
          │   └─> scheduler.postprocess(seqs, token_ids)
          │       └─> 更新序列状态，检查完成条件
          │
          └─> Collect outputs and decode
```

### 4.3 张量并行流程（多 GPU）

```
ModelRunner(rank=0):
  │
  └─> run(seqs, is_prefill):
      ├─> prepare_prefill/decode()
      ├─> distributed_manager.broadcast_data()  [同步数据]
      ├─> run_model()
      │   ├─> embed_tokens (VocabParallelEmbedding)
      │   │   └─> all-reduce
      │   ├─> For each layer:
      │   │   ├─> qkv_proj (ColumnParallelLinear)
      │   │   ├─> attention
      │   │   ├─> o_proj (RowParallelLinear)
      │   │   │   └─> all-reduce
      │   │   ├─> gate_up_proj (MergedColumnParallelLinear)
      │   │   └─> down_proj (RowParallelLinear)
      │   │       └─> all-reduce
      │   └─> lm_head (ParallelLMHead)
      │       └─> gather at rank 0
      └─> distributed_manager.synchronize()

ModelRunner(rank > 0):
  │
  └─> loop():
      └─> While True:
          ├─> distributed_manager.broadcast_data() [接收命令]
          ├─> 执行模型运算
          └─> distributed_manager.synchronize()
```

### 4.4 前缀缓存工作流

```
Sequence 1: [A, B, C, D, E] (prompt tokens)
Sequence 2: [A, B, C, F, G] (same prefix)

allocate(Sequence 1):
  Block 0: [A, B, C, D, E]
    ├─> hash = compute_hash([A, B, C, D, E])
    ├─> Allocate new block_id=0
    └─> hash_to_block_id[hash0] = 0

allocate(Sequence 2):
  Block 0: [A, B, C, F, G]
    ├─> hash_prefix = compute_hash([A, B, C])
    ├─> Match! hash_to_block_id[hash_prefix] = 0
    ├─> Reuse block_id=0 (ref_count += 1)
    └─> num_cached_tokens += 3

During forward:
  ├─> Sequence 1: Process [A, B, C, D, E] → store KV in block 0
  ├─> Sequence 2: Process [A, B, C, F, G]
  │   ├─> KV for [A, B, C] already cached
  │   └─> Only compute KV for [F, G]
  └─> Both share KV cache for common prefix
```

## 五、关键优化总结

| 优化技术 | 位置 | 作用 |
|---------|------|------|
| **前缀缓存** | BlockManager | 相同前缀共享 KV 缓存，减少存储和重复计算 |
| **块化 KV 缓存** | BlockManager, Attention | 灵活管理内存，支持序列长度变化 |
| **张量并行** | Linear 层, embed_head | 多 GPU 分布式推理，扩大吞吐量 |
| **CUDA 图** | ModelRunner | 捕获 decode 图，减少 CPU 开销 |
| **FlashAttention** | Attention | 优化的注意力实现，提升速度 2-4 倍 |
| **统一采样器** | Sampler | 支持多种采样策略，灵活配置 |
| **RMSNorm 融合** | RMSNorm | 融合加法和归一化，减少内存访问 |
| **分布式管理器** | DistributedManager | 统一多设备通信，支持 NCCL/HCCL/Gloo |

## 六、性能特点

### 代码规模对比

| 指标 | mini-vLLM | vLLM |
|------|-----------|------|
| 核心代码行数 | ~3,000 | ~50,000 |
| Python 文件数 | ~35 | ~500+ |
| 核心依赖 | 5 | 20+ |

### 内存效率

| 优化技术 | 内存节省 | 性能提升 |
|---------|---------|---------|
| 块化 KV 缓存 | ~20% | ~10% |
| 前缀缓存 | 最高 ~50% | ~15% |
| FlashAttention | ~30% | 2-4x |

### 扩展性

- **张量并行**：支持 1-8 个 GPU
- **模型支持**：Qwen2/Qwen3、OPT，易于扩展
- **多设备**：CUDA、NPU、XPU、MPS、MLU、MUSA

## 七、使用建议与最佳实践

### 模式选择

| 模式 | 参数 | 适用场景 |
|------|------|---------|
| Eager 模式 | `enforce_eager=True` | 调试和开发，便于查看中间结果 |
| 图模式 | `enforce_eager=False` | 生产环境，最大化性能 |

### 内存优化建议

- **device_memory_utilization**：
  - 内存充足：0.9
  - 内存紧张：0.7-0.8
- **kvcache_block_size**：
  - 长序列：512 或 1024
  - 短序列：256（默认）
- **批处理配置**：
  - 长提示：减少 `max_num_seqs`，增加 `max_num_batched_tokens`
  - 短提示：增加 `max_num_seqs`

### 采样参数调整

| 温度 | 效果 | 适用场景 |
|------|------|---------|
| 0.1-0.5 | 确定性高 | 准确结果、代码生成 |
| 0.6-1.0 | 平衡 | 大多数场景 |
| 1.0-1.5 | 创意性高 | 创意写作、头脑风暴 |

### 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| CUDA OOM | 显存不足 | 降低 `device_memory_utilization`，减少 `max_num_seqs`，启用张量并行 |
| 推理速度慢 | 未启用优化 | 设置 `enforce_eager=False`，启用 FlashAttention |
| 生成质量差 | 采样参数不当 | 调整 `temperature`、`top_p`、`top_k` |

## 八、多设备支持指南

### CUDA (NVIDIA GPU)

```python
llm = LLM(
    model="Qwen/Qwen2-7B",
    device_memory_utilization=0.9
)
```

- 完整支持：FlashAttention、CUDA Graph、张量并行
- 分布式后端：NCCL

### NPU (华为昇腾)

```python
llm = LLM(
    model="Qwen/Qwen2-7B",
    device_memory_utilization=0.9
)
```

- 支持 NPU Flash Attention（transformers 库）
- BNSD 数据布局优化
- 分布式后端：HCCL

### 其他设备 (XPU/MPS/MLU/MUSA)

```python
llm = LLM(
    model="Qwen/Qwen2-7B",
    device_memory_utilization=0.9
)
```

- 基础支持通过设备抽象层
- 自动检测设备类型

### CPU 推理

```python
llm = LLM(
    model="Qwen/Qwen2-1.5B",  # 使用小模型
    device_memory_utilization=0.9
)
```

- 适用于小模型或测试
- 分布式后端：Gloo

## 九、许可证与致谢

### 许可证

本项目基于 [Apache 2.0 许可证](LICENSE) 开源。

### 致谢

- **nano-vllm**：提供宝贵的设计思路和实现参考
- **vLLM**：灵感来源和架构参考
- **Transformers**：模型定义和加载
- **PyTorch**：深度学习框架
- **FlashAttention**：高性能注意力实现

## 十、总结

### 核心优势

1. **简单易用**：统一 API 接口，隐藏底层复杂性
2. **高性能**：集成多种优化技术，速度接近主流框架
3. **易理解**：代码结构清晰，适合学习推理原理
4. **灵活扩展**：模块化设计，便于添加新功能
5. **多设备支持**：CUDA、NPU 等多种后端

### 适用场景

- 学习和研究大语言模型推理
- 快速原型开发和验证
- 轻量级部署和边缘设备
- 定制化和二次开发

### 未来展望

- 支持更多模型架构（LLaMA 3、Mistral 等）
- 流式生成支持
- 更完善的多模态支持
- 容器化和 K8s 部署
