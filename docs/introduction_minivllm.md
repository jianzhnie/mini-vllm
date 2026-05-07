# `mini`-vLLM：轻量级大语言模型推理引擎

## 目录

- [一、仓库概览](#一仓库概览)
  - [1.1 项目背景与定位](#11-项目背景与定位)
  - [1.2 核心特性](#12-核心特性)
  - [1.3 项目信息](#13-项目信息)
  - [1.4 依赖环境](#14-依赖环境)
  - [1.5 安装步骤](#15-安装步骤)
  - [1.6 仓库目录结构](#16-仓库目录结构)
- [二、核心流程与优化技术](#二核心流程与优化技术)
  - [2.1 推理流程概述](#21-推理流程概述)
  - [2.2 核心优化技术](#22-核心优化技术)
- [三、代码文件详细解释](#三代码文件详细解释)
  - [3.1 入口和配置](#31-入口和配置)
  - [3.2 引擎核心模块 (engine/)](#32-引擎核心模块-engine)
  - [3.3 模型实现 (models/)](#33-模型实现-models)
  - [3.4 神经网络层 (models/layers/)](#34-神经网络层-modelslayers)
  - [3.5 采样模块 (sampling/)](#35-采样模块-sampling)
  - [3.6 工具模块 (utils/)](#36-工具模块-utils)
- [四、代码逻辑流程](#四代码逻辑流程)
  - [4.1 初始化流程](#41-初始化流程)
  - [4.2 生成流程](#42-生成流程)
  - [4.3 张量并行流程](#43-张量并行流程)
  - [4.4 前缀缓存工作流](#44-前缀缓存工作流)
- [五、使用指南](#五使用指南)
  - [5.1 基本用法](#51-基本用法)
  - [5.2 高级配置](#52-高级配置)
  - [5.3 多设备支持](#53-多设备支持)
  - [5.4 参数调优建议](#54-参数调优建议)
  - [5.5 故障排除](#55-故障排除)
- [六、许可证与致谢](#六许可证与致谢)

---

## 一、仓库概览

### 1.1 项目背景与定位

mini-vLLM 是一个**从零构建的轻量级大语言模型推理引擎**，灵感来源于 vLLM。项目在保持与主流推理引擎相当性能的同时，特别注重**代码的可读性和可理解性**。

**解决的核心痛点**：

| 痛点 | 说明 |
|------|------|
| 代码复杂 | 主流框架追求极致性能导致结构复杂、依赖众多 |
| 学习门槛高 | 初学者理解内部原理需要大量时间 |
| 定制困难 | 代码复杂度高，按需修改不便 |
| 资源消耗大 | 部署运行需要大量计算资源 |

**适用场景**：学习推理原理、快速原型开发、轻量级部署、研究实验、二次开发定制。

### 1.2 核心特性

**高性能推理**

- **CUDA Graph**：录制 decode 阶段 GPU 操作为计算图，减少调度开销
- **智能批处理**：两阶段调度（Prefill + Decode），动态序列长度感知
- **KV 缓存**：块式管理 + xxhash 前缀缓存，减少内存占用和重复计算

**先进架构**

- **多设备**：CUDA、NPU、XPU、MPS、MLU、MUSA
- **张量并行**：1-8 GPU 分布式推理（NCCL/HCCL/Gloo 后端）
- **FlashAttention**：高性能注意力计算，支持变长序列

**灵活配置**

- 设备内存利用率 10%-100% 可调
- KV 缓存块大小和数量可配置
- 丰富的采样参数（temperature / top-p / top-k / min-p）

### 1.3 项目信息

| 属性 | 值 |
|------|-----|
| 语言 | Python 3.10–3.12 |
| 核心代码 | ~9,100 行 / 41 个 Python 文件 |
| 核心依赖 | torch, transformers, safetensors, xxhash |
| 许可证 | Apache 2.0 |
| 支持模型 | Qwen2、Qwen3、OPT |

### 1.4 依赖环境

```
torch>=2.2.0            # 深度学习框架
transformers>=4.37.0    # HuggingFace 模型加载
safetensors>=0.4.0      # 权重文件格式
xxhash                  # 前缀缓存哈希
colorama                # 终端彩色输出
numpy<2                 # 数值计算
tqdm>=4.62.0            # 进度条

# 可选（CUDA）
flash-attn>=2.6.0       # 高性能注意力
triton>=2.3.0           # GPU 内核编译

# 可选（NPU）
torch-npu>=2.2.0        # 华为昇腾支持
```

### 1.5 安装步骤

```bash
git clone https://github.com/jianzhnie/mini-vllm.git
cd mini-vllm

pip install -e .              # 基础安装
pip install -e ".[cuda]"      # CUDA 优化版本
pip install -e ".[npu]"       # NPU 版本
pip install -e ".[dev]"       # 开发依赖（pytest, ruff, black, mypy）

# 验证
python -c "import minivllm; print('OK')"
```

### 1.6 仓库目录结构

```
mini-vllm/
├── minivllm/                       # 核心包
│   ├── __init__.py                 # 导出核心类
│   ├── llm.py                      # LLM 用户接口
│   ├── config.py                   # Config 配置类
│   ├── sampling_params.py          # SamplingParams 采样参数
│   ├── engine/                     # 引擎核心
│   │   ├── llm_engine.py           # LLMEngine 编排器
│   │   ├── scheduler.py            # 两阶段调度器
│   │   ├── block_manager.py        # KV 缓存块管理
│   │   ├── model_runner.py         # 模型运行器（顶层协调）
│   │   ├── inference_executor.py   # 推理执行（KV 分配/CUDA Graph/采样）
│   │   ├── distributed_manager.py  # 多进程通信协调
│   │   └── sequence.py             # 序列状态
│   ├── models/                     # 模型实现
│   │   ├── __init__.py             # 导出模型类和工厂函数
│   │   ├── registry.py             # 模型注册表 + 工厂函数
│   │   ├── manager.py              # 模型加载和生命周期管理
│   │   ├── qwen_base.py            # Qwen 系列共享基类
│   │   ├── qwen2.py                # Qwen2ForCausalLM
│   │   ├── qwen3.py                # Qwen3ForCausalLM
│   │   ├── opt.py                  # OPTForCausalLM
│   │   └── layers/                 # 神经网络层
│   │       ├── attention.py        # FlashAttention + 回退实现
│   │       ├── attention_backend.py # 注意力后端抽象
│   │       ├── rotary_embedding.py  # RoPE 旋转位置编码
│   │       ├── linear.py           # 张量并行线性层
│   │       ├── layernorm.py        # RMSNorm
│   │       ├── activation.py       # SiLU + Mul
│   │       ├── embed_head.py       # 并行 Embedding / LM Head
│   │       └── npu_flash_attention.py # NPU 专用注意力
│   ├── sampling/                   # 采样模块
│   │   ├── sampler.py              # 统一采样器
│   │   ├── functional.py           # 无状态采样操作（torch.compile）
│   │   ├── config.py               # 内部采样配置
│   │   └── mirostat.py             # Mirostat 采样器
│   └── utils/                      # 工具函数
│       ├── device.py               # 多设备抽象
│       ├── context.py              # 推理上下文（contextvars）
│       ├── loader.py               # 权重加载（safetensors + .bin）
│       ├── memory_pool.py          # 内存池管理
│       ├── random_utils.py         # 跨库随机种子设置
│       ├── logger_utils.py         # 分布式感知日志
│       └── example_utils.py        # 示例辅助函数
├── tests/                          # 测试目录
├── examples/                       # 示例代码
├── .github/workflows/              # CI（pytests.yml + code-quality.yml）
└── pyproject.toml                  # 项目配置
```

---

## 二、核心流程与优化技术

### 2.1 推理流程概述

推理分为六个阶段：

```
初始化 → 请求处理 → 调度 → 推理 → 采样 → 输出
```

1. **初始化**：加载模型权重 → 分配 KV 缓存 → 捕获 CUDA Graph → 初始化分布式
2. **请求处理**：接收请求 → tokenize → 创建 Sequence → 入等待队列
3. **调度**：
   - **Prefill**：首次处理提示词，填充 KV 缓存
   - **Decode**：逐 token 生成，复用已有 KV 缓存
4. **推理**：执行前向传播，计算 logits
5. **采样**：penalties → temperature → top-k → top-p → min-p → multinomial
6. **输出**：decode tokens → 返回结果 → 释放资源

### 2.2 核心优化技术

#### 前缀缓存 (Prefix Caching)

多个请求共享相同前缀时，复用已计算的 KV 缓存块：

- 每个块计算 xxhash 哈希值，建立 `hash → block_id` 映射
- 新请求匹配前缀哈希，跳过重复计算
- 引用计数管理块生命周期，安全共享和回收

#### 块化 KV 缓存 (Block-wise KV Cache)

- 固定大小块（默认 64 tokens，必须被 64 整除）
- 每个序列维护块表（block_table），动态分配/释放
- 减少内存碎片化，支持变长序列

#### 张量并行 (Tensor Parallelism)

- 模型权重按列/行切分到多 GPU
- ColumnParallelLinear：输出分片
- RowParallelLinear：输入分片 + all-reduce 聚合
- 支持 1-8 GPU，突破单卡内存限制

#### CUDA Graph 优化

- 捕获 decode 阶段 GPU 操作为静态计算图
- 为 batch sizes [1, 2, 4, 8, 16, 32, ..., max_num_seqs] 预录制
- 重放时跳过 CPU 调度开销，可通过 `enforce_eager=True` 禁用

#### FlashAttention

- 分块计算注意力矩阵，优化内存访问
- 支持 `flash_attn_varlen_func`（Prefill）和 `flash_attn_with_kvcache`（Decode）
- 自动回退到 PyTorch SDPA

---

## 三、代码文件详细解释

### 3.1 入口和配置

#### `llm.py` — LLM 用户接口

```python
class LLM(LLMEngine):
    """用户直接使用的接口，继承 LLMEngine"""
```

提供 `generate()` 方法进行批量推理，隐藏底层调度、分布式、CUDA Graph 等复杂性。

#### `config.py` — Config 配置类

```python
@dataclass
class Config:
    model: str                          # 模型路径或 HF ID
    max_num_batched_tokens: int = 16384 # 每批最大 token 数
    max_num_seqs: int = 512             # 每批最大序列数
    max_model_len: int = 4096           # 最大序列长度
    device_memory_utilization: float = 0.9  # 设备内存利用率
    tensor_parallel_size: int = 1       # 张量并行数
    enforce_eager: bool = False         # 禁用 CUDA Graph
    kvcache_block_size: int = 64        # KV 缓存块大小
    dtype: str = 'auto'                 # 权重数据类型
```

关键校验：`kvcache_block_size % 64 == 0`，`tensor_parallel_size ∈ [1, 8]`，`device_memory_utilization ∈ [0.1, 1.0]`，`max_num_batched_tokens >= max_model_len`。

#### `sampling_params.py` — 用户采样参数

```python
@dataclass
class SamplingParams:
    temperature: float = 1.0    # >= 0
    top_p: float = 1.0          # (0, 1.0]
    top_k: int = -1             # -1 禁用，或 > 0
    min_p: float = 0.0          # [0, 1.0]
    max_tokens: int = 64        # > 0
    ignore_eos: bool = False
```

### 3.2 引擎核心模块 (engine/)

#### `llm_engine.py` — LLMEngine 编排器

核心职责：
1. **初始化**：创建 Config → 启动多进程 → 初始化 ModelRunner → 加载 Tokenizer → 创建 Scheduler
2. **请求处理**：`add_request()` tokenize 提示 → 创建 Sequence → 入调度队列
3. **推理循环**：`step()` 调度序列 → 执行推理 → 更新状态
4. **生成接口**：`generate()` 批量处理，循环 step() 直到全部完成

#### `sequence.py` — Sequence 状态

```python
class Sequence:
    seq_id: int                    # 唯一标识
    status: SequenceStatus         # WAITING / RUNNING / FINISHED
    token_ids: list[int]           # 所有 token IDs
    num_cached_tokens: int         # 前缀缓存命中数
    block_table: list[int]         # KV 缓存块 ID 列表
```

#### `scheduler.py` — 两阶段调度器

1. **Prefill**：从 `waiting` 队列选取序列 → 检查 `max_num_seqs` 和 `max_num_batched_tokens` → 分配 KV 缓存块 → 状态改为 RUNNING
2. **Decode**：从 `running` 队列选取序列 → 检查块空间 → 空间不足时抢占最新序列
3. **Postprocess**：添加生成 token → 检查完成条件（max_tokens 或 EOS）→ 释放资源

#### `block_manager.py` — KV 缓存块管理

```python
class Block:
    block_id: int          # 块 ID
    ref_count: int         # 引用计数
    hash: int              # 前缀哈希
    token_ids: list[int]   # 包含的 tokens
```

核心方法：
- `allocate(seq)`：分配 KV 缓存，优先复用前缀缓存
- `deallocate(seq)`：释放缓存，引用计数归零时回收
- `may_append(seq)`：Decode 阶段检查是否需要新块

#### `model_runner.py` — 模型运行器

顶层协调器，委托给 ModelManager、DistributedManager、InferenceExecutor：
- 分布式初始化、模型加载、KV 缓存分配、CUDA Graph 捕获
- Rank 0 协调执行，Rank > 0 作为 worker 进程

#### `inference_executor.py` — 推理执行器

- KV 缓存张量分配和 warmup
- Prefill/Decode 数据准备（slot_mapping、block_tables、cu_seqlens）
- 批量前向传播 + CUDA Graph 重放
- 调用 Sampler 生成 token

#### `distributed_manager.py` — 分布式管理器

```python
class DistributedManager:
    rank: int           # 当前进程 rank
    world_size: int     # 总进程数
    backend: str        # nccl / hccl / ccl / gloo
```

自动选择后端（CUDA → NCCL，NPU → HCCL），提供 broadcast/gather/all_gather 数据通信。

### 3.3 模型实现 (models/)

#### `registry.py` — 模型注册表

`create_model(hf_config)` 工厂函数，根据 HF config 的 `architectures` 或 `model_type` 查找并实例化对应模型类。

注册表 `SUPPORTED_MODELS`：
- `Qwen2ForCausalLM` → `qwen2.py`
- `Qwen3ForCausalLM` → `qwen3.py`
- `OPTForCausalLM` → `opt.py`

`TYPE_TO_ARCH` 提供从 `model_type` 到架构名的映射。

#### `qwen_base.py` — Qwen 共享基类

Qwen2 和 Qwen3 共享相同的 Transformer 架构，差异仅在默认参数：
- `qkv_bias`：Qwen2 默认 True，Qwen3 默认 False
- `rope_theta`：Qwen2 默认 1000000，Qwen3 默认 10000
- `attention_bias` config fallback：Qwen2 默认 True，Qwen3 默认 False

包含 `QwenModel`、`QwenAttention`、`QwenMLP`、`QwenDecoderLayer`、`QwenForCausalLM` 基类。

#### `manager.py` — ModelManager

模型加载、架构检测（`_detect_model_type()`）、生命周期管理。通过 `registry.create_model()` 实例化模型。

#### 添加新模型架构

1. 在 `minivllm/models/` 创建模型类（参考 `qwen2.py`）
2. 在 `registry.py` 的 `SUPPORTED_MODELS` 和 `TYPE_TO_ARCH` 中注册
3. 在 `manager.py` 的 `_detect_model_type()` 添加检测逻辑

### 3.4 神经网络层 (models/layers/)

#### `attention.py` — 注意力机制

```python
class Attention(nn.Module):
    def forward(self, q, k, v):
        context = get_context()
        self.backend.store_kv_cache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
        if context.is_prefill:
            o = flash_attn_varlen_func(q, k, v, ...)       # Prefill
        else:
            o = flash_attn_with_kvcache(q, self.k_cache, self.v_cache, ...)  # Decode
```

#### `attention_backend.py` — 注意力后端

ABC 基类 + 多后端实现（FlashAttention、SDPA、NPU Flash Attention）。

#### `linear.py` — 张量并行线性层

```
LinearBase
├── ReplicatedLinear              # 复制层（所有 GPU 相同权重）
├── ColumnParallelLinear          # 列并行（输出分片）
│   ├── MergedColumnParallelLinear  # gate_up 合并列并行
│   └── QKVParallelLinear           # QKV 投影专用
└── RowParallelLinear             # 行并行（输入分片 + all-reduce）
```

#### 其他层

| 文件 | 功能 |
|------|------|
| `rotary_embedding.py` | RoPE 旋转位置编码，预计算 cos/sin cache |
| `layernorm.py` | RMSNorm 归一化层（含 NPU 加速） |
| `activation.py` | SiLU + Mul 门控激活 |
| `embed_head.py` | VocabParallelEmbedding + ParallelLMHead |
| `npu_flash_attention.py` | NPU 专用 FlashAttention（BNSD 布局）|

### 3.5 采样模块 (sampling/)

#### `sampler.py` — 统一采样器

```python
class Sampler(nn.Module):
    def forward(self, logits, ...):
        # penalties → avoid_top_k → temperature → typical
        # → top_k → top_p → min_p → multinomial
```

| 策略 | 参数 | 说明 |
|------|------|------|
| Top-K | top_k | 只保留概率最高的 K 个 |
| Top-P | top_p | 累积概率达到 P 截断 |
| Min-P | min_p | 相对最大概率低于阈值截断 |
| Typical | typical_p | 基于信息熵的典型采样 |
| 重复惩罚 | repetition_penalty | 降低已出现 token 概率 |

#### `config.py` — SamplingConfig（内部配置）

在用户 `SamplingParams` 基础上增加 `repetition_penalty`、`frequency_penalty`、`presence_penalty`、`typical_p`、`avoid_top_k`、`seed`。这些参数存在于采样管线但未从用户接口暴露。

#### `functional.py` — 无状态采样操作

`apply_temperature`、`apply_top_k`、`apply_top_p` 等，CUDA 上自动 `torch.compile`。

#### `mirostat.py` — Mirostat 采样器

独立的状态化采样器（MirostatSampler、MirostatV2Sampler），未接入主采样管线。

### 3.6 工具模块 (utils/)

| 文件 | 功能 |
|------|------|
| `device.py` | 多设备抽象（CUDA/NPU/XPU/MPS/MLU/MUSA/CPU），检测、内存、后端选择 |
| `context.py` | `contextvars` 线程安全推理上下文（is_prefill、slot_mapping、block_tables） |
| `loader.py` | 权重加载（safetensors + .bin），packed module 映射（q/k/v → qkv），HF ID 下载 |
| `memory_pool.py` | 内存池管理 |
| `random_utils.py` | 跨库随机种子设置（Python/numpy/torch/CUDA/NPU） |
| `logger_utils.py` | 彩色日志，分布式感知（仅 rank 0 输出 INFO） |
| `example_utils.py` | 示例脚本辅助函数 |

---

## 四、代码逻辑流程

### 4.1 初始化流程

```
llm = LLM(model, ...)
  │
  └─> LLMEngine.__init__()
      ├─> Config(model, **kwargs)           # 参数校验 + 加载 HF config
      ├─> DistributedManager(config, rank=0) # 初始化分布式后端
      ├─> 启动 ModelRunner worker 进程 (rank > 0)
      ├─> ModelRunner(config, rank=0)
      │   ├─> ModelManager: 加载模型
      │   ├─> InferenceExecutor: warmup → allocate_kv_cache → capture_cudagraph
      │   └─> DistributedManager: 注册 worker
      ├─> AutoTokenizer.from_pretrained()
      └─> Scheduler(config)
          └─> BlockManager(num_blocks, block_size)
```

### 4.2 生成流程

```
outputs = llm.generate(prompts, sampling_params)
  │
  └─> LLMEngine.generate()
      ├─> tokenize prompts
      ├─> for each prompt: add_request() → Sequence → scheduler.add()
      │
      └─> while not is_finished():
          └─> step()
              ├─> scheduler.schedule()
              │   ├─> PREFILL: waiting → running, 分配 KV 缓存
              │   └─> DECODE: running 队列, 检查块空间
              │
              ├─> model_runner.call("run", seqs, is_prefill)
              │   ├─> prepare_prefill/decode()
              │   ├─> model.forward()
              │   │   ├─> embed_tokens
              │   │   ├─> for each layer: self_attn → mlp
              │   │   └─> compute_logits
              │   ├─> sampler(logits)
              │   └─> (CUDA Graph replay if enabled)
              │
              └─> scheduler.postprocess(seqs, token_ids)
                  └─> 更新状态, 检查完成条件, 释放资源
```

### 4.3 张量并行流程

```
Rank 0 (主进程):
  run(seqs, is_prefill)
  ├─> prepare_prefill/decode()
  ├─> broadcast_data()                    # 同步输入数据
  ├─> model.forward()
  │   ├─> embed_tokens (VocabParallelEmbedding)
  │   ├─> per layer:
  │   │   ├─> qkv_proj (ColumnParallelLinear)
  │   │   ├─> attention
  │   │   ├─> o_proj (RowParallelLinear) + all-reduce
  │   │   ├─> gate_up_proj (MergedColumnParallelLinear)
  │   │   └─> down_proj (RowParallelLinear) + all-reduce
  │   └─> lm_head (ParallelLMHead) + gather at rank 0
  └─> synchronize()

Rank > 0 (Worker):
  loop()
  └─> while True:
      ├─> broadcast_data()    # 接收数据
      ├─> 执行模型运算
      └─> synchronize()
```

### 4.4 前缀缓存工作流

```
Sequence 1: [A, B, C, D, E]
Sequence 2: [A, B, C, F, G]    # 共享前缀 [A, B, C]

allocate(Sequence 1):
  hash0 = xxhash([A, B, C, D, E])
  → Allocate block_id=0, hash_to_block[hash0] = 0

allocate(Sequence 2):
  hash_prefix = xxhash([A, B, C])
  → Match! block_id=0 复用 (ref_count += 1)
  → num_cached_tokens += 3

Forward:
  Sequence 1: 计算 [A, B, C, D, E] 的 KV → 存入 block 0
  Sequence 2: 复用 block 0 中 [A, B, C] 的 KV，仅计算 [F, G]
```

---

## 五、使用指南

### 5.1 基本用法

```python
from minivllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", enforce_eager=False)
outputs = llm.generate(
    ["Once upon a time", "The future of AI is"],
    SamplingParams(temperature=0.7, max_tokens=128),
)

for output in outputs:
    print(f"Prompt: {output['prompt']}")
    print(f"Generated: {output['text']}")
```

### 5.2 高级配置

```python
# 多 GPU
llm = LLM(
    model="Qwen/Qwen2-72B-Instruct",
    tensor_parallel_size=4,
    max_model_len=4096,
    enforce_eager=False,  # 启用 CUDA Graph
)

# 自定义采样
params = SamplingParams(
    temperature=0.8,
    top_p=0.9,
    top_k=50,
    min_p=0.05,
    max_tokens=256,
    ignore_eos=True,
)
```

### 5.3 多设备支持

| 设备 | 后端 | 注意力 | 状态 |
|------|------|--------|------|
| NVIDIA CUDA | NCCL | FlashAttention + CUDA Graph + TP | 完整支持 |
| 华为 NPU | HCCL | NPU FlashAttention（BNSD 布局） | 支持 |
| XPU / MPS / MLU / MUSA | Gloo | 基础 SDPA | 基础支持 |
| CPU | Gloo | PyTorch 原生 | 小模型测试 |

设备自动检测，无需手动指定。

### 5.4 参数调优建议

**模式选择**

| 模式 | 参数 | 场景 |
|------|------|------|
| Eager | `enforce_eager=True` | 调试开发 |
| Graph | `enforce_eager=False` | 生产环境 |

**内存优化**

- 内存充足：`device_memory_utilization=0.9`
- 内存紧张：`0.7-0.8`，减少 `max_num_seqs`
- 长序列：增大 `kvcache_block_size`（128/256）
- 长提示：减少 `max_num_seqs`，增大 `max_num_batched_tokens`

**采样温度**

| 范围 | 效果 | 场景 |
|------|------|------|
| 0.1–0.5 | 确定性高 | 代码生成、事实问答 |
| 0.6–1.0 | 平衡 | 通用场景 |
| 1.0–1.5 | 创意性 | 创意写作、头脑风暴 |

### 5.5 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| CUDA OOM | 显存不足 | 降低 `device_memory_utilization`，减少 `max_num_seqs`，启用 TP |
| 推理慢 | 未启用优化 | `enforce_eager=False`，安装 flash-attn |
| 生成质量差 | 采样参数不当 | 调整 temperature / top_p / top_k |
| 模型加载失败 | 路径错误 | 使用 HF model ID 或确认本地路径 |

---

## 六、许可证与致谢

**许可证**：Apache 2.0

**致谢**：nano-vllm（设计参考）、vLLM（架构灵感）、Transformers（模型定义）、PyTorch（框架）、FlashAttention（高性能注意力）。
