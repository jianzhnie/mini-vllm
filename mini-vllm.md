# nano-vllm 代码仓库详细梳理

## 一、仓库概览

### 项目信息
- **项目名称**: nano-vllm
- **项目描述**: 一个从零开始构建的轻量级 vLLM 实现
- **核心特性**:
  - 🚀 快速离线推理 - 推理速度与 vLLM 相当
  - 📖 可读代码库 - 约 1,200 行 Python 代码的清晰实现
  - ⚡ 优化工具集 - 前缀缓存、张量并行、Torch 编译、CUDA 图等

### 依赖环境
```
torch>=2.4.0
triton>=3.0.0
transformers>=4.51.0
flash-attn
xxhash
```

### 仓库目录结构
```
nano-vllm/
├── nanovllm/                    # 主要包目录
│   ├── __init__.py             # 导出 LLM 和 SamplingParams
│   ├── llm.py                  # LLM 类入口（继承 LLMEngine）
│   ├── config.py               # 配置类定义
│   ├── sampling_params.py       # 采样参数定义
│   ├── engine/                 # 推理引擎核心模块
│   │   ├── llm_engine.py       # 推理引擎主类
│   │   ├── model_runner.py     # 模型运行器（处理前缀缓存、KV缓存、CUDA图）
│   │   ├── scheduler.py        # 批次调度器
│   │   ├── sequence.py         # 序列数据结构
│   │   └── block_manager.py    # KV缓存块管理（支持前缀缓存）
│   ├── models/                 # 模型实现
│   │   └── qwen3.py           # Qwen3 大语言模型实现
│   ├── layers/                 # 神经网络层
│   │   ├── attention.py        # 闪注意力实现
│   │   ├── linear.py           # 张量并行线性层
│   │   ├── embed_head.py       # 嵌入层和输出层
│   │   ├── rotary_embedding.py # 旋转位置编码
│   │   ├── layernorm.py        # RMSNorm 层
│   │   ├── activation.py       # SiluAndMul 激活函数
│   │   └── sampler.py          # 采样器
│   └── utils/                  # 工具函数
│       ├── context.py          # 全局上下文管理
│       └── loader.py           # 模型权重加载
├── example.py                  # 使用示例
├── bench.py                    # 性能基准测试
├── pyproject.toml             # 项目配置
└── README.md                  # 项目说明
```

---

## 二、代码主要说明

nano-vllm 是一个完整的大语言模型离线推理框架，核心流程为：

1. **初始化阶段**: 加载模型、初始化 KV 缓存、准备 CUDA 图
2. **请求处理阶段**: 接收用户请求，添加到待处理队列
3. **调度阶段**: 根据资源限制调度请求（Prefill 或 Decode 阶段）
4. **推理阶段**: 执行前向传播，存储 KV 缓存
5. **采样阶段**: 对输出 logits 进行采样生成下一个 token
6. **输出阶段**: 返回完整的生成文本

**核心优化技术**:
- **前缀缓存 (Prefix Caching)**: 通过哈希实现相同前缀的 KV 缓存共享
- **块化 KV 缓存 (Block-wise KV Cache)**: 使用固定大小的块管理 KV 缓存，支持灵活的内存管理
- **张量并行 (Tensor Parallelism)**: 支持多 GPU 的权重分片并行
- **CUDA 图 (CUDA Graph)**: 捕获 decode 阶段的计算图，减少 CPU 开销
- **FlashAttention**: 使用优化的注意力实现提升速度

---

## 三、代码文件详细解释

### 3.1 入口和配置文件

#### [llm.py](llm.py) - LLM 类入口
```python
class LLM(LLMEngine):
    pass
```
- 这是用户接口，继承 LLMEngine，提供统一的 API
- 通过 `generate()` 方法进行批量推理

#### [config.py](config.py) - 配置类
```python
@dataclass
class Config:
    model: str                          # 模型路径
    max_num_batched_tokens: int = 16384 # 最大批次 token 数
    max_num_seqs: int = 512             # 最大序列数
    max_model_len: int = 4096           # 最大序列长度
    gpu_memory_utilization: float = 0.9 # GPU 内存使用率
    tensor_parallel_size: int = 1       # 张量并行大小
    enforce_eager: bool = False         # 是否强制 eager 模式
    eos: int = -1                       # EOS token ID
    kvcache_block_size: int = 256       # KV 缓存块大小
    num_kvcache_blocks: int = -1        # KV 缓存总块数（自动计算）
```

#### [sampling_params.py](sampling_params.py) - 采样参数
```python
@dataclass
class SamplingParams:
    temperature: float = 1.0            # 采样温度（不支持 greedy）
    max_tokens: int = 64                # 最多生成多少个 token
    ignore_eos: bool = False            # 是否忽略 EOS token
```

#### [__init__.py](__init__.py) - 包导出
- 导出 `LLM` 类和 `SamplingParams` 类供外部使用

---

### 3.2 引擎核心模块 (`engine/`)

#### [llm_engine.py](llm_engine.py) - 推理引擎主类

**核心功能**:
1. **初始化阶段**
   - 创建 `Config` 对象
   - 启动多进程用于张量并行
   - 初始化 `ModelRunner`、`Tokenizer`、`Scheduler`

2. **请求处理**
   ```python
   def add_request(prompt, sampling_params):
       # 将 prompt 转为 token IDs
       # 创建 Sequence 对象
       # 添加到调度器
   ```

3. **推理循环**
   ```python
   def step():
       seqs, is_prefill = scheduler.schedule()      # 选择要执行的序列
       token_ids = model_runner.call("run", ...)    # 执行推理
       scheduler.postprocess(seqs, token_ids)       # 更新序列状态
   ```

4. **生成接口**
   ```python
   def generate(prompts, sampling_params):
       # 批量处理多个 prompt
       # 显示吞吐量统计
       # 返回生成的文本
   ```

#### [sequence.py](sequence.py) - 序列数据结构

**Sequence 类表示一个处理中的序列**:
```python
class Sequence:
    seq_id: int                    # 序列 ID（全局自增）
    status: SequenceStatus         # WAITING / RUNNING / FINISHED
    token_ids: list[int]           # 所有 token IDs（prompt + completion）
    num_cached_tokens: int         # 已缓存的 token 数（前缀缓存）
    block_table: list[int]         # 分配的 KV 缓存块 ID 列表
    
    # 采样参数
    temperature: float
    max_tokens: int
    ignore_eos: bool
```

**关键属性**:
- `len(seq)`: 序列总 token 数
- `seq.prompt_token_ids`: Prompt 部分
- `seq.completion_token_ids`: 生成部分
- `seq.num_blocks`: 需要的 KV 缓存块数
- `seq.block(i)`: 第 i 个块的 token IDs

#### [scheduler.py](scheduler.py) - 批次调度器

**核心功能**:
- 维护 `waiting` 队列（待处理）和 `running` 队列（运行中）
- 实现两阶段调度：

1. **Prefill 阶段**（首次处理 prompt）
   - 从 `waiting` 队列选择可以分配块的序列
   - 受限制：
     - `max_num_seqs`: 最多多少个序列
     - `max_num_batched_tokens`: 最多处理多少个 token
   - 为序列分配 KV 缓存块

2. **Decode 阶段**（生成阶段，每步生成 1 个 token）
   - 从 `running` 队列选择可以追加块的序列
   - 若块不足则抢占其他序列
   - 调用 `block_manager.may_append()` 为下一步准备

3. **Postprocess 阶段**
   - 添加新 token 到序列
   - 检查是否完成（达到 max_tokens 或生成 EOS）

#### [block_manager.py](block_manager.py) - KV 缓存块管理器

**核心概念**:
- 将 KV 缓存分成固定大小的块（默认 256 个 token）
- 实现**前缀缓存**: 相同的 token 序列块共享同一个 KV 缓存块

**关键数据结构**:
```python
class Block:
    block_id: int          # 块编号
    ref_count: int         # 引用计数
    hash: int              # 块的哈希值（用于前缀缓存）
    token_ids: list[int]   # 块中的 token IDs
```

**关键方法**:
```python
# 分配 KV 缓存
allocate(seq):
    for each block in seq:
        compute_hash(tokens)  # 计算块的哈希
        if hash matches cached block:
            reuse existing block  # 实现前缀缓存
            seq.num_cached_tokens += block_size
        else:
            allocate new block

# 销毁 KV 缓存
deallocate(seq):
    for each block in seq.block_table:
        decrement ref_count
        if ref_count == 0:
            return block to free pool

# 追加块（decode 阶段）
may_append(seq):
    if crossing block boundary:
        allocate new block
    else:
        update last block hash (for prefix cache)
```

#### [model_runner.py](model_runner.py) - 模型运行器

**核心功能**:
1. **初始化**
   - 初始化分布式进程组（NCCL）
   - 加载模型权重
   - 分配 KV 缓存（根据 GPU 可用内存）
   - 预热模型
   - 捕获 CUDA 图（如果启用）

2. **多进程支持**
   - Rank 0: 主进程，执行模型推理
   - Rank > 0: 辅助进程，接收命令执行计算
   - 通过共享内存和事件同步通信

3. **KV 缓存分配**
   ```python
   def allocate_kv_cache():
       # 计算可用内存
       # 根据模型大小计算能容纳的块数
       # 创建 [2, num_layers, num_blocks, block_size, num_heads, head_dim] 的张量
   ```

4. **Prefill 数据准备** (`prepare_prefill`)
   - 处理多个序列，可能长度不同
   - 使用 cu_seqlens（cumulative sequence lengths）
   - 为注意力计算准备 slot_mapping（KV 缓存位置映射）
   - 检测前缀缓存（若有缓存的前缀则准备 block_tables）

5. **Decode 数据准备** (`prepare_decode`)
   - 只取最后一个 token
   - 准备 context_lens（每个序列的上下文长度）
   - 准备 block_tables（用于 KV 缓存查找）

6. **推理执行** (`run_model`)
   - **Eager 模式**: 直接执行模型
   - **图模式**: 使用 CUDA 图加速（对于 batch size < 512）

7. **CUDA 图捕获** (`capture_cudagraph`)
   - 为不同 batch size (1, 2, 4, 8, 16, 32, ...) 捕获计算图
   - 使用图池共享内存

---

### 3.3 模型实现 (`models/`)

#### [qwen3.py](qwen3.py) - Qwen3 模型实现

**架构组成**:

1. **Qwen3Attention - 多头注意力层**
   - QKV 投影（支持 GQA：分组查询注意力）
   - 旋转位置编码（RoPE）
   - 闪注意力加速
   - 输出投影

2. **Qwen3MLP - 前馈网络**
   - 门控上升投影 (gate_up_proj)
   - SiLU 激活 + Mul 操作
   - 下降投影

3. **Qwen3DecoderLayer - Transformer 解码块**
   ```
   Input -> RMSNorm -> Self-Attention -> Add & Norm
        -> RMSNorm -> MLP -> Output
   ```

4. **Qwen3Model - 完整模型体**
   - 词嵌入层
   - N 层解码器堆栈
   - 最后的 RMSNorm

5. **Qwen3ForCausalLM - 因果语言模型**
   - 继承 Qwen3Model
   - 添加 LM Head（语言建模头）
   - 支持权重绑定（词嵌入和输出层权重共享）
   - 定义了 packed_modules_mapping（用于权重加载的映射）

---

### 3.4 神经网络层 (`layers/`)

#### [attention.py](attention.py) - 注意力机制

**Triton 内核**:
```python
@triton.jit
def store_kvcache_kernel:
    # 将计算得到的 K, V 存储到 KV 缓存对应位置
```

**Attention 类**:
```python
class Attention:
    def forward(q, k, v):
        # 1. 将 K, V 存储到 KV 缓存
        store_kvcache(k, v, k_cache, v_cache, slot_mapping)
        
        # 2. Prefill 阶段
        if is_prefill:
            if block_tables exists:  # 前缀缓存
                use cached k, v
            o = flash_attn_varlen_func(...)  # 处理不同长度序列
        
        # 3. Decode 阶段
        else:
            o = flash_attn_with_kvcache(...)  # 使用缓存的 K, V
```

#### [linear.py](linear.py) - 张量并行线性层

**层次结构**:
```
LinearBase (基类)
├── ReplicatedLinear (复制线性层)
├── ColumnParallelLinear (列并行)
│   ├── MergedColumnParallelLinear (合并列并行)
│   └── QKVParallelLinear (QKV 投影)
└── RowParallelLinear (行并行)
```

**张量并行策略**:
- **列并行**: 输出沿列分片，计算后不通信
  ```
  Weight: [out_size/tp, in_size]  (每个 rank 一部分输出)
  ```
- **行并行**: 输入沿行分片，计算后需 all-reduce
  ```
  Weight: [out_size, in_size/tp]  (每个 rank 一部分输入)
  y = W @ x.T  ->  y = W[rank] @ x[rank].T, then all-reduce
  ```

#### [embed_head.py](embed_head.py) - 嵌入和输出层

1. **VocabParallelEmbedding** - 词汇并行嵌入
   - 词汇表沿 vocab 维度分片
   - 掩码对应词汇范围外的 token
   - 输出后需 all-reduce 聚合

2. **ParallelLMHead** - 并行语言建模头
   - 继承词汇并行嵌入
   - Prefill 阶段只取每个序列的最后 token
   - 输出后通过 gather 在 rank 0 聚合所有词表的 logits

#### [rotary_embedding.py](rotary_embedding.py) - 旋转位置编码

```python
class RotaryEmbedding:
    def forward(positions, q, k):
        cos_sin = cache[positions]  # 查找预计算的 cos/sin
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
```

- 预计算所有位置的 cos/sin 值
- 使用索引查找而不是重复计算

#### [layernorm.py](layernorm.py) - RMSNorm 层

```python
class RMSNorm:
    # 标准 RMSNorm: x = x / sqrt(mean(x^2) + eps) * weight
    
    def forward(x, residual=None):
        if residual is None:
            return rms_forward(x)
        else:
            return add_rms_forward(x, residual)
            # 融合操作：先加后 norm，提升效率
```

#### [activation.py](activation.py) - 激活函数

```python
class SiluAndMul:
    def forward(x):
        x, y = x.chunk(2, -1)
        return silu(x) * y  # 门控激活单元
```

#### [sampler.py](sampler.py) - 采样器

```python
class Sampler:
    @torch.compile
    def forward(logits, temperatures):
        logits = logits / temperatures.unsqueeze(1)
        probs = softmax(logits, -1)
        # Gumbel-max 采样技巧
        sample_tokens = (probs / exp_gumbel_noise).argmax(-1)
```

---

### 3.5 工具模块 (`utils/`)

#### [context.py](context.py) - 全局上下文

```python
@dataclass
class Context:
    is_prefill: bool                # 是否 prefill 阶段
    cu_seqlens_q/k: Tensor         # Query/Key 的累积序列长度
    max_seqlen_q/k: int            # 最大 Query/Key 长度
    slot_mapping: Tensor           # KV 缓存槽位映射
    context_lens: Tensor           # 每个序列的上下文长度
    block_tables: Tensor           # 块表（用于前缀缓存）
```

- 存储当前批次的计算上下文
- 注意力机制通过 `get_context()` 访问这些信息

#### [loader.py](loader.py) - 模型权重加载

```python
def load_model(model, path):
    # 遍历所有 .safetensors 文件
    # 使用 packed_modules_mapping 处理打包权重
    # 调用参数的 weight_loader 方法加载
```

- 支持标准权重和打包权重（如 QKV 合并）
- 支持张量并行权重分片加载

---

### 3.6 使用示例

#### [example.py](example.py) - 使用示例

```python
def main():
    # 1. 初始化模型
    llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1)
    
    # 2. 设置采样参数
    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    
    # 3. 批量生成
    outputs = llm.generate(prompts, sampling_params)
    
    # 4. 处理输出
    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Completion: {output['text']}")
```

---

## 四、代码逻辑流程（调用关系）

### 4.1 初始化流程

```
User: llm = LLM(model_path, ...)
  │
  ├─> LLM.__init__() [llm.py]
  │   │
  │   └─> LLMEngine.__init__() [llm_engine.py]
  │       ├─> Config(model, **kwargs) [config.py]
  │       │   └─> AutoConfig.from_pretrained()
  │       │   └─> Calculate max_model_len
  │       │
  │       ├─> Create ModelRunner processes (rank > 0)
  │       │   └─> dist.init_process_group("nccl")
  │       │   └─> torch.cuda.set_device(rank)
  │       │
  │       ├─> ModelRunner(config, rank=0, events) [model_runner.py]
  │       │   ├─> Load model: Qwen3ForCausalLM() [qwen3.py]
  │       │   ├─> load_model(model, path) [loader.py]
  │       │   ├─> warmup_model()
  │       │   ├─> allocate_kv_cache()
  │       │   │   └─> Calculate num_kvcache_blocks
  │       │   │   └─> Create kv_cache tensor [2, layers, blocks, 256, heads, dim]
  │       │   └─> capture_cudagraph() (if not enforce_eager)
  │       │       └─> Capture graphs for different batch sizes
  │       │
  │       ├─> AutoTokenizer.from_pretrained() [llm_engine.py]
  │       │
  │       └─> Scheduler(config) [scheduler.py]
  │           └─> BlockManager(num_blocks, block_size) [block_manager.py]
  │               └─> Initialize free blocks queue
```

### 4.2 生成流程

```
User: outputs = llm.generate(prompts, sampling_params)
  │
  └─> LLMEngine.generate() [llm_engine.py]
      │
      ├─> Tokenize prompts (if str)
      │   └─> tokenizer.encode(prompt)
      │
      ├─> For each (prompt, sp):
      │   └─> add_request(prompt, sp)
      │       └─> Sequence(token_ids, sp) [sequence.py]
      │       └─> scheduler.add(seq) [scheduler.py]
      │
      └─> While not is_finished():
          │
          ├─> step() [llm_engine.py]
          │   │
          │   ├─> scheduler.schedule() [scheduler.py]
          │   │   │
          │   │   ├─> PREFILL Phase (if waiting seqs exist)
          │   │   │   ├─> Select seqs from waiting queue
          │   │   │   ├─> block_manager.allocate(seq) [block_manager.py]
          │   │   │   │   └─> Check prefix cache hits
          │   │   │   │   └─> Allocate/reuse blocks
          │   │   │   └─> Return (seqs, is_prefill=True)
          │   │   │
          │   │   ├─> DECODE Phase (else)
          │   │   │   ├─> Select seqs from running queue
          │   │   │   ├─> Check if can append (block space)
          │   │   │   ├─> If not, preempt other seqs
          │   │   │   ├─> block_manager.may_append(seq)
          │   │   │   │   └─> Allocate new block if crossing boundary
          │   │   │   │   └─> Update block hash for prefix cache
          │   │   │   └─> Return (seqs, is_prefill=False)
          │   │
          │   ├─> model_runner.call("run", seqs, is_prefill) [model_runner.py]
          │   │   │
          │   │   ├─> If is_prefill:
          │   │   │   ├─> prepare_prefill(seqs)
          │   │   │   │   ├─> Flatten multiple seqs into single batch
          │   │   │   │   ├─> Prepare: input_ids, positions
          │   │   │   │   ├─> Compute: cu_seqlens_q, cu_seqlens_k
          │   │   │   │   ├─> Prepare: slot_mapping (for KV cache)
          │   │   │   │   ├─> set_context(is_prefill=True, ...)
          │   │   │   │   └─> Return (input_ids, positions)
          │   │   │   │
          │   │   │   ├─> run_model(input_ids, positions, is_prefill=True)
          │   │   │   │   └─> model.forward(input_ids, positions) [qwen3.py]
          │   │   │   │       ├─> embed_tokens(input_ids)
          │   │   │   │       ├─> For each layer in model.layers:
          │   │   │   │       │   ├─> layer.self_attn(positions, hidden_states)
          │   │   │   │       │   │   ├─> qkv_proj(hidden_states)
          │   │   │   │       │   │   ├─> rotary_emb(positions, q, k)
          │   │   │   │       │   │   ├─> attn(q, k, v) [attention.py]
          │   │   │   │       │   │   │   ├─> store_kvcache_kernel(k, v, ...)
          │   │   │   │       │   │   │   ├─> flash_attn_varlen_func(q, k, v, ...)
          │   │   │   │       │   │   │   │   (with block_tables for prefix cache)
          │   │   │   │       │   │   │   └─> Output: attention output
          │   │   │   │       │   │   └─> o_proj()
          │   │   │   │       │   ├─> layer.mlp()
          │   │   │   │       │   │   ├─> gate_up_proj()
          │   │   │   │       │   │   ├─> act_fn(SiluAndMul)
          │   │   │   │       │   │   └─> down_proj()
          │   │   │   │       │   └─> Add residual connections
          │   │   │   │       └─> norm(hidden_states)
          │   │   │   │
          │   │   │   └─> model.compute_logits(hidden_states)
          │   │   │       └─> lm_head(hidden_states)
          │   │   │
          │   │   ├─> Else (is_prefill=False): DECODE
          │   │   │   ├─> prepare_decode(seqs)
          │   │   │   │   ├─> Extract last token from each seq
          │   │   │   │   ├─> Prepare: slot_mapping, context_lens
          │   │   │   │   ├─> prepare_block_tables(seqs)
          │   │   │   │   ├─> set_context(is_prefill=False, ...)
          │   │   │   │   └─> Return (input_ids, positions)
          │   │   │   │
          │   │   │   ├─> run_model(input_ids, positions, is_prefill=False)
          │   │   │   │   ├─> If use CUDA graph and batch_size in graph_bs:
          │   │   │   │   │   └─> Setup graph variables and replay
          │   │   │   │   └─> Else:
          │   │   │   │       └─> Direct model.forward()
          │   │   │   │
          │   │   │   └─> model.compute_logits()
          │   │   │
          │   │   ├─> prepare_sample(seqs)
          │   │   │   └─> Prepare temperatures for each seq
          │   │   │
          │   │   ├─> sampler(logits, temperatures) [sampler.py]
          │   │   │   └─> Apply temperature scaling
          │   │   │   └─> Gumbel-max sampling
          │   │   │   └─> Return token_ids
          │   │   │
          │   │   └─> reset_context()
          │   │
          │   ├─> scheduler.postprocess(seqs, token_ids)
          │   │   └─> For each (seq, token_id):
          │   │       ├─> seq.append_token(token_id)
          │   │       ├─> Check if finished:
          │   │       │   ├─> token_id == eos_token_id
          │   │       │   ├─> num_completion_tokens >= max_tokens
          │   │       ├─> If finished:
          │   │       │   ├─> seq.status = FINISHED
          │   │       │   ├─> block_manager.deallocate(seq)
          │   │       │   └─> Remove from running queue
          │   │
          │   └─> Return (outputs, num_tokens) for completed seqs
          │
          └─> Collect all outputs and decode
              └─> tokenizer.decode(token_ids)
              └─> Return [{"text": ..., "token_ids": ...}, ...]
```

### 4.3 张量并行流程（多 GPU）

```
ModelRunner(rank=0):
  │
  └─> run(seqs, is_prefill):
      ├─> prepare_prefill/decode()  [在 rank 0 准备数据]
      ├─> call("run", seqs, is_prefill)
      │   └─> write_shm(method_name, seqs, is_prefill)
      │       └─> Serialize and write to shared memory
      │       └─> Set events for rank > 0
      │
      └─> run_model(input_ids, positions, is_prefill)
          └─> Forward pass with distributed operations
              ├─> embed_tokens (VocabParallelEmbedding)
              │   └─> Each rank owns vocab_size/tp subset
              │   └─> Mask invalid vocab + all-reduce
              │
              ├─> For each layer:
              │   ├─> qkv_proj (ColumnParallelLinear)
              │   │   └─> Output split across ranks
              │   ├─> attention
              │   ├─> o_proj (RowParallelLinear)
              │   │   └─> all-reduce across ranks
              │   ├─> gate_up_proj (MergedColumnParallelLinear)
              │   └─> down_proj (RowParallelLinear)
              │       └─> all-reduce
              │
              └─> lm_head (ParallelLMHead)
                  └─> Each rank computes logits for its vocab
                  └─> gather at rank 0 to combine all vocabs

ModelRunner(rank > 0):
  │
  └─> loop():
      └─> While True:
          ├─> read_shm()  [等待 rank 0 的命令]
          ├─> call(method_name, *args)
          │   └─> Execute model operations
          └─> If method_name == "exit": break
```

### 4.4 前缀缓存工作流

```
Sequence 1: [A, B, C, D, E] (prompt tokens)
Sequence 2: [A, B, C, F, G] (same prefix but different suffix)

allocate(Sequence 1):
  Block 0: [A, B, C, D, E] (256 tokens, actually only 5)
    ├─> hash = compute_hash([A, B, C, D, E])
    ├─> No match in hash_to_block_id
    ├─> Allocate new block_id=0
    ├─> block_table[0] = 0
    └─> hash_to_block_id[hash0] = 0

allocate(Sequence 2):
  Block 0: [A, B, C, F, G]
    ├─> hash_prefix = compute_hash([A, B, C])  (from block 0 of seq 1)
    ├─> Match! hash_to_block_id[hash_prefix] = 0
    ├─> Reuse block_id=0
    ├─> block_table[0] = 0
    └─> num_cached_tokens += 256 (虽然实际只用了 3 个)

During forward:
  ├─> Sequence 1: Process [A, B, C, D, E] → store KV in block 0
  ├─> Sequence 2: Process [A, B, C, F, G]
  │   ├─> KV for [A, B, C] already cached in block 0
  │   ├─> Only compute KV for [F, G]
  │   └─> Saves computation for common prefix

During decode:
  ├─> Sequence 1 next step: Use cached KV from block 0 + new slot
  ├─> Sequence 2 next step: Use cached KV from block 0 + new slot
  └─> Both share the KV cache for the common prefix
```

---

## 五、关键优化总结

| 优化技术            | 位置                    | 作用                                      |
| ------------------- | ----------------------- | ----------------------------------------- |
| **前缀缓存**        | BlockManager            | 相同前缀共享 KV 缓存，减少存储和重复计算  |
| **块化 KV 缓存**    | BlockManager, Attention | 灵活管理内存，支持序列长度变化            |
| **张量并行**        | Linear 层, embed_head   | 多 GPU 分布式推理，扩大吞吐量             |
| **CUDA 图**         | ModelRunner             | 捕获 decode 图，减少 CPU 开销和核函数启动 |
| **FlashAttention**  | Attention               | 优化的注意力实现，提升速度                |
| **Gumbel-max 采样** | Sampler                 | 高效的 Gumbel noise 采样                  |
| **RMSNorm 融合**    | RMSNorm                 | 融合加法和归一化，减少内存访问            |
| **Eager vs Graph**  | ModelRunner             | 根据 batch size 选择执行方式              |

---

## 六、性能特点

- **推理速度**: 与 vLLM 相当
- **代码量**: ~1,200 行代码（注释包含在内）
- **内存效率**: 支持 GPU 内存动态利用率配置
- **吞吐量**: 
  - Prefill 阶段: 受限于 max_num_batched_tokens 和 max_num_seqs
  - Decode 阶段: 高吞吐量（每步处理多个序列）

---

## 七、使用建议

1. **Eager 模式** (`enforce_eager=True`):
   - 调试和开发时使用
   - 消除 CUDA 图的编译开销

2. **图模式** (`enforce_eager=False`):
   - 生产环境使用
   - 加速 decode 阶段（batch_size < 512）

3. **张量并行** (`tensor_parallel_size=N`):
   - GPU 显存充足的环境使用
   - 提升吞吐量，但增加通信开销

4. **内存利用**:
   - 调整 `gpu_memory_utilization` 优化 KV 缓存块数
   - 调整 `max_num_batched_tokens` 控制 prefill 吞吐量

