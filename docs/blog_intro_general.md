# mini-vLLM：轻量级大语言模型推理引擎

> 如果你也想看懂 LLM 推理引擎的内部原理，而不是只会调 API，这个项目值得一看。

## 一、为什么做这个项目？

现在做 LLM 推理，大多数人要么直接调用 OpenAI 接口，要么上 vLLM、TGI、TensorRT-LLM。这些工具很强，但也有一个共同问题：**代码太复杂了**。

vLLM 确实把 Continuous Batching、PagedAttention、Prefix Caching 这些技术做到了工业级，但源码动辄十几万行，模块之间相互嵌套，初学者想搞清楚 "一个请求从输入到输出到底经历了什么"，门槛非常高。

所以我做了 **mini-vLLM**：一个从零手写的轻量级 LLM 推理引擎。

它的目标不是替代 vLLM，而是做一个 **"可读、可改、可跑" 的教学级实现**。核心代码只有约 9100 行、41 个 Python 文件，但 vLLM 里最关键的优化技术基本都有覆盖：CUDA Graph、两阶段调度、块式 KV 缓存、前缀缓存、张量并行、FlashAttention。

## 二、mini-vLLM 是什么？

一句话概括：**用纯 Python + PyTorch 实现的高性能 LLM 推理引擎，灵感来自 vLLM，但代码更轻、结构更清晰。**

### 核心定位

| 目标 | 说明 |
|------|------|
| 可读性优先 | 代码结构清晰，方便理解推理引擎的完整链路 |
| 技术完整 | 覆盖生产级推理引擎的核心优化 |
| 易于扩展 | 添加新模型、新后端、新采样策略都很方便 |
| 轻量部署 | 不依赖过多重型库，适合学习和小规模部署 |

### 支持能力

- **模型**：Qwen2、Qwen3、OPT（后续会持续增加）
- **设备**：CUDA、华为 NPU、XPU、MPS、MLU、MUSA，自动检测
- **并行**：1-8 卡张量并行，支持 NCCL / HCCL / Gloo
- **优化**：CUDA Graph、FlashAttention、前缀缓存、块式 KV 缓存
- **采样**：temperature、top-k、top-p、min-p、典型采样、重复惩罚

## 三、架构设计：一个请求是怎么被处理的？

mini-vLLM 的整体流程可以概括为 6 个阶段：

```
初始化 → 请求处理 → 调度 → 推理 → 采样 → 输出
```

### 1. 初始化

`LLM(model=...)` 被调用时，引擎会依次完成：

- 加载 HuggingFace 模型配置和权重
- 初始化分布式后端（单卡或多卡）
- 分配 KV 缓存空间
- 捕获 CUDA Graph（用于加速 decode）
- 启动调度器和 BlockManager

### 2. 请求处理

用户传入 prompts 后，会先被 tokenizer 转成 token ID，每个 prompt 包装成一个 `Sequence` 对象，进入调度器的等待队列。

### 3. 两阶段调度

这是整个引擎的核心。调度器把推理分成两个阶段：

- **Prefill**：首次处理 prompt，计算并填充 KV 缓存
- **Decode**：逐 token 生成，复用已有的 KV 缓存

调度器会根据 `max_num_seqs` 和 `max_num_batched_tokens` 动态决定每一轮放多少序列进 batch，尽量提高 GPU 利用率。

### 4. 推理与采样

模型前向传播得到 logits 后，进入采样管线：

```
penalties → temperature → top-k → top-p → min-p → multinomial
```

每一步都是独立可配置的，方便实验不同采样策略。

### 5. 输出与资源回收

生成的新 token 会被追加到 `Sequence`，检查是否满足停止条件（EOS 或达到 `max_tokens`），完成的序列会被释放 KV 缓存块。

## 四、几个值得一提的优化点

### 1. 前缀缓存（Prefix Caching）

多个请求共享相同前缀时，比如都用同一个 system prompt，mini-vLLM 会为前缀的 KV 缓存计算 xxhash，建立 `hash → block_id` 映射。后续请求命中后可以直接复用，省去重复计算。

```
请求1: [System, A, B, C]
请求2: [System, A, B, D]
                ↑
         这部分 KV 直接复用
```

### 2. 块式 KV 缓存（Block-wise KV Cache）

KV 缓存按固定大小分块管理（默认 64 tokens），每个序列维护一张 `block_table`。这样做有两个好处：

- 减少内存碎片
- 支持变长序列的动态分配和回收

### 3. CUDA Graph

Decode 阶段 batch size 固定、计算图稳定，非常适合用 CUDA Graph 捕获。mini-vLLM 会预录多个 batch size 的图，运行时再按实际大小重放，显著降低 CPU 调度开销。

### 4. 张量并行

大模型单卡放不下时，mini-vLLM 支持把权重按列/行切分到多张卡上：

- `ColumnParallelLinear`：输出分片
- `RowParallelLinear`：输入分片 + all-reduce 聚合
- `QKVParallelLinear`：QKV 投影专用并行

支持 1-8 卡，自动选择 NCCL / HCCL / Gloo 后端。

## 五、上手有多简单？

安装：

```bash
git clone https://github.com/jianzhnie/mini-vllm.git
cd mini-vllm
pip install -e .
# CUDA 用户建议安装 flash-attn
pip install -e ".[cuda]"
```

推理：

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

多 GPU 只需加一行：

```python
llm = LLM(
    model="Qwen/Qwen2-72B-Instruct",
    tensor_parallel_size=4,
    max_model_len=4096,
)
```

## 六、适合谁用？

- **学生/初学者**：想理解 LLM 推理引擎内部是怎么工作的
- **算法工程师**：需要快速验证某个优化想法，不想在 vLLM 里改一万行代码
- **小团队**：轻量级部署，不需要完整的 vLLM 生态
- **研究者**：做采样策略、调度策略、KV 缓存相关的实验

## 七、写在最后

mini-vLLM 不是 vLLM 的竞品，而是它的 "精读版"。

如果你曾经打开 vLLM 源码看了 5 分钟就放弃了，那 mini-vLLM 可能是一个更好的起点。它用更少的代码、更清晰的结构，把 LLM 推理引擎里最关键的技术都串了起来。

项目已经开源，欢迎大家试用、提 issue、贡献代码：

**GitHub**: https://github.com/jianzhnie/mini-vllm

如果你觉得这个项目对你有帮助，欢迎在 GitHub 点个 star，也欢迎在评论区交流讨论。

---

**许可证**：Apache 2.0

**致谢**：vLLM、nano-vllm、Transformers、PyTorch、FlashAttention
