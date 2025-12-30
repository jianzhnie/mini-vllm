# mini-vLLM

[![Python 3.1+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

🚀 **轻量级高效的大语言模型推理引擎**

mini-vLLM 是一个从零开始构建的轻量级大语言模型推理引擎，目标是提供一个简单、高效、可扩展的推理解决方案， 帮助开发者快速理解大语言模型的推理原理。mini-vLLM 不是 vLLM 的直接替代品，而是一个专注于理解和学习的项目。

## ✨ 核心特性

### 🔥 高性能推理

- **CUDA Graph 优化**: 减少 decode 阶段的调度开销，提升推理速度
- **智能批处理**: 支持动态批处理和序列长度感知调度
- **KV 缓存优化**: 块式 KV 缓存管理和前缀缓存，减少内存占用

### 🧠 先进架构

- **张量并行**: 支持多 GPU 分布式推理（1-8 张量并行度）
- **两阶段调度**: 预填充（Prefill）和解码（Decode）阶段分离调度
- **FlashAttention**: 集成高性能注意力计算库

### 🛠️ 灵活配置
- **内存管理**: 可配置 GPU 内存利用率（10%-100%）
- **缓存配置**: 可调节 KV 缓存块大小和数量
- **采样控制**: 支持温度调节和自定义采样参数


## 📦 安装

### 系统要求

- Python 3.10+
- CUDA 11.8+ (推荐)
- PyTorch 2.0+
- 至少 8GB GPU 内存

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-repo/mini-vllm.git
cd mini-vllm

# 安装依赖
pip install -e .

# 可选：安装 FlashAttention (显著提升性能)
pip install flash-attn --no-build-isolation

# 可选：安装 Triton (启用更多优化)
pip install triton
```

### 验证安装

```bash
python -c "import minivllm; print('Installation successful!')"
```

## 🚀 快速开始

### 基本使用

```python
from minivllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    max_num_seqs=256,
    max_num_batched_tokens=8192,
    gpu_memory_utilization=0.9
)

# 准备输入
prompts = [
    "Once upon a time",
    "The future of AI is",
    "In the deep learning era"
]

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=128,
    ignore_eos=False
)

# 生成文本
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    print(f"Prompt: {output['prompt']}")
    print(f"Generated: {output['text']}")
    print("-" * 50)
```

### 高级配置

```python
# 多 GPU 配置
llm = LLM(
    model="meta-llama/Llama-2-70b-chat-hf",
    tensor_parallel_size=4,  # 使用 4 张 GPU
    max_num_seqs=512,
    max_num_batched_tokens=16384,
    gpu_memory_utilization=0.85,
    enforce_eager=False,  # 使用 CUDA Graph
    max_model_len=4096
)

# 自定义采样参数
sampling_params = SamplingParams(
    temperature=0.8,  # 更高温度 = 更多创意
    max_tokens=256,
    ignore_eos=True  # 忽略结束符，继续生成
)

# 流式生成（如果支持）
for output in llm.generate(prompts, sampling_params):
    print(output['text'], end="", flush=True)
```

## ⚙️ 配置参数

### LLM 配置选项

| 参数                     | 类型  | 默认值 | 说明                               |
| ------------------------ | ----- | ------ | ---------------------------------- |
| `model`                  | str   | 必需   | 模型路径（HuggingFace 格式）       |
| `max_num_seqs`           | int   | 512    | 每批次最大序列数                   |
| `max_num_batched_tokens` | int   | 16384  | 每批次最大 token 数                |
| `max_model_len`          | int   | 4096   | 模型最大上下文长度                 |
| `gpu_memory_utilization` | float | 0.9    | GPU 内存利用率 (0.1-1.0)           |
| `tensor_parallel_size`   | int   | 1      | 张量并行度 (1-8)                   |
| `enforce_eager`          | bool  | False  | 强制使用 eager 模式                |
| `kvcache_block_size`     | int   | 256    | KV 缓存块大小（必须能被 256 整除） |
| `num_kvcache_blocks`     | int   | -1     | KV 缓存块数量（-1 表示自动）       |

### SamplingParams 配置

| 参数          | 类型  | 默认值 | 说明                 |
| ------------- | ----- | ------ | -------------------- |
| `temperature` | float | 1.0    | 采样温度 (1e-10-2.0) |
| `max_tokens`  | int   | 64     | 生成的最大 token 数  |
| `ignore_eos`  | bool  | False  | 是否忽略结束符       |


## 📚 API 参考

### LLM 类

```python
class LLM(LLMEngine):
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: SamplingParams
    ) -> List[Dict]:
        """生成文本

        Args:
            prompts: 输入提示（字符串或列表）
            sampling_params: 采样参数

        Returns:
            生成结果列表，每个结果包含：
            - prompt: 原始提示
            - text: 生成的文本
            - tokens: 生成的 token 列表
            - finished: 是否完成
        """
```

### 高级用法

```python
# 自定义批处理
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    max_num_seqs=128,  # 较小批量，更低延迟
    max_num_batched_tokens=4096
)

# 高吞吐量配置
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    max_num_seqs=1024,  # 大量并发
    max_num_batched_tokens=32768,  # 大批次
    gpu_memory_utilization=0.95  # 高内存使用
)

# 内存受限配置
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    gpu_memory_utilization=0.6,  # 保守内存使用
    num_kvcache_blocks=1000,  # 限制缓存块数
    kvcache_block_size=512  # 更大块，减少管理开销
)
```


## 📄 许可证

本项目基于 [Apache 2.0 许可证](LICENSE) 开源。

## 🤝 致谢

非常感谢 [nanovllm](hhttps://github.com/GeeeekExplorer/nano-vllm)， 它为 mini-vLLM 提供了宝贵的设计思路和实现参考。从中借鉴了许多内存管理和调度机制相关代码，帮助我们更好地理解大语言模型推理引擎的实现细节。 同时，感谢以下开源项目和库，它们为 mini-vLLM 的开发提供了重要支持：

- [vLLM](https://github.com/vllm-project/vllm) - 灵感来源
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [FlashAttention](https://github.com/HazyResearch/flash-attention) - 高性能注意力
- [Triton](https://github.com/openai/triton) - GPU 内核优化


<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个星标！ ⭐**

</div>
