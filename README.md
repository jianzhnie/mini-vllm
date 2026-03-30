# mini-vLLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

🚀 **轻量级高效的大语言模型推理引擎**

mini-vLLM 是一个从零开始构建的轻量级大语言模型推理引擎，目标是提供一个简单、高效、可扩展的推理解决方案，帮助开发者快速理解大语言模型的推理原理。mini-vLLM 不是 vLLM 的直接替代品，而是一个专注于理解和学习的项目。

## ✨ 核心特性

### 🔥 高性能推理

- **CUDA Graph 优化**: 减少 decode 阶段的调度开销，提升推理速度
- **智能批处理**: 支持动态批处理和序列长度感知调度
- **KV 缓存优化**: 块式 KV 缓存管理和前缀缓存，减少内存占用

### 🧠 先进架构

- **多设备支持**: CUDA、NPU、XPU、MPS、MLU、MUSA
- **张量并行**: 支持多 GPU 分布式推理（1-8 张量并行度）
- **两阶段调度**: 预填充（Prefill）和解码（Decode）阶段分离调度
- **FlashAttention**: 集成高性能注意力计算库

### 🛠️ 灵活配置
- **内存管理**: 可配置设备内存利用率（10%-100%）
- **缓存配置**: 可调节 KV 缓存块大小和数量
- **采样控制**: 支持温度调节、top-p、top-k、min-p 等采样参数


## 📦 安装

### 系统要求

- Python 3.10 - 3.12
- PyTorch 2.2+
- 至少 8GB GPU 内存（推荐）

### 安装依赖

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
    model="Qwen/Qwen2-7B-Instruct",
    max_num_seqs=256,
    max_num_batched_tokens=8192,
    device_memory_utilization=0.9
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
    top_p=0.95,
    top_k=40,
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

### 运行示例脚本

```bash
# 使用默认模型
python examples.py

# 指定模型路径
python examples.py --model /path/to/model

# 或通过环境变量
MINIVLLM_MODEL=/path/to/model python examples.py

# 完整参数
python examples.py \
    --model /path/to/model \
    --max-seqs 16 \
    --max-tokens 64 \
    --temperature 0.8 \
    --top-p 0.95 \
    --top-k 40
```

### 高级配置

```python
# 多 GPU 配置
llm = LLM(
    model="Qwen/Qwen2-72B-Instruct",
    tensor_parallel_size=4,  # 使用 4 张 GPU
    max_num_seqs=512,
    max_num_batched_tokens=16384,
    device_memory_utilization=0.85,
    enforce_eager=False,  # 使用 CUDA Graph
    max_model_len=4096
)

# 自定义采样参数
sampling_params = SamplingParams(
    temperature=0.8,      # 更高温度 = 更多创意
    top_p=0.9,           # Nucleus 采样
    top_k=50,            # Top-k 采样
    min_p=0.05,          # 最小相对概率
    max_tokens=256,
    ignore_eos=True      # 忽略结束符，继续生成
)
```

## ⚙️ 配置参数

### LLM 配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | str | 必需 | 模型路径（HuggingFace 格式） |
| `max_num_seqs` | int | 512 | 每批次最大序列数 |
| `max_num_batched_tokens` | int | 16384 | 每批次最大 token 数 |
| `max_model_len` | int | 4096 | 模型最大上下文长度 |
| `device_memory_utilization` | float | 0.9 | 设备内存利用率 (0.1-1.0) |
| `tensor_parallel_size` | int | 1 | 张量并行度 (1-8) |
| `enforce_eager` | bool | False | 强制使用 eager 模式（禁用 CUDA Graph） |
| `kvcache_block_size` | int | 256 | KV 缓存块大小（必须被 256 整除） |
| `num_kvcache_blocks` | int | -1 | KV 缓存块数量（-1 表示自动） |
| `trust_remote_code` | bool | False | 是否信任远程代码 |
| `dtype` | str | 'auto' | 模型权重数据类型 |
| `seed` | int | None | 随机种子 |

### SamplingParams 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | float | 1.0 | 采样温度 (> 1e-10) |
| `top_p` | float | 1.0 | Nucleus 采样阈值 (0-1] |
| `top_k` | int | -1 | Top-k 采样 (-1 禁用) |
| `min_p` | float | 0.0 | 最小相对概率 [0-1] |
| `max_tokens` | int | 64 | 最大生成 token 数 |
| `ignore_eos` | bool | False | 是否忽略结束符 |

## 🧪 测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试文件
python -m pytest tests/test_config.py -v

# 运行匹配特定模式的测试
python -m pytest -k test_sampling

# 运行带覆盖率的测试
python -m pytest tests/ --cov=minivllm --cov-report=term-missing

# 使用测试运行脚本
python tests/run_tests.py --coverage -v
```

## 📚 支持的模型

- **Qwen2/Qwen3** (推荐)
- **OPT**
- **LLaMA 系列**（通过 Qwen2 兼容）

## 📄 许可证

本项目基于 [Apache 2.0 许可证](LICENSE) 开源。

## 🤝 致谢

非常感谢 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，它为 mini-vLLM 提供了宝贵的设计思路和实现参考。同时，感谢以下开源项目：

- [vLLM](https://github.com/vllm-project/vllm) - 灵感来源
- [Transformers](https://github.com/huggingface/transformers) - 模型定义和加载
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [FlashAttention](https://github.com/HazyResearch/flash-attention) - 高性能注意力
- [Triton](https://github.com/openai/triton) - GPU 内核优化


<div align="center">

**⭐ 如果这个项目对您有帮助，请给我们一个星标！ ⭐**

</div>
