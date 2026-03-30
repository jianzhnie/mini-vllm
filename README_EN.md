# mini-vLLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

[简体中文](README.md) | English

🚀 **A Lightweight and Efficient Large Language Model Inference Engine**

mini-vLLM is a lightweight LLM inference engine built from scratch, designed to provide a simple, efficient, and scalable inference solution while helping developers quickly understand the principles of LLM inference. mini-vLLM is not a direct replacement for vLLM, but rather a project focused on understanding and learning.

## ✨ Core Features

### 🔥 High-Performance Inference

- **CUDA Graph Optimization**: Reduces scheduling overhead during the decode phase, improving inference speed
- **Intelligent Batching**: Supports dynamic batching and sequence length-aware scheduling
- **KV Cache Optimization**: Block-based KV cache management with prefix caching to reduce memory usage

### 🧠 Advanced Architecture

- **Multi-Device Support**: CUDA, NPU, XPU, MPS, MLU, MUSA
- **Tensor Parallelism**: Supports multi-GPU distributed inference (1-8 tensor parallelism degrees)
- **Two-Stage Scheduling**: Separate scheduling for Prefill and Decode phases
- **FlashAttention**: Integrated high-performance attention computation library

### 🛠️ Flexible Configuration
- **Memory Management**: Configurable device memory utilization (10%-100%)
- **Cache Configuration**: Adjustable KV cache block size and quantity
- **Sampling Control**: Supports temperature scaling, top-p, top-k, min-p, and other sampling parameters


## 📦 Installation

### System Requirements

- Python 3.10 - 3.12
- PyTorch 2.2+
- At least 8GB GPU memory (recommended)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/jianzhnie/mini-vllm.git
cd mini-vllm

# Basic installation
pip install -e .

# Install CUDA optimized version (recommended)
pip install -e ".[cuda]"

# Install NPU version
pip install -e ".[npu]"

# Install development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "import minivllm; print('Installation successful!')"
```

## 🚀 Quick Start

### Basic Usage

```python
from minivllm import LLM, SamplingParams

# Initialize the model
llm = LLM(
    model="Qwen/Qwen2-7B-Instruct",
    max_num_seqs=256,
    max_num_batched_tokens=8192,
    device_memory_utilization=0.9
)

# Prepare inputs
prompts = [
    "Once upon a time",
    "The future of AI is",
    "In the deep learning era"
]

# Configure sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    top_k=40,
    max_tokens=128,
    ignore_eos=False
)

# Generate text
outputs = llm.generate(prompts, sampling_params)

# Print results
for output in outputs:
    print(f"Prompt: {output['prompt']}")
    print(f"Generated: {output['text']}")
    print("-" * 50)
```

### Run Example Script

```bash
# Use default model
python examples.py

# Specify model path
python examples.py --model /path/to/model

# Or via environment variable
MINIVLLM_MODEL=/path/to/model python examples.py

# Full parameters
python examples.py \
    --model /path/to/model \
    --max-seqs 16 \
    --max-tokens 64 \
    --temperature 0.8 \
    --top-p 0.95 \
    --top-k 40
```

### Advanced Configuration

```python
# Multi-GPU configuration
llm = LLM(
    model="Qwen/Qwen2-72B-Instruct",
    tensor_parallel_size=4,  # Use 4 GPUs
    max_num_seqs=512,
    max_num_batched_tokens=16384,
    device_memory_utilization=0.85,
    enforce_eager=False,  # Use CUDA Graph
    max_model_len=4096
)

# Custom sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,      # Higher temperature = more creativity
    top_p=0.9,           # Nucleus sampling
    top_k=50,            # Top-k sampling
    min_p=0.05,          # Minimum relative probability
    max_tokens=256,
    ignore_eos=True      # Ignore EOS, continue generating
)
```

## ⚙️ Configuration Parameters

### LLM Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | str | Required | Model path (HuggingFace format) |
| `max_num_seqs` | int | 512 | Maximum number of sequences per batch |
| `max_num_batched_tokens` | int | 16384 | Maximum number of tokens per batch |
| `max_model_len` | int | 4096 | Maximum context length of the model |
| `device_memory_utilization` | float | 0.9 | Device memory utilization (0.1-1.0) |
| `tensor_parallel_size` | int | 1 | Tensor parallelism degree (1-8) |
| `enforce_eager` | bool | False | Force eager mode (disable CUDA Graph) |
| `kvcache_block_size` | int | 256 | KV cache block size (must be divisible by 256) |
| `num_kvcache_blocks` | int | -1 | Number of KV cache blocks (-1 for auto) |
| `trust_remote_code` | bool | False | Whether to trust remote code |
| `dtype` | str | 'auto' | Model weight data type |
| `seed` | int | None | Random seed |

### SamplingParams Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | float | 1.0 | Sampling temperature (> 1e-10) |
| `top_p` | float | 1.0 | Nucleus sampling threshold (0-1] |
| `top_k` | int | -1 | Top-k sampling (-1 to disable) |
| `min_p` | float | 0.0 | Minimum relative probability [0-1] |
| `max_tokens` | int | 64 | Maximum number of tokens to generate |
| `ignore_eos` | bool | False | Whether to ignore end-of-sequence token |

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_config.py -v

# Run tests matching a specific pattern
python -m pytest -k test_sampling

# Run tests with coverage
python -m pytest tests/ --cov=minivllm --cov-report=term-missing

# Use test runner script
python tests/run_tests.py --coverage -v
```

## 📚 Supported Models

- **Qwen2/Qwen3** (Recommended)
- **OPT**
- **LLaMA Series** (via Qwen2 compatibility)

## 📄 License

This project is open-sourced under the [Apache 2.0 License](LICENSE).

## 🤝 Acknowledgments

Special thanks to [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) for providing valuable design ideas and implementation references for mini-vLLM. Also, thanks to the following open-source projects:

- [vLLM](https://github.com/vllm-project/vllm) - Source of inspiration
- [Transformers](https://github.com/huggingface/transformers) - Model definitions and loading
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [FlashAttention](https://github.com/HazyResearch/flash-attention) - High-performance attention
- [Triton](https://github.com/openai/triton) - GPU kernel optimization


<div align="center">

**⭐ If this project helps you, please give us a star! ⭐**

</div>
