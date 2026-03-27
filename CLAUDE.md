# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mini-vLLM is a lightweight LLM inference engine built from scratch, inspired by vLLM. It provides a simple, efficient, and extensible inference solution with support for CUDA Graph optimization, tensor parallelism, KV cache management with prefix caching, and multi-device support (CUDA, NPU, XPU, MPS, MLU, MUSA).

## Development Commands

### Environment Setup

```bash
# Install in development mode
pip install -e .

# Install with CUDA optimizations (recommended for GPU)
pip install -e ".[cuda]"

# Install for NPU devices
pip install -e ".[npu]"

# Install development dependencies
pip install -e ".[dev]"
```

### Testing

```bash
# Run all tests (excludes slow tests by default)
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_config.py -v

# Run tests matching a pattern
python -m pytest -k test_sampling

# Run with coverage
python -m pytest tests/ --cov=minivllm --cov-report=term-missing

# Include slow tests
python -m pytest tests/ --slow

# Run integration tests only
python -m pytest -m integration

# Use the test runner script
python tests/run_tests.py --coverage -v
```

### Linting and Formatting

```bash
# Format with black
black minivllm/ tests/

# Lint with ruff
ruff check minivllm/ tests/

# Fix linting issues
ruff check --fix minivllm/ tests/

# Type check with mypy
mypy minivllm/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run on all files
pre-commit run --all-files
```

## Architecture Overview

mini-vLLM implements a two-phase scheduling inference engine with the following core components:

### Entry Points

- **`minivllm/llm.py`**: `LLM` class - thin wrapper around `LLMEngine` for user-facing API
- **`minivllm/config.py`**: `Config` dataclass - validates all engine parameters, loads HuggingFace model config

### Engine Layer (`minivllm/engine/`)

- **`llm_engine.py`**: `LLMEngine` - orchestrates the inference pipeline, manages distributed processes
- **`scheduler.py`**: `Scheduler` - implements two-phase scheduling (prefill + decode), handles KV cache pressure via preemption
- **`block_manager.py`**: `BlockManager` - manages physical KV cache blocks with prefix caching using xxhash
- **`model_runner.py`**: `ModelRunner` - handles model loading, KV cache allocation, CUDA Graph capture, distributed inference
- **`sequence.py`**: `Sequence` - represents individual requests with token tracking and state management

### Model Layer (`minivllm/models/`)

- **`manager.py`**: `ModelManager` - model loading, validation, and lifecycle management
- Supported architectures: `Qwen2ForCausalLM`, `Qwen3ForCausalLM`, `OPTForCausalLM`
- **`layers/attention.py`**: Attention module with FlashAttention integration (CUDA/NPU), fallback to standard attention
- **`layers/rotary_embedding.py`**: RoPE implementation with NPU kernel support

### Sampling (`minivllm/sampling/`)

- **`sampler.py`**: Token sampling with temperature, top-p, top-k, min-p support

### Key Design Patterns

1. **Two-Phase Scheduling**:
   - Prefill: Process new sequences, compute initial KV cache
   - Decode: Generate one token per sequence, reuse cached KV

2. **Block-Based KV Cache**:
   - Fixed-size blocks (default 64 tokens, must be divisible by 64)
   - Prefix caching via xxhash for memory sharing across sequences
   - Copy-on-write for efficient block updates

3. **Tensor Parallelism**:
   - Spawn worker processes for ranks 1-N (rank 0 runs in main process)
   - Shared memory IPC for inter-process communication
   - Events for synchronization

4. **CUDA Graph Optimization**:
   - Capture decode graphs for batch sizes 1 to `max_num_seqs`
   - Replay graphs to reduce kernel launch overhead
   - Disable with `enforce_eager=True`

### Data Flow

```
User Prompt → LLM.generate() → LLMEngine.add_request()
    → Sequence created → Scheduler.waiting queue
    → Prefill phase: ModelRunner.execute_model() → KV cache populated
    → Decode phase: Generate one token per step
    → Sequence.FINISHED → Return to user
```

## Multi-Device Support

The codebase supports multiple accelerator backends:

- **CUDA**: Full support with FlashAttention and CUDA Graph
- **NPU**: Flash Attention via transformers library, BNSD layout
- **XPU, MPS, MLU, MUSA**: Basic support via device abstraction layer

Device detection and backend selection is handled in `minivllm/utils/device.py`.

## Configuration Constraints

Key validation rules in `Config`:

- `kvcache_block_size`: Must be divisible by 64 (was 256, reduced in commit b560cbd)
- `device_memory_utilization`: Range [0.1, 1.0]
- `tensor_parallel_size`: Range [1, 8]
- `max_num_batched_tokens >= max_model_len`
- Model path must be a valid directory

## Sampling Parameters

`SamplingParams` supports:
- `temperature`: Must be > 1e-10 (greedy sampling not permitted)
- `top_p`: Range (0, 1.0]
- `top_k`: -1 (disabled) or > 0
- `min_p`: Range [0, 1.0]
- `max_tokens`: Must be > 0
- `ignore_eos`: Continue past EOS token

## Important Files

- **`minivllm/utils/context.py`**: Thread-local context for model execution state
- **`minivllm/utils/loader.py`**: Model weight loading utilities
- **`minivllm/utils/memory_pool.py`**: Memory pool management for KV cache

## Common Patterns

### Adding a New Model Architecture

1. Create model class in `minivllm/models/` following HuggingFace format
2. Register in `minivllm/models/manager.py` model registry
3. Add model creation logic in `minivllm/models/__init__.py` (`create_model` factory)

### Running Inference

```python
from minivllm import LLM, SamplingParams

llm = LLM(
    model="path/to/model",
    max_num_seqs=256,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,  # Multi-GPU: 2, 4, or 8
    enforce_eager=False,  # Enable CUDA Graph
)

outputs = llm.generate(
    ["Once upon a time"],
    SamplingParams(temperature=0.7, max_tokens=128)
)
```

### Debugging Tips

- Set `enforce_eager=True` to disable CUDA Graph for easier debugging
- Use `logger.setLevel(logging.DEBUG)` for verbose output
- Check KV cache utilization via `scheduler.block_manager.get_num_free_blocks()`
- For NPU issues, check `minivllm/models/layers/npu_flash_attention.py`
