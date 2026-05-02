# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mini-vLLM is a lightweight LLM inference engine built from scratch, inspired by vLLM. It supports CUDA Graph optimization, tensor parallelism, KV cache management with prefix caching, and multi-device backends (CUDA, NPU, XPU, MPS, MLU, MUSA).

- **Language**: Python 3.10–3.12
- **Build system**: setuptools via `pyproject.toml` (no `setup.py`)

## Development Commands

### Setup

```bash
pip install -e ".[dev]"          # Dev dependencies (pytest, ruff, black, mypy)
pip install -e ".[cuda]"         # CUDA: flash-attn, triton
pip install -e ".[npu]"          # NPU: torch-npu
```

### Test

```bash
python -m pytest tests/                              # All tests (excludes slow)
python -m pytest tests/test_config.py -v             # Single file
python -m pytest -k test_sampling                    # Pattern match
python -m pytest tests/ --cov=minivllm               # Coverage
python -m pytest -m integration                      # By marker
python -m pytest tests/ --slow                       # Include slow tests
python tests/run_tests.py --coverage -v              # Runner script
```

Markers: `slow`, `integration`, `cuda`, `npu`.

### Lint & Format

```bash
black minivllm/ tests/
ruff check minivllm/ tests/
ruff check --fix minivllm/ tests/
mypy minivllm/
pre-commit run --all-files
```

Tools configured in `pyproject.toml`: ruff (E, F, UP, B, SIM, I rules), black (line-length 88), isort (black profile), mypy (strict).

### CI

Two GitHub Actions workflows (`.github/workflows/`):
- **pytests.yml**: lint (pre-commit + mypy) then test (Python 3.10/3.11 matrix, CPU torch)
- **code-quality.yml**: black/ruff/bandit/mypy

## Architecture

### Entry Points

- **`minivllm/llm.py`**: `LLM` class — extends `LLMEngine`, thin user-facing API
- **`minivllm/config.py`**: `Config` dataclass — validates all engine parameters, loads HF model config
- **`minivllm/sampling_params.py`**: `SamplingParams` dataclass — user-facing sampling parameters

### Engine Layer (`minivllm/engine/`)

- **`llm_engine.py`**: `LLMEngine` — orchestrates the pipeline, spawns worker processes, manages tokenizer reuse
- **`scheduler.py`**: `Scheduler` — two-phase scheduling (prefill + decode), preempts newest sequences under cache pressure
- **`block_manager.py`**: `BlockManager` — physical KV cache blocks with xxhash-based prefix caching and copy-on-write
- **`model_runner.py`**: `ModelRunner` — top-level coordinator delegating to `ModelManager`, `DistributedManager`, `InferenceExecutor`
- **`inference_executor.py`**: `InferenceExecutor` — KV cache allocation, model warmup, batch execution, CUDA Graph capture/replay, token sampling
- **`distributed_manager.py`**: `DistributedManager` — multi-process coordination via nccl/hccl/ccl/gloo backends
- **`sequence.py`**: `Sequence` — request state with token tracking, block table, pickle-optimized serialization

### Model Layer (`minivllm/models/`)

- **`manager.py`**: `ModelManager` — model loading, architecture detection, lifecycle management
- **`__init__.py`**: `create_model()` factory + model registry (maps HF config `architectures`/`model_type` to classes)
- Supported: `Qwen2ForCausalLM` (`qwen2.py`), `Qwen3ForCausalLM` (`qwen3.py`), `OPTForCausalLM` (`opt.py`)
- **`layers/`**: `attention.py` (FlashAttention + fallback), `attention_backend.py` (ABC + backend implementations), `rotary_embedding.py` (RoPE), `linear.py` (column/row/QKV parallel), `layernorm.py` (RMSNorm), `activation.py` (SiluAndMul), `embed_head.py` (parallel embedding/LM head), `npu_flash_attention.py` (NPU-specific)

### Sampling (`minivllm/sampling/`)

- **`sampler.py`**: `Sampler(nn.Module)` — pipeline: penalties → avoid_top_k → temperature → typical → top_k → top_p → min_p → multinomial
- **`functional.py`**: Stateless ops (`apply_temperature`, `apply_top_k`, `apply_top_p`, etc.), `torch.compile` on CUDA
- **`config.py`**: `SamplingConfig` — internal config with `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `typical_p`, `avoid_top_k`, `seed` (beyond user-facing `SamplingParams`)
- **`mirostat.py`**: `MirostatSampler`, `MirostatV2Sampler` — standalone stateful samplers (not wired into main pipeline)

### Utils (`minivllm/utils/`)

- **`device.py`**: Multi-device abstraction (CUDA/NPU/XPU/MPS/MLU/MUSA/CPU) — detection, memory, distributed backend selection
- **`context.py`**: Thread-safe inference context via `contextvars` — stores is_prefill, seq lengths, slot_mapping, block_tables
- **`loader.py`**: Weight loading (safetensors + .bin), packed module mapping (q/k/v → qkv), HF model ID download support
- **`logger_utils.py`**: Color-coded logging, distributed-aware (only rank 0 logs at INFO)
- **`random_utils.py`**: Cross-library seed setting (Python/numpy/torch/CUDA/NPU)

### Key Design Patterns

1. **Two-Phase Scheduling**: Prefill processes new sequences and populates KV cache; Decode generates one token per step reusing cached KV
2. **Block-Based KV Cache**: Fixed-size blocks (default 64 tokens, must be divisible by 64), xxhash prefix caching, copy-on-write
3. **Tensor Parallelism**: Rank 0 in main process, ranks 1-N as spawned workers; shared memory IPC + event synchronization
4. **CUDA Graph**: Captures decode graphs for batch sizes [1, 2, 4, 8, 16, 32, ..., max_num_seqs]; disable with `enforce_eager=True`

### Data Flow

```
User Prompt → LLM.generate() → LLMEngine.add_request()
  → Sequence → Scheduler.waiting → Prefill: InferenceExecutor.execute_model() → KV cache populated
  → Decode: one token per step → Sequence.FINISHED → Return output
```

## Config Validation Rules

Key constraints enforced in `Config.__post_init__()`:

| Field | Rule | Default |
|---|---|---|
| `model` | Existing local dir or HF model ID | (required) |
| `dtype` | `'auto'`, `'float16'`, `'bfloat16'`, `'float32'` | `'auto'` |
| `device_memory_utilization` | [0.1, 1.0] | 0.9 |
| `kvcache_block_size` | Must be divisible by 64 | 64 |
| `tensor_parallel_size` | [1, 8] | 1 |
| `max_num_batched_tokens` | Must be > 0, and >= `max_model_len` | 16384 |
| `max_num_seqs` | Must be > 0 | 512 |
| `max_model_len` | Must be > 0; auto-adjusted if exceeds model limits | 4096 |

`gpu_memory_utilization` is a backward-compat alias for `device_memory_utilization`.

## Sampling Parameters

**User-facing `SamplingParams`** (6 fields):
- `temperature`: > 1e-10 (greedy not permitted)
- `top_p`: (0, 1.0]
- `top_k`: -1 (disabled) or > 0
- `min_p`: [0, 1.0]
- `max_tokens`: > 0
- `ignore_eos`: bool

**Internal `SamplingConfig`** adds: `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `typical_p`, `avoid_top_k`, `seed`. These exist in the sampler pipeline but are not wired through from user input.

## Adding a New Model Architecture

1. Create model class in `minivllm/models/` following HuggingFace format (see `qwen2.py` as reference)
2. Register in `minivllm/models/__init__.py` model registry (`MODEL_REGISTRY` dict)
3. Add detection logic in `minivllm/models/manager.py` (`_detect_model_type()`)

## Running Inference

```python
from minivllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B", enforce_eager=False)
outputs = llm.generate(
    ["Once upon a time"],
    SamplingParams(temperature=0.7, max_tokens=128),
)
```

## Debugging

- Set `enforce_eager=True` to disable CUDA Graph
- Use `logger.setLevel(logging.DEBUG)` for verbose output
- Check KV cache via `scheduler.block_manager.get_num_free_blocks()`
