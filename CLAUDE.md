# CLAUDE.md

mini-vLLM is a lightweight LLM inference engine built from scratch (inspired by vLLM). Supports CUDA Graph, tensor parallelism, block-based KV cache with prefix caching, and multi-device backends (CUDA, NPU, XPU, MPS, MLU, MUSA).

- **Language**: Python 3.10–3.12
- **Build system**: setuptools via `pyproject.toml`
- **Code conventions**: `.claude/rules/` — Python, Git, Shell, compatibility

## Development Commands

```bash
pip install -e ".[dev]"           # dev: pytest, ruff, black, mypy
pip install -e ".[cuda]"          # CUDA: flash-attn, triton
pip install -e ".[npu]"           # NPU: torch-npu
```

```bash
python -m pytest tests/                       # All tests (excludes slow)
python -m pytest tests/ --cov=minivllm        # Coverage
python -m pytest tests/ --slow                # Include slow tests
python -m pytest -m integration               # By marker
python -m pytest -k test_sampling             # Pattern match
```

Markers: `slow`, `integration`, `cuda`, `npu`.

```bash
black minivllm/ tests/
ruff check minivllm/ tests/
ruff check --fix minivllm/ tests/
mypy minivllm/
pre-commit run --all-files
```

## Architecture

### Entry Points

| File | Purpose |
|------|---------|
| `minivllm/llm.py` | `LLM` class — thin user-facing API over `LLMEngine` |
| `minivllm/config.py` | `Config` dataclass — validates engine params, loads HF config |
| `minivllm/sampling_params.py` | `SamplingParams` dataclass — user-facing sampling params |

### Engine Layer (`minivllm/engine/`)

| File | Purpose |
|------|---------|
| `llm_engine.py` | `LLMEngine` — orchestrates pipeline, spawns workers, manages tokenizer reuse |
| `scheduler.py` | `Scheduler` — two-phase (prefill + decode), preempts newest under cache pressure |
| `block_manager.py` | `BlockManager` — physical KV cache blocks, xxhash prefix caching, copy-on-write |
| `model_runner.py` | `ModelRunner` — delegates to `ModelManager`, `DistributedManager`, `InferenceExecutor` |
| `inference_executor.py` | `InferenceExecutor` — KV allocation, warmup, batch execution, CUDA Graph, token sampling |
| `distributed_manager.py` | `DistributedManager` — multi-process via nccl/hccl/ccl/gloo |
| `sequence.py` | `Sequence` — request state: tokens, block table, serialization |

### Model Layer (`minivllm/models/`)

- `manager.py`: `ModelManager` — loading, architecture detection, lifecycle
- `__init__.py`: `create_model()` factory + `MODEL_REGISTRY` (maps HF `architectures`/`model_type` to classes)
- Models: `Qwen2ForCausalLM` (`qwen2.py`), `Qwen3ForCausalLM` (`qwen3.py`), `OPTForCausalLM` (`opt.py`)
- `layers/`: `attention.py`, `attention_backend.py`, `rotary_embedding.py`, `linear.py`, `layernorm.py`, `activation.py`, `embed_head.py`, `npu_flash_attention.py`

### Sampling (`minivllm/sampling/`)

- `sampler.py`: `Sampler(nn.Module)` — pipeline: penalties → avoid_top_k → temperature → typical → top_k → top_p → min_p → multinomial
- `functional.py`: Stateless ops (`apply_temperature`, `apply_top_k`, `apply_top_p`, etc.)
- `config.py`: `SamplingConfig` — internal config (repetition/frequency/presence penalties, typical_p, avoid_top_k, seed)
- `mirostat.py`: `MirostatSampler`, `MirostatV2Sampler` — standalone, not wired into main pipeline

### Utils (`minivllm/utils/`)

- `device.py` — multi-device detection (CUDA/NPU/XPU/MPS/MLU/MUSA/CPU), memory, distributed backend
- `context.py` — thread-safe `contextvars` (is_prefill, seq lengths, slot_mapping, block_tables)
- `loader.py` — weight loading (safetensors + .bin), packed module mapping (q/k/v → qkv), HF download
- `logger_utils.py` — color-coded logging, distributed-aware (rank 0 only at INFO)
- `random_utils.py` — cross-library seed setting (Python/numpy/torch/CUDA/NPU)

### Data Flow

```
User Prompt → LLM.generate() → LLMEngine.add_request()
  → Sequence → Scheduler.waiting → Prefill: InferenceExecutor.execute_model() → KV cache populated
  → Decode: one token per step → Sequence.FINISHED → Return output
```

## Config Validation Rules

| Field | Rule | Default |
|---|---|---|
| `model` | Existing local dir or HF model ID | (required) |
| `dtype` | `'auto'`, `'float16'`, `'bfloat16'`, `'float32'` | `'auto'` |
| `device_memory_utilization` | [0.1, 1.0] | 0.9 |
| `kvcache_block_size` | Divisible by 64 | 64 |
| `tensor_parallel_size` | [1, 8] | 1 |
| `max_num_batched_tokens` | > 0, >= `max_model_len` | 16384 |
| `max_num_seqs` | > 0 | 512 |
| `max_model_len` | > 0; auto-capped to model max | 4096 |

`gpu_memory_utilization` is a backward-compat alias for `device_memory_utilization`.

## Sampling Parameters

**`SamplingParams`** (user-facing): `temperature` (> 1e-10), `top_p` ((0, 1.0]), `top_k` (-1 or > 0), `min_p` ([0, 1.0]), `max_tokens` (> 0), `ignore_eos` (bool).

**`SamplingConfig`** (internal) adds `repetition_penalty`, `frequency_penalty`, `presence_penalty`, `typical_p`, `avoid_top_k`, `seed` — not wired through from user input.

## Adding a New Model

1. Create model class in `minivllm/models/` following HuggingFace format (reference: `qwen2.py`)
2. Register in `minivllm/models/__init__.py` → `MODEL_REGISTRY` dict
3. Add detection in `minivllm/models/manager.py` → `_detect_model_type()`

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

| Symptom | Fix |
|---------|-----|
| OOM | Lower `device_memory_utilization` or `max_model_len` |
| Slow decode | Ensure `enforce_eager=False` and GPU available |
| KV cache exhaustion | Reduce `max_num_seqs` or increase `kvcache_block_size` |

- `enforce_eager=True` disables CUDA Graph for debugging
- `logger.setLevel(logging.DEBUG)` for verbose output
- `scheduler.block_manager.get_num_free_blocks()` to check KV cache
