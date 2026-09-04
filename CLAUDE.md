# CLAUDE.md

mini-vLLM — lightweight LLM inference engine (inspired by vLLM). Python 3.10–3.13, setuptools via `pyproject.toml`.
Code conventions: `@.claude/rules/`.

## Workflow

- **Verify every change**: run tests, check types, or show evidence. Never claim success without proof.
- **Plan before coding** for multi-file changes. Explore first, then plan, then implement.
- **Keep tasks focused**: `/clear` between unrelated work. One task per session.
- Write a failing test first, then make it pass.
- When compacting, preserve: modified file list, current branch, the last command you ran, and any uncommitted changes.

## What NOT to do

- Do NOT add features, abstractions, or error handling that weren't asked for.
- Do NOT refactor adjacent code while fixing something else.
- Do NOT explain what the code does in comments (well-named identifiers do that).
- Do NOT add configuration flags or backwards-compatibility shims for unrequested scenarios.
- Three similar lines is better than a premature abstraction.

## Commands

```bash
pip install -e ".[dev]"           # pytest, ruff, black, mypy
pip install -e ".[cuda]"          # flash-attn, triton
pip install -e ".[npu]"           # torch-npu

python -m pytest tests/                       # All tests (excludes slow)
python -m pytest tests/ --cov=minivllm        # Coverage
python -m pytest tests/ --slow                # Include slow tests
python -m pytest -k test_sampling             # Pattern match
python -m pytest -m integration               # By marker

black minivllm/ tests/ && ruff check minivllm/ tests/ && mypy minivllm/
pre-commit run --all-files
```

Test markers: `slow`, `integration`, `cuda`, `npu`.

## Common Workflows

### Before starting a task
- `/status` — see current branch, uncommitted changes, recent commits

### Fix a bug
1. **Reproduce**: write a minimal failing test
2. **Root cause**: use `@debugger` agent or `codegraph_trace` to trace the flow
3. **Fix**: minimal diff, only the broken code path
4. **Verify**: run the failing test → run `python -m pytest tests/ -x -q`

### Add a feature
1. **Explore**: use `@architect` for design review on multi-file changes
2. **Test first**: write a test, confirm it fails
3. **Implement**: minimal changes, no extra abstractions
4. **Review**: `/precheck` for lint, `@code-reviewer` for the diff

### Before committing
```bash
pre-commit run --all-files    # lint + format + type check
python -m pytest tests/ -x -q # fast test pass
```

### Code exploration
- **Structural questions** (what calls what): `codegraph_*` tools — see `.claude/CLAUDE.md`
- **Literal search** (strings, comments): `grep`
- **Don't delegate exploration to agents** — answer directly with 2-3 codegraph calls

### When context gets long
- `/context-save` before switching tasks
- `/context-restore` to pick up where you left off

## Architecture summary

```
User Prompt → LLM → LLMEngine → Scheduler (prefill+decode) → BlockManager (KV cache, xxhash prefix, CoW)
  → ModelRunner → InferenceExecutor (CUDA Graph, batch, sampling) → Output
```

- **Engine**: `llm_engine.py`, `scheduler.py` (two-phase), `block_manager.py` (block-based KV), `model_runner.py`, `inference_executor.py`, `distributed_manager.py` (nccl/hccl), `sequence.py`
- **Models**: `registry.py` (`SUPPORTED_MODELS` / `create_model`) + `manager.py` (load/lifecycle). Supported: Qwen3 only (GPT2/OPT/Qwen2 were dropped). `qwen_base.py` is the shared Qwen backbone; `qwen3.py` specializes it.
- **Layers**: `attention.py` (FlashAttention + SDPA fallback), `linear.py` (column/row/QKV parallel), `rotary_embedding.py`, `layernorm.py` (RMSNorm), `activation.py` (SiluAndMul), `attention_backend.py` (NPU/CUDA/SDPA backends), `page_attention.py` (paged decode)
- **Sampling**: `sampler.py` (penalties→temperature→top_k→top_p→min_p→multinomial, wired from `SamplingParams`), `mirostat.py` (standalone, not wired into the engine)
- **Utils**: `device.py`, `context.py` (contextvars), `loader.py` (safetensors), `logger_utils.py`

## Non-obvious rules

- `gpu_memory_utilization` is a backward-compat alias for `device_memory_utilization`.
- `SamplingParams` exposes `temperature/top_p/top_k/min_p/typical_p/repetition_penalty/frequency_penalty/presence_penalty/seed`; the executor forwards per-request penalties/typical-p/prev-tokens and a persistent per-request RNG (seeded) to the `Sampler`.
- KV cache block size must be divisible by 64.
- Rank 0 in main process, ranks 1-N as spawned workers; shared-memory IPC + event sync.
- `enforce_eager=True` disables CUDA Graph capture/replay.
- `Sequence` uses pickle-optimized serialization (not default pickle).

## Adding a new model

1. Create `minivllm/models/<name>.py` (reference: `qwen_base.py` / `qwen3.py`)
2. Register the class + its `model_type` in `SUPPORTED_MODELS` / `TYPE_TO_ARCH` in `minivllm/models/registry.py`
3. Ensure `manager.py`'s `create_model` resolves the architecture from `hf_config`

## NPU Optimizations

NPU-specific kernels (auto-detected when `torch_npu` is available):
- `npu_rms_norm` — fused RMSNorm
- `npu_rotary_mul` — fused RoPE (enabled by default; `MINIVLLM_USE_NPU_ROPE=0` to disable)
- `npu_swiglu` — fused SiluAndMul activation
- `npu_fusion_attention` / `npu_fused_infer_attention_score` — flash attention (opt-in: `MINIVLLM_USE_NPU_FA=1`)
- `npu_incre_flash_attention` — incremental decode attention

Key NPU perf considerations:
- Avoid per-step tensor allocation (pre-allocated decode buffers in `InferenceExecutor`)
- Cache attention masks and position grids (in `NPUAttentionBackend`, `PageAttention`)
- Use `index_copy_` for KV cache scatter (fastest on NPU)
- Use int32 indices for NPU index operations
- Prefer contiguous BNSD layout for attention kernels

Docs: `docs/npu_optimization.md`, `docs/npu_benchmark_report.md`, `docs/npu_flash_attention_guide.md`

## Debugging

| Symptom | Fix |
|---------|-----|
| OOM | Lower `device_memory_utilization` or `max_model_len` |
| Slow decode | Ensure `enforce_eager=False` and GPU/NPU available |
| KV cache exhaustion | Reduce `max_num_seqs` or increase `kvcache_block_size` |
| NPU garbled output | Check GQA model with `MINIVLLM_USE_NPU_FA=0` (unified FA doesn't support GQA) |

`logger.setLevel(logging.DEBUG)` for verbose output. `scheduler.block_manager.get_num_free_blocks()` to check KV cache.
