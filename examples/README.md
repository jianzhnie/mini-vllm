# mini-vLLM Examples

Example scripts demonstrating inference, flash attention, and tensor parallelism on NPU / CPU.

## Quick Start

```bash
# Activate environment
source set_env.sh

# Quick inference (auto-detects NPU)
python examples.py

# NPU inference with Qwen
python examples/npu_inference_example.py --model qwen

# Flash attention benchmark
python examples/npu_flash_attention_example.py --benchmark

# Tensor parallelism TP=2
python examples/npu_tp_example.py --tp 2
```

---

## Environment

Activate the CANN + Python environment before running any example:

```bash
source /home/jianzhnie/llmtuner/llm/mini-vllm/set_env.sh
```

## Model Short Names

All scripts accept the following short names (mapped to local model paths):

| Short name | Path |
|---|---|
| `opt` (default) | `/home/jianzhnie/llmtuner/hfhub/models/facebook/opt-125m` |
| `qwen` / `qwen3` | `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B` |
| `qwen3-1.7b` | `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-1.7B` |
| `qwen3-4b` | `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-4B` |
| `gpt2` | `/home/jianzhnie/llmtuner/hfhub/models/openai-community/gpt2` |

You can also pass a full path to any HuggingFace-format model directory.

---

## Examples

### 1. `examples.py` — Main Example

Auto-detects NPU and runs inference. The simplest entry point.

```bash
python examples.py                          # default: Qwen3-0.6B, float16
python examples.py --model opt              # opt-125m
python examples.py --model /path/to/model   # custom model
python examples.py --dtype float32          # float32 precision
python examples.py --max-tokens 128         # longer output
```

**Options**

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen` | Model short name or path |
| `--dtype` | `float16` | `float16`, `float32`, or `auto` |
| `--max-tokens` | `64` | Max tokens to generate per prompt |
| `--temperature` | `0.7` | Sampling temperature |
| `--max-model-len` | `512` | Max sequence length |
| `--eager` | `True` | Force eager mode (disable CUDA Graph) |

---

### 2. `npu_inference_example.py` — NPU Inference

Comprehensive NPU inference with flash attention and tensor parallelism toggles.

```bash
# Basic eager mode
python examples/npu_inference_example.py

# With flash attention
python examples/npu_inference_example.py --flash-attn

# With tensor parallelism
python examples/npu_inference_example.py --tp 2
python examples/npu_inference_example.py --tp 4

# Combined
python examples/npu_inference_example.py --model qwen --tp 2 --flash-attn

# Custom prompt
python examples/npu_inference_example.py --prompt "What is AI?"
```

**Options**

| Flag | Default | Description |
|---|---|---|
| `--model` | `opt` | Model short name or path |
| `--dtype` | `float16` | `float16`, `float32`, `bfloat16` |
| `--max-tokens` | `64` | Max tokens per prompt |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` | `0.95` | Top-p sampling |
| `--top-k` | `40` | Top-k sampling |
| `--max-model-len` | `512` | Max sequence length |
| `--max-seqs` | `8` | Max concurrent sequences |
| `--tp` | `1` | Tensor parallelism size (1–8) |
| `--flash-attn` | off | Enable NPU flash attention |
| `--no-eager` | off | Disable eager mode (enable graph capture) |
| `--prompt` | — | Add a prompt (repeatable) |

---

### 3. `npu_flash_attention_example.py` — Flash Attention

Compares eager mode vs NPU flash attention. Includes low-level attention layer demos.

```bash
# Quick inference (eager, no benchmark)
python examples/npu_flash_attention_example.py

# Full benchmark (eager vs FA comparison)
python examples/npu_flash_attention_example.py --benchmark
python examples/npu_flash_attention_example.py --benchmark --model qwen

# Skip low-level layer demos
python examples/npu_flash_attention_example.py --skip-low-level
```

**Options**

| Flag | Default | Description |
|---|---|---|
| `--model` | `opt` | Model short name or path |
| `--max-tokens` | `48` | Max tokens per prompt |
| `--skip-low-level` | off | Skip attention layer demos |
| `--benchmark` | off | Run full eager vs FA comparison |

**Note:** Flash attention is enabled via `MINIVLLM_USE_NPU_FA=1`. On CANN 8.2.RC1, the `npu_fused_infer_attention_score` and `npu_incre_flash_attention` APIs have known compatibility issues; the standard PyTorch SDPA path is used by default and is well-optimized on NPU.

---

### 4. `npu_tp_example.py` — Tensor Parallelism

Verifies TP=1, TP=2, and TP=4 correctness and throughput.

```bash
# Baseline TP=1
python examples/npu_tp_example.py

# TP=2 only
python examples/npu_tp_example.py --tp 2

# TP=4 only
python examples/npu_tp_example.py --tp 4

# Test all TP sizes sequentially
python examples/npu_tp_example.py --all

# TP with Qwen
python examples/npu_tp_example.py --tp 2 --model qwen
```

**Options**

| Flag | Default | Description |
|---|---|---|
| `--model` | `opt` | Model short name or path |
| `--tp` | `0` | Single TP size (1/2/4); overrides `--all` |
| `--all` | off | Run TP=1, TP=2, TP=4 sequentially |
| `--max-tokens` | `48` | Max tokens per prompt |
| `--dtype` | `float16` | `float16`, `float32`, or `bfloat16` |

**Note:** TP=4 requires 4 NPU devices with HCCL peer-to-peer connectivity. On some machines, TP=4 in `--all` mode may fail due to HCCL link timeouts between runs; try running TP=4 standalone (`--tp 4`) if this occurs.

---

### 5. `check_npu_graph.py` — NPU Environment Check

Quick diagnostic of NPU runtime and available flash attention APIs.

```bash
python examples/check_npu_graph.py
```

Checks:
- NPU device count and name
- Flash attention API availability (fusion, incremental, unified)
- Functional SDPA test

---

### 6. `cpu_inference_opt.py` — CPU Inference

Forces execution on CPU, useful as a golden reference for output comparison.

```bash
python examples/cpu_inference_opt.py
python examples/cpu_inference_opt.py --model qwen3
python examples/cpu_inference_opt.py --model qwen3-1.7b
```

Accepts `--model` with the same short names as other scripts. Hides all accelerator devices by setting `MINIVLLM_DEVICE=cpu`.

---

### 7. `mp_event_demo.py` — Multiprocessing Event Demo

Demonstrates the `multiprocessing.Event` pattern used by the tensor parallelism worker processes.

```bash
python examples/mp_event_demo.py
```

---

## Common Workflows

### Verify NPU environment

```bash
python examples/check_npu_graph.py
```

### Quick smoke test (all in one)

```bash
# Eager mode with both models
python examples/npu_inference_example.py --model opt --max-tokens 16
python examples/npu_inference_example.py --model qwen --max-tokens 16

# Flash attention benchmark
python examples/npu_flash_attention_example.py --benchmark

# TP=2 verification
python examples/npu_tp_example.py --tp 2
```

### Performance comparison (eager vs flash attention)

```bash
python examples/npu_flash_attention_example.py --benchmark --model qwen
```

### Tensor parallelism verification

```bash
# Sequential test (TP=1, TP=2, TP=4)
python examples/npu_tp_example.py --all --max-tokens 32

# Individual TP sizes
python examples/npu_tp_example.py --tp 2 --model qwen
```

### Debug mode

Set `MINIVLLM_LOG_LEVEL=DEBUG` for verbose logs:

```bash
MINIVLLM_LOG_LEVEL=DEBUG python examples/npu_inference_example.py
```

Enable NPU flash attention:

```bash
MINIVLLM_USE_NPU_FA=1 python examples/npu_inference_example.py --flash-attn
```

Force CPU:

```bash
MINIVLLM_DEVICE=cpu python examples/npu_inference_example.py --dtype float32
```
