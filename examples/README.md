# mini-vLLM Examples

Example scripts demonstrating inference, sampling, flash attention, and tensor
parallelism on NPU / CPU. All scripts share the scaffolding in
[`example_utils.py`](example_utils.py) — model registry, `make_config`, the
common CLI args, and `timed_generate` — so each example stays focused on its own
point.

## Quick Start

```bash
# Activate environment
source set_env.sh

# Main inference (auto-detects NPU / CUDA / CPU)
python examples/inference_example.py

# NPU flash-attention benchmark
python examples/npu_flash_attention_example.py --benchmark

# Tensor parallelism TP=2
python examples/npu_tp_example.py --tp 2

# Force CPU
MINIVLLM_DEVICE=cpu python examples/inference_example.py
```

---

## Environment

Activate the CANN + Python environment before running any NPU example:

```bash
source /home/jianzhnie/llmtuner/llm/mini-vllm/tools/set_env.sh
```

## Model Short Names

The engine currently supports **Qwen3** only (see `minivllm/models/registry.py`).
All scripts share one registry — `example_utils.MODEL_PATHS` — with these short
names (mapped to local paths, so they run offline):

| Short name | Path |
|---|---|
| `qwen3` (default) | `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B` |
| `qwen3-1.7b` | `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-1.7B` |
| `qwen3-4b` | `/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-4B` |

You can also pass a full path to any local HuggingFace-format model directory.

---

## Examples

### 1. `inference_example.py` — Main Inference

The device-agnostic entry point: auto-detects NPU / CUDA / CPU and covers the
common knobs (tensor parallelism, flash attention, graph capture) plus an
optional boxed output mode.

```bash
python examples/inference_example.py
python examples/inference_example.py --model qwen3-4b --max-tokens 128
python examples/inference_example.py --flash-attn --no-eager
python examples/inference_example.py --tp 2
MINIVLLM_DEVICE=cpu python examples/inference_example.py --boxed
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen3` | Model short name or path |
| `--dtype` | `float16` | `float16`, `float32`, `bfloat16`, or `auto` |
| `--max-tokens` | `64` | Max tokens to generate per prompt |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-p` / `--top-k` | `0.95` / `40` | Sampling filters |
| `--max-model-len` | `512` | Max sequence length |
| `--tp` | `1` | Tensor parallelism size (1–8) |
| `--flash-attn` | off | Enable NPU flash attention |
| `--no-eager` | off | Enable graph capture (disable eager) |
| `--max-seqs` | `8` | Max concurrent sequences |
| `--prompt` | — | Add a prompt (repeatable) |
| `--boxed` | off | Render results in boxes |

---

### 2. `sampling_example.py` — Sampling Parameters & Strategies

One model load, two parts: a parameter sweep (temperature / top-k / top-p / min-p
on a single prompt) and a strategy comparison (greedy / balanced / creative over a
small batch, with timing).

```bash
python examples/sampling_example.py
python examples/sampling_example.py --model qwen3-4b --max-tokens 48
```

---

### 3. `npu_tp_example.py` — Tensor Parallelism

Verifies TP=1, TP=2, and TP=4 correctness and throughput.

```bash
python examples/npu_tp_example.py            # TP=1 baseline
python examples/npu_tp_example.py --tp 2     # TP=2 only
python examples/npu_tp_example.py --all      # TP=1,2,4 sequentially
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `qwen3` | Model short name or path |
| `--tp` | `0` | Single TP size (1/2/4); overrides `--all` |
| `--all` | off | Run TP=1, TP=2, TP=4 sequentially |
| `--max-tokens` | `48` | Max tokens per prompt |
| `--dtype` | `float16` | `float16`, `float32`, or `bfloat16` |

**Note:** TP=4 requires 4 NPU devices with HCCL peer-to-peer connectivity. On
some machines, TP=4 in `--all` mode may fail due to HCCL link timeouts between
runs; run TP=4 standalone (`--tp 4`) if this occurs.

---

### 4. `npu_flash_attention_example.py` — Flash Attention

Exercises the low-level `Attention` layer (prefill + decode) on NPU and compares
eager vs NPU flash attention.

```bash
python examples/npu_flash_attention_example.py                     # quick check
python examples/npu_flash_attention_example.py --benchmark         # full bench
python examples/npu_flash_attention_example.py --skip-low-level    # LLM only
```

**Note:** Flash attention is enabled via `MINIVLLM_USE_NPU_FA=1`. On CANN 8.2.RC1
the `npu_fused_infer_attention_score` and `npu_incre_flash_attention` APIs have
known compatibility issues, so the standard PyTorch SDPA path is the default and
is well-optimized on NPU.

---

### 5. `check_npu_graph.py` — NPU Environment Check

Quick diagnostic of NPU runtime and available flash-attention APIs: device
count/name, FA API availability (fusion, incremental, unified), and a functional
SDPA test.

```bash
python examples/check_npu_graph.py
```

---

### 6. `mp_event_demo.py` — Multiprocessing Event Demo

Demonstrates the `multiprocessing.Event` pattern used by the tensor-parallelism
worker processes (no LLM involved).

```bash
python examples/mp_event_demo.py
```

---

## Common Workflows

```bash
# Verify NPU environment
python examples/check_npu_graph.py

# Quick smoke test across two model sizes
python examples/inference_example.py --model qwen3 --max-tokens 16
python examples/inference_example.py --model qwen3-4b --max-tokens 16

# Flash-attention benchmark / TP verification
python examples/npu_flash_attention_example.py --benchmark
python examples/npu_tp_example.py --tp 2

# Verbose logs / force CPU
MINIVLLM_LOG_LEVEL=DEBUG python examples/inference_example.py
MINIVLLM_DEVICE=cpu python examples/inference_example.py --dtype float32
```
