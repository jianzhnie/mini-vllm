"""NPU Flash Attention Example — compare eager vs NPU flash attention.

Runs the same prompts with and without NPU flash attention and reports the
speedup. Also exercises the low-level ``Attention`` layer (prefill + decode)
directly on NPU.

Usage:
    python examples/npu_flash_attention_example.py                     # quick check
    python examples/npu_flash_attention_example.py --benchmark         # full bench
    python examples/npu_flash_attention_example.py --model qwen3-4b --benchmark
    python examples/npu_flash_attention_example.py --skip-low-level    # LLM only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from example_utils import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    MODEL_PATHS,
    make_config,
    print_banner,
    resolve_model,
    timed_generate,
)


def check_npu() -> bool:
    """Verify NPU is available."""
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        print("ERROR: NPU not available.")
        return False
    print(f"NPU: {torch.npu.get_device_name(0)}, {torch.npu.device_count()} device(s)")
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NPU Flash Attention Example")
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model short name ({', '.join(MODEL_PATHS)}) or path",
    )
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--skip-low-level", action="store_true",
                   help="Skip low-level attention layer demos")
    p.add_argument("--benchmark", action="store_true",
                   help="Run full benchmark (eager vs FA comparison)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Low-level attention layer demo
# ---------------------------------------------------------------------------


def demo_attention_prefill_decode() -> None:
    """Exercise Attention layer prefill and decode paths on NPU."""
    from minivllm.models.layers.attention import Attention
    from minivllm.utils.context import reset_context, set_context

    print_banner("Low-level: Attention Layer (Prefill + Decode)")

    import torch

    num_heads, head_dim, num_kv_heads = 8, 64, 8
    scale = 1.0 / (head_dim**0.5)
    device = torch.device("npu:0")

    attn = Attention(num_heads=num_heads, head_dim=head_dim, scale=scale, num_kv_heads=num_kv_heads)
    print(f"  Backend: {attn.backend.__class__.__name__}")

    # --- Prefill ---
    print("\n  --- Prefill (packed, 2 sequences) ---")
    batch_sizes = [4, 6]
    total_tokens = sum(batch_sizes)
    max_s = max(batch_sizes)

    q = torch.randn(total_tokens, num_heads, head_dim, device=device, dtype=torch.float16)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    cum = torch.tensor([0, 4, 10], dtype=torch.int32, device=device)
    set_context(
        is_prefill=True, max_seqlen_q=max_s, max_seqlen_k=max_s,
        cum_seqlens_q=cum, cum_seqlens_k=cum,
        slot_mapping=torch.arange(total_tokens, device=device),
    )
    with torch.no_grad():
        out = attn(q, k, v)
    reset_context()
    print(f"  q={list(q.shape)}  k={list(k.shape)}  v={list(v.shape)}")
    print(f"  out={list(out.shape)}  device={out.device}")

    # --- Decode ---
    print("\n  --- Decode (batch=2, block_size=16) ---")
    block_size, num_blocks = 16, 4
    attn.k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                               device=device, dtype=torch.float16)
    attn.v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                               device=device, dtype=torch.float16)
    attn._cache_initialized = True

    q_d = torch.randn(2, num_heads, head_dim, device=device, dtype=torch.float16)
    k_d = torch.randn(2, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v_d = torch.randn(2, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    set_context(
        is_prefill=False,
        slot_mapping=torch.tensor([0, block_size], dtype=torch.int32, device=device),
        context_lens=torch.tensor([3, 5], dtype=torch.int32, device=device),
        block_tables=torch.tensor([[0, -1], [1, 2]], dtype=torch.int32, device=device),
    )
    with torch.no_grad():
        out = attn(q_d, k_d, v_d)
    reset_context()
    print(f"  q={list(q_d.shape)}  out={list(out.shape)}")
    print("  Low-level attention demo: PASSED")


# ---------------------------------------------------------------------------
# LLM inference benchmark
# ---------------------------------------------------------------------------


def run_llm_benchmark(model_path: str, max_tokens: int, use_fa: bool) -> dict:
    """Run LLM inference and return timing stats."""
    if use_fa:
        os.environ["MINIVLLM_USE_NPU_FA"] = "1"
    else:
        os.environ.pop("MINIVLLM_USE_NPU_FA", None)

    from minivllm import LLM, SamplingParams

    config = make_config(
        model_path, dtype="float16", device_memory_utilization=0.85, enforce_eager=True
    )
    params = SamplingParams(temperature=0.7, top_p=0.95, top_k=40, max_tokens=max_tokens)

    t0 = time.perf_counter()
    llm = LLM(config)
    init_t = time.perf_counter() - t0

    outputs, stats = timed_generate(llm, DEFAULT_PROMPTS, params, use_tqdm=False)
    llm.exit()

    return {
        "init_s": init_t,
        "infer_s": stats["elapsed_s"],
        "tokens": stats["tokens"],
        "tok_s": stats["tok_s"],
        "texts": [o["text"].strip()[:80] for o in outputs],
    }


def demo_llm_benchmark(model_path: str, max_tokens: int) -> None:
    """Compare eager vs NPU flash attention."""
    print_banner("LLM Benchmark: Eager vs Flash-Attention")

    model_name = Path(model_path).name
    print(f"  Model: {model_name}  Prompts: {len(DEFAULT_PROMPTS)}  Max tokens: {max_tokens}")

    print("\n  [1/2] Running in EAGER mode (no flash-attn)...")
    eager = run_llm_benchmark(model_path, max_tokens, use_fa=False)
    print(f"  Init: {eager['init_s']:.1f}s  Inference: {eager['infer_s']:.2f}s  "
          f"Tokens: {eager['tokens']}  Throughput: {eager['tok_s']:.1f} tok/s")

    print("  [2/2] Running with NPU Flash-Attention...")
    fa = run_llm_benchmark(model_path, max_tokens, use_fa=True)
    print(f"  Init: {fa['init_s']:.1f}s  Inference: {fa['infer_s']:.2f}s  "
          f"Tokens: {fa['tokens']}  Throughput: {fa['tok_s']:.1f} tok/s")

    print("\n  --- Comparison ---")
    if eager["infer_s"] > 0:
        speedup = eager["infer_s"] / fa["infer_s"] if fa["infer_s"] > 0 else float("inf")
        label = f"FA is {speedup:.1f}x faster" if speedup >= 1 else f"FA is {1 / speedup:.1f}x slower"
        print(f"  Eager:  {eager['infer_s']:.2f}s  ({eager['tok_s']:.1f} tok/s)")
        print(f"  FA:     {fa['infer_s']:.2f}s  ({fa['tok_s']:.1f} tok/s)")
        print(f"  Result: {label}")

    if eager["texts"]:
        print(f"\n  Eager sample: {eager['texts'][0][:100]!r}")
    if fa["texts"]:
        print(f"  FA sample:    {fa['texts'][0][:100]!r}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    if not check_npu():
        return 1

    torch = __import__("torch")
    torch.npu.set_device(0)

    model_path = resolve_model(args.model)

    if not args.skip_low_level:
        try:
            demo_attention_prefill_decode()
        except Exception as e:
            print(f"\n  Low-level demo skipped: {e}")

    if args.benchmark:
        try:
            demo_llm_benchmark(model_path, args.max_tokens)
        except Exception as e:
            print(f"\n  Benchmark failed: {e}")
            return 1
    else:
        # Quick run: single inference with FA
        print_banner("Quick Inference (Eager, no FA)")
        os.environ.pop("MINIVLLM_USE_NPU_FA", None)
        from minivllm import LLM, SamplingParams

        config = make_config(
            model_path, dtype="float16", device_memory_utilization=0.85, enforce_eager=True
        )
        params = SamplingParams(temperature=0.7, top_p=0.95, top_k=40, max_tokens=args.max_tokens)

        llm = LLM(config)
        outputs, stats = timed_generate(llm, DEFAULT_PROMPTS[:2], params)
        for p, o in zip(DEFAULT_PROMPTS[:2], outputs, strict=True):
            print(f"\n  Q: {p}\n  A: {o['text'].strip()[:150]} ({len(o['token_ids'])} tokens)")
        print(f"\n  {stats['tokens']} tokens in {stats['elapsed_s']:.2f}s "
              f"({stats['tok_s']:.1f} tok/s)")
        llm.exit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
