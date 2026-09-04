"""NPU performance example — flash attention and tensor parallelism.

Merges the two NPU perf demos into one script with two sections:
  * FA  (--fa)          low-level Attention prefill+decode demo, then an
                        eager-vs-flash-attention benchmark (or quick run).
  * TP  (--tp N ...)    TP-size correctness / throughput comparison.

Run either section or both (default = FA).

Usage:
    python examples/npu_perf_example.py                    # FA section
    python examples/npu_perf_example.py --fa --benchmark   # full eager-vs-FA bench
    python examples/npu_perf_example.py --fa --skip-low-level
    python examples/npu_perf_example.py --tp 1 2 4         # TP comparison
    python examples/npu_perf_example.py --fa --benchmark --tp 1 2 4
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


# ---------------------------------------------------------------------------
# Flash-attention section
# ---------------------------------------------------------------------------


def demo_attention_prefill_decode() -> None:
    """Exercise the Attention layer prefill and decode paths on NPU."""
    from minivllm.models.layers.attention import Attention
    from minivllm.utils.context import reset_context, set_context

    print_banner("FA: Low-level Attention Layer (Prefill + Decode)")

    import torch

    num_heads, head_dim, num_kv_heads = 8, 64, 8
    scale = 1.0 / (head_dim**0.5)
    device = torch.device("npu:0")

    attn = Attention(num_heads=num_heads, head_dim=head_dim, scale=scale, num_kv_heads=num_kv_heads)
    print(f"  Backend: {attn.backend.__class__.__name__}")

    # --- Prefill (packed, 2 sequences) ---
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

    # --- Decode (batch=2, paged KV) ---
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


def run_fa_benchmark(model_path: str, max_tokens: int) -> None:
    """Compare eager vs NPU flash attention on the full pipeline."""
    print_banner("FA: Benchmark — Eager vs Flash-Attention")

    model_name = Path(model_path).name
    print(f"  Model: {model_name}  Prompts: {len(DEFAULT_PROMPTS)}  Max tokens: {max_tokens}")

    print("\n  [1/2] Running in EAGER mode (no flash-attn)...")
    eager = _fa_run(model_path, max_tokens, use_fa=False)
    print(f"  Init: {eager['init_s']:.1f}s  Inference: {eager['infer_s']:.2f}s  "
          f"Tokens: {eager['tokens']}  Throughput: {eager['tok_s']:.1f} tok/s")

    print("  [2/2] Running with NPU Flash-Attention...")
    fa = _fa_run(model_path, max_tokens, use_fa=True)
    print(f"  Init: {fa['init_s']:.1f}s  Inference: {fa['infer_s']:.2f}s  "
          f"Tokens: {fa['tokens']}  Throughput: {fa['tok_s']:.1f} tok/s")

    if eager["infer_s"] > 0:
        speedup = eager["infer_s"] / fa["infer_s"] if fa["infer_s"] > 0 else float("inf")
        label = f"FA is {speedup:.1f}x faster" if speedup >= 1 else f"FA is {1 / speedup:.1f}x slower"
        print("\n  --- Comparison ---")
        print(f"  Eager:  {eager['infer_s']:.2f}s  ({eager['tok_s']:.1f} tok/s)")
        print(f"  FA:     {fa['infer_s']:.2f}s  ({fa['tok_s']:.1f} tok/s)")
        print(f"  Result: {label}")


def _fa_run(model_path: str, max_tokens: int, use_fa: bool) -> dict:
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
    return {"init_s": init_t, "infer_s": stats["elapsed_s"], "tokens": stats["tokens"],
            "tok_s": stats["tok_s"]}


def fa_section(model_path: str, args) -> None:
    if not args.skip_low_level:
        try:
            demo_attention_prefill_decode()
        except Exception as e:
            print(f"\n  Low-level demo skipped: {e}")

    if args.benchmark:
        run_fa_benchmark(model_path, args.max_tokens)
    else:
        # Quick run: single inference with FA enabled.
        print_banner("FA: Quick Inference (Flash-Attention on)")
        _stats = _fa_run(model_path, args.max_tokens, use_fa=True)
        print(f"  {_stats['tokens']} tokens in {_stats['infer_s']:.2f}s "
              f"({_stats['tok_s']:.1f} tok/s)")


# ---------------------------------------------------------------------------
# Tensor-parallel section
# ---------------------------------------------------------------------------


def check_tp_available(tp: int) -> bool:
    import torch

    count = torch.npu.device_count() if hasattr(torch, "npu") else 0
    if count < tp:
        print(f"  SKIP: need {tp} NPU devices, found {count}")
        return False
    return True


def run_tp_inference(model_path: str, tp: int, max_tokens: int, dtype: str) -> dict:
    from minivllm import LLM, SamplingParams

    config = make_config(
        model_path, dtype=dtype, tp=tp,
        device_memory_utilization=0.8, enforce_eager=True,
    )
    params = SamplingParams(temperature=0.7, top_p=0.95, top_k=40, max_tokens=max_tokens)

    t0 = time.perf_counter()
    llm = LLM(config)
    init_t = time.perf_counter() - t0

    outputs, stats = timed_generate(llm, DEFAULT_PROMPTS, params, use_tqdm=False)
    llm.exit()

    return {
        "tp": tp, "init_s": round(init_t, 1), "infer_s": round(stats["elapsed_s"], 2),
        "tokens": stats["tokens"], "tok_s": round(stats["tok_s"], 1),
        "texts": [o["text"].strip() for o in outputs],
    }


def print_tp_result(r: dict) -> None:
    print(f"  TP={r['tp']}: init={r['init_s']}s  infer={r['infer_s']}s  "
          f"tokens={r['tokens']}  throughput={r['tok_s']} tok/s")
    for i, (prompt, text) in enumerate(zip(DEFAULT_PROMPTS, r["texts"], strict=True)):
        print(f"    [{i}] Q: {prompt[:60]}")
        print(f"        A: {text[:120]}{'...' if len(text) > 120 else ''}")


def tp_section(model_path: str, args) -> None:
    os.environ.pop("MINIVLLM_USE_NPU_FA", None)  # TP uses standard attention

    model_name = Path(model_path).name
    tp_sizes = args.tp if args.tp else [1]
    print_banner(f"TP: Tensor Parallelism — {model_name}")
    print(f"  Dtype: {args.dtype}   Max tokens: {args.max_tokens}   Sizes: {tp_sizes}")

    results = []
    for tp in tp_sizes:
        if not check_tp_available(tp):
            continue
        print(f"\n  Running TP={tp}...")
        try:
            r = run_tp_inference(model_path, tp, args.max_tokens, args.dtype)
            results.append(r)
            print_tp_result(r)
        except Exception as e:
            print(f"  TP={tp} FAILED: {e}")
            if "HCCL" in str(e) or "port" in str(e).lower():
                print(f"  NOTE: HCCL port conflict — retry standalone:\n"
                      f"         python examples/npu_perf_example.py --tp {tp} --model <model>")

        # Let workers + HCCL release resources between runs.
        if len(tp_sizes) > 1:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
            os.environ.pop("MASTER_PORT", None)
            time.sleep(3)

    if len(results) > 1:
        print(f"\n{'=' * 70}\n  Summary\n{'=' * 70}")
        for r in results:
            print(f"  TP={r['tp']}: {r['tok_s']} tok/s  ({r['tokens']} tokens, {r['infer_s']}s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NPU performance: flash attention and tensor parallelism"
    )
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"Model short name ({', '.join(MODEL_PATHS)}) or path")
    p.add_argument("--dtype", default="float16", choices=("float16", "float32", "bfloat16"))
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--fa", action="store_true", help="Run the flash-attention section")
    p.add_argument("--benchmark", action="store_true",
                   help="FA section: full eager-vs-FA benchmark (default: quick run)")
    p.add_argument("--skip-low-level", action="store_true",
                   help="FA section: skip the low-level attention layer demos")
    p.add_argument("--tp", type=int, nargs="+", default=None, metavar="N",
                   help="Run the TP section for these sizes (e.g. --tp 1 2 4)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not check_npu():
        return 1
    torch = __import__("torch")
    torch.npu.set_device(0)

    model_path = resolve_model(args.model)

    # Default to the FA section when no section is explicitly requested.
    run_fa = args.fa or args.tp is None
    run_tp = args.tp is not None

    if run_fa:
        fa_section(model_path, args)
    if run_tp:
        tp_section(model_path, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
