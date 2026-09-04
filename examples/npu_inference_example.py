"""
NPU Inference Example — mini-vLLM on Huawei Ascend NPU.

Demonstrates end-to-end LLM inference on NPU with configurable model,
dtype, and execution mode. Use --flash-attn to enable NPU flash attention,
or --tp N for tensor parallelism.

Usage:
    python examples/npu_inference_example.py                           # eager, qwen3
    python examples/npu_inference_example.py --flash-attn              # NPU FA
    python examples/npu_inference_example.py --tp 2                    # TP=2
    python examples/npu_inference_example.py --model qwen3-4b --tp 4 --flash-attn
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from minivllm.utils.example_utils import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    make_config,
    print_banner,
    resolve_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-vLLM NPU Inference Example")
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model short name or path (see example_utils.MODEL_PATHS)",
    )
    p.add_argument(
        "--dtype", default="float16", choices=["float16", "float32", "bfloat16"]
    )
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--max-seqs", type=int, default=8)
    p.add_argument("--tp", type=int, default=1, help="Tensor parallelism size (1-8)")
    p.add_argument(
        "--flash-attn", action="store_true", help="Enable NPU flash attention"
    )
    p.add_argument(
        "--no-eager",
        action="store_true",
        help="Disable eager mode (enable graph capture)",
    )
    p.add_argument("--prompt", action="append", help="Add a prompt (repeatable)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Enable NPU FA if requested (must be set before first import of mini-vllm internals)
    if args.flash_attn:
        os.environ["MINIVLLM_USE_NPU_FA"] = "1"

    from minivllm import LLM, SamplingParams

    model_path = resolve_model(args.model)
    config = make_config(
        args.model,
        dtype=args.dtype,
        tp=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_seqs,
        device_memory_utilization=0.85,
        enforce_eager=not args.no_eager,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS

    # Header
    model_short = Path(model_path).name
    fa_status = "ON" if args.flash_attn else "OFF"
    print_banner(f"NPU Inference: {model_short}")
    print(f"  Model path:    {model_path}")
    print(f"  Dtype:         {args.dtype}")
    print(f"  Flash-Attn:    {fa_status}")
    print(f"  TP size:       {args.tp}")
    print(f"  Eager mode:    {not args.no_eager}")
    print(f"  Max tokens:    {args.max_tokens}")
    print(f"  Prompts:       {len(prompts)}")

    # Init
    t0 = time.perf_counter()
    llm = LLM(config)
    init_time = time.perf_counter() - t0
    print(f"\n  Engine init:   {init_time:.1f}s")

    # Inference
    t1 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
    infer_time = time.perf_counter() - t1

    total_tokens = sum(len(o["token_ids"]) for o in outputs)

    # Results
    print_banner("Results")
    for prompt, output in zip(prompts, outputs, strict=False):
        text = output["text"].strip()
        tokens = len(output["token_ids"])
        print(f"\n  [{tokens}t] Q: {prompt[:80]}")
        print(f"         A: {text[:200]}")

    print(
        f"\n  Total: {total_tokens} tokens in {infer_time:.2f}s "
        f"({total_tokens / infer_time:.1f} tok/s)"
    )
    print(f"  Init + Inference: {init_time + infer_time:.1f}s")

    del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
