"""NPU Inference Example — mini-vLLM on Huawei Ascend NPU.

End-to-end LLM inference on NPU with configurable model, dtype, flash
attention (--flash-attn), and tensor parallelism (--tp N).

Usage:
    python examples/npu_inference_example.py
    python examples/npu_inference_example.py --flash-attn
    python examples/npu_inference_example.py --tp 2
    python examples/npu_inference_example.py --model qwen3-4b --tp 4 --flash-attn
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from example_utils import (
    DEFAULT_PROMPTS,
    add_common_args,
    build_llm,
    print_banner,
    sample_params,
    timed_generate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-vLLM NPU Inference Example")
    add_common_args(p)
    p.add_argument("--max-seqs", type=int, default=8)
    p.add_argument("--tp", type=int, default=1, help="Tensor parallelism size (1-8)")
    p.add_argument("--flash-attn", action="store_true", help="Enable NPU flash attention")
    p.add_argument(
        "--no-eager",
        action="store_true",
        help="Disable eager mode (enable graph capture)",
    )
    p.add_argument("--prompt", action="append", help="Add a prompt (repeatable)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Must be set before minivllm internals are imported (build_llm defers it).
    if args.flash_attn:
        os.environ["MINIVLLM_USE_NPU_FA"] = "1"

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS

    print_banner(f"NPU Inference: {args.model}")
    print(f"  Dtype:       {args.dtype}   Flash-Attn: {'ON' if args.flash_attn else 'OFF'}")
    print(f"  TP size:     {args.tp}   Eager mode: {not args.no_eager}")
    print(f"  Max tokens:  {args.max_tokens}   Prompts: {len(prompts)}")

    t0 = time.perf_counter()
    llm, config = build_llm(
        args,
        tp=args.tp,
        max_num_seqs=args.max_seqs,
        device_memory_utilization=0.85,
        enforce_eager=not args.no_eager,
    )
    init_time = time.perf_counter() - t0
    print(f"\n  Model path:    {config.model}")
    print(f"  Engine init:   {init_time:.1f}s")

    params = sample_params(args)
    outputs, stats = timed_generate(llm, prompts, params)

    print_banner("Results")
    for prompt, output in zip(prompts, outputs, strict=True):
        text = output["text"].strip()
        print(f"\n  [{len(output['token_ids'])}t] Q: {prompt[:80]}")
        print(f"         A: {text[:200]}")
    print(
        f"\n  Total: {stats['tokens']} tokens in {stats['elapsed_s']:.2f}s "
        f"({stats['tok_s']:.1f} tok/s)"
    )
    print(f"  Init + Inference: {init_time + stats['elapsed_s']:.1f}s")

    llm.exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
