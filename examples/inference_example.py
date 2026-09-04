"""mini-vLLM inference example — the main, device-agnostic entry point.

Auto-detects the compute device (NPU / CUDA / CPU). To force CPU:
    MINIVLLM_DEVICE=cpu python examples/inference_example.py

Covers the common NPU knobs (--tp, --flash-attn, --no-eager) and an optional
--boxed display mode.

Usage:
    python examples/inference_example.py
    python examples/inference_example.py --model qwen3-4b --max-tokens 128
    python examples/inference_example.py --flash-attn --no-eager
    python examples/inference_example.py --tp 2
    MINIVLLM_DEVICE=cpu python examples/inference_example.py --boxed
"""

from __future__ import annotations

import argparse
import os
import sys

from example_utils import (
    DEFAULT_PROMPTS,
    add_common_args,
    apply_darwin_cpu_fallback,
    build_llm,
    print_banner,
    print_boxed_results,
    resolve_model,
    sample_params,
    timed_generate,
)

apply_darwin_cpu_fallback()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-vLLM inference (device-agnostic)")
    add_common_args(p, dtype_choices=("float16", "float32", "bfloat16", "auto"))
    p.add_argument("--tp", type=int, default=1, help="Tensor parallelism size (1-8)")
    p.add_argument("--flash-attn", action="store_true", help="Enable NPU flash attention")
    p.add_argument("--no-eager", action="store_true", help="Enable graph capture (disable eager)")
    p.add_argument("--max-seqs", type=int, default=8)
    p.add_argument("--prompt", action="append", help="Add a prompt (repeatable)")
    p.add_argument("--boxed", action="store_true", help="Render results in boxes")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Must be set before minivllm internals are imported (build_llm defers it).
    if args.flash_attn:
        os.environ["MINIVLLM_USE_NPU_FA"] = "1"

    prompts = args.prompt if args.prompt else DEFAULT_PROMPTS

    llm, _config = build_llm(
        args, tp=args.tp, max_num_seqs=args.max_seqs, enforce_eager=not args.no_eager
    )
    params = sample_params(args)

    print_banner("mini-vLLM inference")
    print(f"  Model:       {resolve_model(args.model)}")
    print(f"  Dtype:       {args.dtype}   TP: {args.tp}   Eager: {not args.no_eager}")
    print(f"  Flash-Attn:  {'ON' if args.flash_attn else 'OFF'}   Prompts: {len(prompts)}")

    outputs, stats = timed_generate(llm, prompts, params)

    print_banner("Results")
    if args.boxed:
        print_boxed_results(prompts, outputs)
    else:
        for i, (prompt, output) in enumerate(zip(prompts, outputs, strict=True)):
            print(f"\n  [{i}] {prompt}\n      {output['text'].strip()}")

    print(
        f"\n  {stats['elapsed_s']:.1f}s  |  {stats['tokens']} tokens  "
        f"|  {stats['tok_s']:.0f} tok/s"
    )

    llm.exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
