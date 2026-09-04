"""mini-vLLM inference example — the simplest entry point.

Auto-detects the device (NPU/CUDA/CPU) and runs a few prompts.

Usage:
    python examples/inference_example.py
    python examples/inference_example.py --attention-mode buffered
    python examples/inference_example.py --model /path/to/model --max-tokens 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from example_utils import (
    DEFAULT_PROMPTS,
    add_common_args,
    apply_darwin_cpu_fallback,
    build_llm,
    print_banner,
    sample_params,
    timed_generate,
)

apply_darwin_cpu_fallback()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-vLLM inference")
    add_common_args(p, dtype_choices=("float16", "float32", "auto"))
    p.add_argument(
        "--eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force eager mode (disable CUDA/Ascend graph capture)",
    )
    p.add_argument(
        "--attention-mode",
        default="fresh",
        choices=("fresh", "buffered"),
        help="KV gather mode: fresh (per-step buffers) or buffered (reused)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    llm, config = build_llm(
        args,
        enforce_eager=args.eager,
        use_buffered_page_attention=args.attention_mode == "buffered",
    )
    params = sample_params(args, top_p=0.95, top_k=40)

    print_banner("mini-vLLM inference")
    print(f"  Model: {Path(config.model).name}   Dtype: {args.dtype}   Attn: {args.attention_mode}")

    outputs, stats = timed_generate(llm, DEFAULT_PROMPTS, params)

    print_banner("Results")
    for i, (prompt, output) in enumerate(zip(DEFAULT_PROMPTS, outputs, strict=True)):
        print(f"\n  [{i}] {prompt}\n      {output['text'].strip()}")
    print(
        f"\n  {stats['elapsed_s']:.1f}s  |  {stats['tokens']} tokens  "
        f"|  {stats['tok_s']:.0f} tok/s"
    )

    llm.exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
