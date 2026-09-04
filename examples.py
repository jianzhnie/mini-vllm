"""mini-vLLM inference example.

Usage:
    python examples.py
    python examples.py --attention-mode buffered
    python examples.py --model /path/to/model --max-tokens 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from minivllm import LLM, SamplingParams
from minivllm.utils.example_utils import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    apply_darwin_cpu_fallback,
    make_config,
)

apply_darwin_cpu_fallback()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-vLLM inference")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--dtype", default="float16", choices=["float16", "float32", "auto"]
    )
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--eager", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attention-mode", default="fresh", choices=["fresh", "buffered"])
    return p.parse_args()


def main() -> int:
    args = parse_args()

    config = make_config(
        args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=args.eager,
        use_buffered_page_attention=(args.attention_mode == "buffered"),
    )

    llm = LLM(config)
    print(
        f"\n{'=' * 56}\n  mini-vLLM  |  {Path(config.model).name}  |  "
        f"{args.dtype}  |  attn={args.attention_mode}\n{'=' * 56}"
    )

    t0 = time.perf_counter()
    outputs = llm.generate(
        DEFAULT_PROMPTS,
        SamplingParams(
            temperature=args.temperature,
            top_p=0.95,
            top_k=40,
            max_tokens=args.max_tokens,
        ),
        use_tqdm=True,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 56}\n  Results\n{'=' * 56}")
    for i, (p, o) in enumerate(zip(DEFAULT_PROMPTS, outputs, strict=False)):
        print(f"\n  [{i}] {p}\n      {o['text'].strip()}")

    tokens = sum(len(o["token_ids"]) for o in outputs)
    print(f"\n  {elapsed:.1f}s  |  {tokens} tokens  |  {tokens / elapsed:.0f} tok/s")
    del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
