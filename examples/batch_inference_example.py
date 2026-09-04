"""Batch inference example — compare sampling strategies side by side.

Runs the same batch of prompts under different sampling strategies (greedy /
creative / balanced) and reports per-strategy timing and throughput.

Usage:
    python examples/batch_inference_example.py
    python examples/batch_inference_example.py --strategies greedy balanced creative
"""

from __future__ import annotations

import argparse
import sys

from example_utils import (
    DEFAULT_MODEL,
    apply_darwin_cpu_fallback,
    build_llm,
    print_banner,
    timed_generate,
)

apply_darwin_cpu_fallback()

PROMPTS = [
    "Explain quantum computing in one sentence.",
    "Write a Python function to check if a number is prime.",
    "What are the three laws of thermodynamics?",
    "Translate 'hello world' to French, German, and Japanese.",
    "Give me a short poem about the ocean.",
    "What is the difference between a list and a tuple in Python?",
]

STRATEGIES = {
    "greedy": {"temperature": 0.0, "max_tokens": 64},
    "creative": {"temperature": 0.9, "top_p": 0.95, "top_k": 50, "max_tokens": 64},
    "balanced": {"temperature": 0.6, "top_p": 0.9, "top_k": 40, "max_tokens": 64},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch inference comparison")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--dtype", default="float32", choices=("float16", "float32"))
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument(
        "--strategies", nargs="+", choices=list(STRATEGIES), default=["greedy", "balanced"]
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from minivllm import SamplingParams

    llm, config = build_llm(args, enforce_eager=True)

    print_banner(f"Batch Inference  |  {config.model}  |  {args.dtype}")
    print(f"  Prompts: {len(PROMPTS)}  |  Strategies: {', '.join(args.strategies)}")

    for name in args.strategies:
        params = SamplingParams(**STRATEGIES[name])
        outputs, stats = timed_generate(llm, PROMPTS, params, use_tqdm=False)

        print(f"\n--- {name.upper()} (temp={params.temperature}) ---")
        print(
            f"    Time: {stats['elapsed_s']:.2f}s | Tokens: {stats['tokens']} | "
            f"Throughput: {stats['tok_s']:.0f} tok/s"
        )
        for i, (prompt, output) in enumerate(zip(PROMPTS, outputs, strict=True)):
            text = output["text"].strip().replace("\n", " ")
            print(f"\n  [{i}] {prompt}\n      {text[:120]}{'...' if len(text) > 120 else ''}")

    print(f"\n{'=' * 70}")
    llm.exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
