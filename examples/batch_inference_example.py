"""Batch inference example with timing and comparison.

Demonstrates:
- Running batch inference with different sampling strategies
- Comparing greedy vs creative generation
- Per-prompt timing and throughput reporting

Usage:
    python examples/batch_inference_example.py
    python examples/batch_inference_example.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import sys
from time import perf_counter

from minivllm import LLM, SamplingParams
from minivllm.utils.example_utils import (
    DEFAULT_MODEL,
    apply_darwin_cpu_fallback,
    make_config,
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
    "greedy": SamplingParams(temperature=0.0, max_tokens=64),
    "creative": SamplingParams(temperature=0.9, top_p=0.95, top_k=50, max_tokens=64),
    "balanced": SamplingParams(temperature=0.6, top_p=0.9, top_k=40, max_tokens=64),
}


def run_strategy(
    llm: LLM, prompts: list[str], name: str, params: SamplingParams
) -> tuple[list[dict], float]:
    """Run inference with a given strategy and return results + elapsed time."""
    start = perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    elapsed = perf_counter() - start
    return outputs, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch inference comparison")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=list(STRATEGIES),
        default=["greedy", "balanced"],
    )
    args = parser.parse_args()

    config = make_config(
        args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        enforce_eager=True,
    )

    llm = LLM(config)

    print(f"\n{'=' * 70}")
    print(f"  Batch Inference  |  {args.model}  |  {args.dtype}")
    print(f"  Prompts: {len(PROMPTS)}  |  Strategies: {', '.join(args.strategies)}")
    print(f"{'=' * 70}")

    for strategy_name in args.strategies:
        params = STRATEGIES[strategy_name]
        outputs, elapsed = run_strategy(llm, PROMPTS, strategy_name, params)
        total_tokens = sum(len(o["token_ids"]) for o in outputs)

        print(f"\n--- {strategy_name.upper()} (temp={params.temperature}) ---")
        print(f"    Time: {elapsed:.2f}s | Tokens: {total_tokens} | "
              f"Throughput: {total_tokens / elapsed:.0f} tok/s")

        for i, (prompt, output) in enumerate(zip(PROMPTS, outputs, strict=True)):
            text = output["text"].strip().replace("\n", " ")
            if len(text) > 120:
                text = text[:117] + "..."
            print(f"\n  [{i}] {prompt}")
            print(f"      {text}")

    print(f"\n{'=' * 70}")
    del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
