"""Sampling example — how sampling parameters and strategies change output.

One model load, two parts:
  1. Parameter sweep: one prompt under different temperature / top-k / top-p / min-p.
  2. Strategy comparison: a small prompt batch under greedy / balanced / creative,
     with per-strategy timing and throughput.

Usage:
    python examples/sampling_example.py
    python examples/sampling_example.py --model qwen3-4b --max-tokens 48
"""

from __future__ import annotations

import argparse
import sys

from example_utils import (
    add_common_args,
    apply_darwin_cpu_fallback,
    build_llm,
    print_banner,
    timed_generate,
)

apply_darwin_cpu_fallback()

PROMPT = "Once upon a time in a magical kingdom,"

BATCH_PROMPTS = [
    "Explain quantum computing in one sentence.",
    "Write a Python function to check if a number is prime.",
    "What are the three laws of thermodynamics?",
    "Give me a short poem about the ocean.",
]

# (label, params) for the single-prompt sweep.
SWEEP = [
    ("Greedy (temp=0)", {"temperature": 0.0}),
    ("Low temp (0.3)", {"temperature": 0.3}),
    ("Med temp (0.7)", {"temperature": 0.7}),
    ("High temp (1.2)", {"temperature": 1.2}),
    ("Top-k=5", {"temperature": 0.7, "top_k": 5}),
    ("Top-p=0.5", {"temperature": 0.7, "top_p": 0.5}),
    ("Min-p=0.1", {"temperature": 0.7, "min_p": 0.1}),
]

# (label, params) for the batch strategy comparison.
STRATEGIES = [
    ("greedy", {"temperature": 0.0}),
    ("balanced", {"temperature": 0.6, "top_p": 0.9, "top_k": 40}),
    ("creative", {"temperature": 0.9, "top_p": 0.95, "top_k": 50}),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sampling parameters & strategies")
    add_common_args(p, default_dtype="float32", dtype_choices=("float16", "float32"))
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from minivllm import SamplingParams

    llm, config = build_llm(args, enforce_eager=True)

    print_banner("1/2  Parameter sweep (one prompt)")
    print(f"  Model: {config.model}\n  Prompt: {PROMPT!r}\n")
    for label, kwargs in SWEEP:
        params = SamplingParams(max_tokens=args.max_tokens, **kwargs)
        outputs = timed_generate(llm, [PROMPT], params, use_tqdm=False)[0]
        text = outputs[0]["text"].strip().replace("\n", " ")
        print(f"  {label}:  [{len(outputs[0]['token_ids'])}t] {text[:100]}")

    print_banner("2/2  Strategy comparison (batch)")
    for label, kwargs in STRATEGIES:
        params = SamplingParams(max_tokens=args.max_tokens, **kwargs)
        outputs, stats = timed_generate(llm, BATCH_PROMPTS, params, use_tqdm=False)
        print(
            f"\n  {label.upper()}   {stats['elapsed_s']:.2f}s   "
            f"{stats['tokens']} tok   {stats['tok_s']:.0f} tok/s"
        )
        for i, (prompt, output) in enumerate(zip(BATCH_PROMPTS, outputs, strict=True)):
            text = output["text"].strip().replace("\n", " ")
            print(f"    [{i}] {prompt}\n        {text[:90]}{'...' if len(text) > 90 else ''}")

    llm.exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
