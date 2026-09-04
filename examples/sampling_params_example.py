"""Sampling-parameter exploration example.

Shows how different sampling parameters affect a single generation:
temperature (deterministic vs random), top-p, top-k, and min-p.

Usage:
    python examples/sampling_params_example.py
    python examples/sampling_params_example.py --model qwen3
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sampling params exploration")
    add_common_args(p, default_dtype="float32", dtype_choices=("float16", "float32"))
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from minivllm import SamplingParams

    llm, config = build_llm(args, enforce_eager=True)

    mt = args.max_tokens
    experiments = [
        ("Greedy (temp=0)", SamplingParams(temperature=0.0, max_tokens=mt)),
        ("Low temp (0.3)", SamplingParams(temperature=0.3, max_tokens=mt)),
        ("Med temp (0.7)", SamplingParams(temperature=0.7, max_tokens=mt)),
        ("High temp (1.2)", SamplingParams(temperature=1.2, max_tokens=mt)),
        ("Top-k=5", SamplingParams(temperature=0.7, top_k=5, max_tokens=mt)),
        ("Top-p=0.5", SamplingParams(temperature=0.7, top_p=0.5, max_tokens=mt)),
        ("Min-p=0.1", SamplingParams(temperature=0.7, min_p=0.1, max_tokens=mt)),
    ]

    print_banner("Sampling Parameter Exploration")
    print(f"  Model: {config.model}  |  Prompt: {PROMPT!r}")

    for name, params in experiments:
        outputs = timed_generate(llm, [PROMPT], params, use_tqdm=False)[0]
        text = outputs[0]["text"].strip().replace("\n", " ")
        print(f"\n  {name}:")
        print(f"    [{len(outputs[0]['token_ids'])} tokens] {text[:100]}")

    print(f"\n{'=' * 70}")
    llm.exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
