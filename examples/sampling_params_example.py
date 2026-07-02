"""Sampling parameters exploration example.

Demonstrates how different sampling parameters affect generation output:
- Temperature effects (deterministic vs random)
- Top-p (nucleus sampling)
- Top-k filtering
- Min-p filtering

Usage:
    python examples/sampling_params_example.py
    python examples/sampling_params_example.py --model Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from time import perf_counter

if platform.system() == "Darwin" and not os.environ.get("MINIVLLM_DEVICE"):
    os.environ["MINIVLLM_DEVICE"] = "cpu"

from minivllm import LLM, SamplingParams
from minivllm.config import Config

PROMPT = "Once upon a time in a magical kingdom,"


def main() -> int:
    parser = argparse.ArgumentParser(description="Sampling params exploration")
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--max-tokens", type=int, default=40)
    args = parser.parse_args()

    config = Config(
        model=args.model,
        max_num_seqs=8,
        max_model_len=512,
        enforce_eager=True,
        trust_remote_code=True,
        dtype=args.dtype,
    )
    llm = LLM(config)

    experiments = [
        ("Greedy (temp=0)", SamplingParams(temperature=0.0, max_tokens=args.max_tokens)),
        ("Low temp (0.3)", SamplingParams(temperature=0.3, max_tokens=args.max_tokens)),
        ("Med temp (0.7)", SamplingParams(temperature=0.7, max_tokens=args.max_tokens)),
        ("High temp (1.2)", SamplingParams(temperature=1.2, max_tokens=args.max_tokens)),
        ("Top-k=5", SamplingParams(temperature=0.7, top_k=5, max_tokens=args.max_tokens)),
        ("Top-p=0.5", SamplingParams(temperature=0.7, top_p=0.5, max_tokens=args.max_tokens)),
        ("Min-p=0.1", SamplingParams(temperature=0.7, min_p=0.1, max_tokens=args.max_tokens)),
    ]

    print(f"\n{'=' * 70}")
    print(f"  Sampling Parameter Exploration")
    print(f"  Model: {args.model}  |  Prompt: {PROMPT!r}")
    print(f"{'=' * 70}")

    for name, params in experiments:
        outputs = llm.generate([PROMPT], params, use_tqdm=False)
        text = outputs[0]["text"].strip().replace("\n", " ")
        if len(text) > 100:
            text = text[:97] + "..."
        tokens = len(outputs[0]["token_ids"])
        print(f"\n  {name}:")
        print(f"    [{tokens} tokens] {text}")

    print(f"\n{'=' * 70}")
    del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
