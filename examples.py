"""mini-vLLM inference example.

Usage:
    python examples.py
    python examples.py --model qwen --attention-mode buffered
    python examples.py --model /path/to/model --max-tokens 128
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

if platform.system() == "Darwin" and not os.environ.get("MINIVLLM_DEVICE"):
    os.environ["MINIVLLM_DEVICE"] = "cpu"  # MPS instability workaround

from minivllm import LLM, SamplingParams
from minivllm.config import Config

_MODELS = {
    "opt": "facebook/opt-125m",
    "qwen": "Qwen/Qwen3-0.6B",
}
_PROMPTS = [
    "Hello, who are you?",
    "What is your name?",
    "Where are you from?",
    "Where is the capital of France?",
    "Tell me a joke.",
]


def main() -> int:
    p = argparse.ArgumentParser(description="mini-vLLM inference")
    p.add_argument("--model", default="opt")
    p.add_argument("--dtype", default="float16", choices=["float16", "float32", "auto"])
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument("--eager", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attention-mode", default="fresh", choices=["fresh", "buffered"])
    args = p.parse_args()

    model_path = _MODELS.get(args.model, args.model)
    if not Path(model_path).is_dir() and "/" not in model_path:
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        return 1

    llm = LLM(
        Config(
            model=model_path,
            max_num_seqs=8,
            max_model_len=args.max_model_len,
            enforce_eager=args.eager,
            trust_remote_code=True,
            dtype=args.dtype,
            use_buffered_page_attention=(args.attention_mode == "buffered"),
        )
    )

    print(
        f"\n{'=' * 56}\n  mini-vLLM  |  {Path(model_path).name}  |  "
        f"{args.dtype}  |  attn={args.attention_mode}\n{'=' * 56}"
    )

    t0 = time.perf_counter()
    outputs = llm.generate(
        _PROMPTS,
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
    for i, (p, o) in enumerate(zip(_PROMPTS, outputs, strict=False)):
        print(f"\n  [{i}] {p}\n      {o['text'].strip()[:]}")

    tokens = sum(len(o["token_ids"]) for o in outputs)
    print(f"\n  {elapsed:.1f}s  |  {tokens} tokens  |  {tokens / elapsed:.0f} tok/s")
    del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
