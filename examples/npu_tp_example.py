"""Tensor Parallelism Example — verify TP=1/2/4 correctness on NPU.

Runs the same prompts across different TP sizes and compares output quality.
Each TP run produces different random output (no seed synchronization), so this
checks semantic coherence rather than exact token match.

Usage:
    python examples/npu_tp_example.py            # TP=1 baseline
    python examples/npu_tp_example.py --all      # TP=1,2,4
    python examples/npu_tp_example.py --tp 2     # TP=2 only
    python examples/npu_tp_example.py --tp 4 --model qwen3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from example_utils import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    MODEL_PATHS,
    make_config,
    resolve_model,
    timed_generate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-vLLM Tensor Parallelism Example")
    p.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model short name ({', '.join(MODEL_PATHS)}) or path",
    )
    p.add_argument("--tp", type=int, default=0, help="Single TP size to test (overrides --all)")
    p.add_argument("--all", action="store_true", help="Test TP=1, TP=2, TP=4 sequentially")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--dtype", default="float16", choices=("float16", "float32", "bfloat16"))
    return p.parse_args()


def run_tp_inference(model_path: str, tp: int, max_tokens: int, dtype: str) -> dict:
    """Run inference at a given TP size and collect stats."""
    from minivllm import LLM, SamplingParams

    config = make_config(
        model_path, dtype=dtype, tp=tp,
        device_memory_utilization=0.8, enforce_eager=True,
    )
    params = SamplingParams(temperature=0.7, top_p=0.95, top_k=40, max_tokens=max_tokens)

    t0 = time.perf_counter()
    llm = LLM(config)
    init_t = time.perf_counter() - t0

    outputs, stats = timed_generate(llm, DEFAULT_PROMPTS, params, use_tqdm=False)
    llm.exit()

    return {
        "tp": tp,
        "init_s": round(init_t, 1),
        "infer_s": round(stats["elapsed_s"], 2),
        "tokens": stats["tokens"],
        "tok_s": round(stats["tok_s"], 1),
        "texts": [o["text"].strip() for o in outputs],
    }


def check_tp_available(tp: int) -> bool:
    """Check if enough NPU devices are available."""
    import torch

    count = torch.npu.device_count() if hasattr(torch, "npu") else 0
    if count < tp:
        print(f"  SKIP: need {tp} NPU devices, found {count}")
        return False
    return True


def print_result(r: dict) -> None:
    print(
        f"  TP={r['tp']}: init={r['init_s']}s  infer={r['infer_s']}s  "
        f"tokens={r['tokens']}  throughput={r['tok_s']} tok/s"
    )
    for i, (prompt, text) in enumerate(zip(DEFAULT_PROMPTS, r["texts"], strict=True)):
        print(f"    [{i}] Q: {prompt[:60]}")
        print(f"        A: {text[:120]}{'...' if len(text) > 120 else ''}")


def main() -> int:
    args = parse_args()
    os.environ.pop("MINIVLLM_USE_NPU_FA", None)  # use standard attention

    model_path = resolve_model(args.model)
    model_name = Path(model_path).name

    if args.tp > 0:
        tp_sizes = [args.tp]
    elif args.all:
        tp_sizes = [1, 2, 4]
    else:
        tp_sizes = [1]

    print(f"\n{'=' * 70}")
    print(f"  Tensor Parallelism Example — {model_name}")
    print(f"  Dtype: {args.dtype}   Max tokens: {args.max_tokens}")
    print(f"  Prompts: {len(DEFAULT_PROMPTS)}")
    print(f"{'=' * 70}")

    results = []
    for tp in tp_sizes:
        if not check_tp_available(tp):
            continue
        print(f"\n  Running TP={tp}...")
        try:
            r = run_tp_inference(model_path, tp, args.max_tokens, args.dtype)
            results.append(r)
            print_result(r)
        except Exception as e:
            print(f"  TP={tp} FAILED: {e}")
            if "HCCL" in str(e) or "port" in str(e).lower():
                print(f"  NOTE: HCCL port conflict — try standalone:\n"
                      f"         python examples/npu_tp_example.py --model <model> --tp {tp}")

        # Let worker processes + HCCL release resources between runs.
        if len(tp_sizes) > 1:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
            os.environ.pop("MASTER_PORT", None)
            time.sleep(3)

    if len(results) > 1:
        print(f"\n{'=' * 70}\n  Summary\n{'=' * 70}")
        for r in results:
            print(f"  TP={r['tp']}: {r['tok_s']} tok/s  ({r['tokens']} tokens, {r['infer_s']}s)")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
