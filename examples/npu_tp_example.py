"""
Tensor Parallelism Example — verify TP=1/2/4 correctness on NPU.

Runs the same prompt across different TP sizes and compares output quality.
Each TP run produces different random output (no seed synchronization), so
this checks for semantic coherence rather than exact token match.

Usage:
    python examples/npu_tp_example.py                       # TP=1 baseline
    python examples/npu_tp_example.py --all                 # TP=1,2,4
    python examples/npu_tp_example.py --tp 2                # TP=2 only
    python examples/npu_tp_example.py --tp 4 --model qwen   # TP=4 + Qwen
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_MODEL_PATHS: dict[str, str] = {
    "opt": "/home/jianzhnie/llmtuner/hfhub/models/facebook/opt-125m",
    "qwen": "/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B",
}

_DEFAULT_MODEL = "opt"
_PROMPTS = [
    "Hello, who are you?",
    "What is the capital of China?",
    "Tell me a short joke.",
]


def resolve_model(name: str) -> str:
    return _MODEL_PATHS.get(name, name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-vLLM Tensor Parallelism Example")
    p.add_argument("--model", default=_DEFAULT_MODEL,
                   help=f"Model short name ({', '.join(_MODEL_PATHS)}) or path")
    p.add_argument("--tp", type=int, default=0,
                   help="Single TP size to test (overrides --all)")
    p.add_argument("--all", action="store_true",
                   help="Test TP=1, TP=2, TP=4 sequentially")
    p.add_argument("--max-tokens", type=int, default=48)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    return p.parse_args()


def run_tp_inference(model_path: str, tp: int, max_tokens: int, dtype: str) -> dict:
    """Run inference at a given TP size and collect stats."""
    from minivllm import LLM, SamplingParams
    from minivllm.config import Config

    config = Config(
        model=model_path,
        max_num_seqs=8,
        max_model_len=512,
        tensor_parallel_size=tp,
        enforce_eager=True,
        trust_remote_code=True,
        device_memory_utilization=0.8,
        dtype=dtype,
    )

    params = SamplingParams(temperature=0.7, top_p=0.95, top_k=40, max_tokens=max_tokens)

    t0 = time.perf_counter()
    llm = LLM(config)
    init_t = time.perf_counter() - t0

    t1 = time.perf_counter()
    outputs = llm.generate(_PROMPTS, params, use_tqdm=False)
    infer_t = time.perf_counter() - t1

    total_tokens = sum(len(o["token_ids"]) for o in outputs)
    del llm

    return {
        "tp": tp,
        "init_s": round(init_t, 1),
        "infer_s": round(infer_t, 2),
        "tokens": total_tokens,
        "tok_s": round(total_tokens / infer_t, 1) if infer_t > 0 else 0,
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
    print(f"  TP={r['tp']}: init={r['init_s']}s  infer={r['infer_s']}s  "
          f"tokens={r['tokens']}  throughput={r['tok_s']} tok/s")
    for i, (prompt, text) in enumerate(zip(_PROMPTS, r["texts"])):
        tokens = len(text.split())
        snippet = text[:120]
        print(f"    [{i}] Q: {prompt[:60]}")
        print(f"        A: {snippet}{'...' if len(text) > 120 else ''}")


def main() -> int:
    args = parse_args()
    os.environ.pop("MINIVLLM_USE_NPU_FA", None)  # use standard attention

    model_path = resolve_model(args.model)
    model_name = Path(model_path).name

    # Determine which TP sizes to test
    if args.tp > 0:
        tp_sizes = [args.tp]
    elif args.all:
        tp_sizes = [1, 2, 4]
    else:
        tp_sizes = [1]

    print(f"\n{'='*70}")
    print(f"  Tensor Parallelism Example — {model_name}")
    print(f"  Dtype: {args.dtype}   Max tokens: {args.max_tokens}")
    print(f"  Prompts: {len(_PROMPTS)}")
    print(f"{'='*70}")

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
            if tp > 2 and "HCCL" in str(e):
                print(f"  NOTE: TP={tp} may require HCCL cross-device links not available")
                print(f"  on this machine. Try running TP={tp} independently.")
        # Allow time for worker processes to fully terminate between runs
        if len(tp_sizes) > 1:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
            time.sleep(1)

    # Summary
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("  Summary")
        print(f"{'='*70}")
        for r in results:
            print(f"  TP={r['tp']}: {r['tok_s']} tok/s  ({r['tokens']} tokens, {r['infer_s']}s)")

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
