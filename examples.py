"""
Main example — mini-vLLM inference with auto device detection.

Usage:
    python examples.py                                     # auto-detect device
    python examples.py --model /path/to/model              # custom model
    python examples.py --model qwen                        # Qwen3-0.6B on NPU
    MINIVLLM_DEVICE=cpu python examples.py                  # force CPU
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

# On macOS, MPS has numerical instability with the fallback attention path.
# Force CPU unless the user explicitly sets MINIVLLM_DEVICE.
if platform.system() == "Darwin" and not os.environ.get("MINIVLLM_DEVICE"):
    os.environ["MINIVLLM_DEVICE"] = "cpu"

_MODEL_PATHS: dict[str, str] = {
    "opt": "/home/jianzhnie/llmtuner/hfhub/models/facebook/opt-125m",
    "qwen": "/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B",
    "gpt2": "/home/jianzhnie/llmtuner/hfhub/models/openai-community/gpt2",
}

_DEFAULT_MODEL = "qwen"

_PROMPTS = [
    "Hello, who are you?",
    "What is your name?",
    "Where are you from?",
    "Where is the capital of France?",
    "Tell me a joke.",
]


def resolve_model(name: str) -> str:
    if name in _MODEL_PATHS:
        return _MODEL_PATHS[name]
    if os.path.isdir(name):
        return name
    return name


def detect_device() -> str:
    """Auto-detect device type for display purposes."""
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            return f"NPU ({torch.npu.get_device_name(0)})"
        if torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        return "CPU"
    except ImportError:
        return "CPU"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mini-vLLM Inference Example")
    p.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help=f"Model short name ({', '.join(_MODEL_PATHS)}) or path",
    )
    p.add_argument("--dtype", default="float16", choices=["float16", "float32", "auto"])
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-model-len", type=int, default=512)
    p.add_argument(
        "--eager", action="store_true", default=True, help="Force eager mode"
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    from minivllm import LLM, SamplingParams
    from minivllm.config import Config
    from minivllm.utils.logger_utils import get_logger

    logger = get_logger(__name__)

    model_path = resolve_model(args.model)
    device_str = detect_device()

    config = Config(
        model=model_path,
        max_num_seqs=8,
        max_model_len=args.max_model_len,
        enforce_eager=args.eager,
        trust_remote_code=True,
        device_memory_utilization=0.85,
        dtype=args.dtype,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        top_k=40,
        max_tokens=args.max_tokens,
    )

    print(f"\n{'=' * 70}")
    print("  mini-vLLM Inference Example")
    print(f"  Model:  {Path(model_path).name}")
    print(f"  Device: {device_str}")
    print(f"  Dtype:  {args.dtype}")
    print(f"  Prompts: {len(_PROMPTS)}  Max tokens: {args.max_tokens}")
    print(f"{'=' * 70}")

    logger.info("Initializing LLM engine...")
    t0 = time.perf_counter()
    llm = LLM(config)
    init_time = time.perf_counter() - t0
    logger.info(f"Engine initialized in {init_time:.1f}s")

    logger.info(f"Generating completions for {len(_PROMPTS)} prompts...")
    t1 = time.perf_counter()
    outputs = llm.generate(_PROMPTS, sampling_params, use_tqdm=True)
    infer_time = time.perf_counter() - t1

    total_tokens = sum(len(o["token_ids"]) for o in outputs)

    # Results
    print(f"\n{'=' * 70}")
    print("  Results")
    print(f"{'=' * 70}")
    for i, (prompt, output) in enumerate(zip(_PROMPTS, outputs, strict=False)):
        text = output["text"].strip()
        print(f"\n  [{i}] {prompt}")
        print(f"      {text[:200]}")

    print(
        f"\n  Total: {total_tokens} tokens in {infer_time:.2f}s "
        f"({total_tokens / infer_time:.1f} tok/s)"
    )
    print(f"  Overall: {init_time + infer_time:.1f}s")

    del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
