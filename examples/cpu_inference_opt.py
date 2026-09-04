"""CPU inference example — mini-vLLM on CPU (golden reference).

Forces CPU execution (hides every accelerator) and prints results in a
nicely-boxed layout. Useful as a device-agnostic golden reference for
comparing NPU/GPU output.

Usage:
    python examples/cpu_inference_opt.py
    python examples/cpu_inference_opt.py --model qwen3
    python examples/cpu_inference_opt.py --model /path/to/model
"""

import argparse
import os
import sys

# Force CPU execution by hiding other devices (must be done before importing torch).
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ""
os.environ["XPU_VISIBLE_DEVICES"] = ""
os.environ["MINIVLLM_DEVICE"] = "cpu"

from example_utils import (
    DEFAULT_MODEL,
    DEFAULT_PROMPTS,
    format_prompts_with_chat_template,
    make_config,
    resolve_model,
    timed_generate,
)

from minivllm import LLM, SamplingParams
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)


def deduplicate_text(text: str, max_repeat: int = 3) -> str:
    """Drop lines that repeat more than ``max_repeat`` times (model artifacts)."""
    seen: dict[str, int] = {}
    result = []
    for line in text.split("\n"):
        key = line.strip()
        if not key:
            continue
        if seen.get(key, 0) < max_repeat:
            result.append(line)
            seen[key] = seen.get(key, 0) + 1
    return "\n".join(result) if result else text


def wrap_text(text: str, width: int) -> list[str]:
    """Word-wrap text into lines of at most ``width`` characters."""
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in text.split():
        if length + len(word) + len(current) <= width:
            current.append(word)
            length += len(word)
        else:
            lines.append(" ".join(current))
            current, length = [word], len(word)
    if current:
        lines.append(" ".join(current))
    return lines or [text[:width]]


def format_output_box(prompt: str, output: str, index: int, token_count: int) -> str:
    """Render one prompt/output pair inside a fixed-width box."""
    width = 76
    inner = width - 4
    output_clean = output.strip().replace("\n", " ")
    wrapped = wrap_text(output_clean, inner - 2)[:10]
    box = [
        f"┌{'─' * width}┐",
        f"│ [{index}] Prompt: {prompt[: inner - 12]:<{inner - 12}} │",
        f"├{'─' * width}┤",
        f"│{' ' * inner} │",
    ]
    box += [f"│  {line:<{inner - 2}} │" for line in wrapped]
    if len(wrapped) == 10:
        box.append(f"│  {' ' * (inner - 2)} │")
    box += [f"│{' ' * inner} │", f"│ {'Tokens: ' + str(token_count):<{inner}} │", f"└{'─' * width}┘"]
    return "\n".join(box)


def run_inference(model_path: str) -> None:
    config = make_config(
        model_path,
        dtype="float32",
        max_model_len=1024,
        device_memory_utilization=0.9,
        enforce_eager=True,
    )
    params = SamplingParams(temperature=0.6, top_p=0.95, top_k=40, max_tokens=50)

    logger.info("Starting CPU inference with mini-vLLM (model=%s)", config.model)
    llm = LLM(config)

    prompts = format_prompts_with_chat_template(llm.tokenizer, DEFAULT_PROMPTS)
    outputs, stats = timed_generate(llm, prompts, params)

    print("\n" + "=" * 80)
    print("              INFERENCE RESULTS (CPU)")
    print("=" * 80)
    print(f"Model:      {config.model}")
    print(f"Prompts:    {len(prompts)}   Tokens: {stats['tokens']}")
    print(f"Inference:  {stats['elapsed_s']:.2f}s   Throughput: {stats['tok_s']:.1f} tok/s")
    print("=" * 80 + "\n")
    for idx, (prompt, output) in enumerate(zip(prompts, outputs, strict=True)):
        print(
            format_output_box(
                prompt, deduplicate_text(output["text"]), idx, len(output["token_ids"])
            )
        )
        print()
    logger.info("Inference completed successfully.")
    llm.exit()


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU Inference Example")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    try:
        run_inference(resolve_model(args.model))
        return 0
    except (ValueError, KeyboardInterrupt) as e:
        logger.error("Interrupted or misconfigured: %s", e)
        return 1
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
