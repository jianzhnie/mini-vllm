"""Shared utilities for mini-vLLM example scripts.

Single home for the scaffolding the examples share: the model registry, model
name resolution, a ``Config`` factory, a result banner, and the macOS CPU
fallback. Examples import from here rather than redeclaring these.
"""

import argparse
import os
import platform
import time
from pathlib import Path

from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

DEFAULT_PROMPTS = [
    "Hello, who are you?",
    "What is your name?",
    "Where are you from?",
    "Where is the capital of France?",
    "Tell me a joke.",
]

# Qwen3 only — the sole architecture the engine supports today
# (see minivllm/models/registry.py). Paths are local so examples run offline.
MODEL_PATHS: dict[str, str] = {
    "qwen3": "/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B",
    "qwen3-1.7b": "/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-1.7B",
    "qwen3-4b": "/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-4B",
}

DEFAULT_MODEL = "qwen3"


def resolve_model(name_or_path: str) -> str:
    """Map a short name to its path, or pass through a local dir / hub ID."""
    if name_or_path in MODEL_PATHS:
        return MODEL_PATHS[name_or_path]
    if Path(name_or_path).is_dir():
        return name_or_path
    # Otherwise treat it as a HuggingFace hub ID.
    return name_or_path


def make_config(
    model: str,
    dtype: str = "float16",
    *,
    tp: int = 1,
    max_model_len: int = 512,
    max_num_seqs: int = 8,
    device_memory_utilization: float = 0.85,
    enforce_eager: bool = True,
    **overrides,
):
    """Build a ``Config`` for an example, resolving the model name.

    ``**overrides`` lets callers pass Config-specific fields (e.g.
    ``use_buffered_page_attention``) without a bespoke constructor.
    """
    from minivllm.config import Config

    return Config(
        model=resolve_model(model),
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        tensor_parallel_size=tp,
        enforce_eager=enforce_eager,
        trust_remote_code=True,
        device_memory_utilization=device_memory_utilization,
        dtype=dtype,
        **overrides,
    )


def print_banner(title: str, width: int = 70) -> None:
    """Print a centered '='-ruled section banner."""
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ---------------------------------------------------------------------------
# Shared CLI + run scaffolding (removes the boilerplate each example repeated)
# ---------------------------------------------------------------------------


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    default_dtype: str = "float16",
    dtype_choices: tuple[str, ...] = ("float16", "float32", "bfloat16"),
) -> argparse.ArgumentParser:
    """Add the CLI args shared by the inference examples.

    Keeps the ``--model/--dtype/--max-tokens/--temperature/--top-p/--top-k/
    --max-model-len`` declarations in one place. Returns ``parser`` for chaining.
    """
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model short name ({', '.join(MODEL_PATHS)}) or path",
    )
    parser.add_argument("--dtype", default=default_dtype, choices=list(dtype_choices))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-model-len", type=int, default=512)
    return parser


def sample_params(args: argparse.Namespace, **overrides):
    """Build a ``SamplingParams`` from the common sampling args.

    ``**overrides`` lets an example pin specific values (e.g. greedy
    ``temperature=0.0``) on top of the CLI defaults.
    """
    from minivllm import SamplingParams

    kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }
    kwargs.update(overrides)
    return SamplingParams(**kwargs)


def build_llm(args: argparse.Namespace, **config_overrides):
    """Create the ``Config`` from the common args and return ``(llm, config)``.

    The ``minivllm`` import is deferred so examples that set env vars
    (e.g. ``MINIVLLM_USE_NPU_FA``) can do so first.
    """
    from minivllm import LLM

    config = make_config(
        args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        **config_overrides,
    )
    return LLM(config), config


def timed_generate(llm, prompts: list[str], params, use_tqdm: bool = True):
    """Run ``llm.generate`` with timing; return ``(outputs, stats)``.

    ``stats`` = ``{"tokens", "elapsed_s", "tok_s"}`` — the one throughput
    calculation every example was re-deriving.
    """
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=use_tqdm)
    elapsed = time.perf_counter() - t0
    tokens = sum(len(o["token_ids"]) for o in outputs)
    stats = {
        "tokens": tokens,
        "elapsed_s": elapsed,
        "tok_s": tokens / elapsed if elapsed > 0 else 0.0,
    }
    return outputs, stats


def apply_darwin_cpu_fallback() -> None:
    """Force CPU on macOS unless MINIVLLM_DEVICE is already set.

    Call before importing torch/minivllm. MPS has shown instability, so the
    examples fall back to CPU on Darwin by default.
    """
    if platform.system() == "Darwin" and not os.environ.get("MINIVLLM_DEVICE"):
        os.environ["MINIVLLM_DEVICE"] = "cpu"


def format_prompts_with_chat_template(
    tokenizer: object,
    prompts: list[str],
) -> list[str]:
    """Format prompts using the tokenizer's chat template if available.

    Args:
        tokenizer: Tokenizer with optional chat_template attribute.
        prompts: List of raw prompt strings.

    Returns:
        Formatted prompts with chat template applied, or original prompts
        if no chat template is available.
    """
    if not getattr(tokenizer, "chat_template", None):
        logger.info("Chat template not available, using raw prompts.")
        return prompts

    if not hasattr(tokenizer, "apply_chat_template"):
        logger.info("Chat template not available, using raw prompts.")
        return prompts

    logger.info("Applying chat template to prompts...")
    formatted_prompts: list[str] = []
    try:
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(formatted)
    except Exception as e:
        logger.warning("Failed to apply chat template (%s), using raw prompts.", e)
        return prompts

    return formatted_prompts


# ---------------------------------------------------------------------------
# Boxed result formatting (optional --boxed output mode)
# ---------------------------------------------------------------------------


def deduplicate_text(text: str, max_repeat: int = 3) -> str:
    """Drop lines that repeat more than ``max_repeat`` times (model artifacts)."""
    seen: dict[str, int] = {}
    kept: list[str] = []
    for line in text.split("\n"):
        key = line.strip()
        if not key:
            continue
        if seen.get(key, 0) < max_repeat:
            kept.append(line)
            seen[key] = seen.get(key, 0) + 1
    return "\n".join(kept) if kept else text


def _wrap_text(text: str, width: int) -> list[str]:
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


def format_output_box(
    prompt: str, output: str, index: int, token_count: int = 0
) -> str:
    """Render one prompt/output pair inside a fixed-width box."""
    width = 76
    inner = width - 4
    wrapped = _wrap_text(output.strip().replace("\n", " "), inner - 2)[:10]
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


def print_boxed_results(
    prompts: list[str], outputs: list[dict], *, dedupe: bool = True
) -> None:
    """Print each result in a box (the ``--boxed`` display mode)."""
    for idx, (prompt, output) in enumerate(zip(prompts, outputs, strict=True)):
        text = deduplicate_text(output["text"]) if dedupe else output["text"]
        print(format_output_box(prompt, text, idx, len(output["token_ids"])))
        print()
