"""Shared utilities for mini-vLLM example scripts.

Single home for the scaffolding the examples share: the model registry, model
name resolution, a ``Config`` factory, a result banner, and the macOS CPU
fallback. Examples import from here rather than redeclaring these.
"""

import os
import platform
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
