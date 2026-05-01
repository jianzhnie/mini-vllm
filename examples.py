"""Example script demonstrating mini-vLLM inference.

This script shows how to use the mini-vLLM engine for text generation
with support for:
- Configurable model paths via CLI or environment variables
- Chat template formatting for chat models
- Batch inference with configurable sampling parameters
- Progress tracking and throughput logging

Usage:
    # Basic usage with default model
    python examples.py

    # Custom model via environment variable
    MINIVLLM_MODEL=/path/to/model python examples.py

    # Custom model via CLI
    python examples.py --model /path/to/model

    # Advanced usage
    python examples.py \\
        --model /path/to/model \\
        --max-seqs 16 \\
        --max-tokens 64 \\
        --temperature 0.8
"""

import argparse
import contextlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

from minivllm import LLM, SamplingParams
from minivllm.utils.logger_utils import get_logger

# Suppress torch dynamo errors (for compatibility)
with contextlib.suppress(AttributeError, RuntimeError):
    torch._dynamo.config.suppress_errors = True

logger = get_logger(__name__)


@dataclass
class ExampleConfig:
    """Configuration for the example script.

    Attributes:
        model_name_or_path: Path to the model weights (HuggingFace format).
        max_num_seqs: Maximum number of sequences to process in parallel.
        max_model_len: Maximum sequence length for the model.
        device_memory_utilization: Fraction of device memory to use (0.1-1.0).
        kvcache_block_size: Size of KV cache blocks in tokens (must be divisible by 64).
        dtype: Data type for model weights ('auto', 'float16', 'float32', 'bfloat16').
        enforce_eager: Force eager execution mode (disables CUDA Graph).
        temperature: Sampling temperature (> 1e-10, higher = more random).
        top_p: Nucleus sampling probability threshold (0-1).
        top_k: Top-k sampling (0 or -1 to disable).
        max_tokens: Maximum tokens to generate per prompt.
        prompts: List of prompts to generate from.
    """
    model_name_or_path: str = os.environ.get(
        'MINIVLLM_MODEL', 'facebook/opt-125m')
    max_num_seqs: int = 8
    max_model_len: int = 64
    device_memory_utilization: float = 0.4
    kvcache_block_size: int = 64
    dtype: str = 'float32'
    enforce_eager: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 32
    prompts: list[str] = field(default_factory=lambda: [
        'Hello, who are you?',
        'What is your name?',
        'Where are you from?',
        'Where is the capital of France?',
        'Tell me a joke.',
        'Introduce yourself',
        'List all prime numbers within 100',
    ])


def parse_args() -> ExampleConfig:
    """Parse command-line arguments and return configuration.

    Supports both CLI arguments and environment variables:
    - MINIVLLM_MODEL: Override model path via environment variable
    - All other parameters can be set via CLI flags

    Returns:
        ExampleConfig with parsed or default values.
    """
    parser = argparse.ArgumentParser(
        description='Run mini-vLLM inference example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default=os.environ.get(
            'MINIVLLM_MODEL',
            'facebook/opt-125m',
        ),
        help=
        'Path to the model (HuggingFace format). Can also be set via MINIVLLM_MODEL env var.',
    )
    parser.add_argument('--max-seqs',
                        type=int,
                        default=8,
                        help='Maximum sequences per batch')
    parser.add_argument('--max-model-len',
                        type=int,
                        default=64,
                        help='Maximum sequence length')
    parser.add_argument('--memory-utilization',
                        type=float,
                        default=0.4,
                        help='GPU memory fraction (0.1-1.0)')
    parser.add_argument('--dtype',
                        type=str,
                        default='float32',
                        choices=['auto', 'float16', 'float32', 'bfloat16'])

    # Sampling configuration
    parser.add_argument('--temperature',
                        type=float,
                        default=0.6,
                        help='Sampling temperature')
    parser.add_argument('--top-p',
                        type=float,
                        default=0.95,
                        help='Nucleus sampling probability')
    parser.add_argument('--top-k',
                        type=int,
                        default=40,
                        help='Top-k sampling (-1 to disable)')
    parser.add_argument('--max-tokens',
                        type=int,
                        default=32,
                        help='Maximum tokens to generate')

    # Execution mode
    parser.add_argument('--eager',
                        action='store_true',
                        help='Use eager execution (disables CUDA Graph)')

    args = parser.parse_args()

    return ExampleConfig(
        model_name_or_path=args.model,
        max_num_seqs=args.max_seqs,
        max_model_len=args.max_model_len,
        device_memory_utilization=args.memory_utilization,
        dtype=args.dtype,
        enforce_eager=args.eager,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )


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
    if not hasattr(tokenizer, 'apply_chat_template'):
        logger.info('Chat template not available, using raw prompts.')
        return prompts

    if getattr(tokenizer, 'chat_template', None) is None:
        logger.info('Chat template not available, using raw prompts.')
        return prompts

    logger.info('Applying chat template to prompts...')
    formatted_prompts: list[str] = []
    for prompt in prompts:
        messages = [{'role': 'user', 'content': prompt}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(formatted)

    return formatted_prompts


def validate_config(config: ExampleConfig) -> None:
    """Validate configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.
    """
    # Validate model: accept either a local directory or a HuggingFace model ID
    if not Path(config.model_name_or_path).is_dir() and '/' in config.model_name_or_path:
        raise ValueError(
            f'Model path does not exist: {config.model_name_or_path}\n'
            f'Set the model path via --model argument or MINIVLLM_MODEL environment variable.'
        )

    if config.temperature <= 1e-10:
        raise ValueError(
            f'Temperature must be > 1e-10, got {config.temperature}. '
            f'Greedy sampling (temperature=0) is not supported.')

    if not 0.0 < config.top_p <= 1.0:
        raise ValueError(f'top_p must be in (0, 1.0], got {config.top_p}')

    if not 0.1 <= config.device_memory_utilization <= 1.0:
        raise ValueError(
            f'device_memory_utilization must be in [0.1, 1.0], got {config.device_memory_utilization}'
        )

    if config.kvcache_block_size % 64 != 0:
        raise ValueError(
            f'kvcache_block_size must be divisible by 64, got {config.kvcache_block_size}'
        )

    if not Path(config.model_name_or_path).exists():
        # Could be a HuggingFace model ID, only validate local paths
        if os.path.isdir(config.model_name_or_path) or '/' not in config.model_name_or_path:
            pass  # OK: either a HF hub ID or existing directory
        else:
        raise ValueError(
            f'kvcache_block_size must be divisible by 64, got {config.kvcache_block_size}'
        )


def run_inference(config: ExampleConfig) -> None:
    """Run the inference pipeline with the given configuration.

    Args:
        config: Configuration containing model and sampling parameters.
    """
    logger.info('Starting mini-vLLM inference example...')
    logger.info(f'Model: {config.model_name_or_path}')
    logger.info(
        f'Configuration: max_seqs={config.max_num_seqs}, max_tokens={config.max_tokens}'
    )

    # Validate configuration
    validate_config(config)

    # Initialize LLM engine
    try:
        logger.info('Initializing LLM engine...')
        llm = LLM(
            model=config.model_name_or_path,
            max_num_seqs=config.max_num_seqs,
            max_model_len=config.max_model_len,
            enforce_eager=config.enforce_eager,
            trust_remote_code=True,
            device_memory_utilization=config.device_memory_utilization,
            kvcache_block_size=config.kvcache_block_size,
            num_kvcache_blocks=-1,
            dtype=config.dtype,
        )
        logger.info('LLM engine initialized successfully.')

    except Exception as e:
        logger.error(f'Failed to initialize LLM: {e}', exc_info=True)
        raise

    # Prepare prompts with optional chat template
    formatted_prompts = format_prompts_with_chat_template(
        llm.tokenizer, config.prompts)

    if formatted_prompts:
        logger.info(f'First prompt: {formatted_prompts[0]!r}')

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        max_tokens=config.max_tokens,
    )

    # Run inference
    logger.info(
        f'Generating completions for {len(formatted_prompts)} prompts...')
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Display results
    logger.info('=' * 100)
    logger.info('RESULTS')
    logger.info('=' * 100)

    for prompt, output in zip(formatted_prompts, outputs, strict=False):
        logger.info('-' * 100)
        logger.info(f'Prompt: {prompt!r}')
        logger.info(f"Completion: {output['text']!r}")
        logger.info(f"Tokens: {len(output['token_ids'])}")

    logger.info('=' * 100)
    logger.info('Inference completed successfully.')


def main() -> int:
    """Main entry point for the example script.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    try:
        config = parse_args()
        run_inference(config)
        return 0

    except ValueError as e:
        logger.error(f'Configuration error: {e}')
        return 1

    except KeyboardInterrupt:
        logger.info('Interrupted by user.')
        return 1

    except Exception as e:
        logger.error(f'Unexpected error: {e}', exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
