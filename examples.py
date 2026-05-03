"""Example script demonstrating mini-vLLM inference.

This script shows how to use the mini-vLLM engine for text generation
with support for:
- Configurable model paths via environment variables
- Chat template formatting for chat models
- Batch inference with configurable sampling parameters
- Progress tracking and throughput logging

Usage:
    # Basic usage with default model
    python examples.py

    # Custom model via environment variable
    MINIVLLM_MODEL=/path/to/model python examples.py
"""

import contextlib
import os
import platform
import sys

# On macOS, MPS has numerical instability with the fallback attention path.
# Force CPU unless the user explicitly sets MINIVLLM_DEVICE.
if platform.system() == 'Darwin' and not os.environ.get('MINIVLLM_DEVICE'):
    os.environ['MINIVLLM_DEVICE'] = 'cpu'

import torch

from minivllm import LLM, SamplingParams
from minivllm.config import Config
from minivllm.utils.example_utils import format_prompts_with_chat_template
from minivllm.utils.logger_utils import get_logger

# Suppress torch dynamo errors (for compatibility)
with contextlib.suppress(AttributeError, RuntimeError):
    torch._dynamo.config.suppress_errors = True

logger = get_logger(__name__)

prompts = [
    'Hello, who are you?',
    'What is your name?',
    'Where are you from?',
    'Where is the capital of France?',
    'Tell me a joke.',
    'Introduce yourself',
    'List all prime numbers within 100',
]


def run_inference() -> None:
    """Run the inference pipeline."""
    config = Config(
        model='facebook/opt-125m',
        max_num_seqs=8,
        max_model_len=64,
        enforce_eager=True,
        trust_remote_code=True,
        device_memory_utilization=0.4,
        dtype='float32',
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=40,
        max_tokens=32,
    )

    logger.info('Starting mini-vLLM inference example...')
    logger.info(f'Model: {config.model}')
    logger.info(f'Configuration: max_seqs={config.max_num_seqs}, '
                f'max_tokens={sampling_params.max_tokens}')

    # Initialize LLM engine
    logger.info('Initializing LLM engine...')
    llm = LLM(config)
    logger.info('LLM engine initialized successfully.')

    # Prepare prompts with optional chat template
    formatted_prompts = format_prompts_with_chat_template(
        llm.tokenizer, prompts)

    if formatted_prompts:
        logger.info(f'First prompt: {formatted_prompts[0]!r}')
    else:
        logger.warning('No prompts to process.')
        return

    # Run inference
    logger.info(
        f'Generating completions for {len(formatted_prompts)} prompts...')
    outputs = llm.generate(formatted_prompts, sampling_params)

    # Display results
    logger.info('=' * 100)
    logger.info('RESULTS')
    logger.info('=' * 100)

    for prompt, output in zip(formatted_prompts, outputs, strict=True):
        logger.info('-' * 100)
        logger.info(f'Prompt: {prompt!r}')
        logger.info(f"Completion: {output['text']!r}")
        logger.info(f"Tokens: {len(output['token_ids'])}")

    logger.info('=' * 100)
    logger.info('Inference completed successfully.')


def main() -> int:
    """Main entry point for the example script."""
    try:
        run_inference()
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
