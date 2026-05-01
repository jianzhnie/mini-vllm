"""Shared utilities for mini-vLLM example scripts."""

import os
from typing import List

from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = os.environ.get(
    'MINIVLLM_MODEL', 'facebook/opt-125m')


def format_prompts_with_chat_template(
    tokenizer: object,
    prompts: List[str],
) -> List[str]:
    """Format prompts using the tokenizer's chat template if available.

    Args:
        tokenizer: Tokenizer with optional chat_template attribute.
        prompts: List of raw prompt strings.

    Returns:
        Formatted prompts with chat template applied, or original prompts
        if no chat template is available.
    """
    if not getattr(tokenizer, 'chat_template', None):
        logger.info('Chat template not available, using raw prompts.')
        return prompts

    if not hasattr(tokenizer, 'apply_chat_template'):
        logger.info('Chat template not available, using raw prompts.')
        return prompts

    logger.info('Applying chat template to prompts...')
    formatted_prompts: List[str] = []
    try:
        for prompt in prompts:
            messages = [{'role': 'user', 'content': prompt}]
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            formatted_prompts.append(formatted)
    except Exception as e:
        logger.warning(
            f'Failed to apply chat template ({e}), using raw prompts.')
        return prompts

    return formatted_prompts
