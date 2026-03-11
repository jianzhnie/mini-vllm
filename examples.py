from typing import List

import torch

from minivllm import LLM, SamplingParams
from minivllm.utils.logger_utils import get_logger

# Suppress torch dynamo errors
try:
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

logger = get_logger(__name__)


def main() -> None:
    """Run the example text generation."""
    logger.info('Example starting...')
    # Model paths
    model_name_or_path = '/Users/jianzhengnie/hfhub/models/facebook/opt-125m'
    # model_name_or_path = '/home/jianzhnie/llmtuner/hfhub/models/Qwen/Qwen3-0.6B'
    logger.info(f'Loading model from: {model_name_or_path}')

    try:
        # Initialize LLM
        # enforce_eager=True is useful for debugging/development
        llm = LLM(model=model_name_or_path,
                  max_num_seqs=8,
                  max_model_len=64,
                  enforce_eager=True,
                  trust_remote_code=True,
                  device_memory_utilization=0.4,
                  kvcache_block_size=64,
                  num_kvcache_blocks=-1,
                  dtype='float32')

        tokenizer = llm.tokenizer

    except Exception as e:
        logger.error(f'Failed to initialize LLM: {e}', exc_info=True)
        return

    sampling_params = SamplingParams(temperature=0.6,
                                     top_p=0.95,
                                     top_k=40,
                                     max_tokens=32)
    raw_prompts: List[str] = [
        'Hello, who are you?', 'What is your name?', 'Where are you from?',
        'Where is the capital of France?', 'Tell me a joke.',
        'Introduce yourself', 'List all prime numbers within 100'
    ]

    # Apply chat template if available, otherwise use raw prompts
    prompts = []
    if hasattr(tokenizer,
               'apply_chat_template') and tokenizer.chat_template is not None:
        logger.info('Applying chat template to prompts...')
        for p in raw_prompts:
            messages = [{'role': 'user', 'content': p}]
            # tokenize=False to get string
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            prompts.append(formatted)
    else:
        logger.info('Chat template not available, using raw prompts.')
        prompts = raw_prompts

    logger.info(f'First formatted prompt: {prompts[0]!r}')

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        logger.info('=' * 100)
        logger.info(f'Prompt: {prompt!r}')
        # output is a dict with 'text' and 'token_ids'
        logger.info(f"Completion: {output['text']!r}")


if __name__ == '__main__':
    main()
