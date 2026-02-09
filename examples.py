import os
from typing import List

import torch

from minivllm import LLM, SamplingParams
from minivllm.utils.logger_utils import get_logger

# Suppress torch dynamo errors
try:
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

logger = get_logger('__main__')


def main() -> None:
    logger.info('Example starting...')
    # Model paths
    model_name_or_path = os.path.expanduser('~/hfhub/models/facebook/opt-125m')

    logger.info(f'Loading model from: {model_name_or_path}')

    try:
        # Initialize LLM
        # enforce_eager=True is useful for debugging/development
        llm = LLM(model_name_or_path,
                  enforce_eager=True,
                  tensor_parallel_size=1)
    except Exception as e:
        logger.error(f'Failed to initialize LLM: {e}')
        return

    sampling_params = SamplingParams(temperature=0.0001, max_tokens=20)
    prompts: List[str] = [
        'Hello, who are you?',
        'What is your name?',
        'Where are you from?',
        'Where is the capital of France?',
        'Tell me a joke.',
        'Introduce yourself',
        'List all prime numbers within 100',
    ]

    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects that contain the prompt,
    # generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        logger.info('\n')
        logger.info(f'Prompt: {prompt!r}')
        # Assuming output is a dict-like or object with 'text' attribute
        # Based on previous code: output['text']
        logger.info(f"Completion: {output['text']!r}")


if __name__ == '__main__':
    main()
