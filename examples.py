import os

import torch

try:
    torch._dynamo.config.suppress_errors = True
except Exception:
    pass

from transformers import AutoTokenizer

from minivllm import LLM, SamplingParams
from minivllm.utils.logger_utils import get_logger

logger = get_logger('__main__')


def main():
    model_name_or_path = os.path.expanduser('~/hfhub/models/Qwen/Qwen3-8B')
    logger.info(f'Loading model from: {model_name_or_path}')

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    llm = LLM(model_name_or_path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        'introduce yourself',
        'list all prime numbers within 100',
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{
                'role': 'user',
                'content': prompt
            }],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        logger.info('\n')
        logger.info(f'Prompt: {prompt!r}')
        logger.info(f"Completion: {output['text']!r}")


if __name__ == '__main__':
    main()
