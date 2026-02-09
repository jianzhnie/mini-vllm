"""
Example of running facebook/opt-125m on CPU using mini-vLLM.

This script demonstrates how to:
1. Initialize the LLMEngine with a lightweight model (OPT-125M)
2. Force execution on CPU
3. Submit prompts and generate text
"""

import os
import sys
import traceback

from huggingface_hub import snapshot_download

# Force CPU execution by hiding other devices
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ''
os.environ['XPU_VISIBLE_DEVICES'] = ''

import minivllm.engine.llm_engine  # noqa: E402
from minivllm.engine.llm_engine import LLMEngine  # noqa: E402
from minivllm.engine.model_runner_opt import ModelRunner as ModelRunnerOpt  # noqa: E402
from minivllm.sampling_params import SamplingParams  # noqa: E402
from minivllm.utils.logger_utils import get_logger  # noqa: E402

# Patch LLMEngine to use optimized ModelRunner
minivllm.engine.llm_engine.ModelRunner = ModelRunnerOpt

logger = get_logger(__name__)


def main():
    """Main execution function."""
    logger.info('Starting script')

    hfhub = '/Users/jianzhengnie/hfhub/models/'
    model_path = os.path.join(hfhub, 'facebook/opt-125m')
    if not os.path.isdir(model_path):
        logger.info(
            f"Model path '{model_path}' is not a directory. Attempting to download..."
        )
        try:
            model_path = snapshot_download(repo_id=model_path)
            logger.info(f'Model downloaded to: {model_path}')
        except Exception as e:
            logger.error(f'Failed to download model: {e}')
            sys.exit(1)

    logger.info(f'Initializing LLMEngine with model: {model_path}')

    try:
        engine = LLMEngine(model=model_path,
                           max_num_seqs=16,
                           max_model_len=1024,
                           enforce_eager=True,
                           trust_remote_code=True,
                           num_kvcache_blocks=128)
        logger.info('Engine initialized')
    except Exception as e:
        logger.error(f'Engine initialization failed: {e}')
        logger.error(traceback.format_exc())
        sys.exit(1)

    prompts = [
        'Hello, who are you?', 'What is your name?', 'Where are you from?',
        'Where is the capital of France?', 'Tell me a joke.'
    ]

    sampling_params = SamplingParams(temperature=1e-8,
                                     top_p=0.95,
                                     max_tokens=50)

    # Dictionary to store results
    results = {}

    for i, prompt in enumerate(prompts):
        # We assume seq_id will be assigned sequentially starting from 0
        # But let's rely on the engine's assignment if possible or just use the returned seq_id
        engine.add_request(prompt, sampling_params)

    logger.info('Requests added')

    step_count = 0
    # Wait for completion
    while not engine.is_finished():
        step_count += 1
        try:
            step_outputs, _ = engine.step()

            # Log progress every 10 steps
            if step_count % 10 == 0:
                logger.info(f'Step {step_count} completed')

            for seq_id, token_ids in step_outputs:
                text = engine.tokenizer.decode(token_ids,
                                               skip_special_tokens=True)
                logger.info(f'Sequence {seq_id} finished. Output: {text}')
                results[seq_id] = text

        except Exception as e:
            logger.error(f'Step failed: {e}')
            logger.error(traceback.format_exc())
            sys.exit(1)

    logger.info(f'Generation finished in {step_count} steps.')

    # Print final results
    print('-' * 50)
    print('Final Results:')
    for i, prompt in enumerate(prompts):
        # Assuming seq_id corresponds to the order of addition (0, 1, 2...)
        # This is implementation dependent but likely true for simple scheduler
        output_text = results.get(i, 'N/A')
        print(f'Prompt: {prompt}')
        print(f'Output: {output_text}')
        print('-' * 50)

    logger.info('Done.')


if __name__ == '__main__':
    main()
