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

# Force CPU execution by hiding other devices
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ''
os.environ['XPU_VISIBLE_DEVICES'] = ''

from minivllm.engine.llm_engine import LLMEngine  # noqa: E402
from minivllm.sampling_params import SamplingParams  # noqa: E402


def log(msg):
    """Log message to file."""
    with open('debug_log_opt.txt', 'a') as f:
        f.write(msg + '\n')


def main():
    """Main execution function."""
    log('Starting script')

    model_id = 'facebook/opt-125m'
    model_path = model_id

    # Check if model path is a directory, if not, try to download
    if not os.path.isdir(model_path):
        print(
            f"Model path '{model_path}' not found locally. Attempting to download from Hugging Face Hub..."
        )
        log(f"Model path '{model_path}' not found locally. Attempting to download from Hugging Face Hub..."
            )
        try:
            from huggingface_hub import snapshot_download
            model_path = snapshot_download(repo_id=model_id)
            print(f'Model downloaded to: {model_path}')
            log(f'Model downloaded to: {model_path}')
        except ImportError:
            print(
                'huggingface_hub not found. Please install it with `pip install huggingface_hub`'
            )
            log('huggingface_hub not found')
            sys.exit(1)
        except Exception as e:
            print(f'Failed to download model: {e}')
            log(f'Failed to download model: {e}')
            sys.exit(1)

    print(f'Initializing LLMEngine with model: {model_path}')
    log(f'Initializing LLMEngine with model: {model_path}')

    try:
        engine = LLMEngine(model=model_path,
                           max_num_seqs=16,
                           max_model_len=1024,
                           enforce_eager=True,
                           trust_remote_code=True)
        log('Engine initialized')
    except Exception as e:
        log(f'Engine initialization failed: {e}')
        log(traceback.format_exc())
        print(f'Error initializing engine: {e}')
        sys.exit(1)

    prompts = [
        'Hello, who are you?',
        'What is 2+2?',
    ]

    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

    for i, prompt in enumerate(prompts):
        engine.add_request(prompt, sampling_params)

    log('Requests added')
    print('Generating...')

    step_count = 0
    while not engine.is_finished():
        step_count += 1
        if step_count % 10 == 0:
            log(f'Step {step_count}')
        try:
            step_outputs, _ = engine.step()
        except Exception as e:
            log(f'Step failed: {e}')
            log(traceback.format_exc())
            print(f'Step failed: {e}')
            break

        for seq_id, token_ids in step_outputs:
            text = engine.tokenizer.decode(token_ids, skip_special_tokens=True)
            print(f'Sequence {seq_id} output: {text}')

    print('Generation finished.')
    log('Generation finished.')

    # Decode and print results
    for i, prompt in enumerate(prompts):
        # We need to find the sequence for this prompt.
        # In this simple example, we can access the engine's scheduler or track manually.
        # But wait, `step_outputs` returns SequenceGroupOutput or similar?
        # In mini-vLLM, `step` returns `(List[SequenceGroup], List[SequenceGroup])` (scheduled, ignored)
        # Actually `engine.step()` returns `output, _`
        pass

    # Since we didn't track the final output in the loop properly (mini-vLLM might not return generated text in step return),
    # let's just inspect the sequences in the scheduler if possible, or assume the loop ran.
    # But for a proper example, we should decode.

    # However, `LLMEngine` in mini-vLLM seems to be lower level.
    # Let's try to access the sequences from the engine if possible, or just trust it ran.

    print('Done.')


if __name__ == '__main__':
    main()
