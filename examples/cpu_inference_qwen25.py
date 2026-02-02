"""
Example of running Qwen2.5-0.5B-Instruct on CPU using mini-vLLM.

This script demonstrates how to:
1. Initialize the LLMEngine with a lightweight model
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
    with open('debug_log.txt', 'a') as f:
        f.write(msg + '\n')


def main():
    """Main execution function."""
    log('Starting script')

    model_id = 'Qwen/Qwen2.5-0.5B-Instruct'
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

    import minivllm.engine.model_runner
    print(
        f'DEBUG: minivllm.engine.model_runner file: {minivllm.engine.model_runner.__file__}'
    )
    log(f'DEBUG: minivllm.engine.model_runner file: {minivllm.engine.model_runner.__file__}'
        )

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

    # Map seq_id to prompt index
    generated_tokens = {i: [] for i in range(len(prompts))}

    step_count = 0
    while engine.scheduler.has_unfinished_sequences():
        step_count += 1
        if step_count % 10 == 0:
            log(f'Step {step_count}')
        try:
            step_outputs, _ = engine.step()
        except Exception as e:
            log(f'Step failed: {e}')
            log(traceback.format_exc())
            break

        for seq_id, tokens in step_outputs:
            if seq_id < len(prompts):
                generated_tokens[seq_id].extend(tokens)

    log('Inference loop finished')

    # Decode and print
    with open('inference_output.txt', 'w') as f:
        for i, prompt in enumerate(prompts):
            output_ids = generated_tokens[i]
            output_text = engine.tokenizer.decode(output_ids,
                                                  skip_special_tokens=True)
            print(f'\nPrompt: {prompt}')
            print(f'Output: {output_text}')
            f.write(f'Prompt: {prompt}\n')
            f.write(f'Output: {output_text}\n\n')
    print('\nInference completed. Results saved to inference_output.txt')


if __name__ == '__main__':
    main()
