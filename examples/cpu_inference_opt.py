"""
Example of running facebook/opt-125m on CPU using mini-vLLM.

This script demonstrates how to:
1. Force execution on CPU by hiding GPU devices
2. Use the LLM high-level API with Config for simplified inference
3. Format prompts with chat template support
4. Run batch inference with progress tracking

Usage:
    python examples/cpu_inference_opt.py
"""

import os
import sys
from time import perf_counter

# Force CPU execution by hiding other devices (must be done before importing torch)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ''
os.environ['XPU_VISIBLE_DEVICES'] = ''
os.environ['MINIVLLM_DEVICE'] = 'cpu'

from minivllm import LLM, SamplingParams  # noqa: E402
from minivllm.config import Config  # noqa: E402
from minivllm.utils.example_utils import format_prompts_with_chat_template  # noqa: E402
from minivllm.utils.logger_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

prompts = [
    'Hello, who are you?',
    'What is your name?',
    'Where are you from?',
    'Where is the capital of France?',
    'Tell me a joke.',
]


def deduplicate_text(text: str, max_repeat: int = 3) -> str:
    """Remove excessive repetition from generated text."""
    lines = text.split('\n')
    seen = {}
    result = []

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        count = seen.get(line_stripped, 0)
        if count < max_repeat:
            result.append(line)
            seen[line_stripped] = count + 1

    return '\n'.join(result) if result else text


def wrap_text(text: str, width: int) -> list[str]:
    """Wrap text into multiple lines of specified width."""
    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_len = len(word)
        if current_length + word_len + len(current_line) <= width:
            current_line.append(word)
            current_length += word_len
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_len

    if current_line:
        lines.append(' '.join(current_line))

    return lines if lines else [text[:width]]


def format_output_box(prompt: str,
                      output: str,
                      index: int,
                      token_count: int = 0) -> str:
    """Format a single prompt-output pair in a nice box."""
    WIDTH = 76
    CONTENT_WIDTH = WIDTH - 4

    lines = [
        f"┌{'─' * WIDTH}┐",
        f'│ [{index}] Prompt: {prompt[:CONTENT_WIDTH-12]:<{CONTENT_WIDTH-12}} │',
        f"├{'─' * WIDTH}┤",
        f"│{' ' * CONTENT_WIDTH} │",
    ]

    output_clean = output.strip().replace('\n', ' ')
    wrapped_lines = wrap_text(output_clean, CONTENT_WIDTH - 2)

    for wrapped in wrapped_lines[:10]:
        lines.append(f'│  {wrapped:<{CONTENT_WIDTH-2}} │')

    if len(wrapped_lines) > 10:
        lines.append(
            f"│  ... ({len(wrapped_lines) - 10} more lines) {' ' * (CONTENT_WIDTH - 25)} │"
        )

    lines.append(f"│{' ' * CONTENT_WIDTH} │")

    if token_count > 0:
        token_info = f'Tokens: {token_count}'
        lines.append(f'│ {token_info:<{CONTENT_WIDTH}} │')

    lines.append(f"└{'─' * WIDTH}┘")

    return '\n'.join(lines)


def run_inference() -> None:
    """Run the inference pipeline."""
    config = Config(
        model='facebook/opt-125m',
        max_num_seqs=8,
        max_model_len=1024,
        enforce_eager=True,
        trust_remote_code=True,
        device_memory_utilization=0.9,
        dtype='float32',
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=40,
        max_tokens=50,
    )

    start_time = perf_counter()

    logger.info('Starting CPU inference with mini-vLLM')
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

    # Run inference
    logger.info(
        f'Generating completions for {len(formatted_prompts)} prompts...')
    inference_start = perf_counter()
    outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)
    inference_time = perf_counter() - inference_start
    total_time = perf_counter() - start_time

    total_tokens = sum(len(output['token_ids']) for output in outputs)

    # Print summary
    print('\n' + '=' * 80)
    print('              INFERENCE RESULTS (OPT-125M on CPU)')
    print('=' * 80)
    print(f'Model:        {config.model}')
    print('Device:       CPU')
    print(f'Prompts:      {len(prompts)}')
    print(f'Inference:    {inference_time:.2f}s')
    print(f'Total time:   {total_time:.2f}s')
    print(f'Tokens:       {total_tokens}')
    print(f'Throughput:   {total_tokens / inference_time:.1f} tokens/s')
    print('=' * 80 + '\n')

    # Print detailed results
    for idx, (formatted_prompt,
              output) in enumerate(zip(formatted_prompts, outputs)):
        original_prompt = prompts[idx] if idx < len(
            prompts) else formatted_prompt
        output_text = deduplicate_text(output['text'])
        token_count = len(output['token_ids'])

        print(format_output_box(original_prompt, output_text, idx,
                                token_count))
        print()

    print('=' * 80)
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
