"""
Example of running facebook/opt-125m on CPU using mini-vLLM.

This script demonstrates how to:
1. Force execution on CPU by hiding GPU devices
2. Use the LLM high-level API for simplified inference
3. Format prompts with chat template support
4. Run batch inference with progress tracking

Usage:
    python examples/cpu_inference_opt.py

    # Custom model path
    python examples/cpu_inference_opt.py --model /path/to/model

    # More tokens
    python examples/cpu_inference_opt.py --max-tokens 100
"""

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

# Force CPU execution by hiding other devices (must be done before importing torch)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = ''
os.environ['XPU_VISIBLE_DEVICES'] = ''

from minivllm import LLM, SamplingParams  # noqa: E402
from minivllm.utils.logger_utils import get_logger  # noqa: E402

logger = get_logger(__name__)


@dataclass
class CPUInferenceConfig:
    """Configuration for CPU inference example.

    Attributes:
        model_name_or_path: Path to the model weights (HuggingFace format).
        max_num_seqs: Maximum number of sequences to process in parallel.
        max_model_len: Maximum sequence length for the model.
        temperature: Sampling temperature (> 1e-10).
        top_p: Nucleus sampling probability threshold (0-1).
        top_k: Top-k sampling (-1 to disable).
        max_tokens: Maximum tokens to generate per prompt.
        prompts: List of prompts to generate from.
    """
    model_name_or_path: str = '/Users/jianzhengnie/hfhub/models/facebook/opt-125m'
    max_num_seqs: int = 8
    max_model_len: int = 1024
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 40
    max_tokens: int = 50
    prompts: list[str] = field(default_factory=lambda: [
        'Hello, who are you?',
        'What is your name?',
        'Where are you from?',
        'Where is the capital of France?',
        'Tell me a joke.',
    ])


def parse_args() -> CPUInferenceConfig:
    """Parse command-line arguments.

    Returns:
        CPUInferenceConfig with parsed or default values.
    """
    parser = argparse.ArgumentParser(
        description='Run OPT-125M inference on CPU with mini-vLLM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--model',
        type=str,
        default='/Users/jianzhengnie/hfhub/models/facebook/opt-125m',
        help='Path to the model (HuggingFace format)',
    )
    parser.add_argument(
        '--max-seqs',
        type=int,
        default=8,
        help='Maximum sequences per batch',
    )
    parser.add_argument(
        '--max-model-len',
        type=int,
        default=1024,
        help='Maximum sequence length',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.6,
        help='Sampling temperature',
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.95,
        help='Nucleus sampling probability',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='Top-k sampling (-1 to disable)',
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=50,
        help='Maximum tokens to generate',
    )

    args = parser.parse_args()

    return CPUInferenceConfig(
        model_name_or_path=args.model,
        max_num_seqs=args.max_seqs,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )


def validate_config(config: CPUInferenceConfig) -> None:
    """Validate configuration parameters.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If any parameter is invalid.
    """
    if not Path(config.model_name_or_path).exists():
        raise ValueError(
            f'Model path does not exist: {config.model_name_or_path}\n'
            f'Set the model path via --model argument.')

    if config.temperature <= 1e-10:
        raise ValueError(
            f'Temperature must be > 1e-10, got {config.temperature}. '
            f'Greedy sampling (temperature=0) is not supported.')

    if not 0.0 < config.top_p <= 1.0:
        raise ValueError(f'top_p must be in (0, 1.0], got {config.top_p}')


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


def deduplicate_text(text: str, max_repeat: int = 3) -> str:
    """Remove excessive repetition from generated text.

    Args:
        text: Input text that may contain repetitions.
        max_repeat: Maximum allowed repetitions of a sentence/phrase.

    Returns:
        Text with repetitions reduced.
    """
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
    """Wrap text into multiple lines of specified width.

    Args:
        text: Input text to wrap.
        width: Maximum width per line.

    Returns:
        List of wrapped lines.
    """
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
    """Format a single prompt-output pair in a nice box.

    Args:
        prompt: The input prompt.
        output: The generated output text.
        index: The prompt index.
        token_count: Number of tokens generated.

    Returns:
        Formatted string for display.
    """
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


def run_inference(config: CPUInferenceConfig) -> None:
    """Run the inference pipeline with the given configuration.

    Args:
        config: Configuration containing model and sampling parameters.
    """
    start_time = perf_counter()

    logger.info('Starting CPU inference with mini-vLLM')
    logger.info(f'Model: {config.model_name_or_path}')
    logger.info(
        f'Configuration: max_seqs={config.max_num_seqs}, max_tokens={config.max_tokens}'
    )

    # Validate configuration
    validate_config(config)

    # Initialize LLM engine
    logger.info('Initializing LLM engine...')
    llm = LLM(
        model=config.model_name_or_path,
        max_num_seqs=config.max_num_seqs,
        max_model_len=config.max_model_len,
        enforce_eager=True,  # Required for CPU
        trust_remote_code=True,
        device_memory_utilization=0.9,
        dtype='float32',
    )
    logger.info('LLM engine initialized successfully.')

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
    inference_start = perf_counter()
    outputs = llm.generate(formatted_prompts, sampling_params, use_tqdm=True)
    inference_time = perf_counter() - inference_start
    total_time = perf_counter() - start_time

    total_tokens = sum(len(output['token_ids']) for output in outputs)

    # Print summary
    print('\n' + '=' * 80)
    print('              INFERENCE RESULTS (OPT-125M on CPU)')
    print('=' * 80)
    print('Model:        facebook/opt-125m')
    print('Device:       CPU')
    print(f'Prompts:      {len(config.prompts)}')
    print(f'Inference:    {inference_time:.2f}s')
    print(f'Total time:   {total_time:.2f}s')
    print(f'Tokens:       {total_tokens}')
    print(f'Throughput:   {total_tokens / inference_time:.1f} tokens/s')
    print('=' * 80 + '\n')

    # Print detailed results
    for prompt, output in zip(formatted_prompts, outputs):
        prompt_idx = config.prompts.index(
            prompt) if prompt in config.prompts else 0
        # Find original prompt index
        for i, p in enumerate(config.prompts):
            if p in prompt or prompt in p:
                prompt_idx = i
                break

        output_text = deduplicate_text(output['text'])
        token_count = len(output['token_ids'])
        original_prompt = config.prompts[prompt_idx]

        print(
            format_output_box(original_prompt, output_text, prompt_idx,
                              token_count))
        print()

    print('=' * 80)
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
