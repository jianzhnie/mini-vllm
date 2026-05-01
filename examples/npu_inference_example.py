"""
NPU Inference Example - Run mini-vLLM on Huawei Ascend NPU

This example shows how to run LLM inference on NPU devices using mini-vLLM,
with automatic NPU Flash Attention acceleration.

Prerequisites:
    - Huawei Ascend NPU (910B/A2/A3 series)
    - CANN toolkit 7.0+
    - torch_npu installed

Usage:
    python examples/npu_inference_example.py

    # Custom model
    python examples/npu_inference_example.py --model /path/to/model

    # Adjust parameters
    python examples/npu_inference_example.py --max-seqs 8 --max-tokens 64
"""

import argparse
import os
import sys
from time import perf_counter


def check_npu() -> bool:
    """Verify NPU is available and print device info."""
    try:
        import torch_npu
    except ImportError:
        print('Error: torch_npu not installed. '
              'Install CANN toolkit for NPU support.')
        return False

    import torch
    if not torch.npu.is_available():
        print('Error: NPU not available. Check driver/CANN installation.')
        return False

    count = torch.npu.device_count()
    print(f'NPU: {torch.npu.get_device_name(0)}, {count} device(s)')
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run mini-vLLM inference on NPU',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str,
                        default=os.environ.get('MINIVLLM_MODEL',
                                               'facebook/opt-125m'),
                        help='Model path or HuggingFace model ID')
    parser.add_argument('--max-seqs', type=int, default=4,
                        help='Max sequences per batch')
    parser.add_argument('--max-model-len', type=int, default=256,
                        help='Max sequence length')
    parser.add_argument('--max-tokens', type=int, default=32,
                        help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Sampling temperature')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'],
                        help='Model data type')
    return parser.parse_args()


def main():
    args = parse_args()

    if not check_npu():
        return 1

    from minivllm import LLM, SamplingParams
    from minivllm.utils.logger_utils import get_logger

    logger = get_logger(__name__)

    prompts = [
        'Hello, who are you?',
        'What is the capital of China?',
        'Tell me a short joke.',
        'Explain quantum computing in one sentence.',
    ]

    print(f'\nModel: {args.model}')
    print(f'Config: max_seqs={args.max_seqs}, max_len={args.max_model_len}, '
          f'dtype={args.dtype}')
    print(f'Prompts: {len(prompts)}\n')

    # Initialize LLM engine on NPU
    t0 = perf_counter()
    llm = LLM(
        model=args.model,
        max_num_seqs=args.max_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        trust_remote_code=True,
        device_memory_utilization=0.85,
        dtype=args.dtype,
    )
    init_time = perf_counter() - t0
    print(f'Engine initialized in {init_time:.1f}s\n')

    # Run inference
    params = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        top_k=40,
        max_tokens=args.max_tokens,
    )

    t1 = perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=True)
    infer_time = perf_counter() - t1

    # Results
    total_tokens = sum(len(o['token_ids']) for o in outputs)
    throughput = total_tokens / infer_time

    print('\n' + '=' * 60)
    print(f'Results: {total_tokens} tokens in {infer_time:.2f}s '
          f'({throughput:.1f} tokens/s)')
    print('=' * 60)

    for prompt, output in zip(prompts, outputs):
        text = output['text'].strip()
        tokens = len(output['token_ids'])
        print(f'\n  Q: {prompt}')
        print(f'  A: {text} ({tokens} tokens)')

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
