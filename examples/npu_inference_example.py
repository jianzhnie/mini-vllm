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
"""

import sys
from time import perf_counter

from minivllm import LLM, SamplingParams
from minivllm.config import Config
from minivllm.utils.logger_utils import get_logger

logger = get_logger(__name__)

prompts = [
    'Hello, who are you?',
    'What is the capital of China?',
    'Tell me a short joke.',
    'Explain quantum computing in one sentence.',
]


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


def main():
    if not check_npu():
        return 1

    config = Config(
        model='facebook/opt-125m',
        max_num_seqs=4,
        max_model_len=256,
        enforce_eager=True,
        trust_remote_code=True,
        device_memory_utilization=0.85,
        dtype='float16',
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_tokens=32,
    )

    print(f'\nModel: {config.model}')
    print(
        f'Config: max_seqs={config.max_num_seqs}, max_len={config.max_model_len}, '
        f'dtype={config.dtype}')
    print(f'Prompts: {len(prompts)}\n')

    # Initialize LLM engine on NPU
    t0 = perf_counter()
    llm = LLM(config)
    init_time = perf_counter() - t0
    print(f'Engine initialized in {init_time:.1f}s\n')

    # Run inference
    t1 = perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
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
