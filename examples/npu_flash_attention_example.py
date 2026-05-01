"""
NPU Flash Attention Example for mini-vLLM

This example demonstrates NPU Flash Attention on Huawei Ascend NPUs,
showing both low-level attention layer usage and high-level LLM inference.

Usage:
    # On an NPU machine with Ascend CANN toolkit installed:
    python examples/npu_flash_attention_example.py

    # With a specific model:
    python examples/npu_flash_attention_example.py --model /path/to/model
"""

import argparse
import os
import sys
import time
from typing import Optional

import torch


def check_npu_environment() -> bool:
    """Check NPU availability and print diagnostic information."""
    print('=' * 60)
    print('NPU Environment Check')
    print('=' * 60)

    try:
        import torch_npu
        print(f'torch_npu version: {getattr(torch_npu, "__version__", "unknown")}')
    except ImportError:
        print('torch_npu not installed. Install CANN toolkit for NPU support.')
        return False

    if not torch.npu.is_available():
        print('NPU not available. Check CANN driver installation.')
        return False

    device_count = torch.npu.device_count()
    print(f'NPU devices: {device_count}')
    for i in range(device_count):
        name = torch.npu.get_device_name(i)
        print(f'  Device {i}: {name}')

    # Check API availability
    apis = {
        'npu_fusion_attention': 'Training + basic inference',
        'npu_incre_flash_attention': 'Incremental decode (legacy)',
        'npu_prompt_flash_attention': 'Prefill (legacy)',
        'npu_fused_infer_attention_score': 'Unified inference (recommended)',
    }
    for api, desc in apis.items():
        available = hasattr(torch_npu, api)
        status = 'OK' if available else 'NOT AVAILABLE'
        print(f'  {api}: {status} ({desc})')

    print()
    return True


def demo_attention_layer():
    """Demonstrate low-level NPU Flash Attention via the Attention layer.

    This shows prefill and decode phases using mini-vLLM's Attention module
    with the NPU backend automatically selected.
    """
    from minivllm.models.layers.attention import Attention
    from minivllm.utils.context import Context, set_context, reset_context

    print('=' * 60)
    print('Demo: Low-level Attention Layer on NPU')
    print('=' * 60)

    num_heads = 8
    head_dim = 64
    num_kv_heads = 8
    scale = 1.0 / (head_dim ** 0.5)
    device = torch.device('npu:0')

    attn = Attention(
        num_heads=num_heads,
        head_dim=head_dim,
        scale=scale,
        num_kv_heads=num_kv_heads,
    )
    print(f'Attention backend: {attn.backend.__class__.__name__}')

    # --- Prefill Phase ---
    print('\n--- Prefill Phase ---')
    batch_sizes = [4, 6]
    total_tokens = sum(batch_sizes)
    seq_len = max(batch_sizes)

    q = torch.randn(total_tokens, num_heads, head_dim, device=device)
    k = torch.randn(total_tokens, num_kv_heads, head_dim, device=device)
    v = torch.randn(total_tokens, num_kv_heads, head_dim, device=device)

    cum_q = torch.tensor([0, 4, 10], dtype=torch.int32, device=device)
    cum_k = torch.tensor([0, 4, 10], dtype=torch.int32, device=device)

    set_context(
        is_prefill=True,
        max_seqlen_q=seq_len,
        max_seqlen_k=seq_len,
        cum_seqlens_q=cum_q,
        cum_seqlens_k=cum_k,
        slot_mapping=torch.arange(total_tokens, device=device),
    )

    with torch.no_grad():
        out = attn(q, k, v)
    reset_context()

    print(f'Input: q={q.shape}, k={k.shape}, v={v.shape}')
    print(f'Output: {out.shape}, device={out.device}')

    # --- Decode Phase ---
    print('\n--- Decode Phase ---')
    batch_size = 2
    block_size = 16
    num_blocks = 4

    attn.k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                               device=device)
    attn.v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                               device=device)
    attn._cache_initialized = True

    q = torch.randn(batch_size, num_heads, head_dim, device=device)
    k = torch.randn(batch_size, num_kv_heads, head_dim, device=device)
    v = torch.randn(batch_size, num_kv_heads, head_dim, device=device)

    set_context(
        is_prefill=False,
        slot_mapping=torch.tensor([0, block_size], dtype=torch.int32,
                                  device=device),
        context_lens=torch.tensor([3, 5], dtype=torch.int32, device=device),
        block_tables=torch.tensor([[0, -1], [1, 2]], dtype=torch.int32,
                                  device=device),
    )

    with torch.no_grad():
        out = attn(q, k, v)
    reset_context()

    print(f'Input: q={q.shape}')
    print(f'Output: {out.shape}, device={out.device}')
    print()


def demo_npu_inference_backend():
    """Demonstrate the NPUAttentionBackend directly.

    Shows unified_inference API that auto-selects prefill/decode kernels.
    """
    from minivllm.models.layers.attention_backend import NPUAttentionBackend

    print('=' * 60)
    print('Demo: NPUAttentionBackend Unified Inference')
    print('=' * 60)

    device = torch.device('npu:0')
    backend = NPUAttentionBackend()

    num_heads = 8
    num_kv_heads = 8
    head_dim = 64
    scale = 1.0 / (head_dim ** 0.5)

    # Prefill
    batch_size = 2
    seq_len = 16

    query = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    key = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    value = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)

    out = backend.forward(query, key, value, is_causal=True)
    print(f'Prefill output: {out.shape}')

    # Decode
    query = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
    out = backend.forward(query, key, value, is_causal=True)
    print(f'Decode output: {out.shape}')
    print()


def demo_llm_inference(model_path: Optional[str] = None):
    """Demonstrate full LLM inference on NPU using the high-level API.

    Args:
        model_path: Path to a local model or HuggingFace model ID.
    """
    from minivllm import LLM, SamplingParams

    print('=' * 60)
    print('Demo: Full LLM Inference on NPU')
    print('=' * 60)

    model = model_path or os.environ.get(
        'MINIVLLM_MODEL', 'facebook/opt-125m')

    print(f'Model: {model}')
    print(f'Device: NPU')

    llm = LLM(
        model=model,
        max_num_seqs=4,
        max_model_len=128,
        enforce_eager=True,
        trust_remote_code=True,
        device_memory_utilization=0.8,
        dtype='float16',
    )

    prompts = [
        'Hello, who are you?',
        'What is the capital of France?',
        'Tell me a short joke.',
    ]

    params = SamplingParams(temperature=0.7, max_tokens=32)
    outputs = llm.generate(prompts, params, use_tqdm=True)

    for prompt, output in zip(prompts, outputs):
        text = output['text'].strip()
        tokens = len(output['token_ids'])
        print(f'\nPrompt: {prompt!r}')
        print(f'Output ({tokens} tokens): {text!r}')

    print()


def parse_args():
    parser = argparse.ArgumentParser(
        description='NPU Flash Attention Example',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, default=None,
                        help='Model path or HuggingFace model ID')
    parser.add_argument('--skip-low-level', action='store_true',
                        help='Skip low-level attention demos')
    return parser.parse_args()


def main():
    args = parse_args()

    print('mini-vLLM NPU Flash Attention Example\n')

    if not check_npu_environment():
        print('NPU not available. This example requires Huawei Ascend NPU '
              'with CANN toolkit installed.')
        print('\nTo run on other devices, see examples/cpu_inference_opt.py '
              'or the main examples.py')
        return 1

    torch.npu.set_device(0)

    if not args.skip_low_level:
        try:
            demo_attention_layer()
        except Exception as e:
            print(f'Attention layer demo failed: {e}')

        try:
            demo_npu_inference_backend()
        except Exception as e:
            print(f'Backend demo failed: {e}')

    if args.model:
        try:
            demo_llm_inference(args.model)
        except Exception as e:
            print(f'LLM inference demo failed: {e}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
