import sys

import torch

from minivllm.models.layers.attention_backend import NPUAttentionBackend
from minivllm.utils.device import is_torch_npu_available


def check_npu_graph():
    print('Checking NPU Graph execution...', flush=True)

    if not is_torch_npu_available():
        print(
            'NPU is not available on this system. Skipping NPU specific checks.',
            flush=True,
        )
        return

    print('NPU detected. Running NPU Flash Attention check...', flush=True)

    # Setup NPU device
    device = torch.device('npu:0')

    # Parameters
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 64

    # Create dummy tensors (BNSD layout)
    q = torch.randn(batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    device=device,
                    dtype=torch.float16)
    k = torch.randn(batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    device=device,
                    dtype=torch.float16)
    v = torch.randn(batch_size,
                    num_heads,
                    seq_len,
                    head_dim,
                    device=device,
                    dtype=torch.float16)

    backend = NPUAttentionBackend()

    try:
        print('Running forward pass...', flush=True)
        output = backend.forward(q, k, v, is_causal=True)
        print(f'Forward pass successful. Output shape: {output.shape}',
              flush=True)

        # Check output layout (should be BNSD)
        expected_shape = (batch_size, num_heads, seq_len, head_dim)
        assert output.shape == expected_shape, (
            f'Expected shape {expected_shape}, got {output.shape}')
        print('Output shape verification passed.', flush=True)

    except Exception as e:
        print(f'FAILED: NPU Attention check failed with error: {e}',
              flush=True)
        sys.exit(1)

    print('NPU Graph check passed successfully!', flush=True)


if __name__ == '__main__':
    check_npu_graph()
