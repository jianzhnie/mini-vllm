"""Unit tests for Qwen3 model with NPU backend."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from minivllm.models.layers.activation import SiluAndMul
from minivllm.models.layers.attention import Attention, NPUAttentionBackend
from minivllm.models.layers.layernorm import RMSNorm
from minivllm.models.qwen3 import Qwen3ForCausalLM


class TestQwen3NPUIntegration:
    """Tests for Qwen3 model integration with NPU backend."""

    @pytest.fixture
    def mock_npu_environment(self):
        """Mock NPU environment variables and availability."""
        with (
                patch(
                    'minivllm.models.layers.attention._NPU_FLASH_ATTN_AVAILABLE',
                    True),
                patch(
                    'minivllm.models.layers.layernorm._NPU_RMS_NORM_AVAILABLE',
                    True),
                patch(
                    'minivllm.models.layers.activation._NPU_SWIGLU_AVAILABLE',
                    True),
                patch('torch.cuda.is_available', return_value=False),
        ):
            # Also mock device.type to return 'npu' for tensors
            with patch('torch.Tensor.device', MagicMock(type='npu')):
                yield

    @patch('minivllm.models.layers.npu_flash_attention.npu_flash_attn_func')
    def test_qwen3_initialization_on_npu(self, mock_npu_attn,
                                         mock_npu_environment):
        """Test that Qwen3 initializes with NPU backend when NPU is available."""
        config = SimpleNamespace(
            model_type='qwen3',
            hidden_size=4096,
            num_hidden_layers=2,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=32768,
            vocab_size=152064,
            intermediate_size=11008,
            hidden_act='silu',
            rms_norm_eps=1e-6,
            attention_bias=True,
            head_dim=128,
            rope_theta=1000000.0,
            rope_scaling=None,
            tie_word_embeddings=False,
        )

        # Mock linear layers to avoid heavy initialization
        with (
                patch(
                    'minivllm.models.layers.linear.ColumnParallelLinear.forward',
                    return_value=torch.randn(1, 10, 4096),
                ),
                patch(
                    'minivllm.models.layers.linear.RowParallelLinear.forward',
                    return_value=torch.randn(1, 10, 4096),
                ),
        ):
            model = Qwen3ForCausalLM(config)

            # Verify Attention backend
            first_layer_attn = model.model.layers[0].self_attn
            assert isinstance(first_layer_attn, Attention)
            assert isinstance(first_layer_attn.backend, NPUAttentionBackend)

    def test_rmsnorm_npu_dispatch(self, mock_npu_environment):
        """Test that RMSNorm dispatches to NPU kernel."""
        layer = RMSNorm(hidden_size=4096)
        x = torch.randn(1, 10, 4096)
        # Mock device to be NPU
        x.device.type = 'npu'

        # We need to mock the import inside the method or rely on the patch in fixture
        # The method in RMSNorm does: import torch_npu
        # Since we patch _NPU_RMS_NORM_AVAILABLE=True, it will try to import.
        # We need to make sure 'import torch_npu' works or is mocked.
        # Python's sys.modules patching is tricky.
        # Instead, we can rely on the fact that if it tries to use torch_npu.npu_rms_norm,
        # we can mock that call if we can intercept the import.

        with patch.dict('sys.modules', {'torch_npu': MagicMock()}):
            import torch_npu

            torch_npu.npu_rms_norm.return_value = (torch.randn_like(x), x)

            layer(x)
            torch_npu.npu_rms_norm.assert_called_once()

    def test_silu_and_mul_npu_dispatch(self, mock_npu_environment):
        """Test that SiluAndMul dispatches to NPU kernel."""
        layer = SiluAndMul()
        x = torch.randn(1, 10, 8192)  # 2 * 4096
        x.device.type = 'npu'

        with patch.dict('sys.modules', {'torch_npu': MagicMock()}):
            import torch_npu

            torch_npu.npu_swiglu.return_value = torch.randn(1, 10, 4096)

            layer(x)
            torch_npu.npu_swiglu.assert_called_once()

    def test_rope_npu_dispatch(self, mock_npu_environment):
        """Test that RoPE dispatches to NPU kernel."""
        from minivllm.models.layers.rotary_embedding import apply_rotary_emb

        x = torch.randn(2, 8, 64)
        cos = torch.randn(8, 32)
        sin = torch.randn(8, 32)
        x.device.type = 'npu'

        with patch.dict('sys.modules', {'torch_npu': MagicMock()}):
            import torch_npu

            torch_npu.npu_rotary_mul.return_value = torch.randn_like(x)

            apply_rotary_emb(x, cos, sin)
            torch_npu.npu_rotary_mul.assert_called_once()

    def test_rotary_embedding_cache_structure(self, mock_npu_environment):
        """Test that RotaryEmbedding uses separated cos/sin caches."""
        from minivllm.models.layers.rotary_embedding import RotaryEmbedding

        head_size = 128
        rotary_dim = 128
        max_position = 1024
        base = 10000.0

        rope = RotaryEmbedding(head_size, rotary_dim, max_position, base)

        # Check if caches are registered
        assert hasattr(rope, 'cos_cache')
        assert hasattr(rope, 'sin_cache')
        assert not hasattr(rope, 'cos_sin_cache')  # Should be removed

        # Check shapes
        # cos/sin cache should be (max_position, rotary_dim)
        # Note: In implementation, freqs.cos() results in shape (max_position, rotary_dim) if not chunked?
        # Wait, implementation:
        # inv_freq: (rotary_dim // 2)
        # freqs = outer(t, inv_freq) -> (max_pos, rotary_dim // 2)
        # cos = freqs.cos() -> (max_pos, rotary_dim // 2)
        # Wait, let's re-read RotaryEmbedding implementation.
        # It says:
        # inv_freq = 1.0 / (base ** (arange(0, rotary_dim, 2) / rotary_dim))
        # This has size rotary_dim // 2.
        # freqs = einsum(t, inv_freq) -> (max_pos, rotary_dim // 2)
        # cos = freqs.cos()
        # sin = freqs.sin()
        #
        # But `apply_rotary_emb` expects `cos` broadcastable to `x`'s last half shape.
        # If `x` is (..., dim), last half is dim/2.
        # So `cos` should be (..., dim/2).
        # But wait, `RotaryEmbedding` implementation usually repeats cos/sin to match dim?
        #
        # In `RotaryEmbedding.__init__`:
        # cache = torch.cat((cos, sin), dim=-1)
        # No, wait.
        #
        # Standard RoPE:
        # x = [x1, x2]
        # x_rotated = [x1*cos - x2*sin, x2*cos + x1*sin]
        # Here `cos` and `sin` apply to x1 and x2.
        # If x1 has size dim/2, then cos has size dim/2.
        #
        # Let's check `apply_rotary_emb`:
        # x1, x2 = chunk(x, 2, dim=-1)
        # y1 = x1*cos - x2*sin
        #
        # So `cos` must be broadcastable to `x1` which is (..., dim/2).
        # My implementation of `__init__`:
        # inv_freq size is dim/2.
        # freqs size is (max_pos, dim/2).
        # cos size is (max_pos, dim/2).
        #
        # But wait, in `RotaryEmbedding.__init__` before my change:
        # cache = torch.cat((cos, sin), dim=-1) -> (max_pos, dim)
        # In `forward`:
        # cos_sin = cache[positions] -> (..., dim)
        # cos, sin = cos_sin.chunk(2, dim=-1) -> (..., dim/2)
        #
        # So `cos` was (..., dim/2).
        #
        # In my NEW implementation:
        # self.register_buffer('cos_cache', cos) -> cos is (max_pos, dim/2)
        # self.register_buffer('sin_cache', sin) -> sin is (max_pos, dim/2)
        #
        # In `forward`:
        # cos = self.cos_cache[positions] -> (..., dim/2)
        #
        # So `cos_cache` shape should be (max_position, rotary_dim // 2).

        expected_shape = (max_position, rotary_dim // 2)
        assert rope.cos_cache.shape == expected_shape
        assert rope.sin_cache.shape == expected_shape
