"""Unit tests for Qwen3 model with NPU backend."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers.utils import is_torch_npu_available

from minivllm.models.layers.activation import SiluAndMul
from minivllm.models.layers.attention import Attention
from minivllm.models.layers.attention_backend import NPUAttentionBackend
from minivllm.models.layers.layernorm import RMSNorm
from minivllm.models.qwen3 import Qwen3ForCausalLM


def is_npu_available() -> bool:
    """Check if NPU is actually available for testing."""
    return is_torch_npu_available()


class TestQwen3NPUIntegration:
    """Tests for Qwen3 model integration with NPU backend."""

    @pytest.fixture
    def mock_npu_environment(self):
        """Mock NPU environment variables and availability."""
        with (
            patch("minivllm.models.layers.attention._NPU_FLASH_ATTN_AVAILABLE", True),
            patch("minivllm.models.layers.layernorm._NPU_RMS_NORM_AVAILABLE", True),
            patch("minivllm.models.layers.activation._NPU_SWIGLU_AVAILABLE", True),
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.Tensor.device", MagicMock(type="npu")),
        ):
            yield

    @pytest.mark.skipif(
        not is_npu_available(),
        reason="NPU not available - requires actual NPU hardware",
    )
    @patch("minivllm.models.layers.npu_flash_attention.npu_flash_attn_func")
    def test_qwen3_initialization_on_npu(self, mock_npu_attn, mock_npu_environment):
        """Test that Qwen3 initializes with NPU backend when NPU is available.

        This test requires actual NPU hardware as the Attention class
        checks for NPU device availability at initialization time.
        """
        config = SimpleNamespace(
            model_type="qwen3",
            hidden_size=4096,
            num_hidden_layers=2,
            num_attention_heads=32,
            num_key_value_heads=32,
            max_position_embeddings=32768,
            vocab_size=152064,
            intermediate_size=11008,
            hidden_act="silu",
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
                "minivllm.models.layers.linear.ColumnParallelLinear.forward",
                return_value=torch.randn(1, 10, 4096),
            ),
            patch(
                "minivllm.models.layers.linear.RowParallelLinear.forward",
                return_value=torch.randn(1, 10, 4096),
            ),
        ):
            model = Qwen3ForCausalLM(config)

            # Verify Attention backend
            first_layer_attn = model.model.layers[0].self_attn
            assert isinstance(first_layer_attn.attn, Attention)
            assert isinstance(first_layer_attn.attn.backend, NPUAttentionBackend)

    def test_rmsnorm_npu_dispatch(self, mock_npu_environment):
        """Test that RMSNorm dispatches to NPU kernel."""
        layer = RMSNorm(hidden_size=4096)
        x = torch.randn(1, 10, 4096)

        with patch.dict("sys.modules", {"torch_npu": MagicMock()}):
            import torch_npu

            torch_npu.npu_rms_norm.return_value = (torch.randn_like(x), x)

            layer(x)
            torch_npu.npu_rms_norm.assert_called_once()

    def test_silu_and_mul_npu_dispatch(self, mock_npu_environment):
        """Test that SiluAndMul dispatches to NPU kernel."""
        layer = SiluAndMul()
        x = torch.randn(1, 10, 8192)

        with patch.dict("sys.modules", {"torch_npu": MagicMock()}):
            import torch_npu

            torch_npu.npu_swiglu.return_value = torch.randn(1, 10, 4096)

            layer(x)
            torch_npu.npu_swiglu.assert_called_once()

    @patch("minivllm.models.layers.rotary_embedding._USE_NPU_ROPE", True)
    def test_rope_npu_dispatch(self, mock_npu_environment):
        """Test that RoPE dispatches to NPU kernel."""
        from minivllm.models.layers.rotary_embedding import apply_rotary_emb

        x = torch.randn(2, 8, 64)
        cos = torch.randn(8, 32)
        sin = torch.randn(8, 32)

        # Mock x.device to report 'npu'
        mock_device = MagicMock()
        mock_device.type = "npu"

        with patch.dict("sys.modules", {"torch_npu": MagicMock()}):
            import torch_npu

            torch_npu.npu_rotary_mul.return_value = torch.randn_like(x)

            with patch.object(x, "device", mock_device):
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

        assert hasattr(rope, "cos_cache")
        assert hasattr(rope, "sin_cache")
        assert not hasattr(rope, "cos_sin_cache")

        expected_shape = (max_position, rotary_dim // 2)
        assert rope.cos_cache.shape == expected_shape
        assert rope.sin_cache.shape == expected_shape
