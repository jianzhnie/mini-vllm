"""Tests for the weight loader functionality.

Tests various scenarios including:
- Basic weight loading from safetensors
- Error handling for missing/empty directories
- Custom weight loaders on parameters
- Shape validation in default weight loader
"""

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from minivllm.utils.loader import get_default_weight_loader, load_model


class DemoModel(nn.Module):
    """Demo model with various parameter types for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 30)
        self.embedding = nn.Embedding(100, 50)

        self.packed_modules_mapping = {
            "packed_": ("linear1.", 0),
            "sharded_": ("linear2.", 1),
        }

    def get_parameter(self, name: str) -> nn.Parameter:
        for param_name, param in self.named_parameters():
            if param_name == name:
                return param
        raise AttributeError(f"Parameter '{name}' not found")


class CustomWeightLoaderModel(nn.Module):
    """Model with custom weight loader for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 10)
        self.linear.weight.weight_loader = self._custom_weight_loader

    def _custom_weight_loader(
        self,
        param: nn.Parameter,
        tensor: torch.Tensor,
        shard_id: int = 0,
    ) -> None:
        param.data.copy_(tensor * 0.5)


def _create_test_weights(directory: Path) -> None:
    """Create test safetensors files."""
    tensors1 = {
        "linear1.weight": torch.randn(20, 10),
        "linear1.bias": torch.randn(20),
        "embedding.weight": torch.randn(100, 50),
    }
    tensors2 = {
        "linear2.weight": torch.randn(30, 20),
        "linear2.bias": torch.randn(30),
    }
    save_file(tensors1, directory / "model-00001-of-00002.safetensors")
    save_file(tensors2, directory / "model-00002-of-00002.safetensors")


class TestBasicLoading:
    """Test basic weight loading from safetensors files."""

    def test_loads_matching_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            _create_test_weights(temp_path)

            model = DemoModel()
            old_weight = model.linear1.weight.data.clone()

            load_model(model, str(temp_path))

            assert not torch.equal(model.linear1.weight.data, old_weight)

    def test_all_params_have_expected_shapes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            _create_test_weights(temp_path)

            model = DemoModel()
            load_model(model, str(temp_path))

            assert model.linear1.weight.shape == (20, 10)
            assert model.linear2.weight.shape == (30, 20)
            assert model.embedding.weight.shape == (100, 50)


class TestErrorHandling:
    """Test error handling for invalid inputs."""

    def test_nonexistent_directory_raises(self):
        model = DemoModel()
        with pytest.raises(FileNotFoundError):
            load_model(model, "/non/existent/path")

    def test_empty_directory_raises(self):
        model = DemoModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError):
                load_model(model, temp_dir)


class TestCustomWeightLoader:
    """Test custom weight loader callable on parameter attributes."""

    def test_custom_loader_scales_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_tensor = torch.randn(10, 5)
            save_file({"linear.weight": test_tensor}, temp_path / "custom.safetensors")

            model = CustomWeightLoaderModel()
            load_model(model, str(temp_path))

            expected = test_tensor * 0.5
            assert torch.allclose(model.linear.weight.data, expected, atol=1e-6)


class TestShapeValidation:
    """Test shape validation in default_weight_loader."""

    def test_correct_shape_loads_successfully(self):
        default_weight_loader = get_default_weight_loader()
        param = nn.Parameter(torch.randn(5, 3))
        correct_tensor = torch.randn(5, 3)
        default_weight_loader(param, correct_tensor)
        assert torch.allclose(param.data, correct_tensor)

    def test_wrong_shape_warns_and_skips(self):
        default_weight_loader = get_default_weight_loader()
        param = nn.Parameter(torch.randn(5, 3))
        original = param.data.clone()
        wrong_tensor = torch.randn(4, 2)
        default_weight_loader(param, wrong_tensor)
        assert torch.equal(param.data, original)
