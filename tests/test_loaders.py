#!/usr/bin/env python3
"""
Test script to demonstrate the improved loader functionality.

This script tests various scenarios including:
- Basic weight loading
- Packed module handling
- Error handling and validation
- Custom weight loaders
"""
import logging
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import save_file

from minivllm.utils.loader import get_default_weight_loader, load_model

# Configure logging to see the improvements
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class TestModel(nn.Module):
    """Test model with various parameter types for demonstration."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 30)
        self.embedding = nn.Embedding(100, 50)

        # Add packed modules mapping for demonstration
        self.packed_modules_mapping = {
            'packed_': ('linear1.', 0),
            'sharded_': ('linear2.', 1),
        }

    def get_parameter(self, name: str) -> nn.Parameter:
        """Custom parameter getter for testing."""
        for param_name, param in self.named_parameters():
            if param_name == name:
                return param
        raise AttributeError(f"Parameter '{name}' not found")


class CustomWeightLoaderModel(nn.Module):
    """Model with custom weight loader for demonstration."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 10)

        # Add custom weight loader to the parameter
        self.linear.weight.weight_loader = self._custom_weight_loader

    def _custom_weight_loader(self,
                              param: nn.Parameter,
                              tensor: torch.Tensor,
                              shard_id: int = 0) -> None:
        """Custom weight loader that scales the weights."""
        print(f'Custom weight loader called with shard_id: {shard_id}')
        param.data.copy_(tensor * 0.5)  # Scale weights by 0.5


def create_test_weights(directory: Path) -> None:
    """Create test safetensors files."""
    # Create some test tensors
    tensors1 = {
        'linear1.weight': torch.randn(20, 10),
        'linear1.bias': torch.randn(20),
        'embedding.weight': torch.randn(100, 50),
    }

    tensors2 = {
        'linear2.weight': torch.randn(30, 20),
        'linear2.bias': torch.randn(30),
        'packed_extra.weight': torch.randn(15,
                                           10),  # For packed module testing
        'sharded_extra.weight': torch.randn(25,
                                            20),  # For sharded module testing
    }

    # Save to safetensors files
    save_file(tensors1, directory / 'model-00001-of-00002.safetensors')
    save_file(tensors2, directory / 'model-00002-of-00002.safetensors')
    print(f'Created test safetensors files in: {directory}')


def test_basic_loading() -> None:
    """Test basic weight loading functionality."""
    print('\n' + '=' * 60)
    print('Testing Basic Weight Loading')
    print('=' * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        create_test_weights(temp_path)

        model = TestModel()
        print('Model parameters before loading:')
        for name, param in model.named_parameters():
            print(f'  {name}: {param.shape}')

        load_model(model, str(temp_path))
        print('✓ Basic loading completed successfully')


def test_error_handling() -> None:
    """Test error handling for invalid inputs."""
    print('\n' + '=' * 60)
    print('Testing Error Handling')
    print('=' * 60)

    model = TestModel()

    # Test non-existent directory
    try:
        load_model(model, '/non/existent/path')
        print('✗ Should have raised FileNotFoundError')
    except FileNotFoundError as e:
        print(f'✓ Correctly caught FileNotFoundError: {e}')

    # Test empty directory
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            load_model(model, temp_dir)
            print('✗ Should have raised ValueError')
        except ValueError as e:
            print(f'✓ Correctly caught ValueError: {e}')


def test_custom_weight_loader() -> None:
    """Test custom weight loader functionality."""
    print('\n' + '=' * 60)
    print('Testing Custom Weight Loader')
    print('=' * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test tensor
        test_tensor = torch.randn(10, 5)
        save_file({'linear.weight': test_tensor},
                  temp_path / 'custom.safetensors')

        model = CustomWeightLoaderModel()

        print('Original weight statistics:')
        print(f'  Mean: {model.linear.weight.data.mean():.4f}')
        print(f'  Std:  {model.linear.weight.data.std():.4f}')

        load_model(model, str(temp_path))

        print('After custom loading (should be scaled by 0.5):')
        print(f'  Mean: {model.linear.weight.data.mean():.4f}')
        print(f'  Std:  {model.linear.weight.data.std():.4f}')
        print('✓ Custom weight loader test completed')


def test_shape_validation() -> None:
    """Test shape validation in default_weight_loader."""
    print('\n' + '=' * 60)
    print('Testing Shape Validation')
    print('=' * 60)

    default_weight_loader = get_default_weight_loader()

    param = nn.Parameter(torch.randn(5, 3))
    correct_tensor = torch.randn(5, 3)
    wrong_tensor = torch.randn(4, 2)

    # Test correct shape
    try:
        default_weight_loader(param, correct_tensor)
        print('✓ Correct shape loaded successfully')
    except Exception as e:
        print(f'✗ Unexpected error with correct shape: {e}')

    # Test wrong shape
    try:
        default_weight_loader(param, wrong_tensor)
        print('✗ Should have raised RuntimeError for wrong shape')
    except RuntimeError as e:
        print(f'✓ Correctly caught shape mismatch: {e}')


if __name__ == '__main__':
    test_basic_loading()
    test_error_handling()
    test_custom_weight_loader()
    test_shape_validation()
