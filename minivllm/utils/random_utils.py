"""Random number generation utilities.

This module provides utilities for setting random seeds across different
libraries and devices (CPU, GPU, NPU) to ensure reproducibility.
"""

import random

import numpy as np
import torch
from transformers.utils import is_torch_npu_available

__all__ = ['set_random_seed']


def set_random_seed(seed: int) -> None:
    """Set the random seed for reproducibility.

    This function sets the seed for:
    - Python's random module
    - NumPy's random module
    - PyTorch (CPU and all available devices including CUDA and NPU)

    Args:
        seed: The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if is_torch_npu_available():
        try:
            torch.npu.manual_seed_all(seed)
        except AttributeError:
            # Fallback if manual_seed_all is not available
            torch.npu.manual_seed(seed)
