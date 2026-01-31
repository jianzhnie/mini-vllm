"""Activation helper layers.

Provides small activation utilities used by model MLPs.

This module contains activation functions commonly used in transformer models,
particularly for gated architectures like SwiGLU variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SiluAndMul']


class SiluAndMul(nn.Module):
    """Gated SiLU-style activation (SwiGLU variant).

    This activation function splits the input tensor in half along the last
    dimension, applies the SiLU (Sigmoid Linear Unit) activation to the first
    half, and multiplies it element-wise with the second half.

    The SiLU function is defined as: SiLU(x) = x * sigmoid(x)

    This creates a gated mechanism where one part of the input controls the flow
    of information from the other part, commonly used in modern transformer MLPs.

    Args:
        None

    Raises:
        ValueError: If the last dimension of input tensor is not even.

    Shape:
        - Input: (*, 2*D) where * means any number of dimensions and D is any positive integer
        - Output: (*, D), same shape as input but with last dimension halved

    Examples:
        >>> import torch
        >>> from minivllm.models.layers import SiluAndMul
        >>> activation = SiluAndMul()
        >>> x = torch.randn(2, 4, 64)  # Last dimension must be even
        >>> output = activation(x)
        >>> print(output.shape)  # torch.Size([2, 4, 32])
    """

    def __init__(self) -> None:
        super().__init__()

    def extra_repr(self) -> str:
        return ''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated SiLU activation.

        Args:
            x: Input tensor with shape (*, 2*D) where the last dimension
               must be even. The tensor will be split into two equal parts.

        Returns:
            Tensor with shape (*, D) after applying gated SiLU activation.

        Raises:
            ValueError: If the last dimension of x is not even.
        """
        if x.size(-1) % 2 != 0:
            raise ValueError(
                f'Input last dimension must be even for SiluAndMul, got {x.size(-1)}'
            )

        # Split input into two halves along the last dimension
        x1, x2 = x.chunk(2, dim=-1)

        # Apply SiLU to first half and multiply with second half
        return F.silu(x1) * x2
