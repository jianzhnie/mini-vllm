"""RMS Layer Normalization implementation.

This implements RMSNorm (Root Mean Square Layer Normalization) used in some
transformer variants for improved training stability and efficiency. The module
avoids optional torch.compile decorators to keep compatibility across
PyTorch versions and toolchains.

RMSNorm is a computationally efficient alternative to LayerNorm that:
- Only normalizes by the root mean square of activations
- Learns a learned bias term (weight) for each dimension
- Provides similar benefits to LayerNorm with reduced computational cost
"""

from typing import Optional, Tuple

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (RMSNorm).

    RMSNorm normalizes activations by their root mean square instead of the
    traditional mean and variance used in LayerNorm. This approach provides
    similar benefits with reduced computational overhead.

    The normalization is applied as:
    y = x * (1 / sqrt(var + eps)) * weight

    where var = mean(x^2) is the variance computed over the last dimension.

    Args:
        hidden_size: Size of the hidden dimension to normalize
        eps: Small epsilon value for numerical stability (default: 1e-6)

    Attributes:
        eps: Numerical stability term added to variance
        weight: Learnable scaling parameter for each dimension

    Shape:
        - Input: Any tensor with last dimension of size `hidden_size`
        - Output: Same shape as input

    Examples:
        >>> norm = RMSNorm(hidden_size=512)
        >>> x = torch.randn(2, 4, 512)  # Batch, seq_len, hidden_size
        >>> output, _ = norm(x)
        >>> print(output.shape)  # torch.Size([2, 4, 512])

        >>> # With residual connection
        >>> residual = torch.randn(2, 4, 512)
        >>> output, new_residual = norm(x, residual)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps: float = float(eps)
        self.weight: nn.Parameter = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization without residual.

        Args:
            x: Input tensor to normalize

        Returns:
            Normalized tensor with same shape as input
        """
        orig_dtype = x.dtype
        # Convert to float for stable computation
        x = x.float()

        # Compute variance (mean of squares) along last dimension
        var = x.pow(2).mean(dim=-1, keepdim=True)

        # Normalize by RMS: divide by sqrt(var + eps)
        x.mul_(torch.rsqrt(var + self.eps))

        # Restore original dtype and apply learned scaling
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
            self, x: torch.Tensor,
            residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply RMS normalization with residual addition.

        This method efficiently combines the residual connection with
        normalization, reducing memory traffic and computation.

        Args:
            x: Input tensor to normalize
            residual: Residual tensor to add before normalization

        Returns:
            Tuple of (normalized tensor, updated residual)
        """
        orig_dtype = x.dtype

        # Add residual to input and convert to float
        x = x.float().add_(residual.float())

        # Update residual in original dtype
        residual = x.to(orig_dtype)

        # Compute variance and normalize
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))

        # Restore dtype and apply scaling
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional residual connection.

        Args:
            x: Input tensor to normalize
            residual: Optional residual tensor for pre-norm architecture

        Returns:
            Tuple of (output tensor, residual tensor or None)

        Note:
            When residual is None, returns (normalized_x, None)
            When residual is provided, returns (normalized_x, updated_residual)
            The updated residual is the input + residual for potential
            use in residual connections.
        """
        if residual is None:
            return self.rms_forward(x), None
        else:
            return self.add_rms_forward(x, residual)
