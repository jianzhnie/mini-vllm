"""Attention backend abstraction layer.

This module provides a unified interface for different attention backends
(CUDA, NPU, CPU) with proper error handling and type safety.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class AttentionBackend(ABC):
    """Abstract base class for attention implementations."""

    @abstractmethod
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass for attention computation.

        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key: Key tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            value: Value tensor of shape (batch_size, num_kv_heads, seq_len, head_dim)
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            Attention output tensor
        """
        pass

    @abstractmethod
    def store_kv_cache(
        self,
        key: Tensor,
        value: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        slot_mapping: Tensor,
    ) -> None:
        """Store key-value pairs to cache.

        Args:
            key: Key tensor
            value: Value tensor
            k_cache: Key cache tensor
            v_cache: Value cache tensor
            slot_mapping: Slot mapping tensor
        """
        pass


class StandardAttentionBackend(AttentionBackend):
    """Standard PyTorch attention implementation.

    This is the fallback implementation that works on all devices
    but may be slower than specialized backends.
    """

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Standard attention computation using PyTorch operations."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        _, num_kv_heads, kv_seq_len, _ = key.shape

        # Handle GQA/MQA by repeating key/value heads
        if num_kv_heads != num_heads:
            repeat_factor = num_heads // num_kv_heads
            key = key.repeat_interleave(repeat_factor, dim=1)
            value = value.repeat_interleave(repeat_factor, dim=1)

        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim**0.5)

        # Apply causal mask if needed
        if is_causal:
            causal_mask = torch.triu(torch.ones(seq_len, kv_seq_len),
                                     diagonal=1).bool()
            if query.device != torch.device('cpu'):
                causal_mask = causal_mask.to(query.device)
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax and attention weights
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention to values
        output = torch.matmul(attention_weights, value)

        return output

    def store_kv_cache(
        self,
        key: Tensor,
        value: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        slot_mapping: Tensor,
    ) -> None:
        """Store KV cache using PyTorch operations."""
        batch_size, num_heads, head_dim = key.shape
        hidden_size = num_heads * head_dim

        # Validate inputs
        if slot_mapping.numel() != batch_size:
            raise ValueError(
                f'slot_mapping size {slot_mapping.numel()} != batch_size {batch_size}'
            )

        # Store each sequence's KV to cache
        for i in range(batch_size):
            slot = slot_mapping[i].item()
            if slot >= 0:  # Valid slot
                # Flatten and store
                key_flat = key[i].view(hidden_size)
                value_flat = value[i].view(hidden_size)

                cache_start = slot * hidden_size
                cache_end = cache_start + hidden_size

                k_cache.view(-1)[cache_start:cache_end] = key_flat
                v_cache.view(-1)[cache_start:cache_end] = value_flat


class FlashAttentionBackend(AttentionBackend):
    """Flash Attention backend for CUDA devices.

    This implementation uses flash-attn for optimal performance on CUDA.
    """

    def __init__(self):
        """Initialize Flash Attention backend."""
        self._flash_attn_available = self._check_flash_attn()
        if not self._flash_attn_available:
            logger.warning(
                'Flash Attention not available, falling back to standard attention'
            )

    def _check_flash_attn(self) -> bool:
        """Check if Flash Attention is available."""
        try:
            from flash_attn import flash_attn_func
            self.flash_attn_func = flash_attn_func
            return True
        except ImportError:
            return False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass using Flash Attention."""
        if not self._flash_attn_available:
            # Fallback to standard attention
            return StandardAttentionBackend().forward(query, key, value,
                                                      attention_mask,
                                                      is_causal)

        try:
            # Flash Attention expects (batch, seq_len, num_heads, head_dim)
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)

            # Use Flash Attention
            output = self.flash_attn_func(q,
                                          k,
                                          v,
                                          causal=is_causal,
                                          attention_mask=attention_mask)

            # Transpose back to (batch, num_heads, seq_len, head_dim)
            return output.transpose(1, 2)
        except Exception as e:
            logger.error(
                f'Flash Attention failed: {e}, falling back to standard attention'
            )
            return StandardAttentionBackend().forward(query, key, value,
                                                      attention_mask,
                                                      is_causal)

    def store_kv_cache(
        self,
        key: Tensor,
        value: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        slot_mapping: Tensor,
    ) -> None:
        """Store KV cache using Flash Attention utilities."""
        # For now, use standard implementation
        StandardAttentionBackend().store_kv_cache(key, value, k_cache, v_cache,
                                                  slot_mapping)


class NPUAttentionBackend(AttentionBackend):
    """NPU-optimized attention backend.

    This implementation uses NPU-specific optimizations for Ascend hardware.
    """

    def __init__(self):
        """Initialize NPU attention backend."""
        self._npu_available = self._check_npu_availability()
        if not self._npu_available:
            logger.warning(
                'NPU not available, falling back to standard attention')

    def _check_npu_availability(self) -> bool:
        """Check if NPU is available."""
        try:
            from transformers.utils import is_torch_npu_available
            return is_torch_npu_available()
        except ImportError:
            return False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass using NPU optimizations."""
        if not self._npu_available:
            return StandardAttentionBackend().forward(query, key, value,
                                                      attention_mask,
                                                      is_causal)

        try:
            import torch_npu

            # Use NPU-specific attention if available
            if hasattr(torch_npu, 'npu_fused_infer_attention_score'):
                # Use unified inference API
                return torch_npu.npu_fused_infer_attention_score(
                    query,
                    key,
                    value,
                    attention_mask=attention_mask,
                    is_causal=is_causal)
            else:
                # Fallback to standard attention on NPU
                return StandardAttentionBackend().forward(
                    query, key, value, attention_mask, is_causal)
        except Exception as e:
            logger.error(
                f'NPU attention failed: {e}, falling back to standard attention'
            )
            return StandardAttentionBackend().forward(query, key, value,
                                                      attention_mask,
                                                      is_causal)

    def store_kv_cache(
        self,
        key: Tensor,
        value: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        slot_mapping: Tensor,
    ) -> None:
        """Store KV cache using NPU optimizations."""
        # For now, use standard implementation
        StandardAttentionBackend().store_kv_cache(key, value, k_cache, v_cache,
                                                  slot_mapping)


def get_attention_backend(
        device: Optional[torch.device] = None) -> AttentionBackend:
    """Get the appropriate attention backend for the given device.

    Args:
        device: Target device. If None, uses current device.

    Returns:
        Appropriate attention backend instance
    """
    if device is None:
        device = torch.cuda.current_device() if torch.cuda.is_available(
        ) else torch.device('cpu')

    # Check device type and return appropriate backend
    if device.type == 'cuda':
        return FlashAttentionBackend()
    elif device.type == 'npu' or (hasattr(torch, 'npu')
                                  and torch.npu.is_available()):
        return NPUAttentionBackend()
    else:
        return StandardAttentionBackend()


class Attention(nn.Module):
    """Unified attention module with automatic backend selection.

    This module provides a clean interface for attention computation
    with automatic backend selection based on the available hardware.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 4096,
        device: Optional[torch.device] = None,
    ):
        """Initialize attention module.

        Args:
            hidden_size: Hidden dimension of the model
            num_heads: Number of attention heads
            num_kv_heads: Number of key/value heads for GQA/MQA
            head_dim: Dimension of each attention head
            max_position_embeddings: Maximum sequence length
            device: Target device for computations
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.max_position_embeddings = max_position_embeddings

        if self.head_dim * self.num_heads != hidden_size:
            raise ValueError(
                f'head_dim * num_heads ({self.head_dim * self.num_heads}) '
                f'!= hidden_size ({hidden_size})')

        # Initialize attention backend
        self.backend = get_attention_backend(device)

        # Initialize projections
        self.q_proj = nn.Linear(hidden_size,
                                num_heads * self.head_dim,
                                bias=False)
        self.k_proj = nn.Linear(hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(hidden_size,
                                self.num_kv_heads * self.head_dim,
                                bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim,
                                hidden_size,
                                bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor, Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]],
               Optional[List[Tensor]]]:
        """Forward pass through attention module.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional past key-value cache
            output_attentions: Whether to output attention weights
            use_cache: Whether to use key-value cache

        Returns:
            Tuple of (output, present_key_value, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Reshape for attention computation
        query_states = query_states.view(batch_size, seq_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_kv_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len,
                                         self.num_kv_heads,
                                         self.head_dim).transpose(1, 2)

        # Handle past key-value cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = (key_states, value_states) if use_cache else None

        # Compute attention
        attn_output = self.backend.forward(
            query_states,
            key_states,
            value_states,
            attention_mask=attention_mask,
            is_causal=True,
        )

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        # Attention weights not supported for backends other than standard
        attn_weights = None
        if output_attentions and isinstance(self.backend,
                                            StandardAttentionBackend):
            # For debugging purposes, compute attention weights
            # This is expensive and should not be used in production
            scores = torch.matmul(query_states, key_states.transpose(
                -2, -1)) / (self.head_dim**0.5)
            attn_weights = torch.softmax(scores, dim=-1)

        return attn_output, present_key_value, attn_weights
