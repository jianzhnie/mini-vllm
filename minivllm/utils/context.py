"""Context management for inference parameters.

This module provides a global context object that stores inference-related
parameters (such as KV cache mappings and sequence information) that need
to be accessible across different parts of the inference pipeline.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Context:
    """Context information for model inference.

    This dataclass stores inference parameters that are set up during
    the prefill or decode phase and used by model layers during forward pass.

    Attributes:
        is_prefill: Whether currently in prefill (True) or decode (False) phase.
        max_seqlen_q: Maximum query sequence length in current batch.
        max_seqlen_k: Maximum key sequence length in current batch.
        cum_seqlens_q: Cumulative sequence lengths for queries in prefill.
            Shape: [batch_size + 1]. None for decode phase.
        cum_seqlens_k: Cumulative sequence lengths for keys in prefill.
            Shape: [batch_size + 1]. None for decode phase.
        slot_mapping: Mapping from token positions to KV cache slots.
            Shape: [num_tokens] for prefill or
            [batch_size] for decode.
        context_lens: Length of context (prompt + cached) for each sequence.
            Shape: [batch_size]. None for prefill phase.
        block_tables: Block table for each sequence in batch.
            Shape: [batch_size, max_blocks]. Maps to physical KV cache blocks.
    """

    is_prefill: bool = False
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    cum_seqlens_q: Optional[torch.Tensor] = None
    cum_seqlens_k: Optional[torch.Tensor] = None
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None


# Global context instance
_CONTEXT: Context = Context()


def get_context() -> Context:
    """Get the current global inference context.

    Returns:
        The current Context object containing inference parameters.
    """
    return _CONTEXT


def set_context(
    is_prefill: bool,
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    cum_seqlens_q: Optional[torch.Tensor] = None,
    cum_seqlens_k: Optional[torch.Tensor] = None,
    slot_mapping: Optional[torch.Tensor] = None,
    context_lens: Optional[torch.Tensor] = None,
    block_tables: Optional[torch.Tensor] = None,
) -> None:
    """Set the global inference context.

    Updates the global context with new values. This is typically called
    before model inference to set up the attention parameters.

    Args:
        is_prefill: Whether in prefill or decode phase.
        max_seqlen_q: Maximum query sequence length.
        max_seqlen_k: Maximum key sequence length.
        cum_seqlens_q: Cumulative query sequence lengths.
        cum_seqlens_k: Cumulative key sequence lengths.
        slot_mapping: KV cache slot mapping.
        context_lens: Context lengths for each sequence.
        block_tables: Block tables for KV cache.

    Note:
        This function modifies the global context state and should only
        be called from the main inference thread.
    """
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        max_seqlen_q,
        max_seqlen_k,
        cum_seqlens_q,
        cum_seqlens_k,
        slot_mapping,
        context_lens,
        block_tables,
    )


def reset_context() -> None:
    """Reset the global inference context to default state.

    Clears all context parameters, typically done after each inference step
    to avoid stale data affecting the next step.

    Note:
        This function modifies the global context state and should only
        be called from the main inference thread.
    """
    global _CONTEXT
    _CONTEXT = Context()
