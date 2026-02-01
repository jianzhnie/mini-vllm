"""
Stateless functional implementations of sampling strategies.
These functions operate on logits and return modified logits or samples.
"""

from typing import Optional

import torch
from torch import Tensor

# Constants
MIN_TEMPERATURE = 1e-8
MIN_PROB = 1e-10

# Attempt to use torch.compile for kernel fusion on supported devices
try:
    import torch._dynamo

    # Use 'reduce-overhead' or 'max-autotune' if stable, but 'default' is safer for now
    # We only compile if not on MPS (which has issues with compile sometimes) or if explicitly desired
    compile_ops = True
except ImportError:
    compile_ops = False


def compiled(fn):
    if compile_ops:
        return torch.compile(fn)
    return fn


def apply_temperature(logits: Tensor, temperature: Tensor) -> Tensor:
    """
    Apply temperature scaling to logits.

    Args:
        logits: [batch_size, vocab_size]
        temperature: [batch_size] or scalar

    Returns:
        scaled logits
    """
    if isinstance(temperature, float):
        if temperature == 1.0:
            return logits
        temp = torch.tensor(temperature,
                            device=logits.device,
                            dtype=logits.dtype)
    else:
        temp = temperature.to(logits.dtype)

    # Avoid division by zero
    temp = temp.clamp_min(MIN_TEMPERATURE)

    if temp.dim() == 1:
        temp = temp.unsqueeze(1)

    return logits / temp


def apply_top_k(logits: Tensor,
                top_k: Tensor,
                filter_value: float = -float('inf')) -> Tensor:
    """
    Apply Top-K filtering.

    Args:
        logits: [batch_size, vocab_size]
        top_k: [batch_size] or scalar integer.
               If 0 or >= vocab_size, no filtering is applied for that sample.
        filter_value: Value to set for filtered tokens.
    """
    # Handle scalar k
    if isinstance(top_k, int):
        if top_k <= 0 or top_k >= logits.size(-1):
            return logits
        top_k = torch.full((logits.size(0), ),
                           top_k,
                           device=logits.device,
                           dtype=torch.long)

    # Check if any k is active
    active_mask = (top_k > 0) & (top_k < logits.size(-1))
    if not active_mask.any():
        return logits

    logits = logits.clone()

    # Vectorized implementation
    # We use the maximum k across the batch to vectorize the topk call
    max_k = top_k[active_mask].max().item()

    # Get top-k indices for all sequences
    _, top_k_indices = torch.topk(logits, max_k, dim=-1)

    # Create a mask for valid k positions [batch, max_k]
    k_range = torch.arange(max_k, device=logits.device).unsqueeze(0)
    # keep_top_k_mask[i, j] is True if j < top_k[i]
    keep_top_k_mask = k_range < top_k.unsqueeze(1)

    # Create the final mask for logits [batch, vocab]
    mask = torch.zeros_like(logits, dtype=torch.bool)

    # Scatter the keep mask to the original positions
    mask.scatter_(1, top_k_indices, keep_top_k_mask)

    # For sequences where top-k is disabled, keep all tokens
    mask[~active_mask] = True

    # Apply filter
    logits.masked_fill_(~mask, filter_value)

    return logits


@compiled
def apply_top_p(logits: Tensor,
                top_p: Tensor,
                filter_value: float = -float('inf')) -> Tensor:
    """
    Apply Top-P (Nucleus) filtering.

    Args:
        logits: [batch_size, vocab_size]
        top_p: [batch_size] or scalar float.
    """
    # Handle scalar p
    if isinstance(top_p, float):
        if top_p >= 1.0 or top_p <= 0.0:
            return logits
        top_p = torch.full((logits.size(0), ),
                           top_p,
                           device=logits.device,
                           dtype=logits.dtype)

    # Check if any p is active (< 1.0)
    if not (top_p < 1.0).any():
        return logits

    logits = logits.clone()

    # Sort probabilities (descending)
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p.unsqueeze(1)

    # Shift to right to keep first token above threshold
    sorted_indices_to_remove[...,
                             1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Scatter back to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    logits.masked_fill_(indices_to_remove, filter_value)

    return logits


def apply_min_p(logits: Tensor,
                min_p: Tensor,
                filter_value: float = -float('inf')) -> Tensor:
    """
    Apply Min-P filtering.

    Args:
        logits: [batch_size, vocab_size]
        min_p: [batch_size] or scalar float.
    """
    # Handle scalar min_p
    if isinstance(min_p, float):
        if min_p <= 0.0:
            return logits
        min_p = torch.full((logits.size(0), ),
                           min_p,
                           device=logits.device,
                           dtype=logits.dtype)

    if not (min_p > 0.0).any():
        return logits

    logits = logits.clone()
    probs = torch.softmax(logits, dim=-1)
    top_prob, _ = torch.max(probs, dim=-1, keepdim=True)
    scaled_min_p = min_p.unsqueeze(1) * top_prob
    tokens_to_remove = probs < scaled_min_p
    logits.masked_fill_(tokens_to_remove, filter_value)

    return logits


def apply_typical_filtering(logits: Tensor,
                            tau: float = 1.0,
                            filter_value: float = -float('inf')) -> Tensor:
    """
    Apply Typical Sampling filtering.

    Args:
        logits: [batch_size, vocab_size]
        tau: float, typical threshold (default 1.0)
    """
    if tau == 1.0:  # Logic in original code says if tau > 0. But default is 1.0.
        # If typical sampling is disabled, usually we don't call this.
        # But here we assume tau controls the strictness.
        # Wait, if tau is used, we MUST calculate entropy.
        pass

    logits = logits.clone()
    probs = torch.softmax(logits, dim=-1)

    # Compute entropy: H = -sum(p * log(p))
    log_probs = torch.log(probs + 1e-10)
    entropy = -torch.sum(probs * log_probs, dim=-1, keepdim=True)

    # Compute KL divergence term (shifted information)
    # condition: | -log(p) - H | < tau * H
    # equivalent to: | log(p) + H | < tau * H
    shifted_info = log_probs + entropy

    # Filter
    entropy_threshold = tau * entropy
    mask = shifted_info.abs() <= entropy_threshold

    # Ensure at least one token is selected per batch item
    any_selected = mask.any(dim=-1)
    if not any_selected.all():
        # Find indices where nothing is selected
        no_selection_indices = torch.where(~any_selected)[0]

        # For these indices, select the token with min |shifted_info|
        diff = shifted_info[no_selection_indices].abs()
        min_indices = torch.argmin(diff, dim=-1)

        # Update mask
        mask[no_selection_indices, min_indices] = True

    logits.masked_fill_(~mask, filter_value)

    return logits


def apply_top_token_restriction(
        logits: Tensor,
        avoid_top_k: int,
        filter_value: float = -float('inf'),
) -> Tensor:
    """
    Apply Top Token Restriction (avoid top k).
    """
    if avoid_top_k <= 0:
        return logits

    logits = logits.clone()
    _, top_k_indices = torch.topk(logits,
                                  min(avoid_top_k, logits.size(-1)),
                                  dim=-1)
    logits.scatter_(-1, top_k_indices, filter_value)
    return logits


def apply_repetition_penalty(logits: Tensor, prev_tokens: Tensor,
                             penalty: float) -> Tensor:
    """
    Apply repetition penalty.
    """
    if penalty == 1.0 or prev_tokens is None:
        return logits

    logits = logits.clone()
    batch_size, vocab_size = logits.shape

    # Ensure prev_tokens is on the same device
    if prev_tokens.device != logits.device:
        prev_tokens = prev_tokens.to(logits.device)

    # Handle variable length sequences or single sequence
    if prev_tokens.dim() == 1 and batch_size > 1:
        prev_tokens = prev_tokens.unsqueeze(0).expand(batch_size, -1)
    elif prev_tokens.dim() == 1:
        prev_tokens = prev_tokens.unsqueeze(0)

    # Ensure LongTensor for indexing
    if prev_tokens.dtype != torch.long:
        prev_tokens = prev_tokens.long()

    # Create a mask of present tokens
    presence_mask = torch.zeros((batch_size, vocab_size),
                                dtype=torch.bool,
                                device=logits.device)

    valid_tokens_mask = (prev_tokens >= 0) & (prev_tokens < vocab_size)

    if valid_tokens_mask.all():
        presence_mask.scatter_(1, prev_tokens, True)
    else:
        batch_indices = (torch.arange(
            batch_size,
            device=logits.device).unsqueeze(1).expand_as(prev_tokens))
        valid_batch = batch_indices[valid_tokens_mask]
        valid_toks = prev_tokens[valid_tokens_mask]
        presence_mask[valid_batch, valid_toks] = True

    masked_logits = logits[presence_mask]
    if masked_logits.numel() > 0:
        logits[presence_mask] = torch.where(
            masked_logits > 0,
            masked_logits / penalty,
            masked_logits * penalty,
        )

    return logits


def apply_frequency_penalty(logits: Tensor, sequence: Tensor,
                            alpha: float) -> Tensor:
    """Apply frequency penalty."""
    if alpha == 0.0 or sequence is None:
        return logits

    logits = logits.clone()
    batch_size, vocab_size = logits.shape

    if sequence.device != logits.device:
        sequence = sequence.to(logits.device)

    if sequence.dim() == 1 and batch_size > 1:
        sequence = sequence.unsqueeze(0).expand(batch_size, -1)
    elif sequence.dim() == 1:
        sequence = sequence.unsqueeze(0)

    if sequence.dtype != torch.long:
        sequence = sequence.long()

    # Count frequencies
    counts = torch.zeros((batch_size, vocab_size),
                         dtype=logits.dtype,
                         device=logits.device)

    valid_mask = (sequence >= 0) & (sequence < vocab_size)

    if valid_mask.all():
        src = torch.ones_like(sequence, dtype=logits.dtype)
        counts.scatter_add_(1, sequence, src)
    else:
        batch_indices = (torch.arange(
            batch_size, device=logits.device).unsqueeze(1).expand_as(sequence))
        valid_batch = batch_indices[valid_mask]
        valid_toks = sequence[valid_mask]
        values = torch.ones_like(valid_toks, dtype=logits.dtype)
        counts.index_put_((valid_batch, valid_toks), values, accumulate=True)

    mask = counts > 0
    if mask.any():
        logits[mask] -= counts[mask] * alpha

    return logits


@compiled
def apply_presence_penalty(logits: Tensor, sequence: Tensor,
                           penalty: float) -> Tensor:
    """Apply presence penalty."""
    if penalty == 0.0 or sequence is None:
        return logits

    logits = logits.clone()
    batch_size, vocab_size = logits.shape

    if sequence.device != logits.device:
        sequence = sequence.to(logits.device)

    if sequence.dim() == 1 and batch_size > 1:
        sequence = sequence.unsqueeze(0).expand(batch_size, -1)
    elif sequence.dim() == 1:
        sequence = sequence.unsqueeze(0)

    if sequence.dtype != torch.long:
        sequence = sequence.long()

    presence_mask = torch.zeros((batch_size, vocab_size),
                                dtype=torch.bool,
                                device=logits.device)

    valid_mask = (sequence >= 0) & (sequence < vocab_size)

    if valid_mask.all():
        presence_mask.scatter_(1, sequence, True)
    else:
        batch_indices = (torch.arange(
            batch_size, device=logits.device).unsqueeze(1).expand_as(sequence))
        valid_batch = batch_indices[valid_mask]
        valid_toks = sequence[valid_mask]
        presence_mask[valid_batch, valid_toks] = True

    if presence_mask.any():
        logits[presence_mask] -= penalty

    return logits


def sample_from_logits(logits: Tensor,
                       generator: Optional[torch.Generator] = None) -> Tensor:
    """
    Sample one token from logits using multinomial sampling.
    """
    probs = torch.softmax(logits, dim=-1)

    # Handle NaNs (e.g., if logits were all -inf)
    if torch.isnan(probs).any():
        probs = torch.nan_to_num(probs, nan=0.0)

    sum_probs = probs.sum(dim=-1, keepdim=True)

    # Check for zero sum (all filtered or all NaN originally)
    zero_sum_mask = (sum_probs == 0.0).squeeze(-1)
    if zero_sum_mask.any():
        # Fallback to uniform distribution for these cases
        probs[zero_sum_mask] = 1.0 / probs.size(-1)
        sum_probs[zero_sum_mask] = 1.0

    probs = probs / sum_probs

    num_samples = 1
    sample_tokens = torch.multinomial(probs,
                                      num_samples,
                                      replacement=True,
                                      generator=generator).squeeze(1)
    return sample_tokens
