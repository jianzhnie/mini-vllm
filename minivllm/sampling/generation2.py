from typing import Optional, Tensor

import torch


def multinomial_sample_one(probs: Tensor,
                           rng: Optional[torch.Generator] = None) -> Tensor:
    """
    Sample from a multinomial distribution using the Gumbel-max trick.

    Args:
        probs: Probability distribution tensor of shape `[..., vocab_size]`
        rng: Random number generator for reproducibility

    Returns:
        Sampled token indices of shape `[..., 1]`
    """
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tensor:
    """
    Convert logits to probabilities with optional temperature scaling and top-k filtering.

    Args:
        logits: Logits tensor of shape `[..., vocab_size]`
        temperature: Temperature for scaling logits (higher = more random)
        top_k: Number of top tokens to keep (None = keep all)

    Returns:
        Probability distribution tensor of shape `[..., vocab_size]`
    """
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float('Inf'), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def generate_next_token(
    model,
    x: Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
) -> Tensor:
    """
    Generate the next token given the current context.

    Args:
        model: Language model to use for generation
        x: Input tokens of shape `[batch_size, seq_len]`
        temperature: Temperature for sampling (higher = more random)
        top_k: Number of top tokens to consider for sampling
        rng: Random number generator for reproducibility

    Returns:
        Next token indices of shape `[batch_size, 1]`
    """
    logits = model(x)  # (B, T, vocab_size)
    probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
    next_token = multinomial_sample_one(probs, rng=rng)
    return next_token


@torch.no_grad()
def generate(
    model,
    input_ids: Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tensor:
    """
    Generate text from a prompt using autoregressive sampling.

    Args:
        model: Language model to use for generation
        input_ids: Input prompt tokens of shape `[seq_len]` or `[batch_size, seq_len]`
        max_new_tokens: Maximum number of tokens to generate
        temperature: Temperature for sampling (higher = more random)
        top_k: Number of top tokens to consider for sampling
        seed: Random seed for reproducibility

    Returns:
        Generated tokens of shape `[batch_size, seq_len + max_new_tokens]`
    """
    # Ensure batch dimension (T,) --> (B, T)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    rng = None
    if seed is not None:
        rng = torch.Generator(input_ids.device).manual_seed(seed)

    generated_tokens = input_ids.clone()

    for _ in range(max_new_tokens):
        next_token = generate_next_token(
            model,
            x=generated_tokens,
            temperature=temperature,
            top_k=top_k,
            rng=rng,
        )

        generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

    return generated_tokens
