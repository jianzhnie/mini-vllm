from typing import Optional, Tensor

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


class TopKTopPFilter:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).

        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

        filter_value: logits of filtered tokens are set to this value (default -inf).
    """

    def __init__(
            self,
            top_k: int = 0,
            top_p: float = 0.0,
            filter_value: float = -float('Inf'),
    ) -> None:
        self.top_k = top_k
        self.top_p = top_p
        self.filter_value = filter_value

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Apply top-k and top-p filtering to input logits.

        Args:
            logits: (batch_size, vocab_size) logits tensor.

        Returns:
            Filtered logits tensor.
        """
        assert (
            logits.dim() == 1
        )  # batch size 1 for now - could be updated for more but the code would be less clear
        logits = logits.clone()  # avoid in-place modification
        vocab_size = logits.size(-1)

        # Top-K filtering
        if self.top_k > 0:
            top_k = min(self.top_k, vocab_size)
            threshold = torch.topk(logits, top_k).values[-1]
            logits[logits < threshold] = self.filter_value

        # Top-P filtering
        if 0.0 < self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1]
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = self.filter_value

        return logits


def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    eos_token_id: Optional[int] = None,
    device: Optional[str] = None,
) -> str:
    """
    Generate text using a Hugging Face model with custom sampling.

    Args:
        model: Loaded transformers model.
        tokenizer: Corresponding tokenizer.
        prompt: Initial input text.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Controls sampling diversity, higher values are more random.
        top_k: Top-k sampling limit.
        top_p: Top-p (nucleus) sampling limit.
        eos_token_id: End-of-sequence token ID, generation stops if encountered.
        device: Device to use (e.g., 'cuda' or 'cpu'). Automatically detected if None.

    Returns:
        Generated text string.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids.clone()

    # Create filter instance
    sample_filter = TopKTopPFilter(top_k=top_k, top_p=top_p)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated)
            next_token_logits = outputs.logits[0, -1, :] / temperature

            filtered_logits = sample_filter(next_token_logits)
            probs = F.softmax(filtered_logits, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)
