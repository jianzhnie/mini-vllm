import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from minivllm.sampling.base import Sampler


class TopKTopPSampler(Sampler):
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

        dist = Categorical(logits=logits)
        return dist.sample()
