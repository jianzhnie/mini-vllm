"""
---
title: Greedy Sampling
summary: A PyTorch implementation of greedy sampling from language models.
---

# Greedy Sampling

Here we sample the most likely token from the distribution of logits.


"""

from typing import Tensor

from minivllm.sampling.base import Sampler


class GreedySampler(Sampler):
    """
    ## Greedy Sampler
    """

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Sample the most likely token from the distribution of logits

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :return: sampled token indices of shape `[...]`
        """
        return logits.argmax(dim=-1)
