"""
---
title: Sampling Techniques for Language Models
summary: >
 A set of PyTorch implementations/tutorials of sampling techniques for language models.
---

# Sampling Techniques for Language Models

* [Greedy Sampling]
* [Temperature Sampling]
* [Top-k Sampling]
* [Nucleus Sampling]
* [Random Sampling]


"""

from typing import Protocol, Tensor


class Sampler(Protocol):
    """
    ### Sampler base class
    """

    def __call__(self, logits: Tensor) -> Tensor:
        """
        ### Sample from logits

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :return: sampled token indices of shape `[...]`
        """
        raise NotImplementedError()
