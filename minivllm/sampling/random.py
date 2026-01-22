"""
---
title: Random Sampling
summary: A PyTorch implementation of random sampling from language models.
---

# Random Sampling

Here we sample randomly from the probability distribution defined by the logits.
This is equivalent to temperature sampling with temperature = 1.0.


"""

from torch import Tensor
from torch.distributions import Categorical

from minivllm.sampling.base import Sampler


class RandomSampler(Sampler):
    """
    ## Random Sampler
    """

    def __call__(self, logits: Tensor) -> Tensor:
        """
        Sample randomly from the probability distribution defined by logits

        :param logits: are the logits of the distribution of shape `[..., n_tokens]`
        :return: sampled token indices of shape `[...]`
        """
        # Create a categorical distribution from the logits
        dist = Categorical(logits=logits)

        # Sample from the distribution
        return dist.sample()
