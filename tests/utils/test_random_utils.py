"""Tests for minivllm.utils.random_utils.set_random_seed."""

from __future__ import annotations

import torch

from minivllm.utils.random_utils import set_random_seed


def test_same_seed_reproduces_torch_draws():
    set_random_seed(1234)
    a = torch.rand(8)
    set_random_seed(1234)
    b = torch.rand(8)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    set_random_seed(1)
    a = torch.rand(16)
    set_random_seed(2)
    b = torch.rand(16)
    assert not torch.equal(a, b)
