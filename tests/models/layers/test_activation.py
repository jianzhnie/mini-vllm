"""Tests for minivllm.models.layers.activation.SiluAndMul."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from minivllm.models.layers.activation import SiluAndMul


def test_shape_halves_last_dim():
    x = torch.randn(2, 4, 8)
    assert SiluAndMul()(x).shape == (2, 4, 4)


def test_value_matches_silu_gate():
    x = torch.randn(3, 6, dtype=torch.float32)
    out = SiluAndMul()(x)
    x1, x2 = x.chunk(2, dim=-1)
    assert torch.allclose(out, F.silu(x1) * x2)


def test_odd_last_dim_raises():
    with pytest.raises(ValueError):
        SiluAndMul()(torch.randn(2, 3, 5))
