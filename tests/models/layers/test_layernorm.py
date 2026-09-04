"""Tests for minivllm.models.layers.layernorm.RMSNorm."""

from __future__ import annotations

import torch

from minivllm.models.layers.layernorm import RMSNorm


def _manual_rms(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * weight


def test_rms_forward_matches_manual():
    norm = RMSNorm(16)
    x = torch.randn(4, 16, dtype=torch.float32)
    out = norm.rms_forward(x)
    ref = _manual_rms(x, norm.weight, norm.eps)
    assert torch.allclose(out, ref, atol=1e-5)


def test_forward_without_residual_returns_none_residual():
    norm = RMSNorm(8)
    out, residual = norm(torch.randn(2, 8, dtype=torch.float32))
    assert residual is None
    assert out.shape == (2, 8)


def test_add_rms_forward_updates_residual():
    norm = RMSNorm(16)
    x = torch.randn(3, 16, dtype=torch.float32)
    res = torch.randn(3, 16, dtype=torch.float32)
    out, new_res = norm.add_rms_forward(x, res)
    assert torch.allclose(new_res, x + res, atol=1e-6)
    ref = _manual_rms((x + res), norm.weight, norm.eps)
    assert torch.allclose(out, ref, atol=1e-5)


def test_dtype_preserved_fp16():
    norm = RMSNorm(16)
    x = torch.randn(2, 16).half()
    assert norm.rms_forward(x).dtype == torch.float16
