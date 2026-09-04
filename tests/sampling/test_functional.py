"""Tests for minivllm.sampling.functional (stateless sampling ops).

These operate on plain CPU tensors; ``@compiled`` is a no-op here because
torch.compile is disabled on NPU/CPU-only builds.
"""

from __future__ import annotations

import pytest
import torch

from minivllm.sampling import functional as F

NEG_INF = -float("inf")


def _logits(vals):
    return torch.tensor(vals, dtype=torch.float32)


# --- temperature -----------------------------------------------------------


def test_temperature_identity_at_one():
    x = _logits([[1.0, 2.0, 3.0]])
    assert torch.equal(F.apply_temperature(x, 1.0), x)


def test_temperature_scales_scalar():
    x = _logits([[1.0, 2.0, 3.0]])
    assert torch.allclose(F.apply_temperature(x, 0.5), x / 0.5)


def test_temperature_per_row_tensor():
    x = _logits([[1.0, 2.0], [1.0, 2.0]])
    out = F.apply_temperature(x, torch.tensor([1.0, 2.0]))
    assert torch.allclose(out[0], x[0])
    assert torch.allclose(out[1], x[1] / 2.0)


def test_temperature_rejects_1d():
    with pytest.raises(ValueError):
        F.apply_temperature(_logits([1.0, 2.0]), 0.7)


# --- top_k -----------------------------------------------------------------


def test_top_k_zero_is_noop():
    x = _logits([[1.0, 2.0, 3.0, 4.0]])
    assert torch.equal(F.apply_top_k(x, 0), x)


def test_top_k_ge_vocab_is_noop():
    x = _logits([[1.0, 2.0, 3.0]])
    assert torch.equal(F.apply_top_k(x, 10), x)


def test_top_k_masks_all_but_topk():
    x = _logits([[1.0, 5.0, 3.0, 2.0, 4.0]])
    out = F.apply_top_k(x, 2)
    assert out[0, 1] == 5.0 and out[0, 4] == 4.0  # the two largest kept
    assert out[0, 0] == NEG_INF and out[0, 2] == NEG_INF and out[0, 3] == NEG_INF


# --- top_p -----------------------------------------------------------------


def test_top_p_ge_one_is_noop():
    x = _logits([[1.0, 2.0, 3.0]])
    assert torch.equal(F.apply_top_p(x, 1.0), x)


def test_top_p_keeps_dominant_token():
    x = _logits([[10.0, 0.0, 0.0, 0.0]])
    out = F.apply_top_p(x, 0.9)
    assert out[0, 0] == 10.0  # dominant token always kept
    assert (out[0] != NEG_INF).sum() >= 1


# --- min_p -----------------------------------------------------------------


def test_min_p_zero_is_noop():
    x = _logits([[1.0, 2.0, 3.0]])
    assert torch.equal(F.apply_min_p(x, 0.0), x)


def test_min_p_masks_low_probability():
    x = _logits([[10.0, 0.0, 0.0]])
    out = F.apply_min_p(x, 0.5)
    assert out[0, 0] == 10.0
    assert out[0, 1] == NEG_INF and out[0, 2] == NEG_INF


# --- penalties -------------------------------------------------------------


def test_repetition_penalty_one_is_noop():
    x = _logits([[1.0, 2.0]])
    assert torch.equal(F.apply_repetition_penalty(x, torch.tensor([0]), 1.0), x)


def test_repetition_penalty_divides_positive():
    x = _logits([[2.0, 1.0]])
    out = F.apply_repetition_penalty(x, torch.tensor([0]), 1.5)
    assert out[0, 0] == pytest.approx(2.0 / 1.5)
    assert out[0, 1] == 1.0


def test_frequency_penalty_subtracts_counts():
    x = _logits([[1.0, 1.0]])
    out = F.apply_frequency_penalty(x, torch.tensor([0, 0]), 0.5)
    assert out[0, 0] == pytest.approx(1.0 - 2 * 0.5)
    assert out[0, 1] == 1.0


def test_presence_penalty_zero_is_noop():
    x = _logits([[1.0, 2.0]])
    assert torch.equal(F.apply_presence_penalty(x, torch.tensor([0]), 0.0), x)


def test_presence_penalty_subtracts_once():
    x = _logits([[1.0, 2.0]])
    out = F.apply_presence_penalty(x, torch.tensor([1]), 0.3)
    assert out[0, 1] == pytest.approx(2.0 - 0.3)
    assert out[0, 0] == 1.0


def test_top_token_restriction_zero_is_noop():
    x = _logits([[3.0, 1.0, 2.0]])
    assert torch.equal(F.apply_top_token_restriction(x, 0), x)


def test_top_token_restriction_masks_topk():
    x = _logits([[3.0, 1.0, 2.0]])
    out = F.apply_top_token_restriction(x, 1)
    assert out[0, 0] == NEG_INF  # the single top token is avoided
    assert out[0, 1] == 1.0 and out[0, 2] == 2.0


# --- typical ---------------------------------------------------------------


def test_typical_disabled_at_tau_one():
    x = _logits([[1.0, 2.0, 3.0]])
    assert torch.equal(F.apply_typical_filtering(x, tau=1.0), x)


def test_typical_keeps_at_least_one_token():
    x = _logits([[1.0, 2.0, 3.0, 4.0]])
    out = F.apply_typical_filtering(x, tau=0.5)
    assert (out[0] != NEG_INF).sum() >= 1


# --- sampling --------------------------------------------------------------


def test_sample_from_logits_deterministic_with_generator():
    x = _logits([[0.0, 0.0, 100.0, 0.0]])
    s1 = F.sample_from_logits(x, generator=torch.Generator().manual_seed(0))
    s2 = F.sample_from_logits(x, generator=torch.Generator().manual_seed(0))
    assert torch.equal(s1, s2)
    assert s1.item() == 2  # near-certain pick


def test_sample_from_logits_zero_sum_fallback():
    x = torch.full((1, 4), NEG_INF)
    out = F.sample_from_logits(x)
    assert 0 <= out.item() < 4
