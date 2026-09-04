"""Tests for minivllm.sampling.mirostat (Mirostat v1 and v2 samplers).

Both samplers run on plain CPU tensors, so the module is exercised here.

Note on supported inputs: the samplers' state updates call ``.item()`` on a
per-row error tensor, so they currently only support a **single** logits row
``(1, vocab)`` (this matches the only usage the standalone module has ever had).
These tests therefore cover single-row behavior, the temperature / ``mu`` update
recurrence and its clamps, ``reset()`` semantics, and determinism under a fixed
torch seed. Batched and degenerate (single-spike) inputs are intentionally not
asserted on because the module does not yet handle them.
"""

from __future__ import annotations

import pytest
import torch

from minivllm.sampling.mirostat import MirostatSampler, MirostatV2Sampler


def _rand_logits(vocab: int = 100, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(1, vocab)


# --- MirostatSampler (v1) --------------------------------------------------


class TestMirostatSampler:
    def test_forward_output_shape_single_row(self):
        sampler = MirostatSampler()
        token = sampler(_rand_logits())
        assert token.shape == (1,)
        assert 0 <= token.item() < 100

    def test_forward_repeated_calls_stay_valid(self):
        sampler = MirostatSampler(target_perplexity=3.0)
        for _ in range(10):
            token = sampler(_rand_logits(vocab=100, seed=1))
            assert token.shape == (1,)
            assert 0 <= token.item() < 100

    def test_temperature_stays_within_bounds(self):
        max_temp = 2.0
        sampler = MirostatSampler(target_perplexity=3.0, max_temperature=max_temp)
        for _ in range(10):
            sampler(_rand_logits())
            assert 0.1 <= sampler.temperature <= max_temp

    def test_flat_logits_push_temperature_above_initial(self):
        # A flat distribution yields a high surprise, so the update moves the
        # temperature away from its 1.0 start (clamped at max_temperature).
        torch.manual_seed(0)
        sampler = MirostatSampler(
            target_perplexity=3.0, learning_rate=1.0, max_temperature=5.0
        )
        sampler(torch.full((1, 100), 0.0))
        assert sampler.temperature != 1.0

    def test_reset_restores_state(self):
        sampler = MirostatSampler()
        sampler(_rand_logits())
        sampler.previous_surprise = torch.tensor(1.0)
        sampler.reset()
        assert sampler.temperature == 1.0
        assert sampler.previous_surprise is None


# --- MirostatV2Sampler (v2) ------------------------------------------------


class TestMirostatV2Sampler:
    def test_forward_output_shape_single_row(self):
        sampler = MirostatV2Sampler()
        token = sampler(_rand_logits())
        assert token.shape == (1,)
        assert 0 <= token.item() < 100

    def test_forward_repeated_calls_stay_valid(self):
        sampler = MirostatV2Sampler(target_perplexity=3.0)
        for _ in range(10):
            token = sampler(_rand_logits(vocab=100, seed=1))
            assert token.shape == (1,)
            assert 0 <= token.item() < 100

    def test_mu_within_bounds(self):
        sampler = MirostatV2Sampler()
        for _ in range(10):
            sampler(_rand_logits())
            assert 1.0 <= sampler.mu <= 100.0

    def test_temperature_equals_tau_over_mu(self):
        sampler = MirostatV2Sampler(target_perplexity=3.0, tau=5.0)
        sampler(_rand_logits())
        assert abs(sampler.temperature - sampler.tau / sampler.mu) < 1e-6

    def test_reset_restores_state(self):
        sampler = MirostatV2Sampler(target_perplexity=3.0)
        sampler(_rand_logits())
        sampler.reset()
        assert sampler.temperature == 1.0
        assert sampler.mu == sampler.target_perplexity


# --- determinism of the update recurrence ---------------------------------


def _trajectory(sampler_cls, **kwargs):
    sampler = sampler_cls(**kwargs)
    states = []
    for _ in range(6):
        sampler(_rand_logits(vocab=64, seed=123))
        states.append((round(sampler.temperature, 6), getattr(sampler, "mu", None)))
    return states


def test_v1_temperature_trajectory_is_deterministic():
    a = _trajectory(MirostatSampler, target_perplexity=3.0)
    b = _trajectory(MirostatSampler, target_perplexity=3.0)
    assert a == b


def test_v2_state_trajectory_is_deterministic():
    a = _trajectory(MirostatV2Sampler, target_perplexity=3.0)
    b = _trajectory(MirostatV2Sampler, target_perplexity=3.0)
    assert a == b


if __name__ == "__main__":
    pytest.main([__file__])
