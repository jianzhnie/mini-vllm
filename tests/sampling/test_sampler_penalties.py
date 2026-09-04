"""Tests that the penalty / typical-p / seed sampling path is wired end-to-end.

The functional math itself is covered in test_sampler.py; these tests pin the
newly-wired per-row plumbing: penalties now flip the argmax for seen tokens,
typical-p is accepted, seeded generators reproduce, and the executor's
token-history / persistent-generator builders behave correctly.
"""

from __future__ import annotations

import pytest
import torch

from minivllm.engine.inference_executor import InferenceExecutor
from minivllm.engine.sequence import Sequence
from minivllm.sampling.sampler import Sampler
from minivllm.sampling_params import SamplingParams


def _greedy(logits: torch.Tensor, **kwargs) -> torch.Tensor:
    # temperature=0 -> argmax over the (already penalty-adjusted) logits, so
    # the effect of each penalty shows up deterministically.
    return Sampler()(logits, temperatures=torch.tensor([0.0]), **kwargs)


class TestPenaltyWiring:
    def test_repetition_penalty_flips_argmax(self):
        logits = torch.tensor([[10.0, 10.0]])
        # token 0 seen -> penalized to 10/2=5, token 1 wins
        assert _greedy(logits.clone(), prev_tokens=torch.tensor([[0]]),
                      repetition_penalties=torch.tensor([2.0])).item() == 1
        # no penalty -> argmax returns first tie (token 0)
        assert _greedy(logits.clone()).item() == 0

    def test_frequency_penalty_scales_with_count(self):
        logits = torch.tensor([[10.0, 10.0]])
        # token 0 seen twice -> 10 - 2*1 = 8, token 1 wins
        assert _greedy(logits.clone(), prev_tokens=torch.tensor([[0, 0]]),
                      frequency_penalties=torch.tensor([1.0])).item() == 1

    def test_presence_penalty_flips_argmax(self):
        logits = torch.tensor([[10.0, 10.0]])
        # token 0 seen once -> 10 - 5 = 5, token 1 wins
        assert _greedy(logits.clone(), prev_tokens=torch.tensor([[0]]),
                      presence_penalties=torch.tensor([5.0])).item() == 1

    def test_typical_p_accepted(self):
        logits = torch.randn(2, 50)
        out = Sampler()(
            logits,
            temperatures=torch.tensor([1.0, 1.0]),
            typical_ps=torch.tensor([0.5, 1.0]),
        )
        assert out.shape == (2,)
        assert torch.all((out >= 0) & (out < 50))


class TestSeedReproducibility:
    def test_same_seed_same_sample(self):
        logits = torch.randn(1, 20)
        a = Sampler()._sample(logits, [torch.Generator().manual_seed(42)], None)
        b = Sampler()._sample(logits, [torch.Generator().manual_seed(42)], None)
        assert (a == b).all()

    def test_no_generators_uses_batch_path(self):
        logits = torch.randn(4, 20)
        out = Sampler()._sample(logits, None, None)
        assert out.shape == (4,)


class TestExecutorBuilders:
    def test_build_prev_tokens_padding(self):
        s1 = Sequence(token_ids=[1, 2, 3], sampling_params=SamplingParams())
        s2 = Sequence(token_ids=[4, 5], sampling_params=SamplingParams())
        prev = InferenceExecutor._build_prev_tokens([s1, s2], torch.device("cpu"))
        # [batch=2, max_len=3], shorter row -1 padded
        assert prev.shape == (2, 3)
        assert prev[0].tolist() == [1, 2, 3]
        assert prev[1].tolist() == [4, 5, -1]

    def test_build_generators_persistence(self):
        fake = type("Fake", (), {"_sample_generators": {}})()
        seeded = Sequence(
            token_ids=[1], sampling_params=SamplingParams(seed=7)
        )
        unseeded = Sequence(token_ids=[2], sampling_params=SamplingParams())
        dev = torch.device("cpu")

        gens = InferenceExecutor._build_generators(fake, [unseeded, seeded], dev)
        assert gens[0] is None and gens[1] is not None
        # second call reuses the same persistent generator for the same seq
        gens2 = InferenceExecutor._build_generators(fake, [unseeded, seeded], dev)
        assert gens2[1] is gens[1]

    def test_build_generators_all_unseeded_returns_none(self):
        fake = type("Fake", (), {"_sample_generators": {}})()
        unseeded = Sequence(token_ids=[1], sampling_params=SamplingParams())
        assert (
            InferenceExecutor._build_generators(
                fake, [unseeded], torch.device("cpu")
            )
            is None
        )


if __name__ == "__main__":
    pytest.main([__file__])
