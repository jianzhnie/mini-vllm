"""Tests for InferenceExecutor batch execution.

Focus: the prefill last-token logit selection, which must index by each
sequence's *contributed* (uncached) tokens, not its full length, and must
guarantee every prefill sequence yields at least one row to sample from.
"""

import json
from pathlib import Path

import pytest
import torch

from minivllm.config import Config
from minivllm.engine.inference_executor import InferenceExecutor
from minivllm.engine.sequence import Sequence
from minivllm.sampling_params import SamplingParams

V = 16


def _make_config(tmp_path: Path) -> Config:
    model_dir = tmp_path / "test_model"
    model_dir.mkdir()
    cfg = {
        "model_type": "llama",
        "hidden_size": V,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 256,
        "torch_dtype": "float32",
    }
    (model_dir / "config.json").write_text(json.dumps(cfg))
    return Config(str(model_dir), max_model_len=256, max_num_batched_tokens=256)


class _StubModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(V, V)

    def forward(self, *args, **kwargs):  # pragma: no cover - replaced by stub
        raise NotImplementedError


def _row_indexed_execute(ex: InferenceExecutor) -> None:
    """Replace _execute_model so that returned row i is [i, i, ..., i]."""

    def fake_execute(input_ids, positions, prefill):
        rows = input_ids.numel()
        idx = torch.arange(rows, dtype=torch.float32)
        return idx.unsqueeze(1).expand(rows, V).contiguous()

    ex._execute_model = fake_execute  # type: ignore[method-assign]


class TestPrefillLogitSelection:
    def test_selection_respects_prefix_cache(self, tmp_path):
        ex = InferenceExecutor(_make_config(tmp_path), _StubModel())
        _row_indexed_execute(ex)

        s1 = Sequence(token_ids=[1, 2, 3, 4, 5, 6], sampling_params=SamplingParams())
        s1.num_cached_tokens = 4  # 2 contributed rows (indices 0, 1)
        s2 = Sequence(token_ids=[7, 8, 9], sampling_params=SamplingParams())
        s2.num_cached_tokens = 0  # 3 contributed rows (indices 2, 3, 4)

        logits, _ = ex.execute_batch([s1, s2], prefill=True)

        assert logits.shape == (2, V)
        # s1 must sample its own last uncached row (index 1), not a full-length index.
        assert (logits[0] == 1).all()
        # s2's last row is index 4.
        assert (logits[1] == 4).all()

    def test_fully_cached_prefill_yields_one_row(self, tmp_path):
        ex = InferenceExecutor(_make_config(tmp_path), _StubModel())
        _row_indexed_execute(ex)

        # Prompt length is a multiple of nothing here, but set num_cached == len to
        # force the fully-cached edge: without the cap, zero rows would be built.
        s = Sequence(token_ids=[1, 2, 3, 4], sampling_params=SamplingParams())
        s.num_cached_tokens = 4

        logits, _ = ex.execute_batch([s], prefill=True)

        # Exactly the last token must still be processed -> one row (index 0).
        assert logits.shape[0] == 1
        assert (logits[0] == 0).all()

    def test_no_cache_hit_matches_previous_behavior(self, tmp_path):
        ex = InferenceExecutor(_make_config(tmp_path), _StubModel())
        _row_indexed_execute(ex)

        s1 = Sequence(token_ids=[1, 2, 3], sampling_params=SamplingParams())
        s2 = Sequence(token_ids=[4, 5], sampling_params=SamplingParams())

        logits, _ = ex.execute_batch([s1, s2], prefill=True)

        assert logits.shape == (2, V)
        assert (logits[0] == 2).all()  # s1 last row index 2
        assert (logits[1] == 4).all()  # s2 last row index 4


if __name__ == "__main__":
    pytest.main([__file__])
